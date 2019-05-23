/** \file
 *
 *  \author N. Amapane - CERN
 */

#include "DaqFakeReader.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "EventFilter/Utilities/interface/GlobalEventNumber.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CLHEP/Random/RandGauss.h"

#include <cmath>
#include <sys/time.h>
#include <cstring>

using namespace std;
using namespace edm;

////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
DaqFakeReader::DaqFakeReader(const edm::ParameterSet& pset)
    : runNum(1),
      eventNum(1),
      empty_events(pset.getUntrackedParameter<bool>("emptyEvents", false)),
      meansize(pset.getUntrackedParameter<unsigned int>("meanSize", 1024)),
      width(pset.getUntrackedParameter<unsigned int>("width", 1024)),
      injected_errors_per_million_events(pset.getUntrackedParameter<unsigned int>("injectErrPpm", 0)),
      modulo_error_events(injected_errors_per_million_events ? 1000000 / injected_errors_per_million_events
                                                             : 0xffffffff) {
  // mean = pset.getParameter<float>("mean");
  produces<FEDRawDataCollection>();
}

//______________________________________________________________________________
DaqFakeReader::~DaqFakeReader() {}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
int DaqFakeReader::fillRawData(Event& e, FEDRawDataCollection*& data) {
  // a null pointer is passed, need to allocate the fed collection
  data = new FEDRawDataCollection();
  EventID eID = e.id();

  if (!empty_events) {
    // Fill the EventID
    eventNum++;
    // FIXME:

    fillFEDs(FEDNumbering::MINSiPixelFEDID, FEDNumbering::MAXSiPixelFEDID, eID, *data, meansize, width);
    fillFEDs(FEDNumbering::MINSiStripFEDID, FEDNumbering::MAXSiStripFEDID, eID, *data, meansize, width);
    fillFEDs(FEDNumbering::MINDTFEDID, FEDNumbering::MAXDTFEDID, eID, *data, meansize, width);
    fillFEDs(FEDNumbering::MINCSCFEDID, FEDNumbering::MAXCSCFEDID, eID, *data, meansize, width);
    fillFEDs(FEDNumbering::MINRPCFEDID, FEDNumbering::MAXRPCFEDID, eID, *data, meansize, width);
    fillFEDs(FEDNumbering::MINECALFEDID, FEDNumbering::MAXECALFEDID, eID, *data, meansize, width);
    fillFEDs(FEDNumbering::MINHCALFEDID, FEDNumbering::MAXHCALFEDID, eID, *data, meansize, width);

    timeval now;
    gettimeofday(&now, nullptr);
    fillGTPFED(eID, *data, &now);
    //TODO: write fake TCDS FED filler
  }
  return 1;
}

void DaqFakeReader::produce(Event& e, EventSetup const& es) {
  edm::Handle<FEDRawDataCollection> rawdata;
  FEDRawDataCollection* fedcoll = nullptr;
  fillRawData(e, fedcoll);
  std::unique_ptr<FEDRawDataCollection> bare_product(fedcoll);
  e.put(std::move(bare_product));
}

//______________________________________________________________________________
void DaqFakeReader::fillFEDs(
    const int fedmin, const int fedmax, EventID& eID, FEDRawDataCollection& data, float meansize, float width) {
  // FIXME: last ID included?
  for (int fedId = fedmin; fedId <= fedmax; ++fedId) {
    // Generate size...
    float logsiz = CLHEP::RandGauss::shoot(std::log(meansize), std::log(meansize) - std::log(width / 2.));
    size_t size = int(std::exp(logsiz));
    size -= size % 8;  // all blocks aligned to 64 bit words

    FEDRawData& feddata = data.FEDData(fedId);
    // Allocate space for header+trailer+payload
    feddata.resize(size + 16);

    // Generate header
    FEDHeader::set(feddata.data(),
                   1,            // Trigger type
                   eID.event(),  // LV1_id (24 bits)
                   0,            // BX_id
                   fedId);       // source_id

    // Payload = all 0s...

    // Generate trailer
    int crc = 0;  // FIXME : get CRC
    FEDTrailer::set(feddata.data() + 8 + size,
                    size / 8 + 2,  // in 64 bit words!!!
                    crc,
                    0,   // Evt_stat
                    0);  // TTS bits
  }
}

void DaqFakeReader::fillGTPFED(EventID& eID, FEDRawDataCollection& data, timeval* now) {
  uint32_t fedId = FEDNumbering::MINTriggerGTPFEDID;
  FEDRawData& feddata = data.FEDData(fedId);
  uint32_t size = evf::evtn::SLINK_WORD_SIZE * 37 - 16;  //BST52_3BX
  feddata.resize(size + 16);

  FEDHeader::set(feddata.data(),
                 1,            // Trigger type
                 eID.event(),  // LV1_id (24 bits)
                 0,            // BX_id
                 fedId);       // source_id

  int crc = 0;  // FIXME : get CRC
  FEDTrailer::set(feddata.data() + 8 + size,
                  size / 8 + 2,  // in 64 bit words!!!
                  crc,
                  0,   // Evt_stat
                  0);  // TTS bits

  unsigned char* pOffset = feddata.data() + FEDHeader::length;
  //fill in event ID
  *((uint32_t*)(pOffset + evf::evtn::EVM_BOARDID_OFFSET * evf::evtn::SLINK_WORD_SIZE / 2)) =
      evf::evtn::EVM_BOARDID_VALUE << evf::evtn::EVM_BOARDID_SHIFT;
  *((uint32_t*)(pOffset + FEDHeader::length +
                (9 * 2 + evf::evtn::EVM_TCS_TRIGNR_OFFSET) * evf::evtn::SLINK_WORD_SIZE / 2)) = eID.event();
  //fill in timestamp
  *((uint32_t*)(pOffset + evf::evtn::EVM_GTFE_BSTGPS_OFFSET * evf::evtn::SLINK_WORD_SIZE / 2)) = now->tv_sec;
  *((uint32_t*)(pOffset + FEDHeader::length + evf::evtn::EVM_GTFE_BSTGPS_OFFSET * evf::evtn::SLINK_WORD_SIZE / 2 +
                evf::evtn::SLINK_HALFWORD_SIZE)) = now->tv_usec;

  //*( (uint16_t*) (pOffset + (evtn::EVM_GTFE_BLOCK*2 + evtn::EVM_TCS_LSBLNR_OFFSET)*evtn::SLINK_HALFWORD_SIZE)) = (unsigned short)fakeLs_-1;

  //we could also generate lumiblock, bcr, orbit,... but they are not currently used by the FRD input source
}

void DaqFakeReader::beginLuminosityBlock(LuminosityBlock const& iL, EventSetup const& iE) {
  std::cout << "DaqFakeReader begin Lumi " << iL.luminosityBlock() << std::endl;
  fakeLs_ = iL.luminosityBlock();
}
