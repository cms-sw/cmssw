/** \file
 *
 *  \author N. Amapane - CERN
 */

#include "DaqFakeReader.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/TCDS/interface/TCDSRaw.h"

#include "EventFilter/Utilities/interface/GlobalEventNumber.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CLHEP/Random/RandGauss.h"

#include <cmath>
#include <sys/time.h>
#include <cstring>
#include <cstdlib>
#include <chrono>

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
      fillRandom_(pset.getUntrackedParameter<bool>("fillRandom", false)),
      meansize(pset.getUntrackedParameter<unsigned int>("meanSize", 1024)),
      width(pset.getUntrackedParameter<unsigned int>("width", 1024)),
      injected_errors_per_million_events(pset.getUntrackedParameter<unsigned int>("injectErrPpm", 0)),
      tcdsFEDID_(pset.getUntrackedParameter<unsigned int>("tcdsFEDID", 1024)),
      modulo_error_events(injected_errors_per_million_events ? 1000000 / injected_errors_per_million_events
                                                             : 0xffffffff),
      subsystems_(pset.getUntrackedParameter<std::vector<std::string>>("subsystems")) {
  for (auto const& subsystem : subsystems_) {
    if (subsystem == "TCDS")
      haveTCDS_ = true;
    else if (subsystem == "SiPixel")
      haveSiPixel_ = true;
    else if (subsystem == "SiStrip")
      haveSiStrip_ = true;
    else if (subsystem == "ECAL")
      haveECAL_ = true;
    else if (subsystem == "HCAL")
      haveHCAL_ = true;
    else if (subsystem == "DT")
      haveDT_ = true;
    else if (subsystem == "CSC")
      haveCSC_ = true;
    else if (subsystem == "RPC")
      haveRPC_ = true;
  }
  // mean = pset.getParameter<float>("mean");
  if (haveTCDS_ && tcdsFEDID_ < FEDNumbering::MINTCDSuTCAFEDID)
    throw cms::Exception("DaqFakeReader::DaqFakeReader")
        << " TCDS FED ID lower than " << FEDNumbering::MINTCDSuTCAFEDID;
  if (fillRandom_) {
    //intialize random seed
    auto time_count =
        static_cast<long unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    srand(time_count & 0xffffffff);
  }
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
  auto ls = e.luminosityBlock();

  if (!empty_events) {
    // Fill the EventID
    eventNum++;
    // FIXME:

    if (haveSiPixel_)
      fillFEDs(FEDNumbering::MINSiPixelFEDID, FEDNumbering::MAXSiPixelFEDID, eID, *data, meansize, width);
    if (haveSiStrip_)
      fillFEDs(FEDNumbering::MINSiStripFEDID, FEDNumbering::MAXSiStripFEDID, eID, *data, meansize, width);
    if (haveECAL_)
      fillFEDs(FEDNumbering::MINECALFEDID, FEDNumbering::MAXECALFEDID, eID, *data, meansize, width);
    if (haveHCAL_)
      fillFEDs(FEDNumbering::MINHCALFEDID, FEDNumbering::MAXHCALFEDID, eID, *data, meansize, width);
    if (haveDT_)
      fillFEDs(FEDNumbering::MINDTFEDID, FEDNumbering::MAXDTFEDID, eID, *data, meansize, width);
    if (haveCSC_)
      fillFEDs(FEDNumbering::MINCSCFEDID, FEDNumbering::MAXCSCFEDID, eID, *data, meansize, width);
    if (haveRPC_)
      fillFEDs(FEDNumbering::MINRPCFEDID, FEDNumbering::MAXRPCFEDID, eID, *data, meansize, width);

    timeval now;
    gettimeofday(&now, nullptr);
    if (haveTCDS_)
      fillTCDSFED(eID, *data, ls, &now);
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

    if (fillRandom_) {
      //fill FED with random values
      size_t size_ui = size - size % sizeof(unsigned int);
      for (size_t i = 0; i < size_ui; i += sizeof(unsigned int)) {
        *((unsigned int*)(feddata.data() + i)) = (unsigned int)rand();
      }
      //remainder
      for (size_t i = size_ui; i < size; i++) {
        *(feddata.data() + i) = rand() & 0xff;
      }
    }

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

void DaqFakeReader::fillTCDSFED(EventID& eID, FEDRawDataCollection& data, uint32_t ls, timeval* now) {
  uint32_t fedId = tcdsFEDID_;
  FEDRawData& feddata = data.FEDData(fedId);
  uint32_t size = sizeof(tcds::Raw_v1);
  feddata.resize(size + 16);

  uint64_t orbitnr = 0;
  uint16_t bxid = 0;

  FEDHeader::set(feddata.data(),
                 1,            // Trigger type
                 eID.event(),  // LV1_id (24 bits)
                 bxid,         // BX_id
                 fedId);       // source_id

  tcds::Raw_v1* tcds = reinterpret_cast<tcds::Raw_v1*>(feddata.data() + FEDHeader::length);
  tcds::BST_v1* bst = const_cast<tcds::BST_v1*>(&tcds->bst);
  tcds::Header_v1* header = const_cast<tcds::Header_v1*>(&tcds->header);

  const_cast<uint32_t&>(bst->gpstimehigh) = now->tv_sec;
  const_cast<uint32_t&>(bst->gpstimelow) = now->tv_usec;
  const_cast<uint16_t&>(bst->lhcFillHigh) = 0;
  const_cast<uint16_t&>(bst->lhcFillLow) = 0;

  const_cast<uint32_t&>(header->orbitHigh) = orbitnr & 0xffff00;
  const_cast<uint16_t&>(header->orbitLow) = orbitnr & 0xff;
  const_cast<uint16_t&>(header->bxid) = bxid;

  const_cast<uint64_t&>(header->eventNumber) = eID.event();
  const_cast<uint32_t&>(header->lumiSection) = ls;

  int crc = 0;  // only full event crc32c checked in HLT, not FED CRC16
  FEDTrailer::set(feddata.data() + 8 + size,
                  size / 8 + 2,  // in 64 bit words!!!
                  crc,
                  0,   // Evt_stat
                  0);  // TTS bits
}

void DaqFakeReader::beginLuminosityBlock(LuminosityBlock const& iL, EventSetup const& iE) {
  std::cout << "DaqFakeReader begin Lumi " << iL.luminosityBlock() << std::endl;
  fakeLs_ = iL.luminosityBlock();
}

void DaqFakeReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Injector of generated raw FED data for DAQ testing");
  desc.addUntracked<bool>("emptyEvents", false);
  desc.addUntracked<bool>("fillRandom", false);
  desc.addUntracked<unsigned int>("meanSize", 1024);
  desc.addUntracked<unsigned int>("width", 1024);
  desc.addUntracked<unsigned int>("injectErrPpm", 1024);
  desc.addUntracked<unsigned int>("tcdsFEDID", 1024);
  desc.addUntracked<std::vector<std::string>>(
      "subsystems",
      std::initializer_list<std::string>({"TCDS", "SiPixel", "SiStrip", "ECAL", "HCAL", "DT", "CSC", "RPC"}));
  descriptions.add("DaqFakeReader", desc);
}
