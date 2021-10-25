#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripFedCablingFakeESSource.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripModule.h"
#include "CalibTracker/Records/interface/SiStripHashedDetIdRcd.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CalibTracker/SiStripCommon/interface/SiStripFedIdListReader.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>
#include <vector>
#include <map>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripFedCablingFakeESSource::SiStripFedCablingFakeESSource(const edm::ParameterSet& pset)
    : SiStripFedCablingESProducer(pset), fedIds_(pset.getParameter<edm::FileInPath>("FedIdsFile")), pset_(pset) {
  findingRecord<SiStripFedCablingRcd>();
  m_detInfo = SiStripDetInfoFileReader::read(pset.getParameter<edm::FileInPath>("SiStripDetInfoFile").fullPath());
  edm::LogVerbatim("FedCabling") << "[SiStripFedCablingFakeESSource::" << __func__ << "]"
                                 << " Constructing object...";
}

// -----------------------------------------------------------------------------
//
SiStripFedCablingFakeESSource::~SiStripFedCablingFakeESSource() {
  edm::LogVerbatim("FedCabling") << "[SiStripFedCablingFakeESSource::" << __func__ << "]"
                                 << " Destructing object...";
}

// -----------------------------------------------------------------------------
//
SiStripFedCabling* SiStripFedCablingFakeESSource::make(const SiStripFedCablingRcd&) {
  edm::LogVerbatim("FedCabling") << "[SiStripFedCablingFakeESSource::" << __func__ << "]"
                                 << " Building \"fake\" FED cabling map"
                                 << " from real DetIds and FedIds (read from ascii file)";

  // Create FEC cabling object
  SiStripFecCabling* fec_cabling = new SiStripFecCabling();

  // Read DetId list from file
  typedef std::vector<uint32_t> Dets;
  Dets dets = m_detInfo.getAllDetIds();

  // Read FedId list from file
  typedef std::vector<uint16_t> Feds;
  Feds feds = SiStripFedIdListReader(fedIds_.fullPath()).fedIds();

  bool populateAllFeds = pset_.getParameter<bool>("PopulateAllFeds");

  // Iterator through DetInfo objects and populate FEC cabling object
  uint32_t imodule = 0;
  Dets::const_iterator idet = dets.begin();
  Dets::const_iterator jdet = dets.end();
  for (; idet != jdet; ++idet) {
    uint16_t npairs = m_detInfo.getNumberOfApvsAndStripLength(*idet).first / 2;
    for (uint16_t ipair = 0; ipair < npairs; ++ipair) {
      uint16_t addr = 0;
      if (npairs == 2 && ipair == 0) {
        addr = 32;
      } else if (npairs == 2 && ipair == 1) {
        addr = 36;
      } else if (npairs == 3 && ipair == 0) {
        addr = 32;
      } else if (npairs == 3 && ipair == 1) {
        addr = 34;
      } else if (npairs == 3 && ipair == 2) {
        addr = 36;
      } else {
        edm::LogWarning("FedCabling") << "[SiStripFedCablingFakeESSource::" << __func__ << "]"
                                      << " Inconsistent values for nPairs (" << npairs << ") and ipair (" << ipair
                                      << ")!";
      }
      uint32_t module_key =
          SiStripFecKey(fecCrate(imodule), fecSlot(imodule), fecRing(imodule), ccuAddr(imodule), ccuChan(imodule)).key();
      FedChannelConnection conn(fecCrate(imodule),
                                fecSlot(imodule),
                                fecRing(imodule),
                                ccuAddr(imodule),
                                ccuChan(imodule),
                                addr,
                                addr + 1,    // apv i2c addresses
                                module_key,  // dcu id
                                *idet,       // det id
                                npairs);     // apv pairs
      fec_cabling->addDevices(conn);
    }
    imodule++;
  }

  // Assign "dummy" FED ids/chans
  bool insufficient = false;
  Feds::const_iterator ifed = feds.begin();
  uint16_t fed_ch = 0;
  for (std::vector<SiStripFecCrate>::const_iterator icrate = fec_cabling->crates().begin();
       icrate != fec_cabling->crates().end();
       icrate++) {
    for (std::vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++) {
      for (std::vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end();
           iring++) {
        for (std::vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end();
             iccu++) {
          for (std::vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end();
               imod++) {
            if (populateAllFeds) {
              for (uint16_t ipair = 0; ipair < imod->nApvPairs(); ipair++) {
                if (ifed == feds.end()) {
                  fed_ch++;
                  ifed = feds.begin();
                }
                if (fed_ch == 96) {
                  insufficient = true;
                  break;
                }

                std::pair<uint16_t, uint16_t> addr = imod->activeApvPair(imod->lldChannel(ipair));
                SiStripModule::FedChannel fed_channel((*ifed) / 16 + 1,  // 16 FEDs per crate, numbering starts from 1
                                                      (*ifed) % 16 + 2,  // FED slot starts from 2
                                                      *ifed,
                                                      fed_ch);
                const_cast<SiStripModule&>(*imod).fedCh(addr.first, fed_channel);
                ifed++;
              }
            } else {
              // Patch introduced by D.Giordano 2/12/08
              //to reproduce the fake cabling used in 2x
              //that was designed to fill each fed iteratively
              //filling all channels of a fed before going to the next one
              if (96 - fed_ch < imod->nApvPairs()) {
                ifed++;
                fed_ch = 0;
              }  // move to next FED
              for (uint16_t ipair = 0; ipair < imod->nApvPairs(); ipair++) {
                std::pair<uint16_t, uint16_t> addr = imod->activeApvPair((*imod).lldChannel(ipair));
                SiStripModule::FedChannel fed_channel((*ifed) / 16 + 1,  // 16 FEDs per crate, numbering starts from 1
                                                      (*ifed) % 16 + 2,  // FED slot starts from 2
                                                      (*ifed),
                                                      fed_ch);
                const_cast<SiStripModule&>(*imod).fedCh(addr.first, fed_channel);
                fed_ch++;
              }
            }
          }
        }
      }
    }
  }

  if (insufficient) {
    edm::LogWarning(mlCabling_) << "[SiStripFedCablingFakeESSource::" << __func__ << "]"
                                << " Insufficient FED channels to cable entire system!";
  }

  // Some debug
  std::stringstream ss;
  ss << "[SiStripFedCablingFakeESSource::" << __func__ << "]"
     << " First count devices of FEC cabling " << std::endl;
  fec_cabling->countDevices().print(ss);
  LogTrace(mlCabling_) << ss.str();

  // Build FED cabling using FedChannelConnections
  std::vector<FedChannelConnection> conns;
  fec_cabling->connections(conns);
  SiStripFedCabling* cabling = new SiStripFedCabling(conns);

  return cabling;
}

// -----------------------------------------------------------------------------
//
void SiStripFedCablingFakeESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& key,
                                                   const edm::IOVSyncValue& iov_sync,
                                                   edm::ValidityInterval& iov_validity) {
  edm::ValidityInterval infinity(iov_sync.beginOfTime(), iov_sync.endOfTime());
  iov_validity = infinity;
}
