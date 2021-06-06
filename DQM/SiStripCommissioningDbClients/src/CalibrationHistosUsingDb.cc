#include "DQM/SiStripCommissioningDbClients/interface/CalibrationHistosUsingDb.h"
#include "CondFormats/SiStripObjects/interface/CalibrationAnalysis.h"
#include "CondFormats/SiStripObjects/interface/CalibrationScanAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
std::string getBasePath(const std::string& path) {
  return path.substr(0, path.find(std::string(sistrip::root_) + "/") + sizeof(sistrip::root_));
}

// -----------------------------------------------------------------------------
/** */
CalibrationHistosUsingDb::CalibrationHistosUsingDb(const edm::ParameterSet& pset,
                                                   DQMStore* bei,
                                                   SiStripConfigDb* const db,
                                                   edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken,
                                                   const sistrip::RunType& task)
    : CommissioningHistograms(pset.getParameter<edm::ParameterSet>("CalibrationParameters"), bei, task),
      CommissioningHistosUsingDb(db, tTopoToken, task),
      CalibrationHistograms(pset.getParameter<edm::ParameterSet>("CalibrationParameters"), bei, task) {
  LogTrace(mlDqmClient_) << "[CalibrationHistosUsingDb::" << __func__ << "]"
                         << " Constructing object...";

  // Load and dump the current ISHA/VFS values. This is used by the standalone analysis script
  const SiStripConfigDb::DeviceDescriptionsRange& apvDescriptions = db->getDeviceDescriptions(APV25);
  for (SiStripConfigDb::DeviceDescriptionsV::const_iterator apv = apvDescriptions.begin(); apv != apvDescriptions.end();
       ++apv) {
    apvDescription* desc = dynamic_cast<apvDescription*>(*apv);
    if (!desc) {
      continue;
    }
    // Retrieve device addresses from device description
    const SiStripConfigDb::DeviceAddress& addr = db->deviceAddress(*desc);
    std::stringstream bin;
    bin << std::setw(1) << std::setfill('0') << addr.fecCrate_;
    bin << "." << std::setw(2) << std::setfill('0') << addr.fecSlot_;
    bin << "." << std::setw(1) << std::setfill('0') << addr.fecRing_;
    bin << "." << std::setw(3) << std::setfill('0') << addr.ccuAddr_;
    bin << "." << std::setw(2) << std::setfill('0') << addr.ccuChan_;
    bin << "." << desc->getAddress();
    LogTrace(mlDqmClient_) << "Present values for ISHA/VFS of APV " << bin.str() << " : "
                           << static_cast<uint16_t>(desc->getIsha()) << " " << static_cast<uint16_t>(desc->getVfs());
  }

  allowSelectiveUpload_ =
      this->pset().existsAs<bool>("doSelectiveUpload") ? this->pset().getParameter<bool>("doSelectiveUpload") : false;
  if (allowSelectiveUpload_)
    LogTrace(mlDqmClient_) << "[CalibrationHistosUsingDb::" << __func__ << "]"
                           << " Enabling selective update of FED parameters.";
}

// -----------------------------------------------------------------------------
/** */
CalibrationHistosUsingDb::~CalibrationHistosUsingDb() {
  LogTrace(mlDqmClient_) << "[CalibrationHistosUsingDb::" << __func__ << "]"
                         << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
void CalibrationHistosUsingDb::uploadConfigurations() {
  if (!db()) {
    edm::LogWarning(mlDqmClient_) << "[CalibrationHistosUsingDb::" << __func__ << "]"
                                  << " NULL pointer to SiStripConfigDb interface!"
                                  << " Aborting upload...";
    return;
  }

  if (task() == sistrip::CALIBRATION or task() == sistrip::CALIBRATION_DECO) {
    edm::LogWarning(mlDqmClient_)
        << "[CalibrationHistosUsingDb::" << __func__ << "]"
        << " Nothing has to be uploaded to the SiStripConfigDb for CALIBRATION_SCAN or CALIBRATION_SCAN_DECO run-types"
        << " Aborting upload...";
    return;
  } else if (task() == sistrip::CALIBRATION_SCAN or task() == sistrip::CALIBRATION_SCAN_DECO) {
    // Update all APV device descriptions with new ISHA and VFS settings
    SiStripConfigDb::DeviceDescriptionsRange devices = db()->getDeviceDescriptions(APV25);
    update(devices);
    if (doUploadConf()) {
      edm::LogVerbatim(mlDqmClient_) << "[CalibrationHistosUsingDb::" << __func__ << "]"
                                     << " Uploading ISHA/VFS settings to DB...";
      db()->uploadDeviceDescriptions();
      edm::LogVerbatim(mlDqmClient_) << "[CalibrationHistosUsingDb::" << __func__ << "]"
                                     << " Uploaded ISHA/VFS settings to DB!";
    } else {
      edm::LogWarning(mlDqmClient_) << "[CalibrationHistosUsingDb::" << __func__ << "]"
                                    << " TEST only! No ISHA/VFS settings will be uploaded to DB...";
    }

    LogTrace(mlDqmClient_) << "[CalibrationHistosUsingDb::" << __func__ << "]"
                           << " Upload of ISHA/VFS settings to DB finished!";
  }
}

// -----------------------------------------------------------------------------
void CalibrationHistosUsingDb::update(SiStripConfigDb::DeviceDescriptionsRange& devices) {
  if (task() == sistrip::CALIBRATION or task() == sistrip::CALIBRATION_DECO) {
    edm::LogWarning(mlDqmClient_)
        << "[CalibrationHistosUsingDb::" << __func__ << "]"
        << " Nothing has to be uploaded to the SiStripConfigDb for CALIBRATION_SCAN or CALIBRATION_SCAN_DECO run-type"
        << " Aborting upload...";
    return;
  } else if (task() == sistrip::CALIBRATION_SCAN or task() == sistrip::CALIBRATION_SCAN_DECO) {
    // Iterate through devices and update device descriptions
    SiStripConfigDb::DeviceDescriptionsV::const_iterator idevice;
    for (idevice = devices.begin(); idevice != devices.end(); idevice++) {
      // Check device type
      if ((*idevice)->getDeviceType() != APV25) {
        continue;
      }
      // Cast to retrieve appropriate description object
      apvDescription* desc = dynamic_cast<apvDescription*>(*idevice);
      if (!desc) {
        continue;
      }
      // Retrieve the device address from device description
      const SiStripConfigDb::DeviceAddress& addr = db()->deviceAddress(*desc);

      // Retrieve LLD channel and APV numbers
      uint16_t ichan = (desc->getAddress() - 0x20) / 2;
      uint16_t iapv = (desc->getAddress() - 0x20) % 2;

      // Construct key from device description
      SiStripFecKey fec_key(addr.fecCrate_, addr.fecSlot_, addr.fecRing_, addr.ccuAddr_, addr.ccuChan_, ichan + 1);

      // Iterate through all channels and extract LLD settings
      Analyses::const_iterator iter = data(allowSelectiveUpload_).find(fec_key.key());
      if (iter != data(allowSelectiveUpload_).end()) {
        CalibrationScanAnalysis* anal = dynamic_cast<CalibrationScanAnalysis*>(iter->second);

        if (!anal) {
          edm::LogError(mlDqmClient_) << "[CalibrationHistosUsingDb::" << __func__ << "]"
                                      << " NULL pointer to analysis object!";
          continue;
        }

        std::stringstream ss;
        ss << "[CalibrationHistosUsingDb::" << __func__ << "]"
           << " Updating ISHA and VFS setting for FECCrate/FECSlot/FECRing/CCUAddr/LLD/APV " << fec_key.fecCrate()
           << "/" << fec_key.fecSlot() << "/" << fec_key.fecRing() << "/" << fec_key.ccuAddr() << "/"
           << fec_key.ccuChan() << "/" << fec_key.channel() << iapv << " from ISHA "
           << static_cast<uint16_t>(desc->getIsha()) << " and VFS " << static_cast<uint16_t>(desc->getVfs());
        if (iapv == 0) {
          desc->setIsha(anal->bestISHA()[0]);
        }
        if (iapv == 1) {
          desc->setIsha(anal->bestISHA()[1]);
        }
        if (iapv == 0) {
          desc->setVfs(anal->bestVFS()[0]);
        }
        if (iapv == 1) {
          desc->setVfs(anal->bestVFS()[1]);
        }
        ss << " to ISHA " << static_cast<uint16_t>(desc->getIsha()) << " and VFS "
           << static_cast<uint16_t>(desc->getVfs());
        edm::LogWarning(mlDqmClient_) << ss.str();
      } else {
        if (deviceIsPresent(fec_key)) {
          edm::LogWarning(mlDqmClient_) << "[CalibrationHistosUsingDb::" << __func__ << "]"
                                        << " Unable to find FEC key with params FEC/slot/ring/CCU/LLDchan/APV: "
                                        << fec_key.fecCrate() << "/" << fec_key.fecSlot() << "/" << fec_key.fecRing()
                                        << "/" << fec_key.ccuAddr() << "/" << fec_key.ccuChan() << "/"
                                        << fec_key.channel() << "/" << iapv + 1;
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
void CalibrationHistosUsingDb::create(SiStripConfigDb::AnalysisDescriptionsV& desc, Analysis analysis) {
  if (task() == sistrip::CALIBRATION or
      task() == sistrip::CALIBRATION_DECO) {  // calibration run --> pulse shape measurement

    CalibrationAnalysis* anal = dynamic_cast<CalibrationAnalysis*>(analysis->second);
    if (!anal) {
      return;
    }

    SiStripFecKey fec_key(anal->fecKey());
    SiStripFedKey fed_key(anal->fedKey());

    for (uint16_t iapv = 0; iapv < 2; ++iapv) {
      // Create description table with placeholder values for isha and vfs
      CalibrationAnalysisDescription* tmp;
      tmp = new CalibrationAnalysisDescription(anal->amplitudeMean()[iapv],
                                               anal->tailMean()[iapv],
                                               anal->riseTimeMean()[iapv],
                                               anal->decayTimeMean()[iapv],
                                               anal->smearingMean()[iapv],
                                               anal->chi2Mean()[iapv],
                                               anal->deconvMode(),
                                               fec_key.fecCrate(),
                                               fec_key.fecSlot(),
                                               fec_key.fecRing(),
                                               fec_key.ccuAddr(),
                                               fec_key.ccuChan(),
                                               SiStripFecKey::i2cAddr(fec_key.lldChan(), !iapv),
                                               db()->dbParams().partitions().begin()->second.partitionName(),
                                               db()->dbParams().partitions().begin()->second.runNumber(),
                                               anal->isValid(),
                                               "",
                                               fed_key.fedId(),
                                               fed_key.feUnit(),
                                               fed_key.feChan(),
                                               fed_key.fedApv(),
                                               anal->calChan(),
                                               -1,
                                               -1);

      // Add comments
      typedef std::vector<std::string> Strings;
      Strings errors = anal->getErrorCodes();
      Strings::const_iterator istr = errors.begin();
      Strings::const_iterator jstr = errors.end();
      for (; istr != jstr; ++istr) {
        tmp->addComments(*istr);
      }
      // Store description
      desc.push_back(tmp);
    }
  } else if (task() == sistrip::CALIBRATION_SCAN or task() == sistrip::CALIBRATION_SCAN_DECO) {
    CalibrationScanAnalysis* anal = dynamic_cast<CalibrationScanAnalysis*>(analysis->second);
    if (!anal) {
      return;
    }

    SiStripFecKey fec_key(anal->fecKey());
    SiStripFedKey fed_key(anal->fedKey());

    for (uint16_t iapv = 0; iapv < 2; ++iapv) {
      // Create description table with placeholder values for isha and vfs
      CalibrationAnalysisDescription* tmp;
      tmp = new CalibrationAnalysisDescription(anal->tunedAmplitude()[iapv],
                                               anal->tunedTail()[iapv],
                                               anal->tunedRiseTime()[iapv],
                                               anal->tunedDecayTime()[iapv],
                                               anal->tunedSmearing()[iapv],
                                               anal->tunedChi2()[iapv],
                                               anal->deconvMode(),
                                               fec_key.fecCrate(),
                                               fec_key.fecSlot(),
                                               fec_key.fecRing(),
                                               fec_key.ccuAddr(),
                                               fec_key.ccuChan(),
                                               SiStripFecKey::i2cAddr(fec_key.lldChan(), !iapv),
                                               db()->dbParams().partitions().begin()->second.partitionName(),
                                               db()->dbParams().partitions().begin()->second.runNumber(),
                                               anal->isValid(),
                                               "",
                                               fed_key.fedId(),
                                               fed_key.feUnit(),
                                               fed_key.feChan(),
                                               fed_key.fedApv(),
                                               -1,
                                               anal->tunedISHA()[iapv],
                                               anal->tunedVFS()[iapv]);

      // Add comments
      typedef std::vector<std::string> Strings;
      Strings errors = anal->getErrorCodes();
      Strings::const_iterator istr = errors.begin();
      Strings::const_iterator jstr = errors.end();
      for (; istr != jstr; ++istr) {
        tmp->addComments(*istr);
      }
      // Store description
      desc.push_back(tmp);
    }
  }
}
