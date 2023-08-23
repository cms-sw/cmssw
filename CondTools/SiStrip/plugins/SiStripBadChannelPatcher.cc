// system include files
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

// user include files
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/DataRecord/interface/SiStripBadStripRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h" /* for STRIPS_PER_APV*/
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace sistrip {
  static const uint16_t NOT_A_FEDID = static_cast<uint16_t>(FEDNumbering::NOT_A_FEDID);
}

class SiStripBadChannelPatcher : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripBadChannelPatcher(const edm::ParameterSet& iConfig)
      : record_(iConfig.getParameter<std::string>("Record")),
        printDebug_(iConfig.getParameter<bool>("printDebug")),
        detIdsToExclude_(iConfig.getParameter<std::vector<unsigned int>>("detIdsToExclude")),
        detIdsToInclude_(iConfig.getParameter<std::vector<unsigned int>>("detIdsToInclude")),
        fedsToExclude_(iConfig.getParameter<std::vector<unsigned int>>("FEDsToExclude")),
        fedsToInclude_(iConfig.getParameter<std::vector<unsigned int>>("FEDsToInclude")),
        badStripToken_(esConsumes()),
        cablingToken_(esConsumes()) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  ~SiStripBadChannelPatcher() override = default;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  std::unique_ptr<SiStripBadStrip> getNewObject(const edm::EventSetup& iSetup);
  unsigned int fedFromDetId(const uint32_t& detid);
  void addDetIdsFromExcludedFEDs(const std::vector<uint32_t>& allDetIds);

  // member data
  const SiStripDetCabling* detCabling_;
  const std::string record_;
  const bool printDebug_;
  std::vector<uint32_t> detIdsToExclude_;
  std::vector<uint32_t> detIdsToInclude_;
  std::vector<unsigned int> fedsToExclude_;
  std::vector<unsigned int> fedsToInclude_;
  const edm::ESGetToken<SiStripBadStrip, SiStripBadStripRcd> badStripToken_;    /*!< ES token for the bad strips */
  const edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> cablingToken_; /*!< ES token for the cabling */
};

//-----------------------------------------------------------------------------------------------//
void SiStripBadChannelPatcher::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment(
      "create a SiStripBadStrip payload starting from one in DB and excluding or adding entire modules from a list");
  desc.add<std::string>("Record", "SiStripBadStrip")->setComment("Record to match in the PoolDBOutputService");
  desc.add<bool>("printDebug", false)->setComment("full debug printout");
  desc.add<std::vector<unsigned int>>("detIdsToExclude", {})->setComment("list of detIds to exclude");
  desc.add<std::vector<unsigned int>>("detIdsToInclude", {})->setComment("list of detIds to include");
  desc.add<std::vector<unsigned int>>("FEDsToExclude", {})->setComment("list of FEDs to exclude");
  desc.add<std::vector<unsigned int>>("FEDsToInclude", {})->setComment("list of FEDs to include");
  descriptions.addWithDefaultLabel(desc);
}

//-----------------------------------------------------------------------------------------------//
unsigned int SiStripBadChannelPatcher::fedFromDetId(const uint32_t& detid) {
  // For the cabled det_id retrieve the FEDid
  const std::vector<const FedChannelConnection*>& conns = detCabling_->getConnections(detid);
  if (conns.empty()) {
    edm::LogWarning("SiStripBadChannelPatcher")
        << " DetId " << detid << " appears to be uncabled, returning NOT_A_FEDID !";
    return sistrip::NOT_A_FEDID;
  }
  unsigned int lFedId = sistrip::NOT_A_FEDID;
  for (uint32_t ch = 0; ch < conns.size(); ch++) {
    if (conns[ch] && conns[ch]->isConnected()) {
      lFedId = conns[ch]->fedId();
      LogDebug("SiStripBadChannelPatcher") << "obtained FED id " << ch << " " << lFedId;
      if (lFedId < sistrip::FED_ID_MIN || lFedId > sistrip::FED_ID_MAX) {
        edm::LogWarning("SiStripBadChannelPatcher") << lFedId << " for detid " << detid << " connection " << ch;
        continue;
      } else {
        break;
      }
    }
  }
  return lFedId;
}

//-----------------------------------------------------------------------------------------------//
void SiStripBadChannelPatcher::addDetIdsFromExcludedFEDs(const std::vector<uint32_t>& allDetIds) {
  for (const auto& detid : allDetIds) {
    const auto& currentFED = this->fedFromDetId(detid);
    if (std::count(fedsToInclude_.begin(), fedsToInclude_.end(), currentFED)) {
      detIdsToInclude_.push_back(detid);
    }
  }
}

//-----------------------------------------------------------------------------------------------//
void SiStripBadChannelPatcher::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // create the patched bad strips payload
  std::unique_ptr<SiStripBadStrip> theBadStrips = this->getNewObject(iSetup);

  // write out the BadStrip record
  edm::Service<cond::service::PoolDBOutputService> poolDbService;

  if (poolDbService.isAvailable()) {
    poolDbService->writeOneIOV(*theBadStrips, poolDbService->currentTime(), record_);
  } else {
    throw std::runtime_error("PoolDBService required.");
  }
}

//-----------------------------------------------------------------------------------------------//
std::unique_ptr<SiStripBadStrip> SiStripBadChannelPatcher::getNewObject(const edm::EventSetup& iSetup) {
  edm::LogInfo("SiStripBadChannelPatcher") << "... creating dummy SiStripBadStrip Data";

  // this is the output object
  auto obj = std::make_unique<SiStripBadStrip>();

  // get the cabling object
  detCabling_ = &iSetup.getData(cablingToken_);

  // get the strips to add
  std::vector<uint32_t> detids;
  const auto& payload = iSetup.getData(badStripToken_);
  payload.getDetIds(detids);

  // copy the exisiting channels (excluding the ones to remove)
  for (const auto& id : detids) {
    // check on the detids to exclude
    if (std::count(detIdsToExclude_.begin(), detIdsToExclude_.end(), id)) {
      edm::LogInfo("SiStripBadChannelPatcher") << "I AM GOING TO EXCLUDE DETID: " << id;
      continue;
    } else {
      LogDebug("SiStripBadChannelPatcher") << "I AM GOING TO KEEP DETID: " << id;
    }

    // check on the FEDs to exclude
    const auto& currentFED = this->fedFromDetId(id);
    if (std::count(fedsToExclude_.begin(), fedsToExclude_.end(), currentFED)) {
      edm::LogInfo("SiStripBadChannelPatcher") << "I AM GOING TO EXCLUDE DETID: " << id;
      continue;
    } else {
      LogDebug("SiStripBadChannelPatcher") << "I AM GOING TO KEEP DETID: " << id;
    }

    SiStripBadStrip::Range range = payload.getRange(id);
    std::vector<unsigned int> theSiStripVector;
    unsigned int theBadStripRange;
    for (std::vector<unsigned int>::const_iterator badStrip = range.first; badStrip != range.second; ++badStrip) {
      unsigned short firstBadStrip = payload.decode(*badStrip).firstStrip;
      unsigned short nConsecutiveBadStrips = payload.decode(*badStrip).range;
      theBadStripRange = obj->encode(firstBadStrip, nConsecutiveBadStrips);
      theSiStripVector.push_back(theBadStripRange);
    }
    SiStripBadStrip::Range outRange(theSiStripVector.begin(), theSiStripVector.end());
    if (!obj->put(id, outRange))
      edm::LogError("SiStripBadChannelPatcher") << "[SiStripBadChannelPatcher::analyze] detid already exists";
  }  // loop on the detids of the original payload

  const auto& detInfo =
      SiStripDetInfoFileReader::read(edm::FileInPath{SiStripDetInfoFileReader::kDefaultFile}.fullPath());

  // add to the list of DetIds to include also the one from the list of FEDs
  const std::vector<uint32_t>& allDetIds = detInfo.getAllDetIds();
  this->addDetIdsFromExcludedFEDs(allDetIds);

  // add more full bad detids
  if (!detIdsToInclude_.empty()) {
    edm::LogInfo("SiStripBadChannelPatcher") << "I AM GOING TO ADD MORE DETIDS";

    std::stringstream ss;
    for (const auto& detid : detIdsToInclude_) {
      edm::LogInfo("SiStripBadChannelPatcher") << "I AM GOING TO ADD DETID: " << detid;
      const auto& nAPVs = detInfo.getNumberOfApvsAndStripLength(detid).first;

      std::vector<unsigned int> theSiStripVector;
      unsigned short firstBadStrip{0}, nConsecutiveBadStrips{0};
      unsigned int theBadStripRange;

      for (unsigned int n = 0; n < nAPVs; n++) {
        firstBadStrip = n * sistrip::STRIPS_PER_APV;
        nConsecutiveBadStrips = sistrip::STRIPS_PER_APV;
        theBadStripRange = obj->encode(firstBadStrip, nConsecutiveBadStrips);

        if (printDebug_) {
          ss << "detid " << detid << " \t"
             << " firstBadStrip " << firstBadStrip << "\t "
             << " nConsecutiveBadStrips " << nConsecutiveBadStrips << "\t "
             << " packed integer " << std::hex << theBadStripRange << std::dec << std::endl;
        }

        theSiStripVector.push_back(theBadStripRange);
      }

      SiStripBadStrip::Range outRange(theSiStripVector.begin(), theSiStripVector.end());
      if (!obj->put(detid, outRange))
        edm::LogError("SiStripBadChannelPatcher") << "[SiStripBadChannelPatcher::analyze] detid already exists";
    }  // loop on detids to include

    if (printDebug_) {
      // print the added strips
      edm::LogInfo("SiStripBadChannelPatcher") << ss.str();
    }

  }  // if there is any new channel to append

  return obj;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripBadChannelPatcher);
