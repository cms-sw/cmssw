// system includes
#include <iostream>
#include <fstream>

// user includes
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h" /* for STRIPS_PER_APV*/
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

class SiStripNoisesAndBadCompsChecker : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SiStripNoisesAndBadCompsChecker(const edm::ParameterSet& iConfig);

  ~SiStripNoisesAndBadCompsChecker() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const bool writePayload_;
  const edm::FileInPath fp_;
  const uint32_t printdebug_;
  const edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noiseToken_;
  const edm::ESGetToken<SiStripBadStrip, SiStripBadChannelRcd> deadChannelToken_;

  const std::string k_Name_ = "SiStripNoisesAndBadCompsChecker";
  const std::string k_Record_ = "SiStripNoisesRcd";
};

SiStripNoisesAndBadCompsChecker::SiStripNoisesAndBadCompsChecker(const edm::ParameterSet& iConfig)
    : writePayload_(iConfig.getUntrackedParameter<bool>("writePayload", true)),
      fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",
                                                         edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile))),
      printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", std::numeric_limits<unsigned int>::max())),
      noiseToken_(esConsumes()),
      deadChannelToken_(esConsumes()) {
  usesResource(cond::service::PoolDBOutputService::kSharedResource);
}

void SiStripNoisesAndBadCompsChecker::analyze(const edm::Event& evt, const edm::EventSetup& iSetup) {
  SiStripNoises obj; /* this is the output object */

  unsigned int count{0};

  const auto& reader = SiStripDetInfoFileReader::read(fp_.fullPath());
  const auto& DetInfos = reader.getAllData();

  auto const& deadChannel = iSetup.getData(deadChannelToken_);
  auto const& noise = iSetup.getData(noiseToken_);

  for (const auto& it : DetInfos) {
    const auto& nAPVs = it.second.nApvs;

    SiStripNoises::Range detNoiseRange = noise.getRange(it.first);

    // fill in the information about the dead channels
    SiStripBadStrip::Range detBadStripRange = deadChannel.getRange(it.first);
    std::vector<bool> badChannels;
    badChannels.clear();
    badChannels.insert(badChannels.begin(), sistrip::STRIPS_PER_APV * nAPVs, false);
    for (SiStripBadStrip::ContainerIterator it2 = detBadStripRange.first; it2 != detBadStripRange.second; ++it2) {
      SiStripBadStrip::data fs = deadChannel.decode(*it2);
      for (int strip = fs.firstStrip; strip < fs.firstStrip + fs.range; ++strip) {
        badChannels[strip] = true;
      }
    }

    SiStripNoises::InputVector theSiStripVector;
    for (int strip = 0; strip < sistrip::STRIPS_PER_APV * nAPVs; ++strip) {
      const auto& theNoise = noise.getNoise(strip, detNoiseRange);
      if (!badChannels[strip]) {
        try {
          noise.verify(strip, detNoiseRange);
        } catch (cms::Exception& e) {
          if (count < printdebug_) {
            edm::LogPrint(k_Name_) << "WARNING: out-of-range "
                                   << " detid: " << it.first << " strip: " << strip << " noise:" << theNoise;
          }
          count++;
        }
      }  // is strip is not masked
      obj.setData(theNoise, theSiStripVector);
    }  // loop on the strips

    if (!obj.put(it.first, theSiStripVector))
      edm::LogError(k_Name_) << "[SiStripNoisesAndBadCompsChecker::analyze] detid already exists";
  }  // loop on the detids
  edm::LogPrint(k_Name_) << "Found " << count << " strips in out-of-bounds!";

  if (writePayload_) {
    edm::LogInfo(k_Name_) << "Will write an updated fixed payload";

    //And now write sistripnoises data in DB
    edm::Service<cond::service::PoolDBOutputService> mydbservice;

    if (mydbservice.isAvailable()) {
      if (mydbservice->isNewTagRequest("SiStripNoisesRcd")) {
        mydbservice->createOneIOV<SiStripNoises>(obj, mydbservice->beginOfTime(), k_Record_);
      } else {
        mydbservice->appendOneIOV<SiStripNoises>(obj, mydbservice->currentTime(), k_Record_);
      }
    } else {
      edm::LogError("SiStripNoisesBuilder") << "Service is unavailable, will not write any output";
    }
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SiStripNoisesAndBadCompsChecker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.setComment(
      "Given a certain Global Tag, checks that the all the unmasked strips, do have a noise within the payload range");
  desc.addUntracked<bool>("writePayload", true);
  desc.addUntracked<edm::FileInPath>("file", edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile));
  desc.addUntracked<uint32_t>("printDebug", std::numeric_limits<unsigned int>::max());
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(SiStripNoisesAndBadCompsChecker);
