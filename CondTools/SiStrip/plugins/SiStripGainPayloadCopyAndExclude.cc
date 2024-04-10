// -*- C++ -*-
//
// Package:    CondTools/SiStrip
// Class:      SiStripGainPayloadCopyAndExclude
//
/*
 *\class SiStripGainPayloadCopyAndExclude SiStripGainPayloadCopyAndExclude.cc CondTools/SiStrip/plugins/SiStripGainPayloadCopyAndExclude.cc

 Description: This module is meant to copy the content of a SiStrip APV Gain payload (either G1 or G2) from a local sqlite file (that should be feeded to the Event Setup via the SiStripApvGain3Rcd and put in another local sqlite file, excepted for the modules specified in the excludedModules parameter. If the doReverse parameter is true, the opposite action is performed. 

 Implementation: The implemenation takes advantage of the convenience record SiStripApvGain3Rcd in the EventSetup to be able to hold at the same time two instances of the Strip Gains in the same job.

*/
//
// Original Author:  Marco Musich
//         Created:  Fri, 08 Jun 2018 08:28:01 GMT
//
//

// system include files
#include <memory>
#include <iostream>

// user include files
#include "CLHEP/Random/RandGauss.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

//
// class declaration
//
class SiStripGainPayloadCopyAndExclude : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SiStripGainPayloadCopyAndExclude(const edm::ParameterSet&);
  ~SiStripGainPayloadCopyAndExclude() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  std::unique_ptr<SiStripApvGain> getNewObject(const std::map<std::pair<uint32_t, int>, float>& theMap);

  // ----------member data ---------------------------
  const edm::ESGetToken<SiStripGain, SiStripGainRcd> m_gainToken;
  const edm::ESGetToken<SiStripApvGain, SiStripApvGain3Rcd> m_gain3Token;
  std::vector<unsigned int> m_excludedMods;
  const std::string m_Record;
  const uint32_t m_gainType;
  const bool m_reverseSelect;
};

//
// constructors and destructor
//
SiStripGainPayloadCopyAndExclude::SiStripGainPayloadCopyAndExclude(const edm::ParameterSet& iConfig)
    : m_gainToken{esConsumes()},
      m_gain3Token{esConsumes()},
      m_excludedMods{iConfig.getUntrackedParameter<std::vector<unsigned int>>("excludedModules")},
      m_Record{iConfig.getUntrackedParameter<std::string>("record", "SiStripApvGainRcd")},
      m_gainType{iConfig.getUntrackedParameter<uint32_t>("gainType", 1)},
      m_reverseSelect{iConfig.getUntrackedParameter<bool>("reverseSelection", false)} {
  usesResource(cond::service::PoolDBOutputService::kSharedResource);

  //now do what ever initialization is needed
  sort(m_excludedMods.begin(), m_excludedMods.end());

  edm::LogInfo("ExcludedModules") << "Selected module list";
  for (std::vector<unsigned int>::const_iterator mod = m_excludedMods.begin(); mod != m_excludedMods.end(); mod++) {
    edm::LogVerbatim("ExcludedModules") << *mod;
  }
}

//
// member functions
//

// ------------ method called for each event  ------------
void SiStripGainPayloadCopyAndExclude::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // gain to be validated
  edm::ESHandle<SiStripApvGain> gNew = iSetup.getHandle(m_gain3Token);
  edm::ESHandle<SiStripGain> gOld = iSetup.getHandle(m_gainToken);

  std::map<std::pair<uint32_t, int>, float> theMap, oldPayloadMap;

  std::vector<uint32_t> detid;
  gNew->getDetIds(detid);
  for (const auto& d : detid) {
    SiStripApvGain::Range range_new = gNew->getRange(d);
    SiStripApvGain::Range range_old = gOld->getRange(d, m_gainType);
    float nAPV = 0;

    for (int it = 0; it < range_new.second - range_new.first; it++) {
      nAPV += 1;
      float Gain = gNew->getApvGain(it, range_new);
      float patchGain = gOld->getApvGain(it, range_old);
      std::pair<uint32_t, int> index = std::make_pair(d, nAPV);

      oldPayloadMap[index] = Gain;

      bool found(false);
      for (const auto& mod : m_excludedMods) {
        if (d == mod) {
          edm::LogInfo("ModuleFound") << " module " << mod << " found! Excluded... " << std::endl;
          found = true;
          break;
        }
      }

      if (m_reverseSelect)
        found = (!found);

      if (!found) {
        theMap[index] = Gain;
      } else {
        theMap[index] = patchGain;
      }

    }  // loop over APVs
  }    // loop over DetIds

  std::unique_ptr<SiStripApvGain> theAPVGains = this->getNewObject(theMap);

  // write out the APVGains record
  edm::Service<cond::service::PoolDBOutputService> poolDbService;

  if (poolDbService.isAvailable())
    poolDbService->writeOneIOV(theAPVGains.get(), poolDbService->currentTime(), m_Record);
  else
    throw std::runtime_error("PoolDBService required.");
}

//********************************************************************************//
std::unique_ptr<SiStripApvGain> SiStripGainPayloadCopyAndExclude::getNewObject(
    const std::map<std::pair<uint32_t, int>, float>& theMap) {
  std::unique_ptr<SiStripApvGain> obj = std::make_unique<SiStripApvGain>();

  std::vector<float> theSiStripVector;
  uint32_t PreviousDetId = 0;
  for (const auto& element : theMap) {
    uint32_t DetId = element.first.first;
    if (DetId != PreviousDetId) {
      if (!theSiStripVector.empty()) {
        SiStripApvGain::Range range(theSiStripVector.begin(), theSiStripVector.end());
        if (!obj->put(PreviousDetId, range))
          printf("Bug to put detId = %i\n", PreviousDetId);
      }
      theSiStripVector.clear();
      PreviousDetId = DetId;
    }
    theSiStripVector.push_back(element.second);

    edm::LogInfo("SiStripGainPayloadCopyAndExclude")
        << " DetId: " << DetId << " APV:   " << element.first.second << " Gain:  " << element.second << std::endl;
  }

  if (!theSiStripVector.empty()) {
    SiStripApvGain::Range range(theSiStripVector.begin(), theSiStripVector.end());
    if (!obj->put(PreviousDetId, range))
      printf("Bug to put detId = %i\n", PreviousDetId);
  }

  return obj;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SiStripGainPayloadCopyAndExclude::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::vector<unsigned int>>("excludedModules", {});
  desc.addUntracked<std::string>("record", "SiStripApvGainRcd");
  desc.addUntracked<uint32_t>("gainType", 1);
  desc.addUntracked<bool>("reverseSelection", false);
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripGainPayloadCopyAndExclude);
