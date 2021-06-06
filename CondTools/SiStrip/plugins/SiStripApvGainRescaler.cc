// -*- C++ -*-
//
// Package:    CondTools/SiStrip
// Class:      SiStripApvGainRescaler
//
/**\class SiStripApvGainRescaler SiStripApvGainRescaler.cc CondTools/SiStrip/plugins/SiStripApvGainRescaler.cc

 Description: Utility class to rescale the values of SiStrip G2 by the ratio of G1_old/G1_new: this is useful in the case in which a Gain2 payload needs to recycled after a G1 update to keep the G1*G2 product constant

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marco Musich
//         Created:  Tue, 03 Oct 2017 12:57:34 GMT
//
//

// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/EventSetup.h"

//
// class declaration
//

class SiStripApvGainRescaler : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripApvGainRescaler(const edm::ParameterSet&);
  ~SiStripApvGainRescaler() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  std::unique_ptr<SiStripApvGain> getNewObject(const std::map<std::pair<uint32_t, int>, float>& theMap);
  void endJob() override;

  // ----------member data ---------------------------
  const std::string m_Record;

  // take G2_old and G1_old from the regular gain handle
  edm::ESGetToken<SiStripGain, SiStripGainRcd> g1g2Token_;
  // take the additional G1_new from the Gain3Rcd (dirty trick)
  edm::ESGetToken<SiStripApvGain, SiStripApvGain3Rcd> g3Token_;
};

//
// constructors and destructor
//
SiStripApvGainRescaler::SiStripApvGainRescaler(const edm::ParameterSet& iConfig)
    : m_Record(iConfig.getParameter<std::string>("Record")), g1g2Token_(esConsumes()), g3Token_(esConsumes()) {
  //now do what ever initialization is needed
}

SiStripApvGainRescaler::~SiStripApvGainRescaler() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------
void SiStripApvGainRescaler::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  const auto& g1g2 = iSetup.getData(g1g2Token_);
  const auto& g3 = iSetup.getData(g3Token_);

  std::map<std::pair<uint32_t, int>, float> theMap;

  std::vector<uint32_t> detid;
  g1g2.getDetIds(detid);
  for (const auto& d : detid) {
    SiStripApvGain::Range rangeG1_old = g1g2.getRange(d, 0);
    SiStripApvGain::Range rangeG2_old = g1g2.getRange(d, 1);
    SiStripApvGain::Range rangeG1_new = g3.getRange(d);

    int nAPV = 0;
    for (int it = 0; it < rangeG1_old.second - rangeG1_old.first; it++) {
      nAPV++;

      std::pair<uint32_t, int> index = std::make_pair(d, nAPV);

      float G1_old = g1g2.getApvGain(it, rangeG1_old);
      float G2_old = g1g2.getApvGain(it, rangeG2_old);
      float G1G2_old = G1_old * G2_old;
      float G1_new = g3.getApvGain(it, rangeG1_new);

      // this is based on G1_old*G2_old = G1_new * G2_new ==> G2_new = (G1_old*G2_old)/G1_new

      float NewGain = G1G2_old / G1_new;

      // DO NOT RESCALE APVs set to the default value
      if (G2_old != 1.) {
        theMap[index] = NewGain;
      } else {
        theMap[index] = 1.;
      }

    }  // loop over APVs
  }    // loop over DetIds

  std::unique_ptr<SiStripApvGain> theAPVGains = this->getNewObject(theMap);

  // write out the APVGains record
  edm::Service<cond::service::PoolDBOutputService> poolDbService;

  if (poolDbService.isAvailable())
    poolDbService->writeOne(theAPVGains.get(), poolDbService->currentTime(), m_Record);
  else
    throw std::runtime_error("PoolDBService required.");
}

// ------------ method called once each job just before starting event loop  ------------
void SiStripApvGainRescaler::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void SiStripApvGainRescaler::endJob() {}

//********************************************************************************//
std::unique_ptr<SiStripApvGain> SiStripApvGainRescaler::getNewObject(
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

    edm::LogInfo("SiStripApvGainRescaler")
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
void SiStripApvGainRescaler::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.setComment(
      " Utility class to rescale the values of SiStrip G2 by the ratio of G1_old/G1_new: this is useful in the case in "
      "which a Gain2 payload needs to recycled after a G1 update to keep the G1*G2 product constant."
      "PoolDBOutputService must be set up for 'SiStripApvGainRcd'.");

  desc.add<std::string>("Record", "SiStripApvGainRcd");
  descriptions.add("rescaleGain2byGain1", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripApvGainRescaler);
