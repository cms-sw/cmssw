// -*- C++ -*-
//
// Package:    L1Trigger/L1TCaloSummary
// Class:      L1TCaloSummary
//
/**\class L1TCaloSummary L1TCaloSummary.cc L1Trigger/L1TCaloSummary/plugins/L1TCaloSummary.cc

   Description: The package L1Trigger/L1TCaloSummary is prepared for monitoring the CMS Layer-1 Calorimeter Trigger.

   Implementation:
   It prepares region objects and puts them in the event
*/
//
// Original Author:  Sridhara Dasu
//         Created:  Sat, 14 Nov 2015 14:18:27 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "L1Trigger/L1TCaloLayer1/src/UCTLayer1.hh"
#include "L1Trigger/L1TCaloLayer1/src/UCTCrate.hh"
#include "L1Trigger/L1TCaloLayer1/src/UCTCard.hh"
#include "L1Trigger/L1TCaloLayer1/src/UCTRegion.hh"
#include "L1Trigger/L1TCaloLayer1/src/UCTTower.hh"
#include "L1Trigger/L1TCaloLayer1/src/UCTGeometry.hh"

#include "L1Trigger/L1TCaloLayer1/src/UCTObject.hh"
#include "L1Trigger/L1TCaloLayer1/src/UCTSummaryCard.hh"
#include "L1Trigger/L1TCaloLayer1/src/UCTGeometryExtended.hh"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "L1Trigger/L1TCaloLayer1/src/L1TCaloLayer1FetchLUTs.hh"
#include "L1Trigger/L1TCaloLayer1/src/UCTLogging.hh"
#include <bitset>

using namespace l1tcalo;
using namespace l1extra;
using namespace std;

//
// class declaration
//

class L1TCaloSummary : public edm::stream::EDProducer<> {
public:
  explicit L1TCaloSummary(const edm::ParameterSet&);
  ~L1TCaloSummary() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  //void beginJob() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  //void endJob() override;

  void beginRun(edm::Run const&, edm::EventSetup const&) override;

  void print();

  // ----------member data ---------------------------

  uint32_t nPumBins;

  std::vector<std::vector<std::vector<uint32_t>>> pumLUT;

  double caloScaleFactor;

  uint32_t jetSeed;
  uint32_t tauSeed;
  float tauIsolationFactor;
  uint32_t eGammaSeed;
  double eGammaIsolationFactor;
  double boostedJetPtFactor;

  bool verbose;
  int fwVersion;

  edm::EDGetTokenT<L1CaloRegionCollection> regionToken;

  UCTLayer1* layer1;
  UCTSummaryCard* summaryCard;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1TCaloSummary::L1TCaloSummary(const edm::ParameterSet& iConfig)
    : nPumBins(iConfig.getParameter<unsigned int>("nPumBins")),
      pumLUT(nPumBins, std::vector<std::vector<uint32_t>>(2, std::vector<uint32_t>(13))),
      caloScaleFactor(iConfig.getParameter<double>("caloScaleFactor")),
      jetSeed(iConfig.getParameter<unsigned int>("jetSeed")),
      tauSeed(iConfig.getParameter<unsigned int>("tauSeed")),
      tauIsolationFactor(iConfig.getParameter<double>("tauIsolationFactor")),
      eGammaSeed(iConfig.getParameter<unsigned int>("eGammaSeed")),
      eGammaIsolationFactor(iConfig.getParameter<double>("eGammaIsolationFactor")),
      boostedJetPtFactor(iConfig.getParameter<double>("boostedJetPtFactor")),
      verbose(iConfig.getParameter<bool>("verbose")),
      fwVersion(iConfig.getParameter<int>("firmwareVersion")),
      regionToken(consumes<L1CaloRegionCollection>(edm::InputTag("simCaloStage2Layer1Digis"))) {
  std::vector<double> pumLUTData;
  char pumLUTString[10];
  for (uint32_t pumBin = 0; pumBin < nPumBins; pumBin++) {
    for (uint32_t side = 0; side < 2; side++) {
      if (side == 0)
        sprintf(pumLUTString, "pumLUT%2.2dp", pumBin);
      else
        sprintf(pumLUTString, "pumLUT%2.2dn", pumBin);
      pumLUTData = iConfig.getParameter<std::vector<double>>(pumLUTString);
      for (uint32_t iEta = 0; iEta < std::max((uint32_t)pumLUTData.size(), MaxUCTRegionsEta); iEta++) {
        pumLUT[pumBin][side][iEta] = (uint32_t)round(pumLUTData[iEta] / caloScaleFactor);
      }
      if (pumLUTData.size() != (MaxUCTRegionsEta))
        edm::LogError("L1TCaloSummary") << "PUM LUT Data size integrity check failed; Expected size = "
                                        << MaxUCTRegionsEta << "; Provided size = " << pumLUTData.size()
                                        << "; Will use what is provided :(" << std::endl;
    }
  }
  produces<L1JetParticleCollection>("Boosted");
  summaryCard = new UCTSummaryCard(&pumLUT, jetSeed, tauSeed, tauIsolationFactor, eGammaSeed, eGammaIsolationFactor);
}

L1TCaloSummary::~L1TCaloSummary() {
  if (summaryCard != nullptr)
    delete summaryCard;
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void L1TCaloSummary::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  std::unique_ptr<L1JetParticleCollection> bJetCands(new L1JetParticleCollection);

  UCTGeometry g;

  // Here we read region data from the region collection created by L1TCaloLayer1 instead of
  // independently creating regions from TPGs for processing by the summary card. This results
  // in a single region vector of size 252 whereas from independent creation we had 3*6 vectors
  // of size 7*2. Indices are mapped in UCTSummaryCard accordingly.
  summaryCard->clearRegions();
  std::vector<UCTRegion*> inputRegions;
  inputRegions.clear();
  edm::Handle<std::vector<L1CaloRegion>> regionCollection;
  if (!iEvent.getByToken(regionToken, regionCollection))
    edm::LogError("L1TCaloSummary") << "UCT: Failed to get regions from region collection!";
  iEvent.getByToken(regionToken, regionCollection);
  for (const L1CaloRegion& i : *regionCollection) {
    UCTRegionIndex r = g.getUCTRegionIndexFromL1CaloRegion(i.gctEta(), i.gctPhi());
    UCTTowerIndex t = g.getUCTTowerIndexFromL1CaloRegion(r, i.raw());
    uint32_t absCaloEta = std::abs(t.first);
    uint32_t absCaloPhi = std::abs(t.second);
    bool negativeEta = false;
    if (t.first < 0)
      negativeEta = true;
    uint32_t crate = g.getCrate(t.first, t.second);
    uint32_t card = g.getCard(t.first, t.second);
    uint32_t region = g.getRegion(absCaloEta, absCaloPhi);
    UCTRegion* test = new UCTRegion(crate, card, negativeEta, region, fwVersion);
    test->setRegionSummary(i.raw());
    inputRegions.push_back(test);
  }
  summaryCard->setRegionData(inputRegions);

  if (!summaryCard->process()) {
    edm::LogError("L1TCaloSummary") << "UCT: Failed to process summary card" << std::endl;
    exit(1);
  }

  double pt = 0;
  double eta = -999.;
  double phi = -999.;
  double mass = 0;

  std::list<UCTObject*> boostedJetObjs = summaryCard->getBoostedJetObjs();
  for (std::list<UCTObject*>::const_iterator i = boostedJetObjs.begin(); i != boostedJetObjs.end(); i++) {
    const UCTObject* object = *i;
    pt = ((double)object->et()) * caloScaleFactor * boostedJetPtFactor;
    eta = g.getUCTTowerEta(object->iEta());
    phi = g.getUCTTowerPhi(object->iPhi());
    bitset<3> activeRegionEtaPattern = 0;
    for (uint32_t iEta = 0; iEta < 3; iEta++) {
      bool activeStrip = false;
      for (uint32_t iPhi = 0; iPhi < 3; iPhi++) {
        if (object->boostedJetRegionET()[3 * iEta + iPhi] > 30 &&
            object->boostedJetRegionET()[3 * iEta + iPhi] > object->et() * 0.0625)
          activeStrip = true;
      }
      if (activeStrip)
        activeRegionEtaPattern |= (0x1 << iEta);
    }
    bitset<3> activeRegionPhiPattern = 0;
    for (uint32_t iPhi = 0; iPhi < 3; iPhi++) {
      bool activeStrip = false;
      for (uint32_t iEta = 0; iEta < 3; iEta++) {
        if (object->boostedJetRegionET()[3 * iEta + iPhi] > 30 &&
            object->boostedJetRegionET()[3 * iEta + iPhi] > object->et() * 0.0625)
          activeStrip = true;
      }
      if (activeStrip)
        activeRegionPhiPattern |= (0x1 << iPhi);
    }
    string regionEta = activeRegionEtaPattern.to_string<char, std::string::traits_type, std::string::allocator_type>();
    string regionPhi = activeRegionPhiPattern.to_string<char, std::string::traits_type, std::string::allocator_type>();
    if (std::abs(eta) < 2.5 && (regionEta == "010" || regionPhi == "010" || regionEta == "110" || regionPhi == "110" ||
                                regionEta == "011" || regionPhi == "011"))
      bJetCands->push_back(L1JetParticle(math::PtEtaPhiMLorentzVector(pt, eta, phi, mass), L1JetParticle::kCentral));
  }

  iEvent.put(std::move(bJetCands), "Boosted");
}

void L1TCaloSummary::print() {}

// ------------ method called once each job just before starting event loop  ------------
//void L1TCaloSummary::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
//void L1TCaloSummary::endJob() {}

// ------------ method called when starting to processes a run  ------------

void L1TCaloSummary::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {}

// ------------ method called when ending the processing of a run  ------------
/*
  void
  L1TCaloSummary::endRun(edm::Run const&, edm::EventSetup const&)
  {
  }
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
  void
  L1TCaloSummary::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
  {
  }
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
  void
  L1TCaloSummary::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
  {
  }
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1TCaloSummary::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TCaloSummary);
