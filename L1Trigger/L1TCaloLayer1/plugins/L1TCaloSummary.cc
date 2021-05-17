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

class L1TCaloSummary : public edm::EDProducer {
public:
  explicit L1TCaloSummary(const edm::ParameterSet&);
  ~L1TCaloSummary();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob() override;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;
      
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;

  void print();

  // ----------member data ---------------------------

//  edm::EDGetTokenT<EcalTrigPrimDigiCollection> ecalTPSource;
//  std::string ecalTPSourceLabel;
//  edm::EDGetTokenT<HcalTrigPrimDigiCollection> hcalTPSource;
//  std::string hcalTPSourceLabel;
//
//  std::vector<std::array<std::array<std::array<uint32_t, nEtBins>, nCalSideBins>, nCalEtaBins> > ecalLUT;
//  std::vector<std::array<std::array<std::array<uint32_t, nEtBins>, nCalSideBins>, nCalEtaBins> > hcalLUT;
//  std::vector<std::array<std::array<uint32_t, nEtBins>, nHfEtaBins> > hfLUT;
//  std::vector<unsigned int> ePhiMap;
//  std::vector<unsigned int> hPhiMap;
//  std::vector<unsigned int> hfPhiMap;

  uint32_t nPumBins;

  std::vector< std::vector< std::vector < uint32_t > > > pumLUT;

//  std::vector< UCTTower* > twrList;
//
//  bool useLSB;
//  bool useCalib;
//  bool useECALLUT;
//  bool useHCALLUT;
//  bool useHFLUT;

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

  UCTLayer1 *layer1;
  UCTSummaryCard *summaryCard;
  
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
L1TCaloSummary::L1TCaloSummary(const edm::ParameterSet& iConfig) :
//  ecalTPSource(consumes<EcalTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("ecalToken"))),
//  ecalTPSourceLabel(iConfig.getParameter<edm::InputTag>("ecalToken").label()),
//  hcalTPSource(consumes<HcalTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("hcalToken"))),
//  hcalTPSourceLabel(iConfig.getParameter<edm::InputTag>("hcalToken").label()),
//  ePhiMap(72 * 2, 0),
//  hPhiMap(72 * 2, 0),
//  hfPhiMap(72 * 2, 0),
  nPumBins(iConfig.getParameter<unsigned int>("nPumBins")),
  pumLUT(nPumBins, std::vector< std::vector<uint32_t> >(2, std::vector<uint32_t>(13))),
//  useLSB(iConfig.getParameter<bool>("useLSB")),
//  useCalib(iConfig.getParameter<bool>("useCalib")),
//  useECALLUT(iConfig.getParameter<bool>("useECALLUT")),
//  useHCALLUT(iConfig.getParameter<bool>("useHCALLUT")),
//  useHFLUT(iConfig.getParameter<bool>("useHFLUT")),
  caloScaleFactor(iConfig.getParameter<double>("caloScaleFactor")),
  jetSeed(iConfig.getParameter<unsigned int>("jetSeed")),
  tauSeed(iConfig.getParameter<unsigned int>("tauSeed")),
  tauIsolationFactor(iConfig.getParameter<double>("tauIsolationFactor")),
  eGammaSeed(iConfig.getParameter<unsigned int>("eGammaSeed")),
  eGammaIsolationFactor(iConfig.getParameter<double>("eGammaIsolationFactor")),
  boostedJetPtFactor(iConfig.getParameter<double>("boostedJetPtFactor")),
  verbose(iConfig.getParameter<bool>("verbose")),
  fwVersion(iConfig.getParameter<int>("firmwareVersion")),
  regionToken(consumes<L1CaloRegionCollection>( edm::InputTag("l1tCaloLayer1Digis","","L1TCaloSummaryTest") ))
{
  std::vector<double> pumLUTData;
  char pumLUTString[10];
  for(uint32_t pumBin = 0; pumBin < nPumBins; pumBin++) {
    for(uint32_t side = 0; side < 2; side++) {
      if(side == 0) sprintf(pumLUTString, "pumLUT%2.2dp", pumBin);
      else sprintf(pumLUTString, "pumLUT%2.2dn", pumBin);
      pumLUTData = iConfig.getParameter<std::vector < double > >(pumLUTString);
      for(uint32_t iEta = 0; iEta < std::max((uint32_t) pumLUTData.size(), MaxUCTRegionsEta); iEta++) {
	pumLUT[pumBin][side][iEta] = (uint32_t) round(pumLUTData[iEta] / caloScaleFactor);
      }
      if(pumLUTData.size() != (MaxUCTRegionsEta))
	std::cerr << "PUM LUT Data size integrity check failed; Expected size = " << MaxUCTRegionsEta
		  << "; Provided size = " << pumLUTData.size()
		  << "; Will use what is provided :(" << std::endl;
    }
  }
  produces< L1JetParticleCollection >( "Boosted" ) ;
  summaryCard = new UCTSummaryCard(&pumLUT, jetSeed, tauSeed, tauIsolationFactor, eGammaSeed, eGammaIsolationFactor);
//  layer1 = new UCTLayer1(fwVersion);
//  summaryCard = new UCTSummaryCard(layer1, &pumLUT, jetSeed, tauSeed, tauIsolationFactor, eGammaSeed, eGammaIsolationFactor);
//  vector<UCTCrate*> crates = layer1->getCrates();
//  for(uint32_t crt = 0; crt < crates.size(); crt++) {
//    vector<UCTCard*> cards = crates[crt]->getCards();
//    for(uint32_t crd = 0; crd < cards.size(); crd++) {
//      vector<UCTRegion*> regions = cards[crd]->getRegions();
//      for(uint32_t rgn = 0; rgn < regions.size(); rgn++) {
//	vector<UCTTower*> towers = regions[rgn]->getTowers();
//	for(uint32_t twr = 0; twr < towers.size(); twr++) {
//	  twrList.push_back(towers[twr]);
//	}
//      }
//    }
//  }
}

L1TCaloSummary::~L1TCaloSummary() {
//  if(layer1 != 0) delete layer1;
  if(summaryCard != 0) delete summaryCard;
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1TCaloSummary::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

//  edm::Handle<EcalTrigPrimDigiCollection> ecalTPs;
//  iEvent.getByToken(ecalTPSource, ecalTPs);
//  edm::Handle<HcalTrigPrimDigiCollection> hcalTPs;
//  iEvent.getByToken(hcalTPSource, hcalTPs);

  std::unique_ptr<L1JetParticleCollection> bJetCands(new L1JetParticleCollection);

//  uint32_t expectedTotalET = 0;
//
//  if(!layer1->clearEvent()) {
//    std::cerr << "UCT: Failed to clear event" << std::endl;
//    exit(1);
//  }
//
//  for ( const auto& ecalTp : *ecalTPs ) {
//    int caloEta = ecalTp.id().ieta();
//    int caloPhi = ecalTp.id().iphi();
//    int et = ecalTp.compressedEt();
//    bool fgVeto = ecalTp.fineGrain();
//    if(et != 0) {
//      UCTTowerIndex t = UCTTowerIndex(caloEta, caloPhi);
//      if(!layer1->setECALData(t,fgVeto,et)) {
//	std::cerr << "UCT: Failed loading an ECAL tower" << std::endl;
//	return;
//      }
//      expectedTotalET += et;
//    }
//  }
//
//  for ( const auto& hcalTp : *hcalTPs ) {
//    int caloEta = hcalTp.id().ieta();
//    uint32_t absCaloEta = abs(caloEta);
//    // Tower 29 is not used by Layer-1
//    if(absCaloEta == 29) {
//      continue;
//    }
//    // Prevent usage of HF TPs with Layer-1 emulator if HCAL TPs are old style
//    else if(hcalTp.id().version() == 0 && absCaloEta > 29) {
//      continue;
//    }
//    else if(absCaloEta <= 41) {
//      int caloPhi = hcalTp.id().iphi();
//      if(caloPhi <= 72) {
//	int et = hcalTp.SOI_compressedEt();
//	bool fg = hcalTp.SOI_fineGrain();
//	if(et != 0) {
//	  UCTTowerIndex t = UCTTowerIndex(caloEta, caloPhi);
//	  uint32_t featureBits = 0;
//	  if(fg) featureBits = 0x1F; // Set all five feature bits for the moment - they are not defined in HW / FW yet!
//	  if(!layer1->setHCALData(t, featureBits, et)) {
//	    std::cerr << "caloEta = " << caloEta << "; caloPhi =" << caloPhi << std::endl;
//	    std::cerr << "UCT: Failed loading an HCAL tower" << std::endl;
//	    return;
//	    
//	  }
//	  expectedTotalET += et;
//	}
//      }
//      else {
//	std::cerr << "Illegal Tower: caloEta = " << caloEta << "; caloPhi =" << caloPhi << std::endl;	
//      }
//    }
//    else {
//      std::cerr << "Illegal Tower: caloEta = " << caloEta << std::endl;
//    }
//  }
//
//  if(!layer1->process()) {
//    std::cerr << "UCT: Failed to process layer 1" << std::endl;
//    exit(1);
//  }
//
//  // Crude check if total ET is approximately OK!
//  // We can't expect exact match as there is region level saturation to 10-bits
//  // 1% is good enough
//  int diff = abs((int)layer1->et() - (int)expectedTotalET);
//  if(verbose && diff > 0.01 * expectedTotalET ) {
//    print();
//    std::cout << "Expected " 
//	      << std::showbase << std::internal << std::setfill('0') << std::setw(10) << std::hex
//	      << expectedTotalET << std::dec << std::endl;
//  }

  UCTGeometry g;

  summaryCard->clearRegions();
  std::vector<UCTRegion*> inputRegions;
  inputRegions.clear();
  //edm::Handle<L1CaloRegionCollection> regionCollection;
  edm::Handle<std::vector<L1CaloRegion>> regionCollection;
  if(!iEvent.getByToken(regionToken, regionCollection)) std::cerr << "UCT: Failed to get regions!";
  iEvent.getByToken(regionToken, regionCollection);
  //const std::vector<L1CaloRegion> &caloRegion = *regionCollection;
  //for(auto i : caloRegion) {
  for (const L1CaloRegion &i : *regionCollection) {
    cout<<"here L1TCaloSummary: "<<hex<<i.raw()<<endl;
    uint32_t uctEta = 99;
    if(i.gctEta() < 11) { uctEta = i.gctEta() - 11; }
    else if(i.gctEta() > 10) { uctEta = i.gctEta() - 10; }
    //if(uctEta != 99) summaryCard->setRegionData(i.raw(), uctEta, i.gctPhi(), fwVersion);
    UCTRegionIndex r = UCTRegionIndex(uctEta, i.gctPhi());
    UCTTowerIndex t = g.getUCTTowerIndex(r);
    uint32_t absCaloEta = std::abs(t.first);
    uint32_t absCaloPhi = std::abs(t.second);
    bool negativeEta = false;
    if (t.first < 0)
      negativeEta = true;
    uint32_t crate = g.getCrate(t.first, t.second);
    uint32_t card = g.getCard(t.first, t.second);
    uint32_t region = g.getRegion(absCaloEta, absCaloPhi) * 2;
    UCTRegion* test = new UCTRegion(crate, card, negativeEta, region, fwVersion);
    test->setRegionSummary(i.raw());
    cout<<"here L1TCaloSummary test: "<<hex<<test->compressedData()<<endl;
    inputRegions.push_back(test);
  }
  summaryCard->setRegionData(inputRegions);
 
  if(!summaryCard->process()) {
    std::cerr << "UCT: Failed to process summary card" << std::endl;
    exit(1);      
  }

  double pt = 0;
  double eta = -999.;
  double phi = -999.;
  double mass = 0;
  
  std::list<UCTObject*> boostedJetObjs = summaryCard->getBoostedJetObjs();
  for(std::list<UCTObject*>::const_iterator i = boostedJetObjs.begin(); i != boostedJetObjs.end(); i++) {
    const UCTObject* object = *i;
    pt = ((double) object->et()) * caloScaleFactor * boostedJetPtFactor;
    eta = g.getUCTTowerEta(object->iEta());
    phi = g.getUCTTowerPhi(object->iPhi());
    bitset<3> activeRegionEtaPattern = 0;
    for(uint32_t iEta = 0; iEta < 3; iEta++){
      bool activeStrip = false;
      for(uint32_t iPhi = 0; iPhi < 3; iPhi++){
        if(object->boostedJetRegionET()[3*iEta+iPhi] > 30 && object->boostedJetRegionET()[3*iEta+iPhi] > object->et()*0.0625) activeStrip = true;
      }
      if(activeStrip) activeRegionEtaPattern |= (0x1 << iEta);
    }
    bitset<3> activeRegionPhiPattern = 0;
    for(uint32_t iPhi = 0; iPhi < 3; iPhi++){
      bool activeStrip = false;
      for(uint32_t iEta = 0; iEta < 3; iEta++){
        if(object->boostedJetRegionET()[3*iEta+iPhi] > 30 && object->boostedJetRegionET()[3*iEta+iPhi] > object->et()*0.0625) activeStrip = true;
      }
      if(activeStrip) activeRegionPhiPattern |= (0x1 << iPhi);
    }
    string regionEta = activeRegionEtaPattern.to_string<char,std::string::traits_type,std::string::allocator_type>();
    string regionPhi = activeRegionPhiPattern.to_string<char,std::string::traits_type,std::string::allocator_type>();
    if(abs(eta) < 2.5 && (regionEta == "010" || regionPhi == "010" || regionEta == "110" || regionPhi == "110" || regionEta == "011" || regionPhi == "011") ) bJetCands->push_back(L1JetParticle(math::PtEtaPhiMLorentzVector(pt, eta, phi, mass), L1JetParticle::kCentral));
  }

  iEvent.put(std::move(bJetCands), "Boosted");
}

void L1TCaloSummary::print() {
//  vector<UCTCrate*> crates = layer1->getCrates();
//  for(uint32_t crt = 0; crt < crates.size(); crt++) {
//    vector<UCTCard*> cards = crates[crt]->getCards();
//    for(uint32_t crd = 0; crd < cards.size(); crd++) {
//      vector<UCTRegion*> regions = cards[crd]->getRegions();
//      for(uint32_t rgn = 0; rgn < regions.size(); rgn++) {
//	if(regions[rgn]->et() > 10) {
//	  int hitEta = regions[rgn]->hitCaloEta();
//	  int hitPhi = regions[rgn]->hitCaloPhi();
//	  vector<UCTTower*> towers = regions[rgn]->getTowers();
//	  for(uint32_t twr = 0; twr < towers.size(); twr++) {
//	    if(towers[twr]->caloPhi() == hitPhi && towers[twr]->caloEta() == hitEta) {
//	      std::cout << "*";
//	    }
//	    if(towers[twr]->et() > 10) std::cout << *towers[twr];
//	  }
//	  std::cout << *regions[rgn];
//	}
//      }
//      std::cout << *cards[crd];
//    }
//    std::cout << *crates[crt];
//  }
//  std::cout << *layer1;
}

// ------------ method called once each job just before starting event loop  ------------
void 
L1TCaloSummary::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TCaloSummary::endJob() {
}

// ------------ method called when starting to processes a run  ------------

void
L1TCaloSummary::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
//  if (!L1TCaloLayer1FetchLUTs(iSetup,
//                              ecalLUT,
//                              hcalLUT,
//                              hfLUT,
//                              ePhiMap,
//                              hPhiMap,
//                              hfPhiMap,
//                              useLSB,
//                              useCalib,
//                              useECALLUT,
//                              useHCALLUT,
//                              useHFLUT,
//                              fwVersion)) {
//    LOG_ERROR << "L1TCaloLayer1::beginRun: failed to fetch LUTS - using unity" << std::endl;
//    std::array<std::array<std::array<uint32_t, nEtBins>, nCalSideBins>, nCalEtaBins> eCalLayer1EtaSideEtArray;
//    std::array<std::array<std::array<uint32_t, nEtBins>, nCalSideBins>, nCalEtaBins> hCalLayer1EtaSideEtArray;
//    std::array<std::array<uint32_t, nEtBins>, nHfEtaBins> hfLayer1EtaEtArray;
//    ecalLUT.push_back(eCalLayer1EtaSideEtArray);
//    hcalLUT.push_back(hCalLayer1EtaSideEtArray);
//    hfLUT.push_back(hfLayer1EtaEtArray);
//  }
//  for (uint32_t twr = 0; twr < twrList.size(); twr++) {
//    int iphi = twrList[twr]->caloPhi();
//    int ieta = twrList[twr]->caloEta();
//    if (ieta < 0) {
//      iphi -= 1;
//    } else {
//      iphi += 71;
//    }
//    twrList[twr]->setECALLUT(&ecalLUT[ePhiMap[iphi]]);
//    twrList[twr]->setHCALLUT(&hcalLUT[hPhiMap[iphi]]);
//    twrList[twr]->setHFLUT(&hfLUT[hfPhiMap[iphi]]);
//  }
}
 
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
void
L1TCaloSummary::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TCaloSummary);
