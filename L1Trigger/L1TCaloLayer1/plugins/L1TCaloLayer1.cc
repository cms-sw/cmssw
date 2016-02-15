// -*- C++ -*-
//
// Package:    L1Trigger/L1TCaloLayer1
// Class:      L1TCaloLayer1
// 
/**\class L1TCaloLayer1 L1TCaloLayer1.cc L1Trigger/L1TCaloLayer1/plugins/L1TCaloLayer1.cc

   Description: [one line class summary]

   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Sridhara Rao Dasu
//         Created:  Thu, 08 Oct 2015 09:20:16 GMT
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

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

#include "L1Trigger/L1TCaloLayer1/src/L1TCaloLayer1FetchLUTs.hh"

using namespace l1t;

//
// class declaration
//

class L1TCaloLayer1 : public edm::EDProducer {
public:
  explicit L1TCaloLayer1(const edm::ParameterSet&);
  ~L1TCaloLayer1();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob() override;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;
      
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;

  //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;


  // ----------member data ---------------------------

  edm::EDGetTokenT<EcalTrigPrimDigiCollection> ecalTPSource;
  std::string ecalTPSourceLabel;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> hcalTPSource;
  std::string hcalTPSourceLabel;
  
  std::vector< std::vector< std::vector < uint32_t > > > ecalLUT;
  std::vector< std::vector< std::vector < uint32_t > > > hcalLUT;
  
  bool useLSB;
  bool useLUT;
  bool verbose;

  UCTLayer1 *layer1;

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
L1TCaloLayer1::L1TCaloLayer1(const edm::ParameterSet& iConfig) :
  ecalTPSource(consumes<EcalTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("ecalToken"))),
  ecalTPSourceLabel(iConfig.getParameter<edm::InputTag>("ecalToken").label()),
  hcalTPSource(consumes<HcalTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("hcalToken"))),
  hcalTPSourceLabel(iConfig.getParameter<edm::InputTag>("hcalToken").label()),
  ecalLUT(28, std::vector< std::vector<uint32_t> >(2, std::vector<uint32_t>(256))),
  hcalLUT(28, std::vector< std::vector<uint32_t> >(2, std::vector<uint32_t>(256))),
  useLSB(iConfig.getParameter<bool>("useLSB")),
  useLUT(iConfig.getParameter<bool>("useLUT")),
  verbose(iConfig.getParameter<bool>("verbose")) 
{
  produces<CaloTowerBxCollection>();
  layer1 = new UCTLayer1;
}

L1TCaloLayer1::~L1TCaloLayer1() {
  if(layer1 != 0) delete layer1;
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1TCaloLayer1::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  edm::Handle<EcalTrigPrimDigiCollection> ecalTPs;
  iEvent.getByToken(ecalTPSource, ecalTPs);
  edm::Handle<HcalTrigPrimDigiCollection> hcalTPs;
  iEvent.getByToken(hcalTPSource, hcalTPs);

  std::auto_ptr<CaloTowerBxCollection> towersColl (new CaloTowerBxCollection);

  uint32_t expectedTotalET = 0;
  if(!layer1->clearEvent()) {
    std::cerr << "UCT: Failed to clear event" << std::endl;
    return;
  }

  for ( const auto& ecalTp : *ecalTPs ) {
    int caloEta = ecalTp.id().ieta();
    int caloPhi = ecalTp.id().iphi();
    int et = ecalTp.compressedEt();
    bool fgVeto = ecalTp.fineGrain();
    if(et != 0) {
      UCTTowerIndex t = UCTTowerIndex(caloEta, caloPhi);
      if(!layer1->setECALData(t,fgVeto,et)) {
	std::cerr << "UCT: Failed loading an ECAL tower" << std::endl;
	return;
      }
      expectedTotalET += et;
    }
  }

  for ( const auto& hcalTp : *hcalTPs ) {
    int caloEta = hcalTp.id().ieta();
    uint32_t absCaloEta = abs(caloEta);
    // Tower 29 is not used by Layer-1
    if(absCaloEta == 29) {
      continue;
    }
    // Prevent usage of HF TPs with Layer-1 emulator if HCAL TPs are old style
    else if(hcalTp.id().version() == 0 && absCaloEta > 29) {
      continue;
    }
    else {
      int caloPhi = hcalTp.id().iphi();
      int et = hcalTp.SOI_compressedEt();
      bool fg = hcalTp.SOI_fineGrain();
      if(et != 0) {
	UCTTowerIndex t = UCTTowerIndex(caloEta, caloPhi);
	uint32_t featureBits = 0;
	if(fg) featureBits = 0x1F; // Set all five feature bits for the moment - they are not defined in HW / FW yet!
	if(!layer1->setHCALData(t, featureBits, et)) {
	  std::cerr << "caloEta = " << caloEta << "; caloPhi =" << caloPhi << std::endl;
	  std::cerr << "UCT: Failed loading an HCAL tower" << std::endl;
	  return;
	  
	}
	expectedTotalET += et;
      }
    }
  }
  
  
   //Process
  if(!layer1->process()) {
    std::cerr << "UCT: Failed to process layer 1" << std::endl;
  }
  
  
  // Crude check if total ET is approximately OK!
  // We can't expect exact match as there is region level saturation to 10-bits
  // 1% is good enough
  int diff = abs(layer1->et() - expectedTotalET);
  if(verbose && diff > 0.01 * expectedTotalET ) {
    //print();
    //    std::cout << "Expected " 
    //	      << std::showbase << std::internal << std::setfill('0') << std::setw(10) << std::hex
    //	      << expectedTotalET << std::dec << std::endl;
  }

  int theBX = 0; // Currently we only read and process the "hit" BX only
 
  vector<UCTCrate*> crates = layer1->getCrates();
  for(uint32_t crt = 0; crt < crates.size(); crt++) {
    vector<UCTCard*> cards = crates[crt]->getCards();
    for(uint32_t crd = 0; crd < cards.size(); crd++) {
      vector<UCTRegion*> regions = cards[crd]->getRegions();
      for(uint32_t rgn = 0; rgn < regions.size(); rgn++) {
	vector<UCTTower*> towers = regions[rgn]->getTowers();
	for(uint32_t twr = 0; twr < towers.size(); twr++) {
	  CaloTower caloTower;
	  caloTower.setHwPt(towers[twr]->et());               // Bits 0-8 of the 16-bit word per the interface protocol document
	  caloTower.setHwEtRatio(towers[twr]->er());          // Bits 9-11 of the 16-bit word per the interface protocol document
	  caloTower.setHwQual(towers[twr]->miscBits());       // Bits 12-15 of the 16-bit word per the interface protocol document
	  caloTower.setHwEta(towers[twr]->uctEta());          // Same as caloEta = 1-28 and 30-41
	  caloTower.setHwPhi(towers[twr]->uctPhi());          // Same as caloPhi = 1-72 for caloEta = 1-28, but also 1-72 for caloEta = 30-41!
	  caloTower.setHwEtEm(towers[twr]->getEcalET());      // This is provided as a courtesy - not available to hardware
	  caloTower.setHwEtHad(towers[twr]->getHcalET());     // This is provided as a courtesy - not available to hardware
	  towersColl->push_back(theBX, caloTower);
	}
      }
    }
  }  

  iEvent.put(towersColl);

}



// ------------ method called once each job just before starting event loop  ------------
void 
L1TCaloLayer1::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TCaloLayer1::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void
L1TCaloLayer1::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  if(!L1TCaloLayer1FetchLUTs(iSetup, ecalLUT, hcalLUT, useLSB, useLUT)) {
    std::cerr << "beginRun: failed to fetch LUTS" << std::endl;
  }
  vector<UCTCrate*> crates = layer1->getCrates();
  for(uint32_t crt = 0; crt < crates.size(); crt++) {
    vector<UCTCard*> cards = crates[crt]->getCards();
    for(uint32_t crd = 0; crd < cards.size(); crd++) {
      vector<UCTRegion*> regions = cards[crd]->getRegions();
      for(uint32_t rgn = 0; rgn < regions.size(); rgn++) {
	vector<UCTTower*> towers = regions[rgn]->getTowers();
	for(uint32_t twr = 0; twr < towers.size(); twr++) {
	  towers[twr]->setECALLUT(&ecalLUT);
	  towers[twr]->setHCALLUT(&hcalLUT);
	}
      }
    }
  }
}

// ------------ method called when ending the processing of a run  ------------
/*
  void
  L1TCaloLayer1::endRun(edm::Run const&, edm::EventSetup const&)
  {
  }
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
  void
  L1TCaloLayer1::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
  {
  }
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
  void
  L1TCaloLayer1::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
  {
  }
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TCaloLayer1::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TCaloLayer1);
