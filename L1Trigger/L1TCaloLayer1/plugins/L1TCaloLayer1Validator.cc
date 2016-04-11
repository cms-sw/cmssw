// -*- C++ -*-
//
// Package:    L1Trigger/L1TCaloLayer1
// Class:      L1TCaloLayer1Validator
// 
/**\class L1TCaloLayer1Validator L1TCaloLayer1Validator.cc L1Trigger/L1TCaloLayer1/plugins/L1TCaloLayer1Validator.cc

 Description: This ED Analyzer compares output of CMS L1 Trigger Calo Layer-1 output (CaloTowers) from two sources

 Implementation:
              It is expected that we compare CaloTowers from the spy source to that of the emulator.  
              It can be used to compare any two CaloTower collections
*/
//
// Original Author:  Sridhara Dasu
//         Created:  Sun, 11 Oct 2015 08:14:01 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"

#include "L1Trigger/L1TCaloLayer1/src/UCTGeometry.hh"
#include "L1Trigger/L1TCaloLayer1/src/UCTLogging.hh"

using namespace l1t;

//
// class declaration
//

class L1TCaloLayer1Validator : public edm::EDAnalyzer {
   public:
      explicit L1TCaloLayer1Validator(const edm::ParameterSet&);
      ~L1TCaloLayer1Validator();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------

  edm::EDGetTokenT<CaloTowerBxCollection> testTowerToken;
  edm::EDGetTokenT<CaloTowerBxCollection> emulTowerToken;

  edm::EDGetTokenT<L1CaloRegionCollection> testRegionToken;
  edm::EDGetTokenT<L1CaloRegionCollection> emulRegionToken;

  uint32_t eventCount;
  uint32_t badEventCount;
  uint32_t towerCount;
  uint32_t badTowerCount;
  uint32_t nonZeroTowerCount;
  uint32_t badNonZeroTowerCount;
  uint32_t regionCount;
  uint32_t badRegionCount;
  uint32_t nonZeroRegionCount;
  uint32_t badNonZeroRegionCount;

  bool validateTowers;
  bool validateRegions;

  bool verbose;

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
L1TCaloLayer1Validator::L1TCaloLayer1Validator(const edm::ParameterSet& iConfig) :
  testTowerToken(consumes<CaloTowerBxCollection>(iConfig.getParameter<edm::InputTag>("testTowerToken"))),
  emulTowerToken(consumes<CaloTowerBxCollection>(iConfig.getParameter<edm::InputTag>("emulTowerToken"))),
  testRegionToken(consumes<L1CaloRegionCollection>(iConfig.getParameter<edm::InputTag>("testRegionToken"))),
  emulRegionToken(consumes<L1CaloRegionCollection>(iConfig.getParameter<edm::InputTag>("emulRegionToken"))),
  eventCount(0),
  badEventCount(0),
  towerCount(0),
  badTowerCount(0),
  nonZeroTowerCount(0),
  badNonZeroTowerCount(0),
  regionCount(0),
  badRegionCount(0),
  nonZeroRegionCount(0),
  badNonZeroRegionCount(0),
  validateTowers(iConfig.getParameter<bool>("validateTowers")),
  validateRegions(iConfig.getParameter<bool>("validateRegions")),
  verbose(iConfig.getParameter<bool>("verbose")) {}

L1TCaloLayer1Validator::~L1TCaloLayer1Validator() {}

//
// member functions
//

// ------------ method called for each event  ------------
void
L1TCaloLayer1Validator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   using namespace edm;
   bool badEvent = false;

   if(validateTowers) {
     edm::Handle<CaloTowerBxCollection> testTowers;
     iEvent.getByToken(testTowerToken, testTowers);
     edm::Handle<CaloTowerBxCollection> emulTowers;
     iEvent.getByToken(emulTowerToken, emulTowers);
     int theBX = 0;
     for(std::vector<CaloTower>::const_iterator testTower = testTowers->begin(theBX);
	 testTower != testTowers->end(theBX);
	 ++testTower) {
       int test_iEta = testTower->hwEta();
       int test_iPhi = testTower->hwPhi();
       int test_et = testTower->hwPt();
       int test_er = testTower->hwEtRatio();
       int test_fb = testTower->hwQual();
       for(std::vector<CaloTower>::const_iterator emulTower = emulTowers->begin(theBX);
	   emulTower != emulTowers->end(theBX);
	   ++emulTower) {
	 int emul_iEta = emulTower->hwEta();
	 int emul_iPhi = emulTower->hwPhi();
	 int emul_et = emulTower->hwPt();
	 int emul_er = emulTower->hwEtRatio();
	 int emul_fb = emulTower->hwQual();
	 bool success = true;
	 if(test_iEta == emul_iEta && test_iPhi == emul_iPhi) {
	   if(test_et != emul_et) {success = false;}
	   if(test_er != emul_er) {success = false;}
	   if(test_fb != emul_fb) {success = false;}
	   if(!success) {
	     if(test_et != emul_et) {if(verbose) LOG_ERROR << "ET ";}
	     if(test_er != emul_er) {if(verbose) LOG_ERROR << "ER ";}
	     if(test_fb != emul_fb) {if(verbose) LOG_ERROR << "FB ";}
	     if(verbose) LOG_ERROR << "Checks failed for tower ("
				   << std::dec
				   << test_iEta << ", "
				   << test_iPhi << ") : ("
				   << test_et << ", "
				   << test_er << ", "
				   << test_fb << ") != ("
				   << emul_et << ", "
				   << emul_er << ", "
				   << emul_fb << ")" << std::endl;
	     badEvent = true;
	     badTowerCount++;
	     if(test_et > 0) badNonZeroTowerCount++;
	   }
	   towerCount++;
	   if(test_et > 0) nonZeroTowerCount++;
	 }
	 if(!success && test_et == emul_et && test_iPhi == emul_iPhi) {
	   if(verbose) LOG_ERROR << "Incidental match for tower ("
				 << std::dec
				 << test_iEta << ", "
				 << test_iPhi << ") : ("
				 << test_et << ", "
				 << test_er << ", "
				 << test_fb << ") != ("
				 << emul_iEta <<","
				 << emul_iPhi<<") :("
				 << emul_et << ", "
				 << emul_er << ", "
				 << emul_fb << ")" << std::endl;
	 }
       }
     }
   }

   // Region Validation

   if(validateRegions) {
     UCTGeometry g;
     edm::Handle<L1CaloRegionCollection> testRegions;
     iEvent.getByToken(testRegionToken, testRegions);
     edm::Handle<L1CaloRegionCollection> emulRegions;
     iEvent.getByToken(emulRegionToken, emulRegions);
     for(std::vector<L1CaloRegion>::const_iterator testRegion = testRegions->begin();
	 testRegion != testRegions->end();
	 ++testRegion) {
       uint16_t test_raw = testRegion->raw();
       uint32_t test_et = testRegion->et();
       uint32_t test_rEta = testRegion->id().ieta();
       uint32_t test_rPhi = testRegion->id().iphi();
       uint32_t test_iEta = (test_raw >> 12) & 0x3;
       uint32_t test_iPhi = (test_raw >> 14) & 0x3;
       bool test_negativeEta = false;
       int test_cEta = (test_rEta - 11) * 4 + 1;//test_iEta + 1;
       if(test_rEta < 11) {
	 test_negativeEta = true;
	 test_cEta = -((10 - test_rEta) * 4 + 1);//test_iEta + 1);
       }
       int test_cPhi = test_rPhi * 4 + 1;//test_iPhi + 1;
       uint32_t test_crate = g.getCrate(test_cEta, test_cPhi);
       uint32_t test_card = g.getCard(test_cEta, test_cPhi);
       uint32_t test_region = g.getRegion(test_cEta, test_cPhi);
       for(std::vector<L1CaloRegion>::const_iterator emulRegion = emulRegions->begin();
	   emulRegion != emulRegions->end();
	   ++emulRegion) {
	 uint16_t emul_raw = emulRegion->raw();
	 uint32_t emul_et = emulRegion->et();
	 uint32_t emul_rEta = emulRegion->id().ieta();
	 uint32_t emul_rPhi = emulRegion->id().iphi();
	 uint32_t emul_iEta = (emul_raw >> 12) & 0x3;
	 uint32_t emul_iPhi = (emul_raw >> 14) & 0x3;
	 bool emul_negativeEta = false;
	 int emul_cEta = (emul_rEta - 11) * 4 + 1;//emul_iEta + 1;
	 if(emul_rEta < 11) {
	   emul_negativeEta = true;
	   emul_cEta = -((10 - emul_rEta) * 4 + 1);//emul_iEta + 1);
	 }
	 int emul_cPhi = emul_rPhi * 4 + 1;//emul_iPhi + 1;
	 uint32_t emul_crate = g.getCrate(emul_cEta, emul_cPhi);
	 uint32_t emul_card = g.getCard(emul_cEta, emul_cPhi);
	 uint32_t emul_region = g.getRegion(emul_cEta, emul_cPhi);
	 bool success = true;
	 if(test_rEta == emul_rEta && test_rPhi == emul_rPhi) {
	   if(test_et != emul_et) success = false;
	   //if(test_iEta != emul_iEta) success = false;
	   //if(test_iPhi != emul_iPhi) success = false;
	   if(!success) {
	     LOG_ERROR << "Checks failed for region ("
		       << std::dec
		       << test_negativeEta << ", "
		       << test_crate << ", "
		       << test_card << ", "
		       << test_region << ", "
		       << test_iEta << ", "
		       << test_iPhi << ", "
		       << test_et << ") != ("
		       << emul_negativeEta << ", "
		       << emul_crate << ", "
		       << emul_card << ", "
		       << emul_region << ", "
		       << emul_iEta << ", "
		       << emul_iPhi << ", "
		       << emul_et << ")"<< std::endl;
	     badEvent = true;
	     badRegionCount++;
	     if(test_et > 0) badNonZeroRegionCount++;
	   }
	   regionCount++;
	   if(test_et > 0) nonZeroRegionCount++;
	 }
	 if(!success && test_et == emul_et) {// && test_iPhi == emul_iPhi) {
	   if(verbose) LOG_ERROR << "Incidental match for region ("
				 << std::dec
				 << test_rEta << ", "
				 << test_rPhi << ", "
				 << test_iEta << ", "
				 << test_iPhi << ", "
				 << test_et << ") != ("
				 << emul_rEta << ", "
				 << emul_rPhi << ", "
				 << emul_iEta << ", "
				 << emul_iPhi << ", "
				 << emul_et << ")"<< std::endl;
	 }
       }
     }
   }

   // Event counters

   if(badEvent) badEventCount++;
   eventCount++;

}

// ------------ method called once each job just before starting event loop  ------------
void 
L1TCaloLayer1Validator::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TCaloLayer1Validator::endJob() 
{
  if(validateTowers)
    LOG_ERROR << "L1TCaloLayer1Vaidator: Summary is Non-Zero Bad Tower / Bad Tower / Event Count = ("
	      << badNonZeroTowerCount << " of " << nonZeroTowerCount << ") / ("
	      << badTowerCount << " of " << towerCount << ") / ("
	      << badEventCount << " of " << eventCount << ")" << std::endl;
  if(validateRegions)
    LOG_ERROR << "L1TCaloLayer1Vaidator: Summary is Non-Zero Bad Region / Bad Region / Event Count = ("
	      << badNonZeroRegionCount << " of " << nonZeroRegionCount << ") / ("
	      << badRegionCount << " of " << regionCount << ") / ("
	      << badEventCount << " of " << eventCount << ")" << std::endl;
}

// ------------ method called when starting to processes a run  ------------
/*
void 
L1TCaloLayer1Validator::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void 
L1TCaloLayer1Validator::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
L1TCaloLayer1Validator::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
L1TCaloLayer1Validator::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TCaloLayer1Validator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TCaloLayer1Validator);
