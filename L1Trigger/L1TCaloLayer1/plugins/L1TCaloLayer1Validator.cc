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

  uint32_t ngRegion[22];
  uint32_t nbRegion[22];
  uint32_t zgRegion[22];
  uint32_t zbRegion[22];

  uint32_t ngCard[18];
  uint32_t nbCard[18];
  uint32_t zgCard[18];
  uint32_t zbCard[18];

  uint32_t tLrEmulTotET;
  uint32_t tErEmulTotET;
  uint32_t tGrEmulTotET;

  uint32_t tLeTotET;
  uint32_t tEeTotET;
  uint32_t tGeTotET;

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
  tLrEmulTotET(0),
  tErEmulTotET(0),
  tGrEmulTotET(0),
  tLeTotET(0),
  tEeTotET(0),
  tGeTotET(0),
  validateTowers(iConfig.getParameter<bool>("validateTowers")),
  validateRegions(iConfig.getParameter<bool>("validateRegions")),
  verbose(iConfig.getParameter<bool>("verbose")) {
  for(uint32_t r = 0; r < 22; r++) ngRegion[r] = nbRegion[r] = zgRegion[r] = zbRegion[r] = 0; 
  for(uint32_t c = 0; c < 18; c++) ngCard[c] = nbCard[c] = zgCard[c] = zbCard[c] = 0;
}

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
   int theBX = 0;

   // Emulator calo towers and regions should always be available - get them
   // Data will always contain calo regions, but not necessarily calo towers

   edm::Handle<CaloTowerBxCollection> emulTowers;
   iEvent.getByToken(emulTowerToken, emulTowers);
   edm::Handle<L1CaloRegionCollection> testRegions;
   iEvent.getByToken(testRegionToken, testRegions);
   edm::Handle<L1CaloRegionCollection> emulRegions;
   iEvent.getByToken(emulRegionToken, emulRegions);
   
   if(validateTowers) {
     // Test towers will be available for spy and fat events only
     edm::Handle<CaloTowerBxCollection> testTowers;
     iEvent.getByToken(testTowerToken, testTowers);
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
     uint32_t testRegionTotET = 0;
     uint32_t emulRegionTotET = 0;
     for(std::vector<L1CaloRegion>::const_iterator testRegion = testRegions->begin();
	 testRegion != testRegions->end();
	 ++testRegion) {
       //       uint16_t test_raw = testRegion->raw();
       uint32_t test_et = testRegion->et();
       testRegionTotET += test_et;
       uint32_t test_rEta = testRegion->id().ieta();
       uint32_t test_rPhi = testRegion->id().iphi();
       //       uint32_t test_iEta = (test_raw >> 12) & 0x3;
       //       uint32_t test_iPhi = (test_raw >> 14) & 0x3;
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
	 //	 uint16_t emul_raw = emulRegion->raw();
	 uint32_t emul_et = emulRegion->et();
	 if(testRegion == testRegions->begin()) emulRegionTotET += emul_et; // increment only once!
	 uint32_t emul_rEta = emulRegion->id().ieta();
	 uint32_t emul_rPhi = emulRegion->id().iphi();
	 //	 uint32_t emul_iEta = (emul_raw >> 12) & 0x3;
	 //	 uint32_t emul_iPhi = (emul_raw >> 14) & 0x3;
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
	     if(verbose) LOG_ERROR << "Checks failed for region ("
		       << std::dec
		       << test_rEta << ", "
		       << test_rPhi << ") ("
		       << test_negativeEta << ", "
		       << test_crate << ", "
		       << test_card << ", "
		       << test_region << ", "
	       //		       << test_iEta << ", "
	       //                      << test_iPhi << ", "
		       << test_et << ") != ("
		       << emul_negativeEta << ", "
		       << emul_crate << ", "
		       << emul_card << ", "
		       << emul_region << ", "
	       //		       << emul_iEta << ", "
	       //		       << emul_iPhi << ", "
		       << emul_et << ")"<< std::endl;
	     badEvent = true;
	     badRegionCount++;
	     if(test_et > 0) {
	       badNonZeroRegionCount++;
	       nbRegion[test_rEta]++;
	       nbCard[test_rPhi]++;
	     }
	     else {
	       zbRegion[test_rEta]++;
	       zbCard[test_rPhi]++;
	     }
	   }
	   else {
	     if(test_et > 0) {
	       ngRegion[test_rEta]++;
	       ngCard[test_rPhi]++;
	       if(verbose) LOG_ERROR << "Checks passed for region ("
			 << std::dec
			 << test_rEta << ", "
			 << test_rPhi << ") ("
			 << test_negativeEta << ", "
			 << test_crate << ", "
			 << test_card << ", "
			 << test_region << ", "
		 //		       << test_iEta << ", "
		 //                      << test_iPhi << ", "
			 << test_et << ") == ("
			 << emul_negativeEta << ", "
			 << emul_crate << ", "
			 << emul_card << ", "
			 << emul_region << ", "
		 //		       << emul_iEta << ", "
		 //		       << emul_iPhi << ", "
			 << emul_et << ")"<< std::endl;
	     }
	     else {
	       zgRegion[test_rEta]++;
	       zgCard[test_rPhi]++;
	     }
	   }
	   regionCount++;
	   if(test_et > 0) nonZeroRegionCount++;
	 }
	 if(!success && test_et == emul_et) {// && test_iPhi == emul_iPhi) {
	   if(verbose) LOG_ERROR << "Incidental match for region ("
				 << std::dec
				 << test_rEta << ", "
				 << test_rPhi << ", "
			 //				 << test_iEta << ", "
			 //				 << test_iPhi << ", "
				 << test_et << ") != ("
				 << emul_rEta << ", "
				 << emul_rPhi << ", "
			 //				 << emul_iEta << ", "
			 //				 << emul_iPhi << ", "
				 << emul_et << ")"<< std::endl;
	 }
       }
     }
     uint32_t emulTowerTotET = 0;
     for(std::vector<CaloTower>::const_iterator emulTower = emulTowers->begin(theBX);
	 emulTower != emulTowers->end(theBX);
	 ++emulTower) {
       int twr_et = emulTower->hwPt();
       int twr_cEta = emulTower->hwEta();
       int twr_cPhi = emulTower->hwPhi();
       uint32_t twr_region = g.getRegion(twr_cEta, twr_cPhi);
       uint32_t twr_gEta = 10 - twr_region;
       if(twr_cEta > 0) twr_gEta = twr_region + 11;
       uint32_t twr_gPhi = g.getUCTRegionPhiIndex(twr_cPhi);
       if(badEvent && twr_et > 0) {
	 if(verbose) LOG_ERROR << "Non-zero tower in region ("
		   << twr_gEta << ", "
		   << twr_gPhi << ") "
		   << "(cEta, cPhi, et) = (" 
		   << twr_cEta << ", "
		   << twr_cPhi << ", "
		   << twr_et << ")"
		   << std::endl;
       }
       if(std::abs(twr_cEta) <= 28) emulTowerTotET += twr_et; // Exclude HF towers for now
     }
     // Increment counters for emulated total tower ET comparison with total region ET
     if(emulTowerTotET < emulRegionTotET) tLrEmulTotET++;
     else if(emulTowerTotET > emulRegionTotET) tGrEmulTotET++;
     else tErEmulTotET++;
     // Increment counters for emulated total region ET comparison with region test ET
     if(testRegionTotET < emulRegionTotET) tLeTotET++;
     else if(testRegionTotET > emulRegionTotET) tGeTotET++;
     else tEeTotET++;
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
    LOG_ERROR << "L1TCaloLayer1Validator: Summary is Non-Zero Bad Tower / Bad Tower / Event Count = ("
	      << badNonZeroTowerCount << " of " << nonZeroTowerCount << ") / ("
	      << badTowerCount << " of " << towerCount << ") / ("
	      << badEventCount << " of " << eventCount << ")" << std::endl;
  if(validateRegions) {
    LOG_ERROR << "L1TCaloLayer1Validator: Summary is Non-Zero Bad Region / Bad Region / Event Count = ("
	      << badNonZeroRegionCount << " of " << nonZeroRegionCount << ") / ("
	      << badRegionCount << " of " << regionCount << ") / ("
	      << badEventCount << " of " << eventCount << ")" << std::endl;
    LOG_ERROR << "L1TCaloLayer1Validator reTa, non-zero-good / non-zero-bad / zero-good / zero-bad region[rEta] = " << std::endl;
    for(uint32_t r = 0; r < 22; r++) 
      LOG_ERROR << r << ", " << ngRegion[r] << " / " << nbRegion[r] << " / " << zgRegion[r] << " / " << zbRegion[r] << std::endl;
    LOG_ERROR << "L1TCaloLayer1Validator rPhi, non-zero-good / non-zero-bad / zero-good / zero-bad region[rPhi] = " << std::endl;
    for(uint32_t r = 0; r < 18; r++) 
      LOG_ERROR << r << ", " << ngCard[r] << " / " << nbCard[r] << " / " << zgCard[r] << " / " << zbCard[r] << std::endl;
    LOG_ERROR << "L1TCaloLayer1Validator : Total ET emulator tower vs region; less / equal / greater counts: "
	      << tLrEmulTotET << " / " << tErEmulTotET << " / " << tGrEmulTotET << std::endl;
    LOG_ERROR << "L1TCaloLayer1Validator : Total ET region test vs emulator; less / equal / greater counts: "
	      << tLeTotET << " / " << tEeTotET << " / " << tGeTotET << std::endl;
  }
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
