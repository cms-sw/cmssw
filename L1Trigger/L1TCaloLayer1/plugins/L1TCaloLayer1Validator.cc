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

  edm::EDGetTokenT<CaloTowerBxCollection> testSource;
  std::string testLabel;
  edm::EDGetTokenT<CaloTowerBxCollection> emulSource;
  std::string emulLabel;

  uint32_t eventCount;
  uint32_t badEventCount;
  uint32_t towerCount;
  uint32_t badTowerCount;
  uint32_t nonZeroTowerCount;
  uint32_t badNonZeroTowerCount;

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
  testSource(consumes<CaloTowerBxCollection>(iConfig.getParameter<edm::InputTag>("testSource"))),
  emulSource(consumes<CaloTowerBxCollection>(iConfig.getParameter<edm::InputTag>("emulSource"))),
  eventCount(0),
  badEventCount(0),
  towerCount(0),
  badTowerCount(0),
  nonZeroTowerCount(0),
  badNonZeroTowerCount(0),
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
   edm::Handle<CaloTowerBxCollection> testTowers;
   iEvent.getByToken(testSource, testTowers);
   edm::Handle<CaloTowerBxCollection> emulTowers;
   iEvent.getByToken(emulSource, emulTowers);
   int theBX = 0;
   for(std::vector<CaloTower>::const_iterator testTower = testTowers->begin(theBX);
       testTower != testTowers->end(theBX);
       ++testTower) {
     for(std::vector<CaloTower>::const_iterator emulTower = emulTowers->begin(theBX);
	 emulTower != emulTowers->end(theBX);
	 ++emulTower) {
       int test_iEta = testTower->hwEta();
       int test_iPhi = testTower->hwPhi();
       int test_et = testTower->hwPt();
       int test_er = testTower->hwEtRatio();
       int test_fb = testTower->hwQual();
       int emul_iEta = emulTower->hwEta();
       int emul_iPhi = emulTower->hwPhi();
       int emul_et = emulTower->hwPt();
       int emul_er = emulTower->hwEtRatio();
       int emul_fb = emulTower->hwQual();
       bool success = true;
       if(test_iEta == emul_iEta && test_iPhi == emul_iPhi && test_et > 5) {
	 if(test_et != emul_et) {success = false;}
	 //if(test_er != emul_er) {success = false;}
	 //if(test_fb != emul_fb) {success = false;}
	 if(!success) {
	   if(test_et != emul_et) {if(verbose) std::cout << "ET ";}
	   if(test_er != emul_er) {if(verbose) std::cout << "ER ";}
	   if(test_fb != emul_fb) {if(verbose) std::cout << "FB ";}
	   if(verbose) std::cout << "Checks failed for ("
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
       if(!success && test_et == emul_et && test_iPhi == emul_iPhi && test_et > 3) {
	   if(verbose) std::cout << "Incidental match for ("
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
  std::cout << "L1TCaloLayer1Vaidator: Summary is Non-Zero Bad Tower / Bad Tower / Event Count = ("
	    << badNonZeroTowerCount << " of " << nonZeroTowerCount << ") / ("
	    << badTowerCount << " of " << towerCount << ") / ("
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
