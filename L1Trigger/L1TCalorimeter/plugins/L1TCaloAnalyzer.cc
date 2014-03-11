// -*- C++ -*-
//
// Package:    L1Trigger/L1TCalorimeter
// Class:      L1TCaloAnalyzer
// 
/**\class L1TCaloAnalyzer L1TCaloAnalyzer.cc L1Trigger/L1TCalorimeter/plugins/L1TCaloAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  James Brooke
//         Created:  Tue, 11 Mar 2014 14:55:45 GMT
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

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"


//
// class declaration
//

class L1TCaloAnalyzer : public edm::EDAnalyzer {
public:
  explicit L1TCaloAnalyzer(const edm::ParameterSet&);
  ~L1TCaloAnalyzer();
  
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
  edm::EDGetToken m_towerToken;
  edm::EDGetToken m_egToken;
  edm::EDGetToken m_tauToken;
  edm::EDGetToken m_jetToken;
  edm::EDGetToken m_sumToken;
  
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
L1TCaloAnalyzer::L1TCaloAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

  // register what you consume and keep token for later access:
  m_towerToken = consumes<l1t::CaloTowerBxCollection>(iConfig.getParameter<edm::InputTag>("towerToken"));
  m_egToken    = consumes<l1t::EGammaBxCollection>   (iConfig.getParameter<edm::InputTag>("egToken"));
  m_tauToken   = consumes<l1t::TauBxCollection>      (iConfig.getParameter<edm::InputTag>("tauToken"));
  m_jetToken   = consumes<l1t::JetBxCollection>      (iConfig.getParameter<edm::InputTag>("jetToken"));
  m_sumToken   = consumes<l1t::EtSumBxCollection>    (iConfig.getParameter<edm::InputTag>("etSumToken"));


}


L1TCaloAnalyzer::~L1TCaloAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
L1TCaloAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  // get TPs ?
  // get regions ?
  // get RCT clusters ?
  
  
  // get towers
  Handle< BXVector<l1t::CaloTower> > towers;
  iEvent.getByToken(m_towerToken,towers);

  // get EG
  Handle< BXVector<l1t::EGamma> > egs;
  iEvent.getByToken(m_egToken,egs);

  // get tau
  Handle< BXVector<l1t::Tau> > taus;
  iEvent.getByToken(m_tauToken,taus);

  // get jet
  Handle< BXVector<l1t::Jet> > jets;
  iEvent.getByToken(m_jetToken,jets);

  // get sums
  Handle< BXVector<l1t::EtSum> > sums;
  iEvent.getByToken(m_sumToken,sums);

  
//   int bxFirst = towers->getFirstBX();
//   int bxLast = towers->getLastBX();





}


// ------------ method called once each job just before starting event loop  ------------
void 
L1TCaloAnalyzer::beginJob()
{

  edm::Service<TFileService> fs;

  TFileDirectory dir0 = fs->mkdir("towers");
  TFileDirectory dir1 = fs->mkdir("eg");
  TFileDirectory dir2 = fs->mkdir("tau");
  TFileDirectory dir3 = fs->mkdir("jet");
  TFileDirectory dir4 = fs->mkdir("sum");

}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TCaloAnalyzer::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
/*
void 
L1TCaloAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void 
L1TCaloAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
L1TCaloAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
L1TCaloAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TCaloAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TCaloAnalyzer);
