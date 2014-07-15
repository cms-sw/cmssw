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
#include "DataFormats/L1TCalorimeter/interface/CaloCluster.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"


//
// class declaration
//

namespace l1t {

class Stage2CaloAnalyzer : public edm::EDAnalyzer {
public:
  explicit Stage2CaloAnalyzer(const edm::ParameterSet&);
  ~Stage2CaloAnalyzer();
  
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
  edm::EDGetToken m_clusterToken;
  edm::EDGetToken m_egToken;
  edm::EDGetToken m_tauToken;
  edm::EDGetToken m_jetToken;
  edm::EDGetToken m_sumToken;
  
  enum ObjectType{Tower=0x1,
		  Cluster=0x2,
		  EG=0x3,
		  Tau=0x4,
		  Jet=0x5,
		  Sum=0x6};
  
  std::vector< ObjectType > types_;
  std::vector< std::string > typeStr_;
  
  std::map< ObjectType, TFileDirectory > dirs_;
  std::map< ObjectType, TH1F* > het_;
  std::map< ObjectType, TH1F* > heta_;
  std::map< ObjectType, TH1F* > hphi_;
  std::map< ObjectType, TH1F* > hem_;
  std::map< ObjectType, TH1F* > hhad_;
  std::map< ObjectType, TH1F* > hratio_;

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
Stage2CaloAnalyzer::Stage2CaloAnalyzer(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

  // register what you consume and keep token for later access:
  m_towerToken   = consumes<l1t::CaloTowerBxCollection>  (iConfig.getParameter<edm::InputTag>("towerToken"));
  m_clusterToken = consumes<l1t::CaloClusterBxCollection>(iConfig.getParameter<edm::InputTag>("clusterToken"));
  m_egToken      = consumes<l1t::EGammaBxCollection>     (iConfig.getParameter<edm::InputTag>("egToken"));
  m_tauToken     = consumes<l1t::TauBxCollection>        (iConfig.getParameter<edm::InputTag>("tauToken"));
  m_jetToken     = consumes<l1t::JetBxCollection>        (iConfig.getParameter<edm::InputTag>("jetToken"));
  m_sumToken     = consumes<l1t::EtSumBxCollection>      (iConfig.getParameter<edm::InputTag>("etSumToken"));

  types_.push_back( Tower );
  types_.push_back( Cluster );
  types_.push_back( EG );
  types_.push_back( Tau );
  types_.push_back( Jet );
  types_.push_back( Sum );

  typeStr_.push_back( "tower" );
  typeStr_.push_back( "cluster" );
  typeStr_.push_back( "eg" );
  typeStr_.push_back( "tau" );
  typeStr_.push_back( "jet" );
  typeStr_.push_back( "sum" );

}


Stage2CaloAnalyzer::~Stage2CaloAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
Stage2CaloAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  // only does anything for BX=0 right now

  // get TPs ?
  // get regions ?
  // get RCT clusters ?
  
  
  // get towers
  Handle< BXVector<l1t::CaloTower> > towers;
  iEvent.getByToken(m_towerToken,towers);

  for ( auto itr = towers->begin(0); itr != towers->end(0); ++itr ) {
    het_.at(Tower)->Fill( itr->hwPt() );
    heta_.at(Tower)->Fill( itr->hwEta() );
    hphi_.at(Tower)->Fill( itr->hwPhi() );
    hem_.at(Tower)->Fill( itr->hwEtEm() );
    hhad_.at(Tower)->Fill( itr->hwEtHad() );
    hratio_.at(Tower)->Fill( itr->hwEtRatio() );
  }

  // get cluster
  Handle< BXVector<l1t::CaloCluster> > clusters;
  iEvent.getByToken(m_clusterToken,clusters);

  for ( auto itr = clusters->begin(0); itr !=clusters->end(0); ++itr ) {
    het_.at(Cluster)->Fill( itr->hwPt() );
    heta_.at(Cluster)->Fill( itr->hwEta() );
    hphi_.at(Cluster)->Fill( itr->hwPhi() );
  }

  // get EG
  Handle< BXVector<l1t::EGamma> > egs;
  iEvent.getByToken(m_egToken,egs);

  for ( auto itr = egs->begin(0); itr != egs->end(0); ++itr ) {
    het_.at(EG)->Fill( itr->hwPt() );
    heta_.at(EG)->Fill( itr->hwEta() );
    hphi_.at(EG)->Fill( itr->hwPhi() );
  }

  // get tau
  Handle< BXVector<l1t::Tau> > taus;
  iEvent.getByToken(m_tauToken,taus);

  for ( auto itr = taus->begin(0); itr != taus->end(0); ++itr ) {
    het_.at(Tau)->Fill( itr->hwPt() );
    heta_.at(Tau)->Fill( itr->hwEta() );
    hphi_.at(Tau)->Fill( itr->hwPhi() );
  }

  // get jet
  Handle< BXVector<l1t::Jet> > jets;
  iEvent.getByToken(m_jetToken,jets);

  for ( auto itr = jets->begin(0); itr != jets->end(0); ++itr ) {
    het_.at(Jet)->Fill( itr->hwPt() );
    heta_.at(Jet)->Fill( itr->hwEta() );
    hphi_.at(Jet)->Fill( itr->hwPhi() );
  }

  // get sums
  Handle< BXVector<l1t::EtSum> > sums;
  iEvent.getByToken(m_sumToken,sums);

  for ( auto itr = sums->begin(0); itr != sums->end(0); ++itr ) {
    het_.at(Sum)->Fill( itr->hwPt() );
    heta_.at(Sum)->Fill( itr->hwEta() );
    hphi_.at(Sum)->Fill( itr->hwPhi() );
  }

}


// ------------ method called once each job just before starting event loop  ------------
void 
Stage2CaloAnalyzer::beginJob()
{

  edm::Service<TFileService> fs;

  auto itr = types_.cbegin();
  auto str = typeStr_.cbegin();

  for (; itr!=types_.end(); ++itr, ++str ) {
    
    dirs_.insert( std::pair< ObjectType, TFileDirectory >(*itr, fs->mkdir(*str) ) );
    
    het_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("et", "", 50, 0., 100.) ));
    heta_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("eta", "", 70, -35., 35.) ));
    hphi_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("phi", "", 72, 0., 72.) ));
    
    if (*itr==Tower) {
      hem_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("em", "", 50, 0., 100.) ));
      hhad_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("had", "", 50, 0., 100.) ));
      hratio_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("ratio", "", 10, 0., 10.) ));
    }

  }

}

// ------------ method called once each job just after ending the event loop  ------------
void 
Stage2CaloAnalyzer::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
/*
void 
Stage2CaloAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void 
Stage2CaloAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
Stage2CaloAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
Stage2CaloAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
Stage2CaloAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

}

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::Stage2CaloAnalyzer);
