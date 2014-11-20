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

#include "TH1F.h"
#include "TH2F.h"

//
// class declaration
//

namespace l1t {

class L1TStage2CaloAnalyzer : public edm::EDAnalyzer {
public:
  explicit L1TStage2CaloAnalyzer(const edm::ParameterSet&);
  ~L1TStage2CaloAnalyzer();
  
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

  bool m_doTowers;
  bool m_doClusters;
  bool m_doEGs;
  bool m_doTaus;
  bool m_doJets;
  bool m_doSums;
  
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
  std::map< ObjectType, TH1F* > hbx_;
  std::map< ObjectType, TH1F* > hem_;
  std::map< ObjectType, TH1F* > hhad_;
  std::map< ObjectType, TH1F* > hratio_;
  std::map< ObjectType, TH2F* > hetaphi_;

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
L1TStage2CaloAnalyzer::L1TStage2CaloAnalyzer(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

  // register what you consume and keep token for later access:
  edm::InputTag nullTag("None");

  edm::InputTag towerTag = iConfig.getParameter<edm::InputTag>("towerToken");
  m_towerToken         = consumes<l1t::CaloTowerBxCollection>(towerTag);
  m_doTowers           = !(towerTag==nullTag);

  edm::InputTag clusterTag = iConfig.getParameter<edm::InputTag>("clusterToken");
  m_clusterToken         = consumes<l1t::CaloClusterBxCollection>(clusterTag);
  m_doClusters           = !(clusterTag==nullTag);

  edm::InputTag egTag  = iConfig.getParameter<edm::InputTag>("egToken");
  m_egToken          = consumes<l1t::EGammaBxCollection>(egTag);
  m_doEGs            = !(egTag==nullTag);

  edm::InputTag tauTag = iConfig.getParameter<edm::InputTag>("tauToken");
  m_tauToken         = consumes<l1t::TauBxCollection>(tauTag);
  m_doTaus           = !(tauTag==nullTag);

  edm::InputTag jetTag = iConfig.getParameter<edm::InputTag>("jetToken");
  m_jetToken         = consumes<l1t::JetBxCollection>(jetTag);
  m_doJets           = !(jetTag==nullTag);  

  edm::InputTag sumTag = iConfig.getParameter<edm::InputTag>("etSumToken");
  m_sumToken         = consumes<l1t::EtSumBxCollection>(sumTag);
  m_doSums           = !(sumTag==nullTag);

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


L1TStage2CaloAnalyzer::~L1TStage2CaloAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
L1TStage2CaloAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  // get TPs ?
  // get regions ?
  // get RCT clusters ?

  // get towers
  if (m_doTowers) {
    Handle< BXVector<l1t::CaloTower> > towers;
    iEvent.getByToken(m_towerToken,towers);

    for ( int ibx=towers->getFirstBX(); ibx<=towers->getLastBX(); ++ibx) {

      if (ibx>-6 && ibx<6) {
	hbx_.at(Tower)->Fill( ibx );
      }

      for ( auto itr = towers->begin(ibx); itr !=towers->end(ibx); ++itr ) {
	het_.at(Tower)->Fill( itr->hwPt() );
	heta_.at(Tower)->Fill( itr->hwEta() );
	hphi_.at(Tower)->Fill( itr->hwPhi() );
	hem_.at(Tower)->Fill( itr->hwEtEm() );
	hhad_.at(Tower)->Fill( itr->hwEtHad() );
	hratio_.at(Tower)->Fill( itr->hwEtRatio() );
        hetaphi_.at(Tower)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );
      }

    }

  }

  // get cluster
  if (m_doClusters) {
    Handle< BXVector<l1t::CaloCluster> > clusters;
    iEvent.getByToken(m_clusterToken,clusters);

    for ( int ibx=clusters->getFirstBX(); ibx<=clusters->getLastBX(); ++ibx) {

      if (ibx>-6 && ibx<6) {
  	hbx_.at(Cluster)->Fill( ibx );
      }
    
      for ( auto itr = clusters->begin(ibx); itr !=clusters->end(ibx); ++itr ) {
  	het_.at(Cluster)->Fill( itr->hwPt() );
  	heta_.at(Cluster)->Fill( itr->hwEta() );
  	hphi_.at(Cluster)->Fill( itr->hwPhi() );
        hetaphi_.at(Cluster)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );
      }

    }
    
  }

    // get EG
  if (m_doEGs) {
    Handle< BXVector<l1t::EGamma> > egs;
    iEvent.getByToken(m_egToken,egs);
    
    for ( int ibx=egs->getFirstBX(); ibx<=egs->getLastBX(); ++ibx) {

      if (ibx>-6 && ibx<6) {
        hbx_.at(EG)->Fill( ibx );
      }
      
      for ( auto itr = egs->begin(ibx); itr != egs->end(ibx); ++itr ) {
	het_.at(EG)->Fill( itr->hwPt() );
	heta_.at(EG)->Fill( itr->hwEta() );
	hphi_.at(EG)->Fill( itr->hwPhi() );
        hetaphi_.at(EG)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );
      }
      
    }

  }

  // get tau
  if (m_doTaus) {
    Handle< BXVector<l1t::Tau> > taus;
    iEvent.getByToken(m_tauToken,taus);
    
    for ( int ibx=taus->getFirstBX(); ibx<=taus->getLastBX(); ++ibx) {

      if (ibx>-6 && ibx<6) {
        hbx_.at(Tau)->Fill( ibx );
      }

      for ( auto itr = taus->begin(ibx); itr != taus->end(ibx); ++itr ) {
	het_.at(Tau)->Fill( itr->hwPt() );
	heta_.at(Tau)->Fill( itr->hwEta() );
	hphi_.at(Tau)->Fill( itr->hwPhi() );
        hetaphi_.at(Tau)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );
      }
      
    }
    
  }

  // get jet
  if (m_doJets) {
    Handle< BXVector<l1t::Jet> > jets;
    iEvent.getByToken(m_jetToken,jets);
    
    for ( int ibx=jets->getFirstBX(); ibx<=jets->getLastBX(); ++ibx) {

      if (ibx>-6 && ibx<6) {
        hbx_.at(Jet)->Fill( ibx );
      }

      for ( auto itr = jets->begin(ibx); itr != jets->end(ibx); ++itr ) {
	het_.at(Jet)->Fill( itr->hwPt() );
	heta_.at(Jet)->Fill( itr->hwEta() );
	hphi_.at(Jet)->Fill( itr->hwPhi() );
        hetaphi_.at(Jet)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );
      }
      
    }

  }

  // get sums
  if (m_doSums) {
    Handle< BXVector<l1t::EtSum> > sums;
    iEvent.getByToken(m_sumToken,sums);
    
    for ( int ibx=sums->getFirstBX(); ibx<=sums->getLastBX(); ++ibx) {

      if (ibx>-6 && ibx<6) {
        hbx_.at(Sum)->Fill( ibx );
      }
      
      for ( auto itr = sums->begin(ibx); itr != sums->end(ibx); ++itr ) {
	het_.at(Sum)->Fill( itr->hwPt() );
	heta_.at(Sum)->Fill( itr->hwEta() );
	hphi_.at(Sum)->Fill( itr->hwPhi() );
        hetaphi_.at(Sum)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );
      }

    }

  }

}


// ------------ method called once each job just before starting event loop  ------------
void 
L1TStage2CaloAnalyzer::beginJob()
{

  edm::Service<TFileService> fs;

  auto itr = types_.cbegin();
  auto str = typeStr_.cbegin();

  for (; itr!=types_.end(); ++itr, ++str ) {
    
    dirs_.insert( std::pair< ObjectType, TFileDirectory >(*itr, fs->mkdir(*str) ) );
    
    het_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("et", "", 101, -0.5, 100.5) ));
    heta_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("eta", "", 83, -41.5, 41.5) ));
    hphi_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("phi", "", 73, 0.5, 72.5) ));
    hbx_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("bx", "", 11, -5.5, 5.5) ));
    hetaphi_.insert( std::pair< ObjectType, TH2F* >(*itr, dirs_.at(*itr).make<TH2F>("etaphi", "", 83, -41.5, 41.5, 72, .5, 72.5) ));
    
    if (*itr==Tower) {
      hem_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("em", "", 101, -0.5, 100.5) ));
      hhad_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("had", "", 101, -0.5, 100.5) ));
      hratio_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("ratio", "", 11, -0.5, 10.5) ));
    }

  }

}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TStage2CaloAnalyzer::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
/*
void 
L1TStage2CaloAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void 
L1TStage2CaloAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
L1TStage2CaloAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
L1TStage2CaloAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TStage2CaloAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

}

using namespace l1t;

//define this as a plug-in
DEFINE_FWK_MODULE(L1TStage2CaloAnalyzer);
