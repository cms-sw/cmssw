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
// Modified By: Adam Elwood
//        Date: Tue, 22 Apr 2014
//


// system include files
#include <memory>
#include <algorithm>

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
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/METReco/interface/GenMET.h"

#include "TString.h"
#include "TH2F.h"
#include "TGraphAsymmErrors.h"

//
// class declaration
//

class L1TCaloAnalyzer : public edm::EDAnalyzer {
  public:
    explicit L1TCaloAnalyzer(const edm::ParameterSet&);
    ~L1TCaloAnalyzer();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

      void make_cumu_hist(int32_t, TH1F *);

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
    edm::EDGetToken m_towerPreCompressionToken;
    edm::EDGetToken m_clusterToken;
    edm::EDGetToken m_egToken;
    edm::EDGetToken m_tauToken;
    edm::EDGetToken m_jetToken;
    edm::EDGetToken m_sumToken;

    enum ObjectType{Tower=0x1,
      TowerPreCompression=0x2,
      Cluster=0x3,
      EG=0x4,
      Tau=0x5,
      Jet=0x6,
      Sum=0x7,
      HSum=0x8};

    std::vector< ObjectType > types_;
    std::vector< std::string > typeStr_;

    std::map< ObjectType, TFileDirectory > dirs_;
    std::map< ObjectType, TH1F* > het_;
    std::map< ObjectType, TH1F* > hmet_;
    std::map< ObjectType, TH1F* > hrateEt_;
    std::map< ObjectType, TH1F* > hrateMet_;
    std::map< ObjectType, TH1F* > heta_;
    std::map< ObjectType, TH1F* > hphi_;
    std::map< ObjectType, TH1F* > hem_;
    std::map< ObjectType, TH1F* > hhad_;
    std::map< ObjectType, TH1F* > hratio_;

    //My own histograms
    std::map< TString, std::vector<double> > bins_;
    std::map< TString, std::vector<double> > binsESum_;
    std::map< TString, TFileDirectory > jetDirs_;
    std::map< TString, TH1F* > hjets_;//Histograms
    std::vector< TString > vars_;//Variables to plot
    std::vector< TString > categories_;//Categories of jets
    std::vector< TString > eSumCategories_;//Categories of e sums 

    //Comparison histograms
    std::map< TString, TH1F* > h1d_;
    std::map< TString, TH2F* > h2d_;

    //Turn on curves
    std::map< TString, TGraphAsymmErrors* > gTurnons_;
    std::map< TString, TH1F* > h1dTurnons_;
    std::vector< TString > turnonCuts_;
    std::vector< TString > turnonLevel_;

    //Resolution histograms
    std::map< TString, TH1F* > hResolution_;

    std::vector< TString > varLevel_;//The type of variable, eg gen or l1
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//My own function
//
bool L1JetRankDescending ( l1extra::L1JetParticle jet1, l1extra::L1JetParticle jet2 ){ return ( jet1.p4().Pt() > jet2.p4().Pt() ); }
bool GenJetRankDescending ( reco::GenJet jet1, reco::GenJet jet2 ){ return ( jet1.p4().Pt() > jet2.p4().Pt() ); }

const double PI = acos(-1.0);
int PhitoiPhi(double phiMissingEt){
  int intPhiMissingEt;
  if(phiMissingEt > 0)
  {
    intPhiMissingEt = int32_t((36.*(phiMissingEt)+0.5)/PI);
  }
  else
  {
    intPhiMissingEt = int32_t((36.*(phiMissingEt+2.*PI)+0.5)/PI);
  }
  return intPhiMissingEt;
}

double iPhitoPhi(int iPhi){//Based on the map https://twiki.cern.ch/twiki/bin/viewauth/CMS/RCTMap
  if(iPhi<37) return (5.0*iPhi-2.5)*PI/180.;
  else return (5.0*(iPhi-72)-2.5)*PI/180.; 
}

double iEtatoEta(int iEta){
  double etaMapping[57]={-3.0,-2.65,-2.5,-2.322,-2.172,-2.043,-1.93,-1.83,-1.74,-1.653,
    -1.566,-1.4790,-1.3920,-1.3050,-1.2180,-1.1310,-1.0440,-0.9570,
    -0.8700,-0.7830,-0.6950,-0.6090,-0.5220,-0.4350,-0.3480,-0.2610,
    -0.1740,-0.0870,0.0,0.0870,0.1740,0.2610,0.3480,0.4350,0.5220,0.6090,
    0.6950,0.7830,0.8700,0.9570,1.0440,1.1310,1.2180,1.3050,1.3920,
    1.4790,1.566,1.653,1.74,1.83,1.93,2.043,2.172,2.322,2.5,2.65,3.0};
  return etaMapping[iEta+28];
}


//
// constructors and destructor
//
L1TCaloAnalyzer::L1TCaloAnalyzer(const edm::ParameterSet& iConfig)
{
  //now do what ever initialization is needed

  // register what you consume and keep token for later access:
  m_towerToken   = consumes<l1t::CaloTowerBxCollection>  (iConfig.getParameter<edm::InputTag>("towerToken"));
  m_towerPreCompressionToken   = consumes<l1t::CaloTowerBxCollection>  (iConfig.getParameter<edm::InputTag>("towerPreCompressionToken"));
  m_clusterToken = consumes<l1t::CaloClusterBxCollection>(iConfig.getParameter<edm::InputTag>("clusterToken"));
  m_egToken      = consumes<l1t::EGammaBxCollection>     (iConfig.getParameter<edm::InputTag>("egToken"));
  m_tauToken     = consumes<l1t::TauBxCollection>        (iConfig.getParameter<edm::InputTag>("tauToken"));
  m_jetToken     = consumes<l1t::JetBxCollection>        (iConfig.getParameter<edm::InputTag>("jetToken"));
  m_sumToken     = consumes<l1t::EtSumBxCollection>      (iConfig.getParameter<edm::InputTag>("etSumToken"));

  types_.push_back( Tower );
  types_.push_back( TowerPreCompression );
  types_.push_back( Cluster );
  types_.push_back( EG );
  types_.push_back( Tau );
  types_.push_back( Jet );
  types_.push_back( Sum );
  types_.push_back( HSum );

  typeStr_.push_back( "tower" );
  typeStr_.push_back( "towerPreCompression" );
  typeStr_.push_back( "cluster" );
  typeStr_.push_back( "eg" );
  typeStr_.push_back( "tau" );
  typeStr_.push_back( "jet" );
  typeStr_.push_back( "sum" );
  typeStr_.push_back( "hsum" );

  vars_.push_back( "et" );
  bins_["et"].push_back(200.);
  bins_["et"].push_back(-0.5);
  bins_["et"].push_back(1999.5);
  vars_.push_back( "eta" );
  bins_["eta"].push_back(70.);
  bins_["eta"].push_back(-35.);
  bins_["eta"].push_back(35.);
  vars_.push_back( "phi" );
  bins_["phi"].push_back(72.);
  bins_["phi"].push_back(0.);
  bins_["phi"].push_back(72.);
  bins_["real_phi"].push_back(72.);
  bins_["real_phi"].push_back(-3.15);
  bins_["real_phi"].push_back(3.15);
  bins_["real_eta"].push_back(70.);
  bins_["real_eta"].push_back(-3.3);
  bins_["real_eta"].push_back(3.3);
  bins_["real_et"].push_back(200.);
  bins_["real_et"].push_back(-0.5);
  bins_["real_et"].push_back(1999.5);

  categories_.push_back( "lead_jet");
  categories_.push_back( "second_jet");
  categories_.push_back( "third_jet");
  categories_.push_back( "fourth_jet");
  categories_.push_back( "remaining_jets");

  varLevel_.push_back( "gen" );
  varLevel_.push_back( "l1_stage1" );

  eSumCategories_.push_back( "et" );
  binsESum_["et"].push_back(200.);
  binsESum_["et"].push_back(-0.5);
  binsESum_["et"].push_back(1999.5);
  eSumCategories_.push_back( "met" );
  binsESum_["met"].push_back(200.);
  binsESum_["met"].push_back(-0.5);
  binsESum_["met"].push_back(1999.5);
  eSumCategories_.push_back( "met_phi" );
  binsESum_["met_phi"].push_back(72.);
  binsESum_["met_phi"].push_back(-3.15);
  binsESum_["met_phi"].push_back(3.15);

  eSumCategories_.push_back( "ht" );
  binsESum_["ht"].push_back(200.);
  binsESum_["ht"].push_back(-0.5);
  binsESum_["ht"].push_back(1999.5);
  eSumCategories_.push_back( "mht" );
  binsESum_["mht"].push_back(200.);
  binsESum_["mht"].push_back(-0.5);
  binsESum_["mht"].push_back(1999.5);
  eSumCategories_.push_back( "mht_phi" );
  binsESum_["mht_phi"].push_back(72.);
  binsESum_["mht_phi"].push_back(-3.15);
  binsESum_["mht_phi"].push_back(3.15);

  turnonLevel_.push_back("stage1");
  turnonLevel_.push_back("stage2");

  turnonCuts_.push_back("0");
  turnonCuts_.push_back("30");
  turnonCuts_.push_back("40");
  turnonCuts_.push_back("50");
  turnonCuts_.push_back("100");
  turnonCuts_.push_back("200");
  turnonCuts_.push_back("400");

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

  // get towers pre compression
  Handle< BXVector<l1t::CaloTower> > towersPreCompression;
  iEvent.getByToken(m_towerPreCompressionToken,towersPreCompression);

  for ( auto itr = towersPreCompression->begin(0); itr != towersPreCompression->end(0); ++itr ) {
    het_.at(TowerPreCompression)->Fill( itr->hwPt() );
    heta_.at(TowerPreCompression)->Fill( itr->hwEta() );
    hphi_.at(TowerPreCompression)->Fill( itr->hwPhi() );
    hem_.at(TowerPreCompression)->Fill( itr->hwEtEm() );
    hhad_.at(TowerPreCompression)->Fill( itr->hwEtHad() );
    hratio_.at(TowerPreCompression)->Fill( itr->hwEtRatio() );
  }

  //Find the difference pre compression and post compression
  if(towers->size(0) != towersPreCompression->size(0)){
    std::cout << "Difference in compressed and non compressed collections \n";
  }else{
    for(unsigned i=0; i<towers->size(0); ++i){
      if(towers->at(0,i).hwEtEm() != 0 || towersPreCompression->at(0,i).hwEtEm() != 0){ //Don't fill if both 0
	h1d_["tow_em_diff"]->Fill(towers->at(0,i).hwEtEm() - towersPreCompression->at(0,i).hwEtEm());
	h2d_["tow_em_comp"]->Fill(towers->at(0,i).hwEtEm(),towersPreCompression->at(0,i).hwEtEm());
      }
      if(towers->at(0,i).hwEtHad() != 0 || towersPreCompression->at(0,i).hwEtHad() != 0){
	h1d_["tow_had_diff"]->Fill(towers->at(0,i).hwEtHad() - towersPreCompression->at(0,i).hwEtHad());
	h2d_["tow_had_comp"]->Fill(towers->at(0,i).hwEtHad(),towersPreCompression->at(0,i).hwEtHad());
      }
    }
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

  //Get the l1 extra collection

  Handle< std::vector<l1extra::L1EtMissParticle> > l1S1EtMiss;
  iEvent.getByLabel("l1extraParticles","MET",l1S1EtMiss);

  Handle< std::vector<l1extra::L1EtMissParticle> > l1S1HtMiss;
  iEvent.getByLabel("l1extraParticles","MHT",l1S1HtMiss);

  Handle< std::vector<reco::GenMET> > genMet;
  iEvent.getByLabel("genMetCalo", genMet);

  // get l1 sums
  Handle< BXVector<l1t::EtSum> > sums;
  iEvent.getByToken(m_sumToken,sums);

  //Get the l1 jets in the central region
  Handle< std::vector<l1extra::L1JetParticle> > l1GctJet;
  iEvent.getByLabel("l1extraParticles","Central", l1GctJet);

  //Get the gen jets
  Handle< std::vector<reco::GenJet> > genJet;
  iEvent.getByLabel("ak4GenJets", genJet);

  //Make vectors of jets that are within eta of 3

  std::vector< l1extra::L1JetParticle > centralL1GctJet;
  std::vector< reco::GenJet > centralGenJet;

  for(auto itr = l1GctJet->begin(); itr != l1GctJet->end(); itr++){
    if(fabs(itr->eta()) > 3.0) continue;
    centralL1GctJet.push_back(*itr);
  }

  for(auto itr = genJet->begin(); itr != genJet->end(); itr++){
    if(fabs(itr->eta()) > 3.0) continue;
    centralGenJet.push_back(*itr);
  }

  //Sort the jets to make sure they're in energy order

  std::sort( centralL1GctJet.begin(), centralL1GctJet.end(),  L1JetRankDescending );
  std::sort( centralGenJet.begin(), centralGenJet.end(),  GenJetRankDescending );


  //Make the histograms
  //
  //For the energy sums
  for(auto itr = sums->begin(0); itr != sums->end(0); itr++){

    if(itr->getType() == l1t::EtSum::EtSumType::kTotalEt){
      het_.at(Sum)->Fill( itr->hwPt() );
      L1TCaloAnalyzer::make_cumu_hist(itr->hwPt(), hrateEt_.at(Sum));

      double realEt=0.5*itr->hwPt();

      if (l1S1EtMiss->size()>0) h2d_["et_l1_stage1"]->Fill( l1S1EtMiss->at(0).etTotal(), realEt );
      h2d_["et_gen"]->Fill( genMet->at(0).sumEt(), realEt );

      //For the turnons
      for(auto cut=turnonCuts_.begin(); cut!=turnonCuts_.end();cut++ ){
	if(realEt > atof(cut->Data())) h1dTurnons_.at("et_stage2_"+*cut)->Fill( genMet->at(0).sumEt() );
	if(l1S1EtMiss->size() > 0 && l1S1EtMiss->at(0).etTotal() > atof(cut->Data())) h1dTurnons_.at("et_stage1_"+*cut)->Fill( genMet->at(0).sumEt() );
      }
    }
    if(itr->getType() == l1t::EtSum::EtSumType::kMissingEt){
      hmet_.at(Sum)->Fill( itr->hwPt() );
      L1TCaloAnalyzer::make_cumu_hist(itr->hwPt(), hrateMet_.at(Sum));
      double realEt=(1.0/1022.0)*itr->hwPt();

      if(l1S1EtMiss->size()>0) h2d_["met_l1_stage1"]->Fill( l1S1EtMiss->at(0).etMiss(), realEt );
      h2d_["met_gen"]->Fill( genMet->at(0).et(), realEt );
      heta_.at(Sum)->Fill( itr->hwEta() );
      hphi_.at(Sum)->Fill( itr->hwPhi() );
      if(l1S1EtMiss->size()>0) h2d_["met_phi_l1_stage1"]->Fill( l1S1EtMiss->at(0).phi(), iPhitoPhi(itr->hwPhi()) );
      h2d_["met_phi_gen"]->Fill( genMet->at(0).phi(), iPhitoPhi(itr->hwPhi()) );

      //For the turnons
      for(auto cut=turnonCuts_.begin(); cut!=turnonCuts_.end();cut++ ){
	if(realEt > atof(cut->Data())) h1dTurnons_.at("met_stage2_"+*cut)->Fill( genMet->at(0).et() );
	if(l1S1EtMiss->size()>0 && l1S1EtMiss->at(0).etMiss() > atof(cut->Data())) h1dTurnons_.at("met_stage1_"+*cut)->Fill( genMet->at(0).et() );
      }

    }
    if(itr->getType() == l1t::EtSum::EtSumType::kTotalHt){
      het_.at(HSum)->Fill( itr->hwPt() );
      std::cout <<"HT" << itr->hwPt() << std::endl;
      L1TCaloAnalyzer::make_cumu_hist(itr->hwPt(), hrateEt_.at(HSum));
      if (l1S1HtMiss->size()>0) h2d_["ht_l1_stage1"]->Fill( l1S1HtMiss->at(0).etTotal(), 0.5*itr->hwPt() );
      //h2d_["ht_gen"]->Fill( genMHt->at(0).sumEt(), 0.5*itr->hwPt() );
    }
    if(itr->getType() == l1t::EtSum::EtSumType::kMissingHt){
      hmet_.at(HSum)->Fill( itr->hwPt() );
      std::cout <<"MHT" << itr->hwPt() << std::endl;
      L1TCaloAnalyzer::make_cumu_hist(itr->hwPt(), hrateMet_.at(HSum));
      if(l1S1HtMiss->size()>0)      h2d_["mht_l1_stage1"]->Fill( l1S1HtMiss->at(0).etMiss(), (1.0/1022.0)*itr->hwPt() );
      //h2d_["mht_gen"]->Fill( genMht->at(0).et(), (1.0/511.0)*itr->hwPt() );
      heta_.at(HSum)->Fill( itr->hwEta() );
      hphi_.at(HSum)->Fill( itr->hwPhi() );
      if(l1S1HtMiss->size()>0)     h2d_["mht_phi_l1_stage1"]->Fill( l1S1HtMiss->at(0).phi(), -1.0*iPhitoPhi(itr->hwPhi()) );
      //h2d_["mht_phi_gen"]->Fill( genMht->at(0).phi(), iPhitoPhi(itr->hwPhi()) );
    }


  }

  // Categorise the jets and make histograms
  double leadEt=0.0, secondEt=0.0, thirdEt=0.0, fourthEt=0.0;
  double leadEta=0.0, secondEta=0.0, thirdEta=0.0, fourthEta=0.0;
  double leadPhi=0.0, secondPhi=0.0, thirdPhi=0.0, fourthPhi=0.0;

  for ( auto itr = jets->begin(0); itr != jets->end(0); ++itr ) {
    het_.at(Jet)->Fill( itr->hwPt() );
    heta_.at(Jet)->Fill( itr->hwEta() );
    hphi_.at(Jet)->Fill( itr->hwPhi() );

    //Make sure just use central jets for comparison
    if(abs(itr->hwEta()) > 28) continue;

    if(itr->hwPt() > leadEt){
      //If there was a fourth jet, there are remaining jets
      if(fourthEt > 0.01){
	hjets_["remaining_jets_et"]->Fill( fourthEt );
	hjets_["remaining_jets_eta"]->Fill( fourthEta );
	hjets_["remaining_jets_phi"]->Fill( fourthPhi );
      }

      //Find the leading energy and prop through
      fourthEt=thirdEt;
      fourthEta=thirdEta;
      fourthPhi=thirdPhi;
      thirdEt=secondEt;
      thirdEta=secondEta;
      thirdPhi=secondPhi;
      secondEt=leadEt;
      secondEta=leadEta;
      secondPhi=leadPhi;
      leadEt=itr->hwPt();
      leadEta=itr->hwEta();
      leadPhi=itr->hwPhi();
    }
    else if(itr->hwPt() > secondEt){
      //If there was a fourth jet, there are remaining jets
      if(fourthEt > 0.01){
	hjets_["remaining_jets_et"]->Fill( fourthEt );
	hjets_["remaining_jets_eta"]->Fill( fourthEta );
	hjets_["remaining_jets_phi"]->Fill( fourthPhi );
      }
      fourthEt=thirdEt;
      fourthEta=thirdEta;
      fourthPhi=thirdPhi;
      thirdEt=secondEt;
      thirdEta=secondEta;
      thirdPhi=secondPhi;
      secondEt=itr->hwPt();
      secondEta=itr->hwEta();
      secondPhi=itr->hwPhi();
    }
    else if(itr->hwPt() > thirdEt){
      //If there was a fourth jet, there are remaining jets
      if(fourthEt>0.01){
	hjets_["remaining_jets_et"]->Fill( fourthEt );
	hjets_["remaining_jets_eta"]->Fill( fourthEta );
	hjets_["remaining_jets_phi"]->Fill( fourthPhi );
      }
      fourthEt=thirdEt;
      fourthEta=thirdEta;
      fourthPhi=thirdPhi;
      thirdEt=itr->hwPt();
      thirdEta=itr->hwEta();
      thirdPhi=itr->hwPhi();
    }
    else if(itr->hwPt() > fourthEt){
      //If there was a fourth jet, there are remaining jets
      if(fourthEt>0.01){
	hjets_["remaining_jets_et"]->Fill( fourthEt );
	hjets_["remaining_jets_eta"]->Fill( fourthEta );
	hjets_["remaining_jets_phi"]->Fill( fourthPhi );
      }
      fourthEt=itr->hwPt();
      fourthEta=itr->hwEta();
      fourthPhi=itr->hwPhi();
    }


  }

  //Fill the jet histograms, multiply all jet energies by half to take account of l1 units
  if(leadEt>0.01){
    hjets_["lead_jet_et"]->Fill( leadEt );
    hjets_["lead_jet_eta"]->Fill( leadEta );
    hjets_["lead_jet_phi"]->Fill( leadPhi );
    if(centralL1GctJet.size()>0){
      h2d_["lead_jet_l1_stage1_et"]->Fill( centralL1GctJet.at(0).pt(),0.5*leadEt );
      h2d_["lead_jet_l1_stage1_eta"]->Fill( centralL1GctJet.at(0).eta(), iEtatoEta(leadEta) );
      h2d_["lead_jet_l1_stage1_phi"]->Fill( centralL1GctJet.at(0).phi(), iPhitoPhi(leadPhi) );
    }
    if(centralGenJet.size()>0){
      h2d_["lead_jet_gen_et"]->Fill( centralGenJet.at(0).pt(),0.5*leadEt );
      h2d_["lead_jet_gen_eta"]->Fill( centralGenJet.at(0).eta(), iEtatoEta(leadEta) );
      h2d_["lead_jet_gen_phi"]->Fill( centralGenJet.at(0).phi(), iPhitoPhi(leadPhi) );
      //For the turnons
      leadEt = 0.5*leadEt;
      for(auto cut=turnonCuts_.begin(); cut!=turnonCuts_.end();cut++ ){
	if(leadEt > atof(cut->Data())) h1dTurnons_.at("lead_jet_stage2_"+*cut)->Fill( centralGenJet.at(0).pt() );
	if(centralL1GctJet.size()>0){
	  if(centralL1GctJet.at(0).pt() > atof(cut->Data())) h1dTurnons_.at("lead_jet_stage1_"+*cut)->Fill( centralGenJet.at(0).pt() );
	}
      }
    }
  }
  if(secondEt>0.01){
    hjets_["second_jet_et"]->Fill( secondEt );
    hjets_["second_jet_eta"]->Fill( secondEta );
    hjets_["second_jet_phi"]->Fill( secondPhi );
    if(centralL1GctJet.size()>1){
      h2d_["second_jet_l1_stage1_et"]->Fill( centralL1GctJet.at(1).pt(),0.5*secondEt );
      h2d_["second_jet_l1_stage1_eta"]->Fill( centralL1GctJet.at(1).eta(), iEtatoEta(secondEta) );
      h2d_["second_jet_l1_stage1_phi"]->Fill( centralL1GctJet.at(1).phi(), iPhitoPhi(secondPhi) );
    }
    if(centralGenJet.size()>1){
      h2d_["second_jet_gen_eta"]->Fill( centralGenJet.at(1).eta(), iEtatoEta(secondEta) );
      h2d_["second_jet_gen_et"]->Fill( centralGenJet.at(1).pt(),0.5*secondEt );
      h2d_["second_jet_gen_phi"]->Fill( centralGenJet.at(1).phi(), iPhitoPhi(secondPhi) );
      //For the turnons
      secondEt = 0.5*secondEt;
      for(auto cut=turnonCuts_.begin(); cut!=turnonCuts_.end();cut++ ){
	if(secondEt > atof(cut->Data())) h1dTurnons_.at("second_jet_stage2_"+*cut)->Fill( centralGenJet.at(1).pt() );
	if(centralL1GctJet.size()>1){
	  if(centralL1GctJet.at(1).pt() > atof(cut->Data())) h1dTurnons_.at("second_jet_stage1_"+*cut)->Fill( centralGenJet.at(1).pt() );
	}
      }
    }
  }
  if(thirdEt>0.01){
    hjets_["third_jet_et"]->Fill( thirdEt );
    hjets_["third_jet_eta"]->Fill( thirdEta );
    hjets_["third_jet_phi"]->Fill( thirdPhi );
    if(centralL1GctJet.size()>2){
      h2d_["third_jet_l1_stage1_et"]->Fill( centralL1GctJet.at(2).pt(),0.5*thirdEt );
      h2d_["third_jet_l1_stage1_eta"]->Fill( centralL1GctJet.at(2).eta(), iEtatoEta(thirdEta) );
      h2d_["third_jet_l1_stage1_phi"]->Fill( centralL1GctJet.at(2).phi(), iPhitoPhi(thirdPhi) );
    }
    if(centralGenJet.size()>2){
      h2d_["third_jet_gen_eta"]->Fill( centralGenJet.at(2).eta(), iEtatoEta(thirdEta) );
      h2d_["third_jet_gen_et"]->Fill( centralGenJet.at(2).pt(),0.5*thirdEt );
      h2d_["third_jet_gen_phi"]->Fill( centralGenJet.at(2).phi(), iPhitoPhi(thirdPhi) );

      //For the turnons
      thirdEt = 0.5*thirdEt;
      for(auto cut=turnonCuts_.begin(); cut!=turnonCuts_.end();cut++ ){
	if(thirdEt > atof(cut->Data())) h1dTurnons_.at("third_jet_stage2_"+*cut)->Fill( centralGenJet.at(2).pt() );
	if(centralL1GctJet.size()>2){
	  if(centralL1GctJet.at(2).pt() > atof(cut->Data())) h1dTurnons_.at("third_jet_stage1_"+*cut)->Fill( centralGenJet.at(2).pt() );
	}
      }
    }
  }
  if(fourthEt>0.01){
    hjets_["fourth_jet_et"]->Fill( fourthEt );
    hjets_["fourth_jet_eta"]->Fill( fourthEta );
    hjets_["fourth_jet_phi"]->Fill( fourthPhi );
    if(centralL1GctJet.size()>3){
      h2d_["fourth_jet_l1_stage1_et"]->Fill( centralL1GctJet.at(3).pt(),0.5*fourthEt );
      h2d_["fourth_jet_l1_stage1_eta"]->Fill( centralL1GctJet.at(3).eta(), iEtatoEta(fourthEta) );
      h2d_["fourth_jet_l1_stage1_phi"]->Fill( centralL1GctJet.at(3).phi(), iPhitoPhi(fourthPhi) );
    }
    if(centralGenJet.size()>3){
      h2d_["fourth_jet_gen_eta"]->Fill( centralGenJet.at(3).eta(), iEtatoEta(fourthEta) );
      h2d_["fourth_jet_gen_et"]->Fill( centralGenJet.at(3).pt(),0.5*fourthEt );
      h2d_["fourth_jet_gen_phi"]->Fill( centralGenJet.at(3).phi(), iPhitoPhi(fourthPhi) );

      //For the turnons
      fourthEt = 0.5*fourthEt;
      for(auto cut=turnonCuts_.begin(); cut!=turnonCuts_.end();cut++ ){
	if(fourthEt > atof(cut->Data())) h1dTurnons_.at("fourth_jet_stage2_"+*cut)->Fill( centralGenJet.at(3).pt() );
	if(centralL1GctJet.size()>3){
	  if(centralL1GctJet.at(3).pt() > atof(cut->Data())) h1dTurnons_.at("fourth_jet_stage1_"+*cut)->Fill( centralGenJet.at(3).pt() );
	}
      }
    }
  }
}

void L1TCaloAnalyzer::make_cumu_hist(int32_t e_event,TH1F *h)
{
  //   std::cout << e_event << std::endl;
  int nbins = h->GetXaxis()->FindBin(e_event);
  for (int bins = 0; bins < nbins; bins++)
  {
    h->AddBinContent(bins,1);
  }
}

// ------------ method called once each job just before starting event loop  ------------
  void 
L1TCaloAnalyzer::beginJob()
{

  edm::Service<TFileService> fs;

  auto itr = types_.cbegin();
  auto str = typeStr_.cbegin();

  for (; itr!=types_.end(); ++itr, ++str ) {

    dirs_.insert( std::pair< ObjectType, TFileDirectory >(*itr, fs->mkdir(*str) ) );

    het_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("et", "Full ET;ET [l1 units];", 4000, 0., 8000.) ));
    hrateEt_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("et rate", "Full ET;ET [l1 units];", 4000, 0., 8000.) ));
    hmet_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("met", "MET;MET [l1 units];", 5000, -10., 99990.) ));
    hrateMet_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("met rate", "MET;MET [l1 units];", 5000, -10., 99990.) ));
    heta_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("eta", "Eta;Eta [l1 units];", 70, -35., 35.) ));
    hphi_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("phi", "Phi;Phi [l1 units]", 72, 0., 72.) ));

    if (*itr==Tower || *itr==TowerPreCompression) {
      hem_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("em", "EM only; ET [l1 units];", 500, 0., 1000.) ));
      hhad_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("had", "Had only; ET [l1 units]; ", 500, 0., 1000.) ));
      hratio_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("ratio", "", 10, 0., 10.) ));
    }

  }


  //Add histograms exclusively for jets, based on category and vars

  for(auto catIt = categories_.cbegin(); catIt!= categories_.end(); ++catIt){

    jetDirs_[*catIt] = fs->mkdir(catIt->Data());

    for(auto varIt = vars_.cbegin(); varIt!=vars_.end(); ++varIt){
      hjets_[*catIt+"_"+*varIt] = jetDirs_.at(*catIt).make<TH1F>(*varIt, *catIt+";"+*varIt+" [l1 units]", 
	  (int)bins_.at(*varIt)[0],bins_.at(*varIt)[1],bins_.at(*varIt)[2]);
    }

  }

  //Add histograms for comparing towers
  jetDirs_["compression_comp"] = fs->mkdir("compression_comp");
  h2d_["tow_em_comp"]=jetDirs_.at("compression_comp").make<TH2F>("tow_em_comp",
      "Tower em before vs after compression;EM after [L1 units];EM before [L1 units]",100,0.,300.,100,0.,300.);
  h2d_["tow_had_comp"]=jetDirs_.at("compression_comp").make<TH2F>("tow_had_comp",
      "Tower had before vs after compression;Had after [L1 units];Had before [L1 units]",100,0.,300.,100,0.,300.);
  h1d_["tow_em_diff"]=jetDirs_.at("compression_comp").make<TH1F>("tow_em_diff",
      "Tower em (after - before) compression;(after-before) [L1 units];",200,-300.,300.);
  h1d_["tow_had_diff"]=jetDirs_.at("compression_comp").make<TH1F>("tow_had_diff",
      "Tower had (after - before) compression;(after-before) [L1 units];",200,-300.,300.);

  //Add 2d histograms for comparison for the jets, based on var and var level


  for(auto catIt = categories_.cbegin(); catIt!= categories_.end(); ++catIt){
    for(auto lvlIt = varLevel_.cbegin(); lvlIt!= varLevel_.end(); ++lvlIt){
      for(auto varIt = vars_.cbegin(); varIt!=vars_.end(); ++varIt){

	h2d_[*catIt+"_"+*lvlIt+"_"+*varIt] = jetDirs_.at(*catIt).make<TH2F>(*varIt+"_"+*lvlIt, *catIt+"_"+*varIt+";"+*lvlIt+";new l1;[standard units]", 
	    (int)bins_.at("real_"+*varIt)[0],bins_.at("real_"+*varIt)[1],bins_.at("real_"+*varIt)[2],
	    (int)bins_.at("real_"+*varIt)[0],bins_.at("real_"+*varIt)[1],bins_.at("real_"+*varIt)[2]);

      }
    }
  }

  //Add 2d histograms for the energy sums

  jetDirs_["eSums"] = fs->mkdir("eSums");
  for(auto lvlIt = eSumCategories_.cbegin(); lvlIt!= eSumCategories_.end(); ++lvlIt){

    for(auto varIt = varLevel_.cbegin(); varIt!=varLevel_.end(); ++varIt){
      h2d_[*lvlIt+"_"+*varIt] = jetDirs_["eSums"].make<TH2F>(*lvlIt+"_"+*varIt, *lvlIt+";"+*varIt+";new l1", 
	  (int)binsESum_.at(*lvlIt)[0],binsESum_.at(*lvlIt)[1],binsESum_.at(*lvlIt)[2],
	  (int)binsESum_.at(*lvlIt)[0],binsESum_.at(*lvlIt)[1],binsESum_.at(*lvlIt)[2]);
    }

  }


  //Add turnon curves for the et of the jets, and energy sums


  for(auto cut = turnonCuts_.cbegin(); cut!=turnonCuts_.end(); ++cut){
    for(auto lvl = turnonLevel_.cbegin(); lvl!=turnonLevel_.end(); ++lvl){
      for(auto catIt = categories_.cbegin(); catIt!= categories_.end(); ++catIt){
	h1dTurnons_[*catIt+"_"+*lvl+"_"+*cut] = jetDirs_.at(*catIt).make<TH1F>(*catIt+"_"+*lvl+"_cut"+*cut,*catIt+"_"+*lvl+"_cut"+*cut,
	    bins_.at("real_et")[0], bins_.at("real_et")[1], bins_.at("real_et")[2]);
	gTurnons_[*catIt+"_"+*lvl+"_"+*cut] = jetDirs_.at(*catIt).make<TGraphAsymmErrors>();
	gTurnons_[*catIt+"_"+*lvl+"_"+*cut]->SetName(*catIt+"_"+*lvl+"_turnon"+*cut);
	gTurnons_[*catIt+"_"+*lvl+"_"+*cut]->SetTitle(*catIt+"_"+*lvl+"_turnon"+*cut+";Jet ET (GeV)");
      }
      h1dTurnons_["et_"+*lvl+"_"+*cut] = jetDirs_.at("eSums").make<TH1F>("et_"+*lvl+"_cut"+*cut,"et_"+*lvl+"_cut"+*cut,
	  bins_.at("real_et")[0], bins_.at("real_et")[1], bins_.at("real_et")[2]);
      gTurnons_["et_"+*lvl+"_"+*cut] = jetDirs_.at("eSums").make<TGraphAsymmErrors>();
      gTurnons_["et_"+*lvl+"_"+*cut]->SetName("etTotal_"+*lvl+"_turnon"+*cut);
      gTurnons_["et_"+*lvl+"_"+*cut]->SetTitle("etTotal_"+*lvl+"_turnon"+*cut+";Total ET (GeV)");
      h1dTurnons_["met_"+*lvl+"_"+*cut] = jetDirs_.at("eSums").make<TH1F>("met_"+*lvl+"_cut"+*cut,"met_"+*lvl+"_cut"+*cut,
	  bins_.at("real_et")[0], bins_.at("real_et")[1], bins_.at("real_et")[2]);
      gTurnons_["met_"+*lvl+"_"+*cut] = jetDirs_.at("eSums").make<TGraphAsymmErrors>();
      gTurnons_["met_"+*lvl+"_"+*cut]->SetName("met_"+*lvl+"_turnon"+*cut);
      gTurnons_["met_"+*lvl+"_"+*cut]->SetTitle("met_"+*lvl+"_turnon"+*cut+";MET (GeV)");
    }
  }

  //Add turnon curves for the met and et

}

// ------------ method called once each job just after ending the event loop  ------------
  void 
L1TCaloAnalyzer::endJob() 
{
  //Fill the turnons
  for(auto cut = turnonCuts_.cbegin(); cut!=turnonCuts_.end(); ++cut){
    for(auto lvl = turnonLevel_.cbegin(); lvl!=turnonLevel_.end(); ++lvl){

      for(auto catIt = categories_.cbegin(); catIt!= categories_.end(); ++catIt){
	gTurnons_[*catIt+"_"+*lvl+"_"+*cut]->Divide(h1dTurnons_[*catIt+"_"+*lvl+"_"+*cut], h1dTurnons_[*catIt+"_"+*lvl+"_0"]);
      }
      gTurnons_["et_"+*lvl+"_"+*cut]->Divide(h1dTurnons_["et_"+*lvl+"_"+*cut], h1dTurnons_["et_"+*lvl+"_0"]);
      gTurnons_["met_"+*lvl+"_"+*cut]->Divide(h1dTurnons_["met_"+*lvl+"_"+*cut], h1dTurnons_["met_"+*lvl+"_0"]);
    }
  }

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
