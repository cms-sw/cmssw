#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

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
    ~L1TStage2CaloAnalyzer() override;
  
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  
  private:
    void beginJob() override;
    void analyze(const edm::Event&, const edm::EventSetup&) override;
    void endJob() override;
  
    //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
    //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
    //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  
    // ----------member data ---------------------------
    edm::EDGetToken m_towerToken;
    edm::EDGetToken m_clusterToken;
    edm::EDGetToken m_mpEGToken;
    edm::EDGetToken m_mpTauToken;
    edm::EDGetToken m_mpJetToken;
    edm::EDGetToken m_mpSumToken;
    edm::EDGetToken m_egToken;
    edm::EDGetToken m_tauToken;
    edm::EDGetToken m_jetToken;
    edm::EDGetToken m_sumToken;

    bool m_doTowers;
    bool m_doClusters;
    bool m_doMPEGs;
    bool m_doMPTaus;
    bool m_doMPJets;
    bool m_doMPSums;
    bool m_doEGs;
    bool m_doTaus;
    bool m_doJets;
    bool m_doSums;
    
    bool doText_;
    bool doHistos_;
 
    enum ObjectType{
      Tower=1,
      Cluster=2,
      EG=3,
      Tau=4,
      Jet=5,
      SumET=6,
      SumETEm=7,
      SumHT=8,
      SumMET=9,
      SumMETHF=10,
      SumMHT=11,
      SumMHTHF=12,
      MPEG=13,
      MPTau=14,
      MPJet=15,
      MPSum=16,
      MPSumET=17,
      MPSumETHF=18,
      MPSumMETx=19,
      MPSumMETxHF=20,
      MPSumMETy=21,
      MPSumMETyHF=22,
      MPSumHT=23,
      MPSumHTHF=24,
      MPSumMHTx=25,
      MPSumMHTxHF=26,
      MPSumMHTy=27,
      MPSumMHTyHF=28,
      MPMinBiasHFP1=29,
      MPMinBiasHFM1=30,
      MPMinBiasHFP0=31,
      MPMinBiasHFM0=32,
      MinBiasHFP1=33,
      MinBiasHFM1=34,
      MinBiasHFP0=35,
      MinBiasHFM0=36,
      MPSumETEm = 37,
      MPSumHITowCount = 38,
      SumHITowCount = 39,
      SumCentrality = 40,
      SumAsymEt = 41,
      SumAsymEtHF = 42,
      SumAsymHt = 43,
      SumAsymHtHF = 44
    };
    
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
    std::map< ObjectType, TH1F* > hiso_;

    TFileDirectory evtDispDir_;

    TH1F *hsortMP_, *hsort_;

    int m_mpBx = 0;
    int m_dmxBx = 0;
    bool m_allBx = false;
    bool m_doEvtDisp = false;

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
  L1TStage2CaloAnalyzer::L1TStage2CaloAnalyzer(const edm::ParameterSet& iConfig) :
    doText_(iConfig.getUntrackedParameter<bool>("doText", true)),
    doHistos_(iConfig.getUntrackedParameter<bool>("doHistos", true))
  {
    //now do what ever initialization is needed

    m_mpBx  = iConfig.getParameter<int>("mpBx");
    m_dmxBx = iConfig.getParameter<int>("dmxBx");
    m_allBx = iConfig.getParameter<bool>("allBx");
    m_doEvtDisp = iConfig.getParameter<bool>("doEvtDisp");
    
    // register what you consume and keep token for later access:
    edm::InputTag nullTag("None");

    edm::InputTag towerTag = iConfig.getParameter<edm::InputTag>("towerToken");
    m_towerToken         = consumes<l1t::CaloTowerBxCollection>(towerTag);
    m_doTowers           = !(towerTag==nullTag);

    edm::InputTag clusterTag = iConfig.getParameter<edm::InputTag>("clusterToken");
    m_clusterToken         = consumes<l1t::CaloClusterBxCollection>(clusterTag);
    m_doClusters           = !(clusterTag==nullTag);

    edm::InputTag mpEGTag  = iConfig.getParameter<edm::InputTag>("mpEGToken");
    m_mpEGToken          = consumes<l1t::EGammaBxCollection>(mpEGTag);
    m_doMPEGs            = !(mpEGTag==nullTag);

    edm::InputTag mpTauTag = iConfig.getParameter<edm::InputTag>("mpTauToken");
    m_mpTauToken         = consumes<l1t::TauBxCollection>(mpTauTag);
    m_doMPTaus           = !(mpTauTag==nullTag);

    edm::InputTag mpJetTag = iConfig.getParameter<edm::InputTag>("mpJetToken");
    m_mpJetToken         = consumes<l1t::JetBxCollection>(mpJetTag);
    m_doMPJets           = !(mpJetTag==nullTag);  

    edm::InputTag mpSumTag = iConfig.getParameter<edm::InputTag>("mpEtSumToken");
    m_mpSumToken         = consumes<l1t::EtSumBxCollection>(mpSumTag);
    m_doMPSums           = !(mpSumTag==nullTag);

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

    std::cout << "Processing " << sumTag.label() << std::endl;
  
    types_.push_back( Tower );
    types_.push_back( Cluster );
    types_.push_back( MPEG );
    types_.push_back( MPTau );
    types_.push_back( MPJet );
    types_.push_back( MPSumET );
    types_.push_back( MPSumETHF );
    types_.push_back( MPSumETEm );
    types_.push_back( MPSumMETx );
    types_.push_back( MPSumMETxHF );
    types_.push_back( MPSumMETy );
    types_.push_back( MPSumMETyHF );
    types_.push_back( MPSumHT );
    types_.push_back( MPSumHTHF );
    types_.push_back( MPSumMHTx );
    types_.push_back( MPSumMHTxHF );
    types_.push_back( MPSumMHTy );
    types_.push_back( MPSumMHTyHF );
    types_.push_back( EG );
    types_.push_back( Tau );
    types_.push_back( Jet );
    types_.push_back( SumET );
    types_.push_back( SumETEm );
    types_.push_back( SumHT );
    types_.push_back( SumMET );
    types_.push_back( SumMETHF );
    types_.push_back( SumMHT );
    types_.push_back( SumMHTHF );
    types_.push_back( MPMinBiasHFP0 );
    types_.push_back( MPMinBiasHFM0 );
    types_.push_back( MPMinBiasHFP1 );
    types_.push_back( MPMinBiasHFM1 );
    types_.push_back( MinBiasHFP0 );
    types_.push_back( MinBiasHFM0 );
    types_.push_back( MinBiasHFP1 );
    types_.push_back( MinBiasHFM1 );
    types_.push_back( MPSumHITowCount );
    types_.push_back( SumHITowCount );
    types_.push_back( SumCentrality );
    types_.push_back( SumAsymEt );
    types_.push_back( SumAsymEtHF );
    types_.push_back( SumAsymHt );
    types_.push_back( SumAsymHtHF );

    typeStr_.push_back( "tower" );
    typeStr_.push_back( "cluster" );
    typeStr_.push_back( "mpeg" );
    typeStr_.push_back( "mptau" );
    typeStr_.push_back( "mpjet" );
    typeStr_.push_back( "mpsumet" );
    typeStr_.push_back( "mpsumethf" );
    typeStr_.push_back( "mpsumetem" );
    typeStr_.push_back( "mpsummetx" );
    typeStr_.push_back( "mpsummetxhf" );
    typeStr_.push_back( "mpsummety" );
    typeStr_.push_back( "mpsummetyhf" );
    typeStr_.push_back( "mpsumht" );
    typeStr_.push_back( "mpsumhthf" );
    typeStr_.push_back( "mpsummhtx" );
    typeStr_.push_back( "mpsummhtxhf" );
    typeStr_.push_back( "mpsummhty" );
    typeStr_.push_back( "mpsummhtyhf" );
    typeStr_.push_back( "eg" );
    typeStr_.push_back( "tau" );
    typeStr_.push_back( "jet" );
    typeStr_.push_back( "sumet" );
    typeStr_.push_back( "sumetem" );
    typeStr_.push_back( "sumht" );
    typeStr_.push_back( "summet" );
    typeStr_.push_back( "summethf" );
    typeStr_.push_back( "summht" );
    typeStr_.push_back( "summhthf" );
    typeStr_.push_back( "mpminbiashfp0" );
    typeStr_.push_back( "mpminbiashfm0" );
    typeStr_.push_back( "mpminbiashfp1" );
    typeStr_.push_back( "mpminbiashfm1" );
    typeStr_.push_back( "minbiashfp0" );
    typeStr_.push_back( "minbiashfm0" );
    typeStr_.push_back( "minbiashfp1" );
    typeStr_.push_back( "minbiashfm1" );
    typeStr_.push_back( "mpsumhitowercount");
    typeStr_.push_back( "sumhitowercount");
    typeStr_.push_back( "sumcentrality" );
    typeStr_.push_back( "sumasymet" );
    typeStr_.push_back( "sumasymethf" );
    typeStr_.push_back( "sumasymht" );
    typeStr_.push_back( "sumasymhthf" );
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

    std::stringstream text;

    TH2F* hEvtTow = new TH2F();
    TH2F* hEvtMPEG = new TH2F();
    TH2F* hEvtMPTau = new TH2F();
    TH2F* hEvtMPJet = new TH2F();
    TH2F* hEvtDemuxEG = new TH2F();
    TH2F* hEvtDemuxTau = new TH2F();
    TH2F* hEvtDemuxJet = new TH2F();


    if (m_doEvtDisp) {
      std::stringstream ss;
      ss << iEvent.run() << "-" << iEvent.id().event();
      TFileDirectory dir = evtDispDir_.mkdir(ss.str());
      hEvtTow = dir.make<TH2F>("Tower", "", 83, -41.5, 41.5, 72, .5, 72.5);
      hEvtMPEG = dir.make<TH2F>("MPEG", "", 83, -41.5, 41.5, 72, .5, 72.5);
      hEvtMPTau = dir.make<TH2F>("MPTau", "", 83, -41.5, 41.5, 72, .5, 72.5);
      hEvtMPJet = dir.make<TH2F>("MPJet", "", 83, -41.5, 41.5, 72, .5, 72.5);
      hEvtDemuxEG = dir.make<TH2F>("DemuxEG", "", 227, -113.5, 113.5, 144, -0.5, 143.5);
      hEvtDemuxTau = dir.make<TH2F>("DemuxTau", "", 227, -113.5, 113.5, 144, -0.5, 143.5);
      hEvtDemuxJet = dir.make<TH2F>("DemuxJet", "", 227, -113.5, 113.5, 144, -0.5, 143.5);
    }

    // get TPs ?
    // get regions ?
    // get RCT clusters ?

    //check mpbx and dmxbx
    if(m_mpBx < -2 || m_mpBx > 2 || m_dmxBx < -2 || m_dmxBx > 2)
      edm::LogError("L1T") << "Selected MP Bx or Demux Bx to fill histograms is outside of range -2,2. Histos will be empty!";
   
 
    // get towers
    if (m_doTowers) {
      Handle< BXVector<l1t::CaloTower> > towers;
      iEvent.getByToken(m_towerToken,towers);

      for ( int ibx=towers->getFirstBX(); ibx<=towers->getLastBX(); ++ibx) {

        if ( !m_allBx && ibx != m_mpBx ) continue;

        for ( auto itr = towers->begin(ibx); itr !=towers->end(ibx); ++itr ) {

          if (itr->hwPt()<=0) continue;

          hbx_.at(Tower)->Fill( ibx );
          het_.at(Tower)->Fill( itr->hwPt() );
          heta_.at(Tower)->Fill( itr->hwEta() );
          hphi_.at(Tower)->Fill( itr->hwPhi() );
          hem_.at(Tower)->Fill( itr->hwEtEm() );
          hhad_.at(Tower)->Fill( itr->hwEtHad() );
          hratio_.at(Tower)->Fill( itr->hwEtRatio() );
          hetaphi_.at(Tower)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );

	  text << "Tower : " << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi() << " iem=" << itr->hwEtEm() << " ihad=" << itr->hwEtHad() << " iratio=" << itr->hwEtRatio() << std::endl;
	
          if (m_doEvtDisp) hEvtTow->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );

        }

      }

    }
    // get cluster
    if (m_doClusters) {
      Handle< BXVector<l1t::CaloCluster> > clusters;
      iEvent.getByToken(m_clusterToken,clusters);

      for ( int ibx=clusters->getFirstBX(); ibx<=clusters->getLastBX(); ++ibx) {

        if (  !m_allBx && ibx != m_mpBx ) continue;

        for ( auto itr = clusters->begin(ibx); itr !=clusters->end(ibx); ++itr ) {
          hbx_.at(Cluster)->Fill( ibx );
          het_.at(Cluster)->Fill( itr->hwPt() );
          heta_.at(Cluster)->Fill( itr->hwEta() );
          hphi_.at(Cluster)->Fill( itr->hwPhi() );
          hetaphi_.at(Cluster)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );
          text << "Cluster : " << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi() << std::endl;
        }

      }
    }

    // get EG
    if (m_doMPEGs) {
      Handle< BXVector<l1t::EGamma> > mpegs;
      iEvent.getByToken(m_mpEGToken,mpegs);

      for ( int ibx=mpegs->getFirstBX(); ibx<=mpegs->getLastBX(); ++ibx) {

        if (  !m_allBx && ibx != m_mpBx ) continue;

        for ( auto itr = mpegs->begin(ibx); itr != mpegs->end(ibx); ++itr ) {
          hbx_.at(MPEG)->Fill( ibx );
          het_.at(MPEG)->Fill( itr->hwPt() );
          heta_.at(MPEG)->Fill( itr->hwEta() );
          hphi_.at(MPEG)->Fill( itr->hwPhi() );
          hetaphi_.at(MPEG)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );
	  hiso_.at(MPEG)->Fill( itr->hwIso() );

          text << "MP EG : " << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi() << std::endl;

          if (m_doEvtDisp) hEvtMPEG->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );

        }
      }

    }

    // get tau
    if (m_doMPTaus) {
      Handle< BXVector<l1t::Tau> > mptaus;
      iEvent.getByToken(m_mpTauToken,mptaus);
    
      for ( int ibx=mptaus->getFirstBX(); ibx<=mptaus->getLastBX(); ++ibx) {

        if (  !m_allBx && ibx != m_mpBx ) continue;

        for ( auto itr = mptaus->begin(ibx); itr != mptaus->end(ibx); ++itr ) {
          hbx_.at(MPTau)->Fill( ibx );
          het_.at(MPTau)->Fill( itr->hwPt() );
          heta_.at(MPTau)->Fill( itr->hwEta() );
          hphi_.at(MPTau)->Fill( itr->hwPhi() );
          hetaphi_.at(MPTau)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );

	  text << "MP Tau : " << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi() << std::endl;      

          if (m_doEvtDisp) hEvtMPTau->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );
        }
      
      }
    
    }

    // get jet
    int njetmp=0; 
    std::vector<l1t::Jet> thejets_poseta;
    std::vector<l1t::Jet> thejets_negeta;

    if (m_doMPJets) {
      Handle< BXVector<l1t::Jet> > mpjets;
      iEvent.getByToken(m_mpJetToken,mpjets);

      //Handle<BXVector<l1t::Jet>> jets;
      //iEvent.getByToken(m_jetToken,jets);
      //if (mpjets->size(0) == jets->size(0)) std::cout<<"******notequal"<<std::endl;
      for ( int ibx=mpjets->getFirstBX(); ibx<=mpjets->getLastBX(); ++ibx) {

        if (  !m_allBx && ibx != m_mpBx ) continue;

        for ( auto itr = mpjets->begin(ibx); itr != mpjets->end(ibx); ++itr ) {
          njetmp+=1;
          hbx_.at(MPJet)->Fill( ibx );
          het_.at(MPJet)->Fill( itr->hwPt() );
          heta_.at(MPJet)->Fill( itr->hwEta() );
          hphi_.at(MPJet)->Fill( itr->hwPhi() );
          hetaphi_.at(MPJet)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );

          text << "MP Jet : " << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi() << std::endl;

          if (m_doEvtDisp) hEvtMPJet->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );

          itr->hwEta()>0 ? thejets_poseta.push_back(*itr) : thejets_negeta.push_back(*itr);
        }
      
      }

    }

    if (!thejets_poseta.empty()) {
      for (unsigned int i=0; i<thejets_poseta.size()-1; i++) {
        for (unsigned int j=i+1; j<thejets_poseta.size(); j++) {
          hsortMP_->Fill(thejets_poseta.at(i).hwPt()-thejets_poseta.at(j).hwPt());
        }
      }
    }

    if (!thejets_negeta.empty()) {
      for (unsigned int i=0; i<thejets_negeta.size()-1; i++) {
        for (unsigned int j=i+1; j<thejets_negeta.size(); j++) {
          hsortMP_->Fill(thejets_negeta.at(i).hwPt()-thejets_negeta.at(j).hwPt());
        }
      }
    }

    //std::cout<<"njetmp "<<njetmp<<std::endl;

    // get sums
    if (m_doMPSums) {
      Handle< BXVector<l1t::EtSum> > mpsums;
      iEvent.getByToken(m_mpSumToken,mpsums);

      for ( int ibx=mpsums->getFirstBX(); ibx<=mpsums->getLastBX(); ++ibx) {

        if (  !m_allBx && ibx != m_mpBx ) continue;

        for ( auto itr = mpsums->begin(ibx); itr != mpsums->end(ibx); ++itr ) {
	  
          switch(itr->getType()){
          case l1t::EtSum::EtSumType::kTotalEt:     het_.at(MPSumET)       ->Fill( itr->hwPt() ); break;
          case l1t::EtSum::EtSumType::kTotalEtHF:    het_.at(MPSumETHF)      ->Fill( itr->hwPt() ); break;
          case l1t::EtSum::EtSumType::kTotalEtEm:     het_.at(MPSumETEm)       ->Fill( itr->hwPt() ); break;
          case l1t::EtSum::EtSumType::kTotalEtx:    het_.at(MPSumMETx)     ->Fill( itr->hwPt() ); break;
          case l1t::EtSum::EtSumType::kTotalEtxHF:   het_.at(MPSumMETxHF)    ->Fill( itr->hwPt() ); break;
          case l1t::EtSum::EtSumType::kTotalEty:    het_.at(MPSumMETy)     ->Fill( itr->hwPt() ); break;
          case l1t::EtSum::EtSumType::kTotalEtyHF:   het_.at(MPSumMETyHF)    ->Fill( itr->hwPt() ); break;
          case l1t::EtSum::EtSumType::kTotalHt:      het_.at(MPSumHT)       ->Fill( itr->hwPt() ); break;
	  case l1t::EtSum::EtSumType::kTotalHtHF:    het_.at(MPSumHTHF)      ->Fill( itr->hwPt() ); break;
          case l1t::EtSum::EtSumType::kTotalHtx:    het_.at(MPSumMHTx)     ->Fill( itr->hwPt() ); break;
          case l1t::EtSum::EtSumType::kTotalHtxHF:   het_.at(MPSumMHTxHF)    ->Fill( itr->hwPt() ); break;
          case l1t::EtSum::EtSumType::kTotalHty:    het_.at(MPSumMHTy)     ->Fill( itr->hwPt() ); break;
	  case l1t::EtSum::EtSumType::kTotalHtyHF:   het_.at(MPSumMHTyHF)    ->Fill( itr->hwPt() ); break;
          case l1t::EtSum::EtSumType::kMinBiasHFP0: het_.at(MPMinBiasHFP0) ->Fill( itr->hwPt() ); break;
          case l1t::EtSum::EtSumType::kMinBiasHFM0: het_.at(MPMinBiasHFM0) ->Fill( itr->hwPt() ); break;
          case l1t::EtSum::EtSumType::kMinBiasHFP1: het_.at(MPMinBiasHFP1) ->Fill( itr->hwPt() ); break;
	  case l1t::EtSum::EtSumType::kMinBiasHFM1: het_.at(MPMinBiasHFM1) ->Fill( itr->hwPt() ); break;
	  case l1t::EtSum::EtSumType::kTowerCount:  het_.at(MPSumHITowCount)  ->Fill( itr->hwPt() ); break;
	    
          default: std::cout<<"wrong type of MP sum"<<std::endl;
          }
	  text << "MP Sum : " << " type=" << itr->getType() << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi() << std::endl;
        }

      }

    }

    // get EG
    if (m_doEGs) {
      Handle< BXVector<l1t::EGamma> > egs;
      iEvent.getByToken(m_egToken,egs);

      for ( int ibx=egs->getFirstBX(); ibx<=egs->getLastBX(); ++ibx) {

        if (  !m_allBx && ibx != m_dmxBx ) continue;

        for ( auto itr = egs->begin(ibx); itr != egs->end(ibx); ++itr ) {
          hbx_.at(EG)->Fill( ibx );
          het_.at(EG)->Fill( itr->hwPt() );
          heta_.at(EG)->Fill( itr->hwEta() );
          hphi_.at(EG)->Fill( itr->hwPhi() );
          hetaphi_.at(EG)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );
	  hiso_.at(EG)->Fill( itr->hwIso() );
	  
	  text << "EG : " << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi() << std::endl;

          if (m_doEvtDisp) hEvtDemuxEG->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );
        }

      }

    }

    // get tau
    if (m_doTaus) {
      Handle< BXVector<l1t::Tau> > taus;
      iEvent.getByToken(m_tauToken,taus);

      for ( int ibx=taus->getFirstBX(); ibx<=taus->getLastBX(); ++ibx) {

        if (  !m_allBx && ibx != m_dmxBx ) continue;

        for ( auto itr = taus->begin(ibx); itr != taus->end(ibx); ++itr ) {
          hbx_.at(Tau)->Fill( ibx );
          het_.at(Tau)->Fill( itr->hwPt() );
          heta_.at(Tau)->Fill( itr->hwEta() );
          hphi_.at(Tau)->Fill( itr->hwPhi() );
          hetaphi_.at(Tau)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );

	  text << "Tau : " << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi() << std::endl;

          if (m_doEvtDisp) hEvtDemuxTau->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );
        }

      }

    }


    // get jet
    int njetdem=0;
    std::vector<l1t::Jet> thejets;

    if (m_doJets) {
      Handle< BXVector<l1t::Jet> > jets;
      iEvent.getByToken(m_jetToken,jets);

      for ( int ibx=jets->getFirstBX(); ibx<=jets->getLastBX(); ++ibx) {

        if (  !m_allBx && ibx != m_dmxBx ) continue;

        for ( auto itr = jets->begin(ibx); itr != jets->end(ibx); ++itr ) {
          njetdem+=1;
          hbx_.at(Jet)->Fill( ibx );
          het_.at(Jet)->Fill( itr->hwPt() );
          heta_.at(Jet)->Fill( itr->hwEta() );
          hphi_.at(Jet)->Fill( itr->hwPhi() );
          hetaphi_.at(Jet)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );

	  text << "Jet : " << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi() << std::endl;

          if (m_doEvtDisp) hEvtDemuxJet->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );

          thejets.push_back(*itr);
        }

      }

    }

    if (!thejets.empty()) {
      for (unsigned int i=0; i<thejets.size()-1; i++) {
        for (unsigned int j=i+1; j<thejets.size(); j++) {
          hsort_->Fill(thejets.at(i).hwPt()-thejets.at(j).hwPt());
        }
      }
    }

    // get sums
    if (m_doSums) {
      Handle< BXVector<l1t::EtSum> > sums;
      iEvent.getByToken(m_sumToken,sums);
    
      for ( int ibx=sums->getFirstBX(); ibx<=sums->getLastBX(); ++ibx) {

        if (  !m_allBx && ibx != m_dmxBx ) continue;

        for ( auto itr = sums->begin(ibx); itr != sums->end(ibx); ++itr ) {

          switch(itr->getType()){
          case l1t::EtSum::EtSumType::kTotalEt:     het_.at(SumET)       ->Fill( itr->hwPt() ); break;
          case l1t::EtSum::EtSumType::kTotalEtHF:   break;
          case l1t::EtSum::EtSumType::kTotalEtEm:   het_.at(SumETEm)     ->Fill( itr->hwPt() ); break;
          case l1t::EtSum::EtSumType::kTotalHt:     het_.at(SumHT)       ->Fill( itr->hwPt() ); break;
          case l1t::EtSum::EtSumType::kTotalHtHF:   break;
          case l1t::EtSum::EtSumType::kMissingEt:   het_.at(SumMET)      ->Fill( itr->hwPt() ); hphi_.at(SumMET)  ->Fill( itr->hwPhi() ); break;
          case l1t::EtSum::EtSumType::kMissingEtHF:  het_.at(SumMETHF)     ->Fill( itr->hwPt() ); hphi_.at(SumMETHF) ->Fill( itr->hwPhi() ); break;
          case l1t::EtSum::EtSumType::kMissingHt:   het_.at(SumMHT)      ->Fill( itr->hwPt() ); hphi_.at(SumMHT)  ->Fill( itr->hwPhi() ); break;
	  case l1t::EtSum::EtSumType::kMissingHtHF:  het_.at(SumMHTHF)     ->Fill( itr->hwPt() ); hphi_.at(SumMHTHF) ->Fill( itr->hwPhi() ); break;
          case l1t::EtSum::EtSumType::kMinBiasHFP0: het_.at(MinBiasHFP0) ->Fill( itr->hwPt() ); break;
          case l1t::EtSum::EtSumType::kMinBiasHFM0: het_.at(MinBiasHFM0) ->Fill( itr->hwPt() ); break;
          case l1t::EtSum::EtSumType::kMinBiasHFP1: het_.at(MinBiasHFP1) ->Fill( itr->hwPt() ); break;
          case l1t::EtSum::EtSumType::kMinBiasHFM1: het_.at(MinBiasHFM1) ->Fill( itr->hwPt() ); break;
	  case l1t::EtSum::EtSumType::kTowerCount:  het_.at(SumHITowCount) ->Fill( itr->hwPt() ); break;
	  case l1t::EtSum::EtSumType::kCentrality:  het_.at(SumCentrality) ->Fill( itr->hwPt() ); break;  
	  case l1t::EtSum::EtSumType::kAsymEt:      het_.at(SumAsymEt)     ->Fill( itr->hwPt() ); break;
	  case l1t::EtSum::EtSumType::kAsymHt:      het_.at(SumAsymHt)     ->Fill( itr->hwPt() ); break;
	  case l1t::EtSum::EtSumType::kAsymEtHF:    het_.at(SumAsymEtHF)   ->Fill( itr->hwPt() ); break;
	  case l1t::EtSum::EtSumType::kAsymHtHF:    het_.at(SumAsymHtHF)   ->Fill( itr->hwPt() ); break;

          default: std::cout<<"wrong type of demux sum: "<< itr->getType() << std::endl;
          }

          text << "Sum : " << " type=" << itr->getType() << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi() << std::endl;
        }
      }
    }
    
    if (doText_) edm::LogVerbatim("L1TCaloEvents") << text.str();

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

      if (*itr==MPSumMETx || *itr == MPSumMETxHF || *itr==MPSumMETy || *itr==MPSumMETyHF){
        het_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("et", "", 2200000, -2199999500, 2200000500) ));
      }
      else if (*itr==MPSumMHTx  || *itr==MPSumMHTxHF  || *itr==MPSumMHTy || *itr==MPSumMHTyHF ) {
	het_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("et", "",  2200000, -2199999500, 2200000500) ));
      }
      else if (*itr==SumET || *itr==SumETEm || *itr==MPSumETEm || *itr==MPSumET || *itr==MPSumETHF || *itr==SumHT || *itr==MPSumHT || *itr==MPSumHTHF ) {
        het_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("et", "", 100000, -0.5, 99999.5) )); 
      }
      else if (*itr==MPMinBiasHFP0 ||
               *itr==MPMinBiasHFM0 ||
               *itr==MPMinBiasHFP1 ||
               *itr==MPMinBiasHFM1 ||
               *itr==MinBiasHFP0 ||
               *itr==MinBiasHFM0 ||
               *itr==MinBiasHFP1 ||
               *itr==MinBiasHFM1)  {
        het_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("et", "", 16, -0.5, 15.5) )); 
      }
      else if (*itr==MPSumHITowCount || *itr==SumHITowCount){
	het_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("et", "", 5904, -0.5, 5903.5)));
      }
      else if (*itr==EG || *itr==Tau || *itr==MPEG || *itr==MPTau){
        het_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("et", "", 1500, -0.5, 1499.5) ));
      }
      else {
        het_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("et", "", 100000, -0.5, 99999.5) ));
      }

      hbx_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("bx", "", 11, -5.5, 5.5) ));

      if (*itr==EG || *itr==Jet || *itr==Tau) {
        heta_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("eta", "", 227, -113.5, 113.5) ));
        hphi_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("phi", "", 144, -0.5, 143.5) ));
        hetaphi_.insert( std::pair< ObjectType, TH2F* >(*itr, dirs_.at(*itr).make<TH2F>("etaphi", "", 227, -113.5, 113.5, 144, -0.5, 143.5) ));
	if(*itr==EG) hiso_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("iso", "", 4, -0.5, 3.5) ));
      }
      else if (*itr==Tower || *itr==Cluster || *itr==MPEG || *itr==MPJet || *itr==MPTau) {
        heta_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("eta", "", 83, -41.5, 41.5) ));
        hphi_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("phi", "", 72, 0.5, 72.5) ));
        hetaphi_.insert( std::pair< ObjectType, TH2F* >(*itr, dirs_.at(*itr).make<TH2F>("etaphi", "", 83, -41.5, 41.5, 72, .5, 72.5) ));
	if(*itr==MPEG) hiso_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("iso", "", 4, -0.5, 3.5) ));
      }
      else if (*itr==SumMET || *itr==SumMETHF || *itr==SumMHT || *itr==SumMHTHF) {
        hphi_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("phi", "", 1008, -0.5, 1007.5) ));
      }

      if (*itr==Tower) {
        hem_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("em", "", 101, -0.5, 100.5) ));
        hhad_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("had", "", 101, -0.5, 100.5) ));
        hratio_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("ratio", "", 11, -0.5, 10.5) ));
      }

    }

    if (m_doEvtDisp) {
      evtDispDir_ = fs->mkdir("Events");
    }

    hsort_ = fs->make<TH1F>("sort","",201,-100.5,100.5);
    hsortMP_ = fs->make<TH1F>("sortMP","",201,-100.5,100.5);

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
