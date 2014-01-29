
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DQM/Physics/src/SingleTopTChannelLeptonDQM.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"
#include <iostream>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
using namespace std;
namespace SingleTopTChannelLepton {
  
  // maximal number of leading jets 
  // to be used for top mass estimate
  static const unsigned int MAXJETS = 4;
  // nominal mass of the W boson to 
  // be used for the top mass estimate
  static const double WMASS = 80.4;
  
  MonitorEnsemble::MonitorEnsemble(const char* label, const edm::ParameterSet& cfg, const edm::VParameterSet& vcfg, edm::ConsumesCollector && iC) : 
    label_(label), pvSelect_(0), jetIDSelect_(0), includeBTag_(false), lowerEdge_(-1.), upperEdge_(-1.), logged_(0)
    
  {
    // sources have to be given; this PSet is not optional
    edm::ParameterSet sources=cfg.getParameter<edm::ParameterSet>("sources");
    muons_ = iC.consumes<edm::View<reco::PFCandidate> >(sources.getParameter<edm::InputTag>("muons"));
    elecs_gsf_ = iC.consumes<edm::View<reco::GsfElectron> >(sources.getParameter<edm::InputTag>("elecs_gsf"));
    elecs_ = iC.consumes<edm::View<reco::PFCandidate> >(sources.getParameter<edm::InputTag>("elecs"));
    jets_ = iC.consumes<edm::View<reco::Jet> >(sources.getParameter<edm::InputTag>("jets" ));
    for (edm::InputTag const & tag : sources.getParameter<std::vector<edm::InputTag> >("mets"))
      mets_.push_back( iC.consumes<edm::View<reco::MET> >(tag) ); 
    pvs_ = iC.consumes<edm::View<reco::Vertex> >(sources.getParameter<edm::InputTag>("pvs"));
    // electronExtras are optional; they may be omitted or 
    // empty
    if( cfg.existsAs<edm::ParameterSet>("elecExtras") ){
      edm::ParameterSet elecExtras=cfg.getParameter<edm::ParameterSet>("elecExtras");
      // select is optional; in case it's not found no
      // selection will be applied
      if( elecExtras.existsAs<std::string>("select") ){
	//	elecSelect_= new StringCutObjectSelector<reco::GsfElectron>(elecExtras.getParameter<std::string>("select"));
	elecSelect_ = vcfg[1].getParameter<std::string>("select");
      }
      // isolation is optional; in case it's not found no
      // isolation will be applied
      if( elecExtras.existsAs<std::string>("isolation") ){
	//elecIso_= new StringCutObjectSelector<reco::GsfElectron>(elecExtras.getParameter<std::string>("isolation"));
	elecIso_= elecExtras.getParameter<std::string>("isolation");
      }
      // electronId is optional; in case it's not found the 
      // InputTag will remain empty
      edm::ParameterSet elecId=vcfg[1].getParameter<edm::ParameterSet>("electronId");
      electronId_= iC.consumes<edm::ValueMap<float> >(elecId.getParameter<edm::InputTag>("src"));
      eidPattern_= elecId.getParameter<int>("pattern");
    }
    // pvExtras are opetional; they may be omitted or empty
    if(cfg.existsAs<edm::ParameterSet>("pvExtras")){
      edm::ParameterSet pvExtras=cfg.getParameter<edm::ParameterSet>("pvExtras");
      // select is optional; in case it's not found no
      // selection will be applied
      if( pvExtras.existsAs<std::string>("select") ){
	pvSelect_= new StringCutObjectSelector<reco::Vertex>(pvExtras.getParameter<std::string>("select"));
      }
    }
    // muonExtras are optional; they may be omitted or empty
    if( cfg.existsAs<edm::ParameterSet>("muonExtras")){ // && vcfg.existsAs<std::vector<edm::ParameterSet> >("selection")){
      edm::ParameterSet muonExtras=cfg.getParameter<edm::ParameterSet>("muonExtras");
      
      // select is optional; in case it's not found no
      // selection will be applied
      if( muonExtras.existsAs<std::string>("select") ){
	//	muonSelect_= new StringCutObjectSelector<reco::Muon>(muonExtras.getParameter<std::string>("select"));
	//	muonSelect_= muonExtras.getParameter<std::string>("select");
	muonSelect_ = vcfg[1].getParameter<std::string>("select");
      }
      // isolation is optional; in case it's not found no
      // isolation will be applied
      if( muonExtras.existsAs<std::string>("isolation") ){
	//	muonIso_= new StringCutObjectSelector<reco::Muon>(muonExtras.getParameter<std::string>("isolation"));
	muonIso_= muonExtras.getParameter<std::string>("isolation");
      }
    }
    
    // jetExtras are optional; they may be omitted or 
    // empty
    if( cfg.existsAs<edm::ParameterSet>("jetExtras") ){
      edm::ParameterSet jetExtras=cfg.getParameter<edm::ParameterSet>("jetExtras");
      // jetCorrector is optional; in case it's not found 
      // the InputTag will remain empty
      if( jetExtras.existsAs<std::string>("jetCorrector") ){
	jetCorrector_= jetExtras.getParameter<std::string>("jetCorrector");
      }
      // read jetID information if it exists
      if(jetExtras.existsAs<edm::ParameterSet>("jetID")){
	edm::ParameterSet jetID=jetExtras.getParameter<edm::ParameterSet>("jetID");
	jetIDLabel_ = iC.consumes<reco::JetIDValueMap>(jetID.getParameter<edm::InputTag>("label"));
	jetIDSelect_= new StringCutObjectSelector<reco::JetID>(jetID.getParameter<std::string>("select"));
      }
      // select is optional; in case it's not found no
      // selection will be applied (only implemented for 
      // CaloJets at the moment)
      if( jetExtras.existsAs<std::string>("select") ){
	
	jetSelect_= jetExtras.getParameter<std::string>("select");
	jetSelect_ = vcfg[2].getParameter<std::string>("select");
      }
      // jetBDiscriminators are optional; in case they are
      // not found the InputTag will remain empty; they 
      // consist of pairs of edm::JetFlavorAssociation's & 
      // corresponding working points
      includeBTag_=jetExtras.existsAs<edm::ParameterSet>("jetBTaggers");
      if( includeBTag_ ){
	edm::ParameterSet btagEff=jetExtras.getParameter<edm::ParameterSet>("jetBTaggers").getParameter<edm::ParameterSet>("trackCountingEff");
	btagEff_= iC.consumes<reco::JetTagCollection>(btagEff.getParameter<edm::InputTag>("label")); btagEffWP_= btagEff.getParameter<double>("workingPoint");
	edm::ParameterSet btagPur=jetExtras.getParameter<edm::ParameterSet>("jetBTaggers").getParameter<edm::ParameterSet>("trackCountingPur");
	btagPur_= iC.consumes<reco::JetTagCollection>(btagPur.getParameter<edm::InputTag>("label")); btagPurWP_= btagPur.getParameter<double>("workingPoint");
	edm::ParameterSet btagVtx=jetExtras.getParameter<edm::ParameterSet>("jetBTaggers").getParameter<edm::ParameterSet>("secondaryVertex" );
	btagVtx_= iC.consumes<reco::JetTagCollection>(btagVtx.getParameter<edm::InputTag>("label")); btagVtxWP_= btagVtx.getParameter<double>("workingPoint");
	edm::ParameterSet btagCombVtx=jetExtras.getParameter<edm::ParameterSet>("jetBTaggers").getParameter<edm::ParameterSet>("combinedSecondaryVertex" );
	btagCombVtx_= iC.consumes<reco::JetTagCollection>(btagCombVtx.getParameter<edm::InputTag>("label")); btagCombVtxWP_= btagCombVtx.getParameter<double>("workingPoint");
      }
    }
    
    // triggerExtras are optional; they may be omitted or empty
    if( cfg.existsAs<edm::ParameterSet>("triggerExtras") ){
      edm::ParameterSet triggerExtras=cfg.getParameter<edm::ParameterSet>("triggerExtras");
      triggerTable_=iC.consumes<edm::TriggerResults>(triggerExtras.getParameter<edm::InputTag>("src"));
      triggerPaths_=triggerExtras.getParameter<std::vector<std::string> >("paths");
    }
    
    // massExtras is optional; in case it's not found no mass
    // window cuts are applied for the same flavor monitor
    // histograms
    if( cfg.existsAs<edm::ParameterSet>("massExtras") ){
      edm::ParameterSet massExtras=cfg.getParameter<edm::ParameterSet>("massExtras");
      lowerEdge_= massExtras.getParameter<double>("lowerEdge");
      upperEdge_= massExtras.getParameter<double>("upperEdge");
    }
    
    // setup the verbosity level for booking histograms;
    // per default the verbosity level will be set to 
    // STANDARD. This will also be the chosen level in
    // the case when the monitoring PSet is not found
    verbosity_=STANDARD;
    if( cfg.existsAs<edm::ParameterSet>("monitoring") ){
      edm::ParameterSet monitoring=cfg.getParameter<edm::ParameterSet>("monitoring");
      if(monitoring.getParameter<std::string>("verbosity") == "DEBUG"   )
	verbosity_= DEBUG;
      if(monitoring.getParameter<std::string>("verbosity") == "VERBOSE" )
	verbosity_= VERBOSE;
      if(monitoring.getParameter<std::string>("verbosity") == "STANDARD")
	verbosity_= STANDARD;
    }
    // and don't forget to do the histogram booking
    book(cfg.getParameter<std::string>("directory"));
  }
  
  void 
  MonitorEnsemble::book(std::string directory)
  {
    //set up the current directory path
    std::string current(directory); current+=label_;
    store_=edm::Service<DQMStore>().operator->();
    store_->setCurrentFolder(current);
    
    // determine number of bins for trigger monitoring
    unsigned int nPaths=triggerPaths_.size();
    
    // --- [STANDARD] --- //
    // number of selected primary vertices
    hists_["pvMult_"     ] = store_->book1D("PvMult"     , "N_{pvs}"          ,     100,     0.,    100.);  
    // pt of the leading muon
    hists_["muonPt_"     ] = store_->book1D("MuonPt"     , "pt(#mu)"          ,     50,     0.,    250.);   
    // muon multiplicity before std isolation
    hists_["muonMult_"   ] = store_->book1D("MuonMult"   , "N_{20}(#mu)"     ,     10,     0.,     10.);   
    // muon multiplicity after  std isolation
    hists_["muonMultIso_"] = store_->book1D("MuonMultIso", "N_{Iso}(#mu)"     ,     10,     0.,     10.);   
    // pt of the leading electron
    hists_["elecPt_"     ] = store_->book1D("ElecPt"     , "pt(e)"            ,     50,     0.,    250.);   
    // electron multiplicity before std isolation
    hists_["elecMult_"   ] = store_->book1D("ElecMult"   , "N_{30}(e)"       ,     10,     0.,     10.);   
    // electron multiplicity after  std isolation
    hists_["elecMultIso_"] = store_->book1D("ElecMultIso", "N_{Iso}(e)"       ,     10,     0.,     10.);   
    // multiplicity of jets with pt>20 (corrected to L2+L3)
    hists_["jetMult_"    ] = store_->book1D("JetMult"    , "N_{30}(jet)"      ,     10,     0.,     10.);   
    // trigger efficiency estimates for single lepton triggers
    hists_["triggerEff_" ] = store_->book1D("TriggerEff" , "Eff(trigger)"     , nPaths,     0.,  nPaths);
    // monitored trigger occupancy for single lepton triggers
    hists_["triggerMon_" ] = store_->book1D("TriggerMon" , "Mon(trigger)"     , nPaths,     0.,  nPaths);
    // MET (calo)
    hists_["metCalo_"    ] = store_->book1D("METCalo"    , "MET_{Calo}"       ,     50,     0.,    200.);   
    // W mass estimate
    hists_["massW_"      ] = store_->book1D("MassW"      , "M(W)"             ,     60,     0.,    300.);   
    // Top mass estimate
    hists_["massTop_"    ] = store_->book1D("MassTop"    , "M(Top)"           ,     50,     0.,    500.);   
    // W mass transverse estimate mu
    hists_["MTWm_"       ] = store_->book1D("MTWm"       , "M_{T}^{W}(#mu)"   ,     60,     0.,    300.);
    // Top mass transverse estimate mu
    hists_["mMTT_"       ] = store_->book1D("mMTT"       , "M_{T}^{t}(#mu)"   ,     50,     0.,    500.);

    // W mass transverse estimate e
    hists_["MTWe_"       ] = store_->book1D("MTWe"       , "M_{T}^{W}(e)"     ,     60,     0.,    300.);
    // Top mass transverse estimate e
    hists_["eMTT_"       ] = store_->book1D("eMTT"       , "M_{T}^{t}(e)"     ,     50,     0.,    500.);
    
    // set bin labels for trigger monitoring
    triggerBinLabels(std::string("trigger"), triggerPaths_);

    if( verbosity_==STANDARD) return;

    // --- [VERBOSE] --- //

    // eta of the leading muon
    hists_["muonEta_"    ] = store_->book1D("MuonEta"    , "#eta(#mu)"        ,     30,    -3.,      3.);   
    // std isolation variable of the leading muon
    hists_["muonPFRelIso_" ] = store_->book1D("MuonPFRelIso","PFIso_{Rel}(#mu)",    50,     0.,      1.);   
    hists_["muonRelIso_" ] = store_->book1D("MuonRelIso" , "Iso_{Rel}(#mu)"   ,     50,     0.,      1.);   

    // eta of the leading electron
    hists_["elecEta_"    ] = store_->book1D("ElecEta"    , "#eta(e)"          ,     30,    -3.,      3.);   
    // std isolation variable of the leading electron
    hists_["elecRelIso_" ] = store_->book1D("ElecRelIso" , "Iso_{Rel}(e)"     ,     50,     0.,      1.);   
    hists_["elecPFRelIso_" ] = store_->book1D("ElecPFRelIso" , "PFIso_{Rel}(e)",    50,     0.,      1.);   

    // multiplicity of btagged jets (for track counting high efficiency) with pt(L2L3)>30
    hists_["jetMultBEff_"] = store_->book1D("JetMultBEff", "N_{30}(b/eff)"    ,     10,     0.,     10.);   
    // btag discriminator for track counting high efficiency for jets with pt(L2L3)>30
    hists_["jetBDiscEff_"] = store_->book1D("JetBDiscEff", "Disc_{b/eff}(jet)",     100,     0.,     10.);   
    
    
    // eta of the 1. leading jet 
    hists_["jet1Eta_"     ] = store_->book1D("Jet1Eta"   , "#eta (jet1)" ,    50,     -5.,    5.);   
    // eta of the 2. leading jet 
    hists_["jet2Eta_"     ] = store_->book1D("Jet2Eta"   , "#eta (jet2)" ,    50,     -5.,    5.);   
    
    // pt of the 1. leading jet (corrected to L2+L3)
    hists_["jet1Pt_"     ] = store_->book1D("Jet1Pt"     , "pt_{L2L3}(jet1)"  ,     60,     0.,    300.);   
    // pt of the 2. leading jet (corrected to L2+L3)
    hists_["jet2Pt_"     ] = store_->book1D("Jet2Pt"     , "pt_{L2L3}(jet2)"  ,     60,     0.,    300.);   
    
    
    // eta and pt of the b-tagged jet (filled only when nJets==2)
    hists_["TaggedJetEta_"     ] = store_->book1D("TaggedJetEta"   , "#eta (Tagged jet)"  ,   50,   -5.,  5.);
    hists_["TaggedJetPt_"      ] = store_->book1D("TaggedJetPt"     , "pt_{L2L3}(Tagged jet)"   ,   60,    0.,  300.);
    
    // eta and pt of the jet not passing b-tag (filled only when nJets==2)
    hists_["UnTaggedJetEta_"     ] = store_->book1D("UnTaggedJetEta"   , "#eta (UnTagged jet)"  ,   50,   -5.,  5.);
    hists_["UnTaggedJetPt_"      ] = store_->book1D("UnTaggedJetPt"     , "pt_{L2L3}(UnTagged jet)"   ,   60,    0.,  300.);
    
    // eta and pt of the most forward jet in the event with nJets==2
    hists_["FwdJetEta_"     ] = store_->book1D("FwdJetEta"   , "#eta (Fwd jet)"  ,   50,   -5.,  5.);
    hists_["FwdJetPt_"      ] = store_->book1D("FwdJetPt"     , "pt_{L2L3}(Fwd jet)"   ,   60,    0.,  300.);
    
    
    // 2D histogram (pt,eta) of the b-tagged jet (filled only when nJets==2)
    hists_["TaggedJetPtEta_"   ] = store_->book2D("TaggedJetPt_Eta"     , "(pt vs #eta)_{L2L3}(Tagged jet)" , 60, 0., 300., 50, -5., 5.);   
    
    // 2D histogram (pt,eta) of the not-b tagged jet (filled only when nJets==2)
    hists_["UnTaggedJetPtEta_"   ] = store_->book2D("UnTaggedJetPt_Eta"     , "(pt vs #eta)_{L2L3}(UnTagged jet)" , 60, 0., 300., 50, -5., 5.);   
    
    
    
    // MET (tc)
    hists_["metTC_"      ] = store_->book1D("METTC"      , "MET_{TC}"         ,     50,     0.,    200.);   
    // MET (pflow)
    hists_["metPflow_"   ] = store_->book1D("METPflow"   , "MET_{Pflow}"      ,     50,     0.,    200.);   
    
    // dz for muons (to suppress cosmis)
    hists_["muonDelZ_"    ] = store_->book1D("MuonDelZ"  , "d_{z}(#mu)"       ,     50,   -25.,     25.);
    // dxy for muons (to suppress cosmics)
    hists_["muonDelXY_"   ] = store_->book2D("MuonDelXY" , "d_{xy}(#mu)"      ,     50,   -0.1,     0.1,   50,   -0.1,   0.1);
    
    // set axes titles for dxy for muons
    hists_["muonDelXY_"   ]->setAxisTitle( "x [cm]", 1); hists_["muonDelXY_"   ]->setAxisTitle( "y [cm]", 2);
    
    if( verbosity_==VERBOSE) return;

    // --- [DEBUG] --- //

    // relative muon isolation from charged hadrons  for the leading muon
    hists_["muonChHadIso_" ] = store_->book1D("MuonChHadIso" , "Iso_{ChHad}(#mu)"   ,     100,     0.,      1.);   
    // relative muon isolation from neutral hadrons for the leading muon
    hists_["muonNeuHadIso_" ] = store_->book1D("MuonNeuHadIso" , "Iso_{NeuHad}(#mu)"  ,  100,     0.,      1.);   
    // relative muon isolation from photons for the leading muon
    hists_["muonPhIso_" ] = store_->book1D("MuonPhIso" , "Iso_{Ph}(#mu)"  ,  100,     0.,      1.);   

    // relative electron isolation from charged hadrons for the leading electron
    hists_["elecChHadIso_" ] = store_->book1D("ElecChHadIso" , "Iso_{ChHad}(e)"     ,     100,     0.,      1.);   
    // relative electron isolation from neutral hadrons for the leading electron
    hists_["elecNeuHadIso_" ] = store_->book1D("ElecNeuHadIso" , "Iso_{NeuHad}(e)"    ,     100,     0.,      1.);   
    // relative electron isolation from photons for the leading electron
    hists_["elecPhIso_" ] = store_->book1D("ElecPhIso" , "Iso_{Ph}(e)"    ,     100,     0.,      1.);   
    
    // multiplicity of btagged jets (for track counting high purity) with pt(L2L3)>30
    hists_["jetMultBPur_"] = store_->book1D("JetMultBPur", "N_{30}(b/pur)"    ,     10,     0.,     10.);   
    // btag discriminator for track counting high purity
    hists_["jetBDiscPur_"] = store_->book1D("JetBDiscPur", "Disc_{b/pur}(Jet)",     200,     -10.,     10.);   
    // btag discriminator for track counting high purity for 1. leading jet
    hists_["jet1BDiscPur_"] = store_->book1D("Jet1BDiscPur", "Disc_{b/pur}(Jet1)",     200,     -10.,     10.);   
    // btag discriminator for track counting high purity for 2. leading jet
    hists_["jet2BDiscPur_"] = store_->book1D("Jet2BDiscPur", "Disc_{b/pur}(Jet2)",     200,     -10.,     10.);   


    // multiplicity of btagged jets (for simple secondary vertex) with pt(L2L3)>30
    hists_["jetMultBVtx_"] = store_->book1D("JetMultBVtx", "N_{30}(b/vtx)"    ,     10,     0.,     10.);   
    // btag discriminator for simple secondary vertex
    hists_["jetBDiscVtx_"] = store_->book1D("JetBDiscVtx", "Disc_{b/vtx}(Jet)",     35,    -1.,      6.);   


    // multiplicity of btagged jets (for combined secondary vertex) with pt(L2L3)>30
    hists_["jetMultBCombVtx_"] = store_->book1D("JetMultBCombVtx", "N_{30}(b/CSV)"    ,     10,     0.,     10.);
    // btag discriminator for combined secondary vertex
    hists_["jetBDiscCombVtx_"] = store_->book1D("JetBDiscCombVtx", "Disc_{b/CSV}(Jet)",     60,    -1.,      2.);
    // btag discriminator for combined secondary vertex for 1. leading jet
    hists_["jet1BDiscCombVtx_"] = store_->book1D("Jet1BDiscCombVtx", "Disc_{b/CSV}(Jet1)",     60,    -1.,      2.);   
    // btag discriminator for combined secondary vertex for 2. leading jet
    hists_["jet2BDiscCombVtx_"] = store_->book1D("Jet2BDiscCombVtx", "Disc_{b/CSV}(Jet2)",     60,    -1.,      2.);   
    
    
    // pt of the 1. leading jet (uncorrected)
    hists_["jet1PtRaw_"  ] = store_->book1D("Jet1PtRaw"  , "pt_{Raw}(jet1)"   ,     60,     0.,    300.);   
    // pt of the 2. leading jet (uncorrected)
    hists_["jet2PtRaw_"  ] = store_->book1D("Jet2PtRaw"  , "pt_{Raw}(jet2)"   ,     60,     0.,    300.);   
    
    // selected events
    hists_["eventLogger_"] = store_->book2D("EventLogger", "Logged Events"    ,      9,     0.,      9.,   10,   0.,   10.);
    
    // set axes titles for selected events
    hists_["eventLogger_"]->getTH1()->SetOption("TEXT");
    hists_["eventLogger_"]->setBinLabel( 1 , "Run"             , 1);
    hists_["eventLogger_"]->setBinLabel( 2 , "Block"           , 1);
    hists_["eventLogger_"]->setBinLabel( 3 , "Event"           , 1);
    hists_["eventLogger_"]->setBinLabel( 4 , "pt_{L2L3}(jet1)" , 1);
    hists_["eventLogger_"]->setBinLabel( 5 , "pt_{L2L3}(jet2)" , 1);
    hists_["eventLogger_"]->setBinLabel( 6 , "pt_{L2L3}(jet3)" , 1);
    hists_["eventLogger_"]->setBinLabel( 7 , "pt_{L2L3}(jet4)" , 1);
    hists_["eventLogger_"]->setBinLabel( 8 , "M_{W}"           , 1);
    hists_["eventLogger_"]->setBinLabel( 9 , "M_{Top}"         , 1);
    hists_["eventLogger_"]->setAxisTitle("logged evts"         , 2);
    return;
  }
  
  void 
  MonitorEnsemble::fill(const edm::Event& event, const edm::EventSetup& setup)
  {
    // fetch trigger event if configured such 
    edm::Handle<edm::TriggerResults> triggerTable;
    if(!triggerTable_.isUninitialized()) {
      if( !event.getByToken(triggerTable_, triggerTable) ) return;

    }
    
    /*
      ------------------------------------------------------------
      
      Primary Vertex Monitoring
      
      ------------------------------------------------------------
    */
    
    // fill monitoring plots for primary vertices
    edm::Handle<edm::View<reco::Vertex> > pvs;
    if( !event.getByToken(pvs_, pvs) ) return;
    unsigned int pvMult = 0;
    for(edm::View<reco::Vertex>::const_iterator pv=pvs->begin(); pv!=pvs->end(); ++pv){
      if(!pvSelect_ || (*pvSelect_)(*pv))
	pvMult++;
    }
    fill("pvMult_",    pvMult   );
    
    /* 
       ------------------------------------------------------------
       
       Electron Monitoring
       
       ------------------------------------------------------------
    */
    
    /*
    reco::BeamSpot beamSpot;
    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
    if( !event.getByToken("offlineBeamSpot",recoBeamSpotHandle)) return;
    beamSpot = *recoBeamSpotHandle;
    */
    
    // fill monitoring plots for electrons
    edm::Handle<edm::View<reco::GsfElectron> > elecs_gsf;
    edm::Handle<edm::View<reco::PFCandidate> > elecs;
    edm::View<reco::PFCandidate>::const_iterator elec_it;
    StringCutObjectSelector<reco::PFCandidate, true> *elecSelect = new StringCutObjectSelector<reco::PFCandidate, true>(elecSelect_); 
    StringCutObjectSelector<reco::PFCandidate, true> *elecIso = new StringCutObjectSelector<reco::PFCandidate, true>(elecIso_);
    reco::GsfElectronRef elec;
    
    if( !event.getByToken(elecs_, elecs) ) return;
    if( !event.getByToken(elecs_gsf_, elecs_gsf) ) return;
    // check availability of electron id
    edm::Handle<edm::ValueMap<float> > electronId; 
    if(!electronId_.isUninitialized()){
      if( !event.getByToken(electronId_, electronId) ) return;
    }
    // loop electron collection
    unsigned int eMult=0, eMultIso=0;
    std::vector<const reco::GsfElectron*> isoElecs;
    reco::GsfElectron e;
    
    unsigned int idx_gsf = 0;
    for(elec_it=elecs->begin(); elec_it!=elecs->end(); ++elec_it){
      if(elec_it->gsfElectronRef().isNull()) continue ; 
      
      reco::GsfElectronRef elec   = elec_it->gsfElectronRef(); 
      if(elec->gsfTrack().isNull()) continue ;
      
      // restrict to electrons with good electronId
      int eID = 0;
      if (!electronId_.isUninitialized()) 
	eID = (int)(*electronId)[elecs_gsf->refAt(idx_gsf)];

      if( electronId_.isUninitialized()  ? true : ( (eID  & eidPattern_) && (eID >=5)) ){ 

	if(!elecSelect || (*elecSelect)(*elec_it)){
	  double isolationRel = (elec->dr03TkSumPt()+elec->dr03EcalRecHitSumEt()+elec->dr03HcalTowerSumEt())/elec->pt();

	  double isolationChHad = elec->pt()/(elec->pt()+elec->pfIsolationVariables().sumChargedHadronPt);
	  double isolationNeuHad = elec->pt()/(elec->pt()+elec->pfIsolationVariables().sumNeutralHadronEt);
	  double isolationPhoton = elec->pt()/(elec->pt()+elec->pfIsolationVariables().sumPhotonEt);
	  double PFisolationRel = (elec->pfIsolationVariables().sumChargedHadronPt+elec->pfIsolationVariables().sumNeutralHadronEt+elec->pfIsolationVariables().sumPhotonEt)/elec->pt(); 
 	  
	  if( eMult==0 ){
	    // restrict to the leading electron
	    fill("elecPt_" , elec->pt() );
	    fill("elecEta_", elec->eta());
	    fill("elecRelIso_" , isolationRel );
	    fill("elecPFRelIso_",PFisolationRel );
	    fill("elecChHadIso_" , isolationChHad );
	    fill("elecNeuHadIso_" , isolationNeuHad );
	    fill("elecPhIso_" , isolationPhoton );
	    
	  }
	  // in addition to the multiplicity counter buffer the iso 
	  // electron candidates for later overlap check with jets
	  ++eMult; if( !elecIso || (*elecIso)(*elec_it)){  if(eMultIso == 0) e = *elec; isoElecs.push_back(&(*elec)); ++eMultIso; }
	}
      }
      idx_gsf++;
    }
    
    fill("elecMult_",    eMult   );
    fill("elecMultIso_", eMultIso);
    

    /* 
       ------------------------------------------------------------
       
       Muon Monitoring
       
       ------------------------------------------------------------
    */
    
    // fill monitoring plots for muons
    unsigned int mMult=0, mMultIso=0;
    
    edm::Handle<edm::View<reco::PFCandidate> > muons;
    edm::View<reco::PFCandidate>::const_iterator muonit;
    StringCutObjectSelector<reco::PFCandidate, true> *muonSelect = new StringCutObjectSelector<reco::PFCandidate, true>(muonSelect_); 
    StringCutObjectSelector<reco::PFCandidate, true> *muonIso = new StringCutObjectSelector<reco::PFCandidate, true>(muonIso_);
    reco::MuonRef muon;
    reco::Muon mu;
    

    /*
      if (muons_.label() == "muons"){
      edm::Handle<edm::View<reco::Muon> > muons;
      edm::View<reco::Muon>::const_iterator muon;
      StringCutObjectSelector<reco::Muon> *muonSelect = new StringCutObjectSelector<reco::Muon>(muonSelect_); 
      StringCutObjectSelector<reco::Muon> *muonIso = new StringCutObjectSelector<reco::Muon>(muonIso_);
      }
    */
    
    if( !event.getByToken(muons_, muons )) return;
    for(muonit = muons->begin(); muonit != muons->end(); ++muonit){    // for now, to use Reco::Muon need to substitute  muonit with muon
                                                                       // and comment the MuonRef and PFCandidate parts

      if(muonit->muonRef().isNull()) continue ; 
      reco::MuonRef muon = muonit->muonRef();

      if(muon->innerTrack().isNull()) continue ; 
      
      
      // restrict to globalMuons
      if( muon->isGlobalMuon() ){ 
	fill("muonDelZ_" , muon->globalTrack()->vz());
	fill("muonDelXY_", muon->globalTrack()->vx(), muon->globalTrack()->vy());
	
	// apply selection
	if( !muonSelect || (*muonSelect)(*muonit)) {

	
	  
	  double isolationRel = (muon->isolationR03().sumPt+muon->isolationR03().emEt+muon->isolationR03().hadEt)/muon->pt();
	  double isolationChHad  = muon->pt()/(muon->pt()+muon->pfIsolationR04().sumChargedHadronPt);
	  double isolationNeuHad = muon->pt()/(muon->pt()+muon->pfIsolationR04().sumNeutralHadronEt);
	  double isolationPhoton    = muon->pt()/(muon->pt()+muon->pfIsolationR04().sumPhotonEt);
	  double PFisolationRel = (muon->pfIsolationR04().sumChargedHadronPt + muon->pfIsolationR04().sumNeutralHadronEt + muon->pfIsolationR04().sumPhotonEt)/muon->pt();
	  
	  
	  
	  if( mMult==0 ){
	    // restrict to leading muon
	    fill("muonPt_"     , muon->pt() );
	    fill("muonEta_"    , muon->eta());
	    fill("muonRelIso_" , isolationRel );
	    fill("muonChHadIso_" , isolationChHad );
	    fill("muonNeuHadIso_" , isolationNeuHad );
	    fill("muonPhIso_" , isolationPhoton );
	    fill("muonPFRelIso_" , PFisolationRel );
	    
	  }
	  ++mMult; 
	  
	  if( !muonIso || (*muonIso)(*muonit)) {if(mMultIso == 0)  mu = *muon; ++mMultIso;}
	}
      }
    }
    fill("muonMult_",    mMult   );
    fill("muonMultIso_", mMultIso);
    
    /* 
       ------------------------------------------------------------
       
       Jet Monitoring
       
       ------------------------------------------------------------
    */
    // check availability of the btaggers
    edm::Handle<reco::JetTagCollection> btagEff, btagPur, btagVtx, btagCombVtx;
    if( includeBTag_ ){ 
      if( !event.getByToken(btagEff_, btagEff) ) return;
      if( !event.getByToken(btagPur_, btagPur) ) return;
      if( !event.getByToken(btagVtx_, btagVtx) ) return;
      if( !event.getByToken(btagCombVtx_, btagCombVtx) ) return;
    }

    // load jet corrector if configured such
    const JetCorrector* corrector=0;
    if(!jetCorrector_.empty()){
      // check whether a jet correcto is in the event setup or not
      if(setup.find( edm::eventsetup::EventSetupRecordKey::makeKey<JetCorrectionsRecord>() )){
	corrector = JetCorrector::getJetCorrector(jetCorrector_, setup);
      }
      else{ 
	edm::LogVerbatim( "SingleTopTChannelLeptonDQM" ) 
	  << "\n"
	  << "------------------------------------------------------------------------------------- \n"
	  << " No JetCorrectionsRecord available from EventSetup:                                   \n" 
	  << "  - Jets will not be corrected.                                                       \n"
	  << "  - If you want to change this add the following lines to your cfg file:              \n"
	  << "                                                                                      \n"
	  << "  ## load jet corrections                                                             \n"
	  << "  process.load(\"JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff\") \n"
	  << "  process.prefer(\"ak5CaloL2L3\")                                                     \n"
	  << "                                                                                      \n"
	  << "------------------------------------------------------------------------------------- \n";
      }
    }

    // loop jet collection
    std::vector<reco::Jet> correctedJets;
    unsigned int mult=0, multBEff=0, multBPur=0, multNoBPur=0,  multBVtx=0, multBCombVtx=0;
    
    
    edm::Handle<edm::View<reco::Jet> > jets; 
    if( !event.getByToken(jets_, jets) ) return;
    

    edm::Handle<reco::JetIDValueMap> jetID; 
    if(jetIDSelect_){ 
      if( !event.getByToken(jetIDLabel_, jetID) ) return;
    }
    
    vector<double> bJetDiscVal; 
    vector<double> NobJetDiscVal;
    reco::Jet TaggedJetCand;
    reco::Jet UnTaggedJetCand;
    reco::Jet FwdJetCand;
    for(edm::View<reco::Jet>::const_iterator jet=jets->begin(); jet!=jets->end(); ++jet){
      // check jetID for calo jets
      unsigned int idx = jet-jets->begin();
      if(dynamic_cast<const reco::CaloJet*>(&*jet)){
	if( jetIDSelect_ && dynamic_cast<const reco::CaloJet*>(jets->refAt(idx).get())){
	  if(!(*jetIDSelect_)((*jetID)[jets->refAt(idx)])) continue;
	}
      }
      
      // check additional jet selection for calo, pf and bare reco jets
      if(dynamic_cast<const reco::CaloJet*>(&*jet)){
	reco::CaloJet sel = dynamic_cast<const reco::CaloJet&>(*jet); sel.scaleEnergy(corrector ? corrector->correction(*jet) : 1.);
	StringCutObjectSelector<reco::CaloJet> jetSelect(jetSelect_); 
	if(!jetSelect(sel)){ continue;}
      }
      else if(dynamic_cast<const reco::PFJet*>(&*jet)){
	reco::PFJet sel= dynamic_cast<const reco::PFJet&>(*jet); sel.scaleEnergy(corrector ? corrector->correction(*jet) : 1.);
	StringCutObjectSelector<reco::PFJet> jetSelect(jetSelect_); 
	if(!jetSelect(sel)) continue;
      } 
      else{
	reco::Jet sel = *jet; sel.scaleEnergy(corrector ? corrector->correction(*jet) : 1.);
	StringCutObjectSelector<reco::Jet> jetSelect(jetSelect_); 
	if(!jetSelect(sel)) continue;
      }
      // check for overlaps -- comment this to be synchronous with the selection
      //bool overlap=false;
      //for(std::vector<const reco::GsfElectron*>::const_iterator elec=isoElecs.begin(); elec!=isoElecs.end(); ++elec){
      //  if(reco::deltaR((*elec)->eta(), (*elec)->phi(), jet->eta(), jet->phi())<0.4){overlap=true; break;}
      //} if(overlap){continue;}
      
      
      // prepare jet to fill monitor histograms
      reco::Jet monitorJet = *jet; monitorJet.scaleEnergy(corrector ? corrector->correction(*jet) : 1.);
      correctedJets.push_back(monitorJet);
      
      
      ++mult; // determine jet multiplicity
      if( includeBTag_ ){
	// fill b-discriminators
	edm::RefToBase<reco::Jet> jetRef = jets->refAt(idx);	
	if( (*btagVtx)[jetRef]>btagVtxWP_ ) ++multBVtx; 
	if( (*btagCombVtx)[jetRef]>btagCombVtxWP_ ) ++multBCombVtx; 
	if( (*btagPur)[jetRef]>btagPurWP_ ){
	  if (multBPur == 0){
	    TaggedJetCand = monitorJet;
	    // TaggedJetCand = *jet;
	    bJetDiscVal.push_back((*btagPur)[jetRef]);
	    
	  }
 	  else if (multBPur == 1){
	    bJetDiscVal.push_back((*btagPur)[jetRef]);
	    if (bJetDiscVal[1]>bJetDiscVal[0])
	      TaggedJetCand = monitorJet;
	    //TaggedJetCand = *jet;
	  }
	  ++multBPur;  
	}
	
	else{
	  if (multNoBPur == 0){
	    UnTaggedJetCand = monitorJet;
	    NobJetDiscVal.push_back((*btagPur)[jetRef]);
	    
	  }
	  else if (multNoBPur == 1){
            NobJetDiscVal.push_back((*btagPur)[jetRef]);
            if (NobJetDiscVal[1]<NobJetDiscVal[0])
              UnTaggedJetCand = monitorJet;
          }
	  
	  ++multNoBPur;
	}
	
	if( (*btagEff)[jetRef]>btagEffWP_ ) ++multBEff; 
	
	if(mult==1) {
	  fill("jet1BDiscPur_", (*btagPur)[jetRef]); 
	  fill("jet1BDiscCombVtx_", (*btagCombVtx)[jetRef]); 
	}
	
	else if(mult==2) {
          fill("jet2BDiscPur_", (*btagPur)[jetRef]);
	  fill("jet2BDiscCombVtx_", (*btagCombVtx)[jetRef]); 
        }
	
	fill("jetBDiscEff_", (*btagEff)[jetRef]); 
	fill("jetBDiscPur_", (*btagPur)[jetRef]); 
	fill("jetBDiscVtx_", (*btagVtx)[jetRef]); 
	fill("jetBDiscCombVtx_", (*btagCombVtx)[jetRef]); 
	
      }
      // fill pt (raw or L2L3) for the leading jets  
      if(mult==1) {
	fill("jet1Pt_" , monitorJet.pt()); 
	fill("jet1Eta_", monitorJet.eta()); 
	fill("jet1PtRaw_", jet->pt() );
	FwdJetCand = monitorJet;
	
      }
      
      if(mult==2) {
	fill("jet2Pt_" , monitorJet.pt()); 
	fill("jet2Eta_", monitorJet.eta()); 
	fill("jet2PtRaw_", jet->pt() );
	
	if ( abs(monitorJet.eta()) > abs(FwdJetCand.eta()) ){
	  FwdJetCand = monitorJet;
	}
	
	fill("FwdJetPt_" , FwdJetCand.pt());
	fill("FwdJetEta_", FwdJetCand.eta());
	
      }
      
      
      
    }
    
    if (multNoBPur == 1 && multBPur == 1){
      
      fill("TaggedJetPtEta_" , TaggedJetCand.pt(), TaggedJetCand.eta());
      fill("UnTaggedJetPtEta_" , UnTaggedJetCand.pt(), UnTaggedJetCand.eta());
      
      
      fill("TaggedJetPt_" , TaggedJetCand.pt());
      fill("TaggedJetEta_", TaggedJetCand.eta());
      fill("UnTaggedJetPt_" , UnTaggedJetCand.pt());
      fill("UnTaggedJetEta_", UnTaggedJetCand.eta());
    }
    
    fill("jetMult_"    , mult    );
    fill("jetMultBEff_", multBEff);
    fill("jetMultBPur_", multBPur);
    fill("jetMultBVtx_", multBVtx);
    fill("jetMultBCombVtx_", multBCombVtx);
    

    /* 
    ------------------------------------------------------------

    MET Monitoring

    ------------------------------------------------------------
    */
    
    // fill monitoring histograms for met
    reco::MET mET;
    for(std::vector<edm::EDGetTokenT<edm::View<reco::MET> > >::const_iterator met_=mets_.begin(); met_!=mets_.end(); ++met_){
      edm::Handle<edm::View<reco::MET> > met;
      if( !event.getByToken(*met_, met) ) continue;
      if(met->begin()!=met->end()){
	unsigned int idx=met_-mets_.begin();
	if(idx==0) { fill("metCalo_" , met->begin()->et()); }
	if(idx==1) { fill("metTC_"   , met->begin()->et()); }
	if(idx==2) { fill("metPflow_", met->begin()->et());   mET = *(met->begin()); }
      }
    }
    
    
    /* 
       ------------------------------------------------------------
       
       Event Monitoring
       
       ------------------------------------------------------------
    */

    // fill W boson and top mass estimates
    Calculate eventKinematics(MAXJETS, WMASS);
    double wMass   = eventKinematics.massWBoson  (correctedJets);
    double topMass = eventKinematics.massTopQuark(correctedJets);
    if(wMass>=0 && topMass>=0) {fill("massW_" , wMass  ); fill("massTop_" , topMass);}
    // fill plots for trigger monitoring
    if((lowerEdge_==-1. && upperEdge_==-1.) || (lowerEdge_<wMass && wMass<upperEdge_) ){
      if(!triggerTable_.isUninitialized()) fill(event, *triggerTable, "trigger", triggerPaths_);
      if(logged_<=hists_.find("eventLogger_")->second->getNbinsY()){
	// log runnumber, lumi block, event number & some
	// more pysics infomation for interesting events
	fill("eventLogger_", 0.5, logged_+0.5, event.eventAuxiliary().run()); 
	fill("eventLogger_", 1.5, logged_+0.5, event.eventAuxiliary().luminosityBlock()); 
	fill("eventLogger_", 2.5, logged_+0.5, event.eventAuxiliary().event()); 
	if(correctedJets.size()>0) fill("eventLogger_", 3.5, logged_+0.5, correctedJets[0].pt()); 
	if(correctedJets.size()>1) fill("eventLogger_", 4.5, logged_+0.5, correctedJets[1].pt()); 
	if(correctedJets.size()>2) fill("eventLogger_", 5.5, logged_+0.5, correctedJets[2].pt()); 
	if(correctedJets.size()>3) fill("eventLogger_", 6.5, logged_+0.5, correctedJets[3].pt()); 
	fill("eventLogger_", 7.5, logged_+0.5, wMass  ); 
	fill("eventLogger_", 8.5, logged_+0.5, topMass); 
	++logged_;
      }
    }
    if(multBPur != 0 && mMultIso == 1 ){
      
      double mtW = eventKinematics.tmassWBoson(&mu,mET,TaggedJetCand); fill("MTWm_",mtW);
      double MTT = eventKinematics.tmassTopQuark(&mu,mET,TaggedJetCand); fill("mMTT_", MTT);
      
    }
    
    if(multBPur != 0 && eMultIso == 1 ){
      double mtW = eventKinematics.tmassWBoson(&e,mET,TaggedJetCand); fill("MTWe_",mtW);
      double MTT = eventKinematics.tmassTopQuark(&e,mET,TaggedJetCand); fill("eMTT_", MTT);
    }

  }
  
}


SingleTopTChannelLeptonDQM::SingleTopTChannelLeptonDQM(const edm::ParameterSet& cfg):  vertexSelect_(0), beamspot_(""), beamspotSelect_(0),
	MuonStep(0), PFMuonStep(0), ElectronStep(0), PFElectronStep(0), PvStep(0), METStep(0)

{
  JetSteps.clear();
  CaloJetSteps.clear();
  PFJetSteps.clear();

  // configure preselection
  edm::ParameterSet presel=cfg.getParameter<edm::ParameterSet>("preselection");
  if( presel.existsAs<edm::ParameterSet>("trigger") ){
    edm::ParameterSet trigger=presel.getParameter<edm::ParameterSet>("trigger");
    triggerTable__=consumes<edm::TriggerResults>(trigger.getParameter<edm::InputTag>("src"));
    triggerPaths_=trigger.getParameter<std::vector<std::string> >("select");
  } 
  if( presel.existsAs<edm::ParameterSet>("vertex" ) ){
    edm::ParameterSet vertex=presel.getParameter<edm::ParameterSet>("vertex");
    vertex_= vertex.getParameter<edm::InputTag>("src");
    vertex__= consumes<reco::Vertex>(vertex.getParameter<edm::InputTag>("src"));
    vertexSelect_= new StringCutObjectSelector<reco::Vertex>(vertex.getParameter<std::string>("select"));
  }
  if( presel.existsAs<edm::ParameterSet>("beamspot" ) ){
    edm::ParameterSet beamspot=presel.getParameter<edm::ParameterSet>("beamspot");
    beamspot_= beamspot.getParameter<edm::InputTag>("src");
    beamspot__= consumes<reco::BeamSpot>(beamspot.getParameter<edm::InputTag>("src"));
    beamspotSelect_= new StringCutObjectSelector<reco::BeamSpot>(beamspot.getParameter<std::string>("select"));
  }
  // conifgure the selection
  std::vector<edm::ParameterSet> sel=cfg.getParameter<std::vector<edm::ParameterSet> >("selection");
  for(unsigned int i=0; i<sel.size(); ++i){
    selectionOrder_.push_back(sel.at(i).getParameter<std::string>("label"));
    selection_[selectionStep(selectionOrder_.back())] = std::make_pair(sel.at(i), new SingleTopTChannelLepton::MonitorEnsemble(selectionStep(selectionOrder_.back()).c_str(), cfg.getParameter<edm::ParameterSet>("setup"), cfg.getParameter<std::vector<edm::ParameterSet> >("selection"), consumesCollector()));
  }
  for(std::vector<std::string>::const_iterator selIt=selectionOrder_.begin(); selIt!=selectionOrder_.end(); ++selIt){
    std::string key = selectionStep(*selIt), type = objectType(*selIt);
    if(selection_.find(key)!=selection_.end()){
      if(type=="muons"){
	MuonStep = new SelectionStep<reco::Muon>(selection_[key].first, consumesCollector());
      } 
      if(type=="muons/pf"){
	PFMuonStep = new SelectionStep<reco::PFCandidate>(selection_[key].first, consumesCollector());
      } 
      if(type=="elecs"){
	ElectronStep = new SelectionStep<reco::GsfElectron>(selection_[key].first, consumesCollector());
      }
      if(type=="elecs/pf"){
	PFElectronStep = new SelectionStep<reco::PFCandidate>(selection_[key].first, consumesCollector());
      }
      if(type=="pvs"){
	PvStep = new SelectionStep<reco::Vertex>(selection_[key].first, consumesCollector());
      }
      if(type=="jets" ){
	JetSteps.push_back(new SelectionStep<reco::Jet>(selection_[key].first, consumesCollector()));
      }
      if(type=="jets/pf" ){
	PFJetSteps.push_back(new SelectionStep<reco::PFJet>(selection_[key].first, consumesCollector()));
      }
      if(type=="jets/calo" ){
	CaloJetSteps.push_back(new SelectionStep<reco::CaloJet>(selection_[key].first, consumesCollector()));
      }
      if(type=="met"){
	METStep = new SelectionStep<reco::MET>(selection_[key].first, consumesCollector());
      } 
    }
  }
}


void 
SingleTopTChannelLeptonDQM::analyze(const edm::Event& event, const edm::EventSetup& setup)
{ 
  if(!triggerTable__.isUninitialized()){
    edm::Handle<edm::TriggerResults> triggerTable;
    if(!event.getByToken(triggerTable__, triggerTable) ) return;
    if(!accept(event, *triggerTable, triggerPaths_)) return;
  }
  if(!beamspot__.isUninitialized()){
    edm::Handle<reco::BeamSpot> beamspot;
    if( !event.getByToken(beamspot__, beamspot) ) return;
    if(!(*beamspotSelect_)(*beamspot)) return;
  }
  
  if(!vertex__.isUninitialized()){
    edm::Handle<edm::View<reco::Vertex>> vertex;
    if( !event.getByToken(vertex__, vertex) ) return;
    edm::View<reco::Vertex>::const_iterator pv = vertex->begin();
    //if ((pv->isFake()) || (pv->ndof() < 4) || (abs(pv->z())>24.) || (pv->position().Rho() > 2.0))
    if(!(*vertexSelect_)(*pv)) return;
  }
  
  
  // apply selection steps
  unsigned int passed=0;
  unsigned int nJetSteps = -1;
  unsigned int nPFJetSteps = -1;
  unsigned int nCaloJetSteps = -1;
 for(std::vector<std::string>::const_iterator selIt=selectionOrder_.begin(); selIt!=selectionOrder_.end(); ++selIt){
    std::string key = selectionStep(*selIt), type = objectType(*selIt);
    if(selection_.find(key)!=selection_.end()){
      if(type=="empty"){
	selection_[key].second->fill(event, setup);
      }
      if(type=="presel" ){
      	selection_[key].second->fill(event, setup);
      }
      if(type=="elecs" && ElectronStep != 0){
	if(ElectronStep->select(event)){ ++passed;
	  selection_[key].second->fill(event, setup);
	} else break;
      }
      if(type=="elecs/pf" && PFElectronStep != 0){

	if(PFElectronStep->select(event, "electron")){ ++passed;

	  selection_[key].second->fill(event, setup);

        } else break;
      }
      if(type=="muons" && MuonStep != 0){
        if(MuonStep->select(event)){ ++passed;
	  selection_[key].second->fill(event, setup);
	} else break;
      }
      if(type=="muons/pf" && PFMuonStep != 0){
	//	cout << "MUON SELECTION" << endl;
        if(PFMuonStep->select(event, "muon")){ ++passed;
          selection_[key].second->fill(event, setup);
	} else break;
      }
      if(type=="jets" ){
	nJetSteps++;
        if(JetSteps[nJetSteps] != NULL){
	  if(JetSteps[nJetSteps]->select(event, setup)){ ++passed;
	    selection_[key].second->fill(event, setup);
	  } else break;
	}
      }
      if(type=="jets/pf" ){
	//	cout << "JET SELECTION" << endl;
	nPFJetSteps++;
        if(PFJetSteps[nPFJetSteps] != NULL){
	  if(PFJetSteps[nPFJetSteps]->select(event, setup)){ ++passed;
	    selection_[key].second->fill(event, setup);
	  }else break;
	}
      }
      if(type=="jets/calo" ){
	nCaloJetSteps++;
        if(CaloJetSteps[nCaloJetSteps] != NULL){
	  if(CaloJetSteps[nCaloJetSteps]->select(event, setup)){ ++passed;
	    selection_[key].second->fill(event, setup);
	  } else break;
	}
      }
      if(type=="met" && METStep != 0 ){
        if(METStep->select(event)){ ++passed;
	  selection_[key].second->fill(event, setup);
	} else break;
      }
    }
 }
}


