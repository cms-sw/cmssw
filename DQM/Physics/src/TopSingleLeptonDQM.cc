#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DQM/Physics/src/TopSingleLeptonDQM.h"
#include "DataFormats/Math/interface/deltaR.h"
#include <iostream>
using namespace std;
namespace TopSingleLepton {

  // maximal number of leading jets 
  // to be used for top mass estimate
  static const unsigned int MAXJETS = 4;
  // nominal mass of the W boson to 
  // be used for the top mass estimate
  static const double WMASS = 80.4;

  MonitorEnsemble::MonitorEnsemble(const char* label, const edm::ParameterSet& cfg) : 
    label_(label), elecIso_(0), elecSelect_(0), pvSelect_(0), muonIso_(0), muonSelect_(0), jetIDSelect_(0), includeBTag_(false), lowerEdge_(-1.), upperEdge_(-1.), logged_(0)
  {
    // sources have to be given; this PSet is not optional
    edm::ParameterSet sources=cfg.getParameter<edm::ParameterSet>("sources");
    muons_= sources.getParameter<edm::InputTag>("muons");
    elecs_= sources.getParameter<edm::InputTag>("elecs");
    jets_ = sources.getParameter<edm::InputTag>("jets" );
    mets_ = sources.getParameter<std::vector<edm::InputTag> >("mets" );
    pvs_ = sources.getParameter<edm::InputTag>("pvs" );
    // electronExtras are optional; they may be omitted or 
    // empty
    if( cfg.existsAs<edm::ParameterSet>("elecExtras") ){
      edm::ParameterSet elecExtras=cfg.getParameter<edm::ParameterSet>("elecExtras");
      // select is optional; in case it's not found no
      // selection will be applied
      if( elecExtras.existsAs<std::string>("select") ){
	elecSelect_= new StringCutObjectSelector<reco::GsfElectron>(elecExtras.getParameter<std::string>("select"));
      }
      // isolation is optional; in case it's not found no
      // isolation will be applied
      if( elecExtras.existsAs<std::string>("isolation") ){
	elecIso_= new StringCutObjectSelector<reco::GsfElectron>(elecExtras.getParameter<std::string>("isolation"));
      }
      // electronId is optional; in case it's not found the 
      // InputTag will remain empty
      if( elecExtras.existsAs<edm::ParameterSet>("electronId") ){
	edm::ParameterSet elecId=elecExtras.getParameter<edm::ParameterSet>("electronId");
	electronId_= elecId.getParameter<edm::InputTag>("src");
	eidPattern_= elecId.getParameter<int>("pattern");
      }
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
    if( cfg.existsAs<edm::ParameterSet>("muonExtras") ){
      edm::ParameterSet muonExtras=cfg.getParameter<edm::ParameterSet>("muonExtras");
      // select is optional; in case it's not found no
      // selection will be applied
      if( muonExtras.existsAs<std::string>("select") ){
	muonSelect_= new StringCutObjectSelector<reco::Muon>(muonExtras.getParameter<std::string>("select"));
      }
      // isolation is optional; in case it's not found no
      // isolation will be applied
      if( muonExtras.existsAs<std::string>("isolation") ){
	muonIso_= new StringCutObjectSelector<reco::Muon>(muonExtras.getParameter<std::string>("isolation"));
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
	jetIDLabel_ =jetID.getParameter<edm::InputTag>("label");
	jetIDSelect_= new StringCutObjectSelector<reco::JetID>(jetID.getParameter<std::string>("select"));
      }
      // select is optional; in case it's not found no
      // selection will be applied (only implemented for 
      // CaloJets at the moment)
      if( jetExtras.existsAs<std::string>("select") ){
	jetSelect_= jetExtras.getParameter<std::string>("select");
      }
      // jetBDiscriminators are optional; in case they are
      // not found the InputTag will remain empty; they 
      // consist of pairs of edm::JetFlavorAssociation's & 
      // corresponding working points
      includeBTag_=jetExtras.existsAs<edm::ParameterSet>("jetBTaggers");
      if( includeBTag_ ){
	edm::ParameterSet btagEff=jetExtras.getParameter<edm::ParameterSet>("jetBTaggers").getParameter<edm::ParameterSet>("trackCountingEff");
	btagEff_= btagEff.getParameter<edm::InputTag>("label"); btagEffWP_= btagEff.getParameter<double>("workingPoint");
	edm::ParameterSet btagPur=jetExtras.getParameter<edm::ParameterSet>("jetBTaggers").getParameter<edm::ParameterSet>("trackCountingPur");
	btagPur_= btagPur.getParameter<edm::InputTag>("label"); btagPurWP_= btagPur.getParameter<double>("workingPoint");
	edm::ParameterSet btagVtx=jetExtras.getParameter<edm::ParameterSet>("jetBTaggers").getParameter<edm::ParameterSet>("secondaryVertex" );
	btagVtx_= btagVtx.getParameter<edm::InputTag>("label"); btagVtxWP_= btagVtx.getParameter<double>("workingPoint");
      }
    }

    // triggerExtras are optional; they may be omitted or empty
    if( cfg.existsAs<edm::ParameterSet>("triggerExtras") ){
      edm::ParameterSet triggerExtras=cfg.getParameter<edm::ParameterSet>("triggerExtras");
      triggerTable_=triggerExtras.getParameter<edm::InputTag>("src");
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
    hists_["muonMult_"   ] = store_->book1D("MuonMult"   , "N_{All}(#mu)"     ,     10,     0.,     10.);   
    // muon multiplicity after  std isolation
    hists_["muonMultIso_"] = store_->book1D("MuonMultIso", "N_{Iso}(#mu)"     ,     10,     0.,     10.);   
    // pt of the leading electron
    hists_["elecPt_"     ] = store_->book1D("ElecPt"     , "pt(e)"            ,     50,     0.,    250.);   
    // electron multiplicity before std isolation
    hists_["elecMult_"   ] = store_->book1D("ElecMult"   , "N_{All}(e)"       ,     10,     0.,     10.);   
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

    // set bin labels for trigger monitoring
    triggerBinLabels(std::string("trigger"), triggerPaths_);

    if( verbosity_==STANDARD) return;

    // --- [VERBOSE] --- //
    // eta of the leading muon
    hists_["muonEta_"    ] = store_->book1D("MuonEta"    , "#eta(#mu)"        ,     30,    -3.,      3.);   
    // std isolation variable of the leading muon
    hists_["muonRelIso_" ] = store_->book1D("MuonRelIso" , "Iso_{Rel}(#mu)"   ,     50,     0.,      1.);   
    // eta of the leading electron
    hists_["elecEta_"    ] = store_->book1D("ElecEta"    , "#eta(e)"          ,     30,    -3.,      3.);   
    // std isolation variable of the leading electron
    hists_["elecRelIso_" ] = store_->book1D("ElecRelIso" , "Iso_{Rel}(e)"     ,     50,     0.,      1.);   
    // multiplicity of btagged jets (for track counting high efficiency) with pt(L2L3)>30
    hists_["jetMultBEff_"] = store_->book1D("JetMultBEff", "N_{30}(b/eff)"    ,     10,     0.,     10.);   
    // btag discriminator for track counting high efficiency for jets with pt(L2L3)>30
    hists_["jetBDiscEff_"] = store_->book1D("JetBDiscEff", "Disc_{b/eff}(jet)",     100,     0.,     10.);   
    // pt of the 1. leading jet (corrected to L2+L3)
    hists_["jet1Pt_"     ] = store_->book1D("Jet1Pt"     , "pt_{L2L3}(jet1)"  ,     60,     0.,    300.);   
    // pt of the 2. leading jet (corrected to L2+L3)
    hists_["jet2Pt_"     ] = store_->book1D("Jet2Pt"     , "pt_{L2L3}(jet2)"  ,     60,     0.,    300.);   
    // pt of the 3. leading jet (corrected to L2+L3)
    hists_["jet3Pt_"     ] = store_->book1D("Jet3Pt"     , "pt_{L2L3}(jet3)"  ,     60,     0.,    300.);   
    // pt of the 4. leading jet (corrected to L2+L3)
    hists_["jet4Pt_"     ] = store_->book1D("Jet4Pt"     , "pt_{L2L3}(jet4)"  ,     60,     0.,    300.);   
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
    // relative muon isolation in tracker for the leading muon
    hists_["muonTrkIso_" ] = store_->book1D("MuonTrkIso" , "Iso_{Trk}(#mu)"   ,     50,     0.,      1.);   
    // relative muon isolation in ecal+hcal for the leading muon
    hists_["muonCalIso_" ] = store_->book1D("MuonCalIso" , "Iso_{Ecal}(#mu)"  ,     50,     0.,      1.);   
    // relative electron isolation in tracker for the leading electron
    hists_["elecTrkIso_" ] = store_->book1D("ElecTrkIso" , "Iso_{Trk}(e)"     ,     50,     0.,      1.);   
    // relative electron isolation in ecal+hcal for the leading electron
    hists_["elecCalIso_" ] = store_->book1D("ElecCalIso" , "Iso_{Ecal}(e)"    ,     50,     0.,      1.);   
    // multiplicity of btagged jets (for track counting high purity) with pt(L2L3)>30
    hists_["jetMultBPur_"] = store_->book1D("JetMultBPur", "N_{30}(b/pur)"    ,     10,     0.,     10.);   
    // btag discriminator for track counting high purity
    hists_["jetBDiscPur_"] = store_->book1D("JetBDiscPur", "Disc_{b/pur}(Jet)",     100,     0.,     10.);   
    // multiplicity of btagged jets (for simple secondary vertex) with pt(L2L3)>30
    hists_["jetMultBVtx_"] = store_->book1D("JetMultBVtx", "N_{30}(b/vtx)"    ,     10,     0.,     10.);   
    // btag discriminator for simple secondary vertex
    hists_["jetBDiscVtx_"] = store_->book1D("JetBDiscVtx", "Disc_{b/vtx}(Jet)",     35,    -1.,      6.);   
    // pt of the 1. leading jet (uncorrected)
    hists_["jet1PtRaw_"  ] = store_->book1D("Jet1PtRaw"  , "pt_{Raw}(jet1)"   ,     60,     0.,    300.);   
    // pt of the 2. leading jet (uncorrected)
    hists_["jet2PtRaw_"  ] = store_->book1D("Jet2PtRaw"  , "pt_{Raw}(jet2)"   ,     60,     0.,    300.);   
    // pt of the 3. leading jet (uncorrected)
    hists_["jet3PtRaw_"  ] = store_->book1D("Jet3PtRaw"  , "pt_{Raw}(jet3)"   ,     60,     0.,    300.);   
    // pt of the 4. leading jet (uncorrected)
    hists_["jet4PtRaw_"  ] = store_->book1D("Jet4PtRaw"  , "pt_{Raw}(jet4)"   ,     60,     0.,    300.);   
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
    if(!triggerTable_.label().empty()) {
      if( !event.getByLabel(triggerTable_, triggerTable) ) return;
    }

    /*
    ------------------------------------------------------------

    Primary Vertex Monitoring

    ------------------------------------------------------------
    */
    // fill monitoring plots for primary verices
    edm::Handle<edm::View<reco::Vertex> > pvs;
    if( !event.getByLabel(pvs_, pvs) ) return;
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

    // fill monitoring plots for electrons
    edm::Handle<edm::View<reco::GsfElectron> > elecs;
    if( !event.getByLabel(elecs_, elecs) ) return;

    // check availability of electron id
    edm::Handle<edm::ValueMap<float> > electronId; 
    if(!electronId_.label().empty()) {
      if( !event.getByLabel(electronId_, electronId) ) return;
    }

    // loop electron collection
    unsigned int eMult=0, eMultIso=0;
    std::vector<const reco::GsfElectron*> isoElecs;
    for(edm::View<reco::GsfElectron>::const_iterator elec=elecs->begin(); elec!=elecs->end(); ++elec){
      unsigned int idx = elec-elecs->begin();
      // restrict to electrons with good electronId
      if( electronId_.label().empty() ? true : ((int)(*electronId)[elecs->refAt(idx)] & eidPattern_) ){
	if(!elecSelect_ || (*elecSelect_)(*elec)){
	  double isolationTrk = elec->pt()/(elec->pt()+elec->dr03TkSumPt());
	  double isolationCal = elec->pt()/(elec->pt()+elec->dr03EcalRecHitSumEt()+elec->dr03HcalTowerSumEt());
	  double isolationRel = (elec->dr03TkSumPt()+elec->dr03EcalRecHitSumEt()+elec->dr03HcalTowerSumEt())/elec->pt();
	  if( eMult==0 ){
	    // restrict to the leading electron
	    fill("elecPt_" , elec->pt() );
	    fill("elecEta_", elec->eta());
	    fill("elecRelIso_" , isolationRel );
	    fill("elecTrkIso_" , isolationTrk );
	    fill("elecCalIso_" , isolationCal );
	  }
	  // in addition to the multiplicity counter buffer the iso 
	  // electron candidates for later overlap check with jets
	  ++eMult; if(!elecIso_ || (*elecIso_)(*elec)){ isoElecs.push_back(&(*elec)); ++eMultIso;}
	}
      }
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

    edm::Handle<edm::View<reco::Muon> > muons;
    if( !event.getByLabel(muons_, muons) ) return;

    for(edm::View<reco::Muon>::const_iterator muon=muons->begin(); muon!=muons->end(); ++muon){
      // restrict to globalMuons
      if( muon->isGlobalMuon() ){ 
	fill("muonDelZ_" , muon->globalTrack()->vz());
	fill("muonDelXY_", muon->globalTrack()->vx(), muon->globalTrack()->vy());
	// apply preselection
	if(!muonSelect_ || (*muonSelect_)(*muon)){
	  double isolationTrk = muon->pt()/(muon->pt()+muon->isolationR03().sumPt);
	  double isolationCal = muon->pt()/(muon->pt()+muon->isolationR03().emEt+muon->isolationR03().hadEt);
	  double isolationRel = (muon->isolationR03().sumPt+muon->isolationR03().emEt+muon->isolationR03().hadEt)/muon->pt();
	  if( mMult==0 ){
	    // restrict to leading muon
	    fill("muonPt_"     , muon->pt() );
	    fill("muonEta_"    , muon->eta());
	    fill("muonRelIso_" , isolationRel );
	    fill("muonTrkIso_" , isolationTrk );
	    fill("muonCalIso_" , isolationCal );
	  }
	   ++mMult; if(!muonIso_ || (*muonIso_)(*muon)) ++mMultIso;
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
    edm::Handle<reco::JetTagCollection> btagEff, btagPur, btagVtx;
    if( includeBTag_ ){ 
      if( !event.getByLabel(btagEff_, btagEff) ) return;
      if( !event.getByLabel(btagPur_, btagPur) ) return;
      if( !event.getByLabel(btagVtx_, btagVtx) ) return;
    }
    // load jet corrector if configured such
    const JetCorrector* corrector=0;
    if(!jetCorrector_.empty()){
      // check whether a jet correcto is in the event setup or not
      if(setup.find( edm::eventsetup::EventSetupRecordKey::makeKey<JetCorrectionsRecord>() )){
	corrector = JetCorrector::getJetCorrector(jetCorrector_, setup);
      }
      else{ 
	edm::LogVerbatim( "TopSingleLeptonDQM" ) 
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
    unsigned int mult=0, multBEff=0, multBPur=0, multBVtx=0;

    edm::Handle<edm::View<reco::Jet> > jets; 
    if( !event.getByLabel(jets_, jets) ) return;

    edm::Handle<reco::JetIDValueMap> jetID; 
    if(jetIDSelect_){ 
      if( !event.getByLabel(jetIDLabel_, jetID) ) return;
    }

    for(edm::View<reco::Jet>::const_iterator jet=jets->begin(); jet!=jets->end(); ++jet){
      // check jetID for calo jets
      unsigned int idx = jet-jets->begin();
      if( jetIDSelect_ && dynamic_cast<const reco::CaloJet*>(jets->refAt(idx).get())){
	if(!(*jetIDSelect_)((*jetID)[jets->refAt(idx)])) continue;
      }
      // chekc additional jet selection for calo, pf and bare reco jets
      if(dynamic_cast<const reco::CaloJet*>(&*jet)){
	reco::CaloJet sel = dynamic_cast<const reco::CaloJet&>(*jet); sel.scaleEnergy(corrector ? corrector->correction(*jet) : 1.);
	StringCutObjectSelector<reco::CaloJet> jetSelect(jetSelect_); if(!jetSelect(sel)){ continue;}
      }
      else if(dynamic_cast<const reco::PFJet*>(&*jet)){
	reco::PFJet sel= dynamic_cast<const reco::PFJet&>(*jet); sel.scaleEnergy(corrector ? corrector->correction(*jet) : 1.);
	StringCutObjectSelector<reco::PFJet> jetSelect(jetSelect_); if(!jetSelect(sel)) continue;
      } 
      else{
	reco::Jet sel = *jet; sel.scaleEnergy(corrector ? corrector->correction(*jet) : 1.);
	StringCutObjectSelector<reco::Jet> jetSelect(jetSelect_); if(!jetSelect(sel)) continue;
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
	fill("jetBDiscEff_", (*btagEff)[jetRef]); if( (*btagEff)[jetRef]>btagEffWP_ ) ++multBEff; 
	fill("jetBDiscPur_", (*btagPur)[jetRef]); if( (*btagPur)[jetRef]>btagPurWP_ ) ++multBPur; 
	fill("jetBDiscVtx_", (*btagVtx)[jetRef]); if( (*btagVtx)[jetRef]>btagVtxWP_ ) ++multBVtx; 
      }
      // fill pt (raw or L2L3) for the leading four jets  
      if(idx==0) {fill("jet1Pt_" , monitorJet.pt()); fill("jet1PtRaw_", jet->pt() );}
      if(idx==1) {fill("jet2Pt_" , monitorJet.pt()); fill("jet2PtRaw_", jet->pt() );}
      if(idx==2) {fill("jet3Pt_" , monitorJet.pt()); fill("jet3PtRaw_", jet->pt() );}
      if(idx==3) {fill("jet4Pt_" , monitorJet.pt()); fill("jet4PtRaw_", jet->pt() );}
    }
    fill("jetMult_"    , mult    );
    fill("jetMultBEff_", multBEff);
    fill("jetMultBPur_", multBPur);
    fill("jetMultBVtx_", multBVtx);
    
    /* 
    ------------------------------------------------------------

    MET Monitoring

    ------------------------------------------------------------
    */

    // fill monitoring histograms for met
    for(std::vector<edm::InputTag>::const_iterator met_=mets_.begin(); met_!=mets_.end(); ++met_){
      edm::Handle<edm::View<reco::MET> > met;
      if( !event.getByLabel(*met_, met) ) continue;
      if(met->begin()!=met->end()){
	unsigned int idx=met_-mets_.begin();
	if(idx==0) fill("metCalo_" , met->begin()->et());
	if(idx==1) fill("metTC_"   , met->begin()->et());
	if(idx==2) fill("metPflow_", met->begin()->et());
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
      if(!triggerTable_.label().empty()) fill(event, *triggerTable, "trigger", triggerPaths_);
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
  }
  
}


TopSingleLeptonDQM::TopSingleLeptonDQM(const edm::ParameterSet& cfg): triggerTable_(""), vertexSelect_(0), beamspot_(""), beamspotSelect_(0)
{
  // configure preselection
  edm::ParameterSet presel=cfg.getParameter<edm::ParameterSet>("preselection");
  if( presel.existsAs<edm::ParameterSet>("trigger") ){
    edm::ParameterSet trigger=presel.getParameter<edm::ParameterSet>("trigger");
    triggerTable_=trigger.getParameter<edm::InputTag>("src");
    triggerPaths_=trigger.getParameter<std::vector<std::string> >("select");
  } 
  if( presel.existsAs<edm::ParameterSet>("vertex" ) ){
    edm::ParameterSet vertex=presel.getParameter<edm::ParameterSet>("vertex");
    vertex_= vertex.getParameter<edm::InputTag>("src");
    vertexSelect_= new StringCutObjectSelector<reco::Vertex>(vertex.getParameter<std::string>("select"));
  }
  if( presel.existsAs<edm::ParameterSet>("beamspot" ) ){
    edm::ParameterSet beamspot=presel.getParameter<edm::ParameterSet>("beamspot");
    beamspot_= beamspot.getParameter<edm::InputTag>("src");
    beamspotSelect_= new StringCutObjectSelector<reco::BeamSpot>(beamspot.getParameter<std::string>("select"));
  }

  // conifgure the selection
  std::vector<edm::ParameterSet> sel=cfg.getParameter<std::vector<edm::ParameterSet> >("selection");
  for(unsigned int i=0; i<sel.size(); ++i){
    selectionOrder_.push_back(sel.at(i).getParameter<std::string>("label"));
    selection_[selectionStep(selectionOrder_.back())] = std::make_pair(sel.at(i), new TopSingleLepton::MonitorEnsemble(selectionStep(selectionOrder_.back()).c_str(), cfg.getParameter<edm::ParameterSet>("setup")));
  }
}

void 
TopSingleLeptonDQM::analyze(const edm::Event& event, const edm::EventSetup& setup)
{ 
  if(!triggerTable_.label().empty()){
    edm::Handle<edm::TriggerResults> triggerTable;
    if( !event.getByLabel(triggerTable_, triggerTable) ) return;
    if(!accept(event, *triggerTable, triggerPaths_)) return;
  }
  //cout<<"trig passed"<<endl;
  if(!beamspot_.label().empty()){
    edm::Handle<reco::BeamSpot> beamspot;
    if( !event.getByLabel(beamspot_, beamspot) ) return;
    if(!(*beamspotSelect_)(*beamspot)) return;
  }
  // apply selection steps
  unsigned int passed=0;
  for(std::vector<std::string>::const_iterator selIt=selectionOrder_.begin(); selIt!=selectionOrder_.end(); ++selIt){
    std::string key = selectionStep(*selIt), type = objectType(*selIt);
    if(selection_.find(key)!=selection_.end()){
      if(type=="empty"){
	selection_[key].second->fill(event, setup);
      }
      if(type=="muons"){
	SelectionStep<reco::Muon> step(selection_[key].first);
	if(step.select(event)){ ++passed;
	  selection_[key].second->fill(event, setup);
	} else break;
      }
      if(type=="elecs"){
	SelectionStep<reco::GsfElectron> step(selection_[key].first);
	if(step.select(event)){ ++passed;
	  selection_[key].second->fill(event, setup);
	} else break;
      }
      if(type=="pvs" ){
	SelectionStep<reco::Vertex> step(selection_[key].first);
	if(step.selectVertex(event)){ ++passed;
	  selection_[key].second->fill(event, setup);
	} else break;
      }
      if(type=="jets" ){
	SelectionStep<reco::Jet> step(selection_[key].first);
	if(step.select(event, setup)){ ++passed;
	  selection_[key].second->fill(event, setup);
	} else break;
      }
      if(type=="jets/pf" ){
	SelectionStep<reco::PFJet> step(selection_[key].first);
	if(step.select(event, setup)){ ++passed;
	  selection_[key].second->fill(event, setup);
	} else break;
      }
      if(type=="jets/calo" ){
	SelectionStep<reco::CaloJet> step(selection_[key].first);
	if(step.select(event, setup)){ ++passed;
	  selection_[key].second->fill(event, setup);
	} else break;
      }
      if(type=="met" ){
	SelectionStep<reco::MET> step(selection_[key].first);
	if(step.select(event)){ ++passed;
	  selection_[key].second->fill(event, setup);
	} else break;
      }
    }
  }
}


