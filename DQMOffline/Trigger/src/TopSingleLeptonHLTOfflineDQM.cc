#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DQMOffline/Trigger/interface/TopSingleLeptonHLTOfflineDQM.h"
#include "DQMOffline/Trigger/interface/TopHLTOfflineDQMHelper.h"
#include <iostream>

#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "CommonTools/UtilAlgos/interface/DeltaR.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <cassert>

/*Originally from DQM/Physics by R. Wolf and J. Andrea*/
using namespace std;
namespace HLTOfflineDQMTopSingleLepton {

  // maximal number of leading jets 
  // to be used for top mass estimate
  static const unsigned int MAXJETS = 4;
  // nominal mass of the W boson to 
  // be used for the top mass estimate
  static const double WMASS = 80.4;
  // maximal Delta to consider
  // hlt and reco objects matched
  static const double DRMIN = 0.05;

  MonitorSingleLepton::MonitorSingleLepton(const char* label, const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC) : 
    label_(label), elecIso_(0), elecSelect_(0), pvSelect_(0), muonIso_(0), muonSelect_(0), jetIDSelect_(0), includeBTag_(false), lowerEdge_(-1.), upperEdge_(-1.), logged_(0)
  {
    // sources have to be given; this PSet is not optional
    edm::ParameterSet sources=cfg.getParameter<edm::ParameterSet>("sources");
    muons_= iC.consumes< edm::View<reco::Muon> >(sources.getParameter<edm::InputTag>("muons"));
    elecs_= iC.consumes< edm::View<reco::GsfElectron> >(sources.getParameter<edm::InputTag>("elecs"));
    jets_ = iC.consumes< edm::View<reco::Jet> >(sources.getParameter<edm::InputTag>("jets" ));

    const auto& mets = sources.getParameter<std::vector<edm::InputTag>>("mets");
    for (const auto& met: mets) {
      mets_.push_back(iC.consumes<edm::View<reco::MET>>(met));
    }

    pvs_ = iC.consumes< edm::View<reco::Vertex> >(sources.getParameter<edm::InputTag>("pvs" ));

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
        electronId_= iC.consumes< edm::ValueMap<float> >(elecId.getParameter<edm::InputTag>("src"));
        eidPattern_= elecId.getParameter<int>("pattern");
      }
    }
    // pvExtras are optional; they may be omitted or empty
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
        jetIDLabel_ = iC.consumes< reco::JetIDValueMap >(jetID.getParameter<edm::InputTag>("label"));
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
        btagEff_= iC.consumes< reco::JetTagCollection >(btagEff.getParameter<edm::InputTag>("label")); btagEffWP_= btagEff.getParameter<double>("workingPoint");
        edm::ParameterSet btagPur=jetExtras.getParameter<edm::ParameterSet>("jetBTaggers").getParameter<edm::ParameterSet>("trackCountingPur");
        btagPur_= iC.consumes< reco::JetTagCollection >(btagPur.getParameter<edm::InputTag>("label")); btagPurWP_= btagPur.getParameter<double>("workingPoint");
        edm::ParameterSet btagVtx=jetExtras.getParameter<edm::ParameterSet>("jetBTaggers").getParameter<edm::ParameterSet>("secondaryVertex" );
        btagVtx_= iC.consumes< reco::JetTagCollection >(btagVtx.getParameter<edm::InputTag>("label")); btagVtxWP_= btagVtx.getParameter<double>("workingPoint");
      }
    }

    // triggerExtras are optional; they may be omitted or empty
    processName_ = "HLT";
    if( cfg.existsAs<edm::ParameterSet>("triggerExtras") ){
      edm::ParameterSet triggerExtras=cfg.getParameter<edm::ParameterSet>("triggerExtras");
      triggerTable_= iC.consumes< edm::TriggerResults >(triggerExtras.getParameter<edm::InputTag>("src"));
      processName_ = triggerExtras.getParameter<edm::InputTag>("src").process();
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

    // and don't forget to do the histogram booking
    folder_=cfg.getParameter<std::string>("directory");

    triggerEventWithRefsTag_ = iC.consumes< trigger::TriggerEventWithRefs >(edm::InputTag("hltTriggerSummaryRAW","",processName_));

  }

  void 
    MonitorSingleLepton::book(DQMStore::IBooker& store_)
    {
      //set up the current directory path
      std::string current(folder_); current+=label_;
      store_.setCurrentFolder(current);

      // determine number of bins for trigger monitoring
      unsigned int nPaths=triggerPaths_.size();

      // number of selected primary vertices
      hists_["pvMult_"     ] = store_.book1D("PvMult"     , "N_{pvs}"          ,     100,     0.,    100.);  
      // multiplicity of jets with pt>20 (corrected to L2+L3)
      hists_["jetMult_"    ] = store_.book1D("JetMult"    , "N_{20}(jet)"      ,     10,     0.,     10.);   
      // // trigger efficiency estimates for single lepton triggers
      // hists_["triggerEff_" ] = store_.book1D("TriggerEff" , "Eff(trigger)"     , nPaths,     0.,  nPaths);
      // monitored trigger occupancy for single lepton triggers
      hists_["triggerMon_" ] = store_.book1D("TriggerMon" , "Mon(trigger)"     , nPaths,     0.,  nPaths);
      // W mass estimate
      hists_["massW_"      ] = store_.book1D("MassW"      , "M(W)"             ,     60,     0.,    300.);   
      // Top mass estimate
      hists_["massTop_"    ] = store_.book1D("MassTop"    , "M(Top)"           ,     50,     0.,    500.);   
      // Mlb mu 
      hists_["mMub_"       ] = store_.book1D("mMub"       , "m_{#mub}"         ,     50,     0.,    500.);
      // W mass transverse estimate mu
      hists_["MTWm_"       ] = store_.book1D("MTWm"       , "M_{T}^{W}(#mu)"   ,     60,     0.,    300.);
      // Top mass transverse estimate mu
      hists_["mMTT_"       ] = store_.book1D("mMTT"       , "M_{T}^{t}(#mu)"   ,     50,     0.,    500.);
      // Mlb e 
      hists_["mEb_"        ] = store_.book1D("mEb"        , "m_{eb}"           ,     50,     0.,    500.);
      // W mass transverse estimate e
      hists_["MTWe_"       ] = store_.book1D("MTWe"       , "M_{T}^{W}(e)"     ,     60,     0.,    300.);
      // Top mass transverse estimate e
      hists_["eMTT_"       ] = store_.book1D("eMTT"       , "M_{T}^{t}(e)"     ,     50,     0.,    500.);
      // set bin labels for trigger monitoring
      triggerBinLabels(std::string("trigger"), triggerPaths_);
      // multiplicity of btagged jets (for track counting high efficiency) with pt(L2L3)>20
      hists_["jetMultBEff_"] = store_.book1D("JetMultBProb", "N_{20}(b/prob)"    ,     10,     0.,     10.);   
      // btag discriminator for track counting high efficiency for jets with pt(L2L3)>20
      hists_["jetBDiscEff_"] = store_.book1D("JetBDiscProb", "Disc_{b/prob}(jet)",     25,     0.,     2.5);   
      // multiplicity of btagged jets (for track counting high purity) with pt(L2L3)>20
      hists_["jetMultBPur_"] = store_.book1D("JetMultBPur", "N_{20}(b/pur)"    ,     10,     0.,     10.);   
      // btag discriminator for track counting high purity
      hists_["jetBDiscPur_"] = store_.book1D("JetBDiscPur", "Disc_{b/pur}(Jet)",     100,     0.,     10.);   
      // multiplicity of btagged jets (for simple secondary vertex) with pt(L2L3)>20
      hists_["jetMultBVtx_"] = store_.book1D("JetMultBVtx", "N_{20}(b/vtx)"    ,     10,     0.,     10.);   
      // btag discriminator for simple secondary vertex
      hists_["jetBDiscVtx_"] = store_.book1D("JetBDiscVtx", "Disc_{b/vtx}(Jet)",     35,    -1.,      6.);   
      // selected events
      hists_["eventLogger_"] = store_.book2D("EventLogger", "Logged Events"    ,      3,     0.,      3.,   4,   0.,   4.);
      // set axes titles for selected events
      hists_["eventLogger_"]->getTH1()->SetOption("TEXT");
      hists_["eventLogger_"]->setBinLabel( 1 , "Run"             , 1);
      hists_["eventLogger_"]->setBinLabel( 2 , "Block"           , 1);
      hists_["eventLogger_"]->setBinLabel( 3 , "Event"           , 1);
      hists_["eventLogger_"]->setAxisTitle("logged evts"         , 2);

      // deltaR min between hlt iso lepton and reco iso lepton wrt eta
      hists_["leptDeltaREta_"] = store_.book2D("DeltaRMinEtaLepton", "#Delta R_{min}(leptons) wrt #eta", 30, -3, 3, 10, 0., 0.1);   
      // deltaR min between hlt jets and reco jets wrt eta
      hists_["jetDeltaREta_"] = store_.book2D("DeltaRMinEtaJet", "#Delta R_{min}(jets) wrt #eta", 30, -3, 3, 10, 0., 0.1);   
      // resolution in pT for matched isolated leptons
      hists_["leptResolution_"] = store_.book1D("ResIsoLeptons", "#Delta p_{T}/p_{T}(matched leptons)", 20, 0., 0.1);   
      // resolution in pT for matched jets
      hists_["jetResolution_"] = store_.book1D("ResIsoJets", "#Delta p_{T}/p_{T}(matched jets)", 20, 0., 0.1);   
      // matching monitoring
      hists_["matchingMon_"] = store_.book1D("MatchingMon", "Mon(matching)", 5, 0., 5.);   
      // set axes titles for matching monitoring
      hists_["matchingMon_"]->setBinLabel( 1 , "iso lepton" );
      hists_["matchingMon_"]->setBinLabel( 2 , "1st jet" );
      hists_["matchingMon_"]->setBinLabel( 3 , "2nd jet" );
      hists_["matchingMon_"]->setBinLabel( 4 , "3rd jet" );
      hists_["matchingMon_"]->setBinLabel( 5 , "all " );

      return;
    }

  void 
    MonitorSingleLepton::fill(const edm::Event& event, const edm::EventSetup& setup, const HLTConfigProvider& hltConfig, const std::vector<std::string> triggerPaths)
    {
      // fetch trigger event if configured such 
      edm::Handle<edm::TriggerResults> triggerTable;
      if(!triggerTable_.isUninitialized()) {
        if( !event.getByToken(triggerTable_, triggerTable) ) return;
      }

      edm::Handle<trigger::TriggerEventWithRefs> triggerEventWithRefsHandle;
      if(!event.getByToken(triggerEventWithRefsTag_,triggerEventWithRefsHandle)) return;

      /*
         ------------------------------------------------------------

         Primary Vertex Monitoring

         ------------------------------------------------------------
         */
      // fill monitoring plots for primary verices
      edm::Handle<edm::View<reco::Vertex> > pvs;
      if( !event.getByToken(pvs_, pvs) ) {
        edm::LogWarning( "TopSingleLeptonHLTOfflineDQM" ) 
            << "Vertex collection not found \n";
        return;
      }
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
      if( !event.getByToken(elecs_, elecs) ) {
        edm::LogWarning( "TopSingleLeptonHLTOfflineDQM" ) 
            << "Electron collection not found \n";
        return;
      }

      // check availability of electron id
      edm::Handle<edm::ValueMap<float> > electronId; 
      if(!electronId_.isUninitialized()) {
        if( !event.getByToken(electronId_, electronId) ) return;
      }

      // loop electron collection
      unsigned int eMultIso=0;
      std::vector<const reco::GsfElectron*> isoElecs;
      reco::GsfElectron e;
      for(edm::View<reco::GsfElectron>::const_iterator elec=elecs->begin(); elec!=elecs->end(); ++elec){
        unsigned int idx = elec-elecs->begin();
        // restrict to electrons with good electronId
        if( electronId_.isUninitialized() ? true : ((int)(*electronId)[elecs->refAt(idx)] & eidPattern_) ){
          if(!elecSelect_ || (*elecSelect_)(*elec)){
            if(!elecIso_ || (*elecIso_)(*elec)){ if(eMultIso == 0) e = *elec;  isoElecs.push_back(&(*elec)); ++eMultIso;}
          }
        }
      }

      /* 
         ------------------------------------------------------------

         Muon Monitoring

         ------------------------------------------------------------
         */

      // fill monitoring plots for muons
      unsigned int mMultIso=0;
      edm::Handle<edm::View<reco::Muon> > muons;
      std::vector<const reco::Muon*> isoMuons;
      if( !event.getByToken(muons_, muons) ) {
        edm::LogWarning( "TopSingleLeptonHLTOfflineDQM" ) 
            << "Muon collection not found \n";
        return;
      }
      reco::Muon mu;
      for(edm::View<reco::Muon>::const_iterator muon=muons->begin(); muon!=muons->end(); ++muon){
        // restrict to globalMuons
        if( muon->isGlobalMuon() ){ 
          // apply preselection
          if(!muonSelect_ || (*muonSelect_)(*muon)){
            if(!muonIso_ || (*muonIso_)(*muon)) {if(mMultIso == 0) mu = *muon; isoMuons.push_back(&(*muon)); ++mMultIso;}
          }
        }
      }

      /* 
         ------------------------------------------------------------

         Jet Monitoring

         ------------------------------------------------------------
         */

      // check availability of the btaggers
      edm::Handle<reco::JetTagCollection> btagEff, btagPur, btagVtx;
      if( includeBTag_ ){ 
        if( !event.getByToken(btagEff_, btagEff) ) return;
        if( !event.getByToken(btagPur_, btagPur) ) return;
        if( !event.getByToken(btagVtx_, btagVtx) ) return;
      }
      // load jet corrector if configured such
      const JetCorrector* corrector=0;
      if(!jetCorrector_.empty()){
        // check whether a jet corrector is in the event setup or not
        if(setup.find( edm::eventsetup::EventSetupRecordKey::makeKey<JetCorrectionsRecord>() )){
          corrector = JetCorrector::getJetCorrector(jetCorrector_, setup);
        }
        else{ 
          edm::LogVerbatim( "TopSingleLeptonHLTOfflineDQM" ) 
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
      if( !event.getByToken(jets_, jets) ) {
        edm::LogWarning( "TopSingleLeptonHLTOfflineDQM" ) 
            << "Jet collection not found \n";
        return;
      }

      edm::Handle<reco::JetIDValueMap> jetID; 
      if(jetIDSelect_){ 
        if( !event.getByToken(jetIDLabel_, jetID) ) return;
      }
      reco::Jet bJetCand;	
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

        // prepare jet to fill monitor histograms
        reco::Jet monitorJet = *jet; monitorJet.scaleEnergy(corrector ? corrector->correction(*jet) : 1.);
        correctedJets.push_back(monitorJet);
        ++mult; // determine jet multiplicity
        if( includeBTag_ ){
          // fill b-discriminators
          edm::RefToBase<reco::Jet> jetRef = jets->refAt(idx);	
          // for the btagEff collection 
          double btagEffDisc = (*btagEff)[jetRef];
          fill("jetBDiscEff_", btagEffDisc); 
          if( (*btagEff)[jetRef]>btagEffWP_ ) ++multBEff; 
          // for the btagPur collection
          double btagPurDisc = (*btagPur)[jetRef];
          fill("jetBDiscPur_", btagPurDisc); 
          if( (*btagPur)[jetRef]>btagPurWP_ ) {if(multBPur == 0) bJetCand = *jet; ++multBPur;} 
          // for the btagVtx collection
          double btagVtxDisc = (*btagVtx)[jetRef];
          fill("jetBDiscVtx_", btagVtxDisc);
          if( (*btagVtx)[jetRef]>btagVtxWP_ ) ++multBVtx; 
        }
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
      reco::MET mET;
      for(std::vector< edm::EDGetTokenT< edm::View<reco::MET> > >::const_iterator met_=mets_.begin(); met_!=mets_.end(); ++met_){
        edm::Handle<edm::View<reco::MET> > met;
        if( !event.getByToken(*met_, met) ) continue;
        if(met->begin()!=met->end()){
          mET = *(met->begin());
        }
      }

      /* 
         ------------------------------------------------------------

         Event Monitoring

         ------------------------------------------------------------
         */

      // fill W boson and top mass estimates
      CalculateHLT eventKinematics(MAXJETS, WMASS);
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
          ++logged_;
        }
      }
      if(multBPur != 0 && mMultIso == 1 ){
        double mtW = eventKinematics.tmassWBoson(&mu,mET,bJetCand); if (mtW == mtW) fill("MTWm_",mtW);
        double Mlb = eventKinematics.masslb(&mu,mET,bJetCand); if (Mlb == Mlb) fill("mMub_", Mlb);
        double MTT = eventKinematics.tmassTopQuark(&mu,mET,bJetCand); if (MTT == MTT) fill("mMTT_", MTT);
      }

      if(multBPur != 0 && eMultIso == 1 ){
        double mtW = eventKinematics.tmassWBoson(&mu,mET,bJetCand); if (mtW == mtW)fill("MTWe_",mtW);
        double Mlb = eventKinematics.masslb(&mu,mET,bJetCand); if (Mlb == Mlb) fill("mEb_", Mlb);
        double MTT = eventKinematics.tmassTopQuark(&mu,mET,bJetCand); if (MTT == MTT) fill("eMTT_", MTT);
      }


      /* 
         ------------------------------------------------------------

         HLT Objects Monitoring

         ------------------------------------------------------------
         */

      const edm::TriggerNames& triggerNames = event.triggerNames(*triggerTable);
      // loop over trigger paths 
      for(unsigned int i=0; i<triggerNames.triggerNames().size(); ++i){
        // consider only path from triggerPaths
        string name = triggerNames.triggerNames()[i];
        bool isInteresting = false;
        for (unsigned int j=0; j<triggerPaths.size(); j++) {
          if (TString(name.c_str()).Contains(TString(triggerPaths[j]), TString::kIgnoreCase)) isInteresting = true; 
        }
        if (!isInteresting) continue;
        // dump infos on the considered trigger path 
        const unsigned int triggerIndex = triggerNames.triggerIndex(name);
        // get modules for the considered trigger path
        const vector<string>& moduleLabels(hltConfig.moduleLabels(triggerIndex));
        const unsigned int moduleIndex(triggerTable->index(triggerIndex));
        // Results from TriggerEventWithRefs product
        electronIds_.clear(); electronRefs_.clear();
        muonIds_.clear();     muonRefs_.clear();
        pfjetIds_.clear();    pfjetRefs_.clear();
        // look only for modules actually run in this path
        unsigned int kElec=0;
        unsigned int kMuon=0;
        unsigned int kJet=0;
        for (unsigned int k=0; k<=moduleIndex; ++k) {
          const string& moduleLabel(moduleLabels[k]);
          const string  moduleType(hltConfig.moduleType(moduleLabel));
          // check whether the module is packed up in TriggerEventWithRef product
          const unsigned int filterIndex(triggerEventWithRefsHandle->filterIndex(edm::InputTag(moduleLabel,"",processName_)));
          if (filterIndex<triggerEventWithRefsHandle->size()) {
            triggerEventWithRefsHandle->getObjects(filterIndex,electronIds_,electronRefs_);
            const unsigned int nElectrons(electronIds_.size());
            if (nElectrons>0) kElec = k;

            triggerEventWithRefsHandle->getObjects(filterIndex,muonIds_,muonRefs_);
            const unsigned int nMuons(muonIds_.size());
            if (nMuons>0) kMuon = k;

            triggerEventWithRefsHandle->getObjects(filterIndex,pfjetIds_,pfjetRefs_);
            const unsigned int nPFJets(pfjetIds_.size());
            if (nPFJets>0) kJet = k;
          }
        }
        bool isMatched = true;
        bool lMatched = false;
        bool j1Matched = false;
        bool j2Matched = false;
        bool j3Matched = false;
        // access to hlt elecs
        double eDeltaRMin = 500.;
        unsigned int eIndMatched = 500;
        electronIds_.clear(); electronRefs_.clear();
        if (kElec > 0) {
          const string& moduleLabelElec(moduleLabels[kElec]);
          const string  moduleTypeElec(hltConfig.moduleType(moduleLabelElec));
          const unsigned int filterIndexElec(triggerEventWithRefsHandle->filterIndex(edm::InputTag(moduleLabelElec,"",processName_)));
          triggerEventWithRefsHandle->getObjects(filterIndexElec,electronIds_,electronRefs_);
          for (unsigned int inde = 0; inde < isoElecs.size(); inde++) {
            double deltar = deltaR(*electronRefs_[0],*isoElecs[inde]); 
            if (deltar < eDeltaRMin) {
              eDeltaRMin = deltar;
              eIndMatched = inde;
            }
          }
          if (eDeltaRMin < DRMIN) lMatched = true;
        }
        // access to hlt muons
        muonIds_.clear(); muonRefs_.clear();
        double mDeltaRMin = 500.;
        unsigned int mIndMatched = 500;
        if (kMuon > 0) {
          const string& moduleLabelMuon(moduleLabels[kMuon]);
          const string  moduleTypeMuon(hltConfig.moduleType(moduleLabelMuon));
          const unsigned int filterIndexMuon(triggerEventWithRefsHandle->filterIndex(edm::InputTag(moduleLabelMuon,"",processName_)));
          triggerEventWithRefsHandle->getObjects(filterIndexMuon,muonIds_,muonRefs_);
          for (unsigned int indm = 0; indm < isoMuons.size(); indm++) {
            double deltar = deltaR(*muonRefs_[0],*isoMuons[indm]); 
            if (deltar < mDeltaRMin) {
              mDeltaRMin = deltar;
              mIndMatched = indm;
            }
          }
          if (mDeltaRMin < DRMIN) lMatched = true;
        }
        // access to hlt pf jets
        const unsigned int nPFJets(pfjetIds_.size());
        pfjetIds_.clear();    pfjetRefs_.clear();
        double j1DeltaRMin = 500.;   
        double j2DeltaRMin = 500.;  
        double j3DeltaRMin = 500.; 
        unsigned int j1IndMatched = 500;
        unsigned int j2IndMatched = 500;
        unsigned int j3IndMatched = 500;
        if (kJet > 0) {
          const string& moduleLabelJet(moduleLabels[kJet]);
          const string  moduleTypeJet(hltConfig.moduleType(moduleLabelJet));
          const unsigned int filterIndexJet(triggerEventWithRefsHandle->filterIndex(edm::InputTag(moduleLabelJet,"",processName_)));
          triggerEventWithRefsHandle->getObjects(filterIndexJet,pfjetIds_,pfjetRefs_);
          for (unsigned int indj = 0; indj < correctedJets.size(); indj++) {
            double deltar1 = deltaR(*pfjetRefs_[0],correctedJets[indj]); 
            if (deltar1 < j1DeltaRMin) {j1DeltaRMin = deltar1; j1IndMatched = indj;}
            if (nPFJets > 1) { 
              double deltar2 = deltaR(*pfjetRefs_[1],correctedJets[indj]); 
              if (deltar2 < j2DeltaRMin) {j2DeltaRMin = deltar2; j2IndMatched = indj;}
              if (nPFJets > 2) {
                double deltar3 = deltaR(*pfjetRefs_[2],correctedJets[indj]); 
                if (deltar3 < j3DeltaRMin) {j3DeltaRMin = deltar3; j3IndMatched = indj;}
              }
            }
          }
          if (nPFJets > 0 && j1DeltaRMin < DRMIN) j1Matched = true;
          if (nPFJets > 1 && j2DeltaRMin < DRMIN) j2Matched = true;
          if (nPFJets > 2 && j3DeltaRMin < DRMIN) j3Matched = true;
        }
        if (eIndMatched < 500) {
          fill("leptDeltaREta_", isoElecs[eIndMatched]->eta(), eDeltaRMin);   
          if (lMatched) fill("leptResolution_", fabs(isoElecs[eIndMatched]->pt()-electronRefs_[0]->pt())/isoElecs[eIndMatched]->pt() );   
        }
        if (mIndMatched < 500) {
          fill("leptDeltaREta_", isoMuons[mIndMatched]->eta(), mDeltaRMin);   
          if (lMatched) fill("leptResolution_", fabs(isoMuons[mIndMatched]->pt()-muonRefs_[0]->pt())/isoMuons[mIndMatched]->pt() );   
        }
        if (lMatched) fill("matchingMon_", 0.5 );
        else isMatched = false;
        if (j1IndMatched < 500) {
          fill("jetDeltaREta_", correctedJets[j1IndMatched].eta(), j1DeltaRMin);   
          if (j1Matched) {
            fill("jetResolution_", fabs(correctedJets[j1IndMatched].pt()-pfjetRefs_[0]->pt())/correctedJets[j1IndMatched].pt() );   
            fill("matchingMon_", 1.5 );
          }
          else isMatched = false;
          if (j2IndMatched < 500) {
            fill("jetDeltaREta_", correctedJets[j2IndMatched].eta(), j2DeltaRMin);   
            if (j2Matched) {
              fill("jetResolution_", fabs(correctedJets[j2IndMatched].pt()-pfjetRefs_[1]->pt())/correctedJets[j2IndMatched].pt() );   
              fill("matchingMon_", 2.5 );
            }
            else isMatched = false;
            if (j3IndMatched < 500) {
              fill("jetDeltaREta_", correctedJets[j3IndMatched].eta(), j3DeltaRMin);   
              if (j3Matched) {
                fill("jetResolution_", fabs(correctedJets[j3IndMatched].pt()-pfjetRefs_[2]->pt())/correctedJets[j3IndMatched].pt() );   
                fill("matchingMon_", 3.5 );
              }
              else isMatched = false;
            }
          }
        }
        if (isMatched) fill("matchingMon_", 4.5 );

      }

    }

}

///===========================================================================================================

TopSingleLeptonHLTOfflineDQM::TopSingleLeptonHLTOfflineDQM(const edm::ParameterSet& cfg): vertexSelect_(0), beamspotSelect_(0)
{
  // configure preselection
  edm::ParameterSet presel=cfg.getParameter<edm::ParameterSet>("preselection");
  if( presel.existsAs<edm::ParameterSet>("trigger") ){
    edm::ParameterSet trigger=presel.getParameter<edm::ParameterSet>("trigger");
    triggerTable_= consumes< edm::TriggerResults >(trigger.getParameter<edm::InputTag>("src"));
    triggerPaths_=trigger.getParameter<std::vector<std::string> >("select");
  } 
  if( presel.existsAs<edm::ParameterSet>("vertex" ) ){
    edm::ParameterSet vertex=presel.getParameter<edm::ParameterSet>("vertex");
    vertex_= consumes< std::vector<reco::Vertex> >(vertex.getParameter<edm::InputTag>("src"));
    vertexSelect_= new StringCutObjectSelector<reco::Vertex>(vertex.getParameter<std::string>("select"));
  }
  if( presel.existsAs<edm::ParameterSet>("beamspot" ) ){
    edm::ParameterSet beamspot=presel.getParameter<edm::ParameterSet>("beamspot");
    beamspot_= consumes< reco::BeamSpot >(beamspot.getParameter<edm::InputTag>("src"));
    beamspotSelect_= new StringCutObjectSelector<reco::BeamSpot>(beamspot.getParameter<std::string>("select"));
  }

  // configure the selection
  std::vector<edm::ParameterSet> sel=cfg.getParameter<std::vector<edm::ParameterSet> >("selection");
  for(unsigned int i=0; i<sel.size(); ++i){
    selectionOrder_.push_back(sel.at(i).getParameter<std::string>("label"));
    selection_[selectionStep(selectionOrder_.back())] = std::make_pair(sel.at(i), new HLTOfflineDQMTopSingleLepton::MonitorSingleLepton(selectionStep(selectionOrder_.back()).c_str(), cfg.getParameter<edm::ParameterSet>("setup"), consumesCollector()));
  }

  for (const std::string& s: selectionOrder_) {
    std::string key = selectionStep(s), type = objectType(s);

    if (selection_.find(key) == selection_.end())
      continue;

    if (type == "muons"){
      selectmap_[type] = new SelectionStepHLT<reco::Muon>(selection_[key].first, consumesCollector());
    }
    if (type == "elecs"){
      selectmap_[type] = new SelectionStepHLT<reco::GsfElectron>(selection_[key].first, consumesCollector());
    }
    if (type == "jets"){
      selectmap_[type] = new SelectionStepHLT<reco::Jet>(selection_[key].first, consumesCollector());
    }
    if (type == "jets/pf"){
      selectmap_[type] = new SelectionStepHLT<reco::PFJet>(selection_[key].first, consumesCollector());
    }
    if (type == "jets/calo"){
      selectmap_[type] = new SelectionStepHLT<reco::CaloJet>(selection_[key].first, consumesCollector());
    }
    if (type == "met"){
      selectmap_[type] = new SelectionStepHLT<reco::MET>(selection_[key].first, consumesCollector());
    }
  }
}

  void
TopSingleLeptonHLTOfflineDQM::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;

  bool changed(true);
  if (!hltConfig_.init(iRun,iSetup,"*",changed)) {
        edm::LogWarning( "TopSingleLeptonHLTOfflineDQM" ) 
            << "Config extraction failure with process name "
            << hltConfig_.processName()
            << "\n";
        return;
  }
}

  void 
TopSingleLeptonHLTOfflineDQM::analyze(const edm::Event& event, const edm::EventSetup& setup)
{ 
  if(!triggerTable_.isUninitialized()){
    edm::Handle<edm::TriggerResults> triggerTable;
    if( !event.getByToken(triggerTable_, triggerTable) ) return;
    if(!acceptHLT(event, *triggerTable, triggerPaths_)) return;
  }
  if(!vertex_.isUninitialized()){
    edm::Handle<std::vector<reco::Vertex> > vertex;
    if( !event.getByToken(vertex_, vertex) ) return;
    if(vertex->empty() || !(*vertexSelect_)(vertex->front())) return;
   }
  if(!beamspot_.isUninitialized()){
    edm::Handle<reco::BeamSpot> beamspot;
    if( !event.getByToken(beamspot_, beamspot) ) return;
    if(!(*beamspotSelect_)(*beamspot)) return;
  }
   // apply selection steps
  for(std::vector<std::string>::const_iterator selIt=selectionOrder_.begin(); selIt!=selectionOrder_.end(); ++selIt){
    std::string key = selectionStep(*selIt), type = objectType(*selIt);
    if(selection_.find(key)!=selection_.end()){

      if(type=="empty"){
        selection_[key].second->fill(event, setup, hltConfig_, triggerPaths_);
        continue;
      }
      if(type=="Hlt" ){
        selection_[key].second->fill(event, setup, hltConfig_, triggerPaths_);
        continue;
      }

      bool passSel = true;

      for(std::vector<std::string>::const_iterator selIt2=selectionOrder_.begin(); selIt2<=selIt; ++selIt2){
        std::string key2 = selectionStep(*selIt2), type2 = objectType(*selIt2);
        if(selection_.find(key2)==selection_.end()) continue;

        if(type2=="Hlt" || type2=="empty" ) continue;
        if (!selectmap_[type2]->select(event)) passSel=false;

      } // end 2nd loop

      // Apply cumulative event selection
      if ( !passSel ) continue;

      selection_[key].second->fill(event, setup, hltConfig_, triggerPaths_);

    }
  }
}

void
TopSingleLeptonHLTOfflineDQM::bookHistograms(DQMStore::IBooker &i, edm::Run const&, edm::EventSetup const&)
{
  for (auto& sel: selection_) {
    sel.second.second->book(i);
  }
}
