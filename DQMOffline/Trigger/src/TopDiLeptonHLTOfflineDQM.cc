//#include <algorithm>
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DQMOffline/Trigger/interface/TopDiLeptonHLTOfflineDQM.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DQMOffline/Trigger/interface/TopHLTOfflineDQMHelper.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "CommonTools/UtilAlgos/interface/DeltaR.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"

/*Originally from DQM/Physics by R. Wolf and J. Andrea*/
using namespace std;
namespace HLTOfflineDQMTopDiLepton {

  // maximal Delta to consider
  // hlt and reco objects matched
  static const double DRMIN = 0.05;

  MonitorDiLepton::MonitorDiLepton(const char* label, const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC) : 
    label_(label), eidPattern_(0), elecIso_(0), elecSelect_(0), muonIso_(0), muonSelect_(0), jetIDSelect_(0), 
    lowerEdge_(-1.), upperEdge_(-1.), elecMuLogged_(0), diMuonLogged_(0), diElecLogged_(0)
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

    // elecExtras are optional; they may be omitted or empty
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
    // jetExtras are optional; they may be omitted or empty
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
    }
    // triggerExtras are optional; they may be omitted or empty
    processName_ = "HLT";
    if( cfg.existsAs<edm::ParameterSet>("triggerExtras") ){
      edm::ParameterSet triggerExtras=cfg.getParameter<edm::ParameterSet>("triggerExtras");
      triggerTable_= iC.consumes< edm::TriggerResults >(triggerExtras.getParameter<edm::InputTag>("src"));
      processName_ = triggerExtras.getParameter<edm::InputTag>("src").process();
      elecMuPaths_ = triggerExtras.getParameter<std::vector<std::string> >("pathsELECMU");
      diMuonPaths_ = triggerExtras.getParameter<std::vector<std::string> >("pathsDIMUON");
      diElecPaths_ = triggerExtras.getParameter<std::vector<std::string> >("pathsDIELEC");
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
    MonitorDiLepton::book(DQMStore::IBooker& store_)
    {
      //set up the current directory path
      std::string current(folder_); current+=label_;
      store_.setCurrentFolder(current);

      // determine number of bins for trigger monitoring
      unsigned int nElecMu=elecMuPaths_.size();
      unsigned int nDiMuon=diMuonPaths_.size();
      unsigned int nDiElec=diElecPaths_.size();

      // invariant mass of opposite charge lepton pair (only filled for same flavor)
      hists_["invMass_"     ] = store_.book1D("InvMass"     , "M(lep1, lep2)"           ,       80,   0.,     320.); //OK
      // invariant mass of same charge lepton pair (only filled for same flavor)
      hists_["invMassWC_"   ] = store_.book1D("InvMassWC"   , "M_{WC}(L1, L2)"          ,       80,   0.,     320.); //OK
      // decay channel [1]: muon/muon, [2]:elec/elec, [3]:elec/muon 
      hists_["decayChannel_"] = store_.book1D("DecayChannel", "Decay Channel"           ,        3,    0,        3); //OK
      // // trigger efficiency estimates for the electron muon channel
      // hists_["elecMuEff_"   ] = store_.book1D("ElecMuEff"   , "Eff(e/#mu paths)"        ,  nElecMu,   0.,  nElecMu);
      // monitored trigger occupancy for the electron muon channel
      hists_["elecMuMon_"   ] = store_.book1D("ElecMuMon"   , "Mon(e/#mu paths)"        ,  nElecMu,   0.,  nElecMu);
      // // trigger efficiency estimates for the di muon channel
      // hists_["diMuonEff_"   ] = store_.book1D("DiMuonEff"   , "Eff(#mu/#mu paths)"      ,  nDiMuon,   0.,  nDiMuon);
      // monitored trigger occupancy for the di muon channel
      hists_["diMuonMon_"   ] = store_.book1D("DiMuonMon"   , "Mon(#mu/#mu paths)"      ,  nDiMuon,   0.,  nDiMuon);
      // // trigger efficiency estimates for the di electron channel
      // hists_["diElecEff_"   ] = store_.book1D("DiElecEff"   , "Eff(e/e paths)"          ,  nDiElec,   0.,  nDiElec);
      // monitored trigger occupancy for the di electron channel
      hists_["diElecMon_"   ] = store_.book1D("DiElecMon"   , "Mon(e/e paths)"          ,  nDiElec,   0.,  nDiElec);
      // multiplicity of jets with pt>30 (corrected to L2+L3)
      hists_["jetMult_"     ] = store_.book1D("JetMult"     , "N_{30}(jet)"             ,       21, -0.5,      20.5); //OK

      // set bin labels for trigger monitoring
      triggerBinLabels(std::string("elecMu"), elecMuPaths_);
      triggerBinLabels(std::string("diMuon"), diMuonPaths_);
      triggerBinLabels(std::string("diElec"), diElecPaths_);
      // set bin labels for decayChannel_
      hists_["decayChannel_"]->setBinLabel( 1, "#mu e"  , 1);
      hists_["decayChannel_"]->setBinLabel( 2, "#mu #mu", 1);
      hists_["decayChannel_"]->setBinLabel( 3, "e e"    , 1);

      // selected dimuon events
      hists_["diMuonLogger_"] = store_.book2D("DiMuonLogger", "Logged DiMuon Events"    ,        8,   0.,       8.,   10,   0.,   10.); //OK
      // selected dielec events
      hists_["diElecLogger_"] = store_.book2D("DiElecLogger", "Logged DiElec Events"    ,        8,   0.,       8.,   10,   0.,   10.); //OK
      // selected elemu events
      hists_["elecMuLogger_"] = store_.book2D("ElecMuLogger", "Logged ElecMu Events"    ,        8,   0.,       8.,   10,   0.,   10.); //OK

      // set bin labels for trigger monitoring
      loggerBinLabels(std::string("diMuonLogger_")); 
      loggerBinLabels(std::string("diElecLogger_")); 
      loggerBinLabels(std::string("elecMuLogger_"));

      // deltaR min between hlt iso lepton and reco iso lepton wrt eta
      hists_["leptDeltaREta_"] = store_.book2D("DeltaRMinEtaLepton", "#Delta R_{min}(leptons) wrt #eta", 30, -3, 3, 10, 0., 0.1);   
      // resolution in pT for matched isolated leptons
      hists_["leptResolution_"] = store_.book1D("ResIsoLeptons", "#Delta p_{T}/p_{T}(matched leptons)", 20, 0., 0.1);   
      // matching monitoring
      hists_["matchingMon_"] = store_.book1D("MatchingMon", "Mon(matching)", 3, 0., 3.);   
      // set axes titles for matching monitoring
      hists_["matchingMon_"]->setBinLabel( 1 , "1st lepton" );
      hists_["matchingMon_"]->setBinLabel( 2 , "2nd lepton" );
      hists_["matchingMon_"]->setBinLabel( 3 , "both " );

      return;
    }

  void 
    MonitorDiLepton::fill(const edm::Event& event, const edm::EventSetup& setup, const HLTConfigProvider& hltConfig, const std::vector<std::string> triggerPaths)
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

         Run information 

         ------------------------------------------------------------
         */

      if (!event.eventAuxiliary().run()) return;

      /* 
         ------------------------------------------------------------

         Muon Selection

         ------------------------------------------------------------
         */

      // buffer isolated muons
      std::vector<const reco::Muon*> isoMuons;

      edm::Handle<edm::View<reco::Muon> > muons;
      if( !event.getByToken(muons_, muons) ) {
        edm::LogWarning( "TopSingleLeptonHLTOfflineDQM" ) 
            << "Muon collection not found \n";
        return;
      }

      for(edm::View<reco::Muon>::const_iterator muon=muons->begin(); muon!=muons->end(); ++muon){
        // restrict to globalMuons
        if( muon->isGlobalMuon() ){ 
          // apply preselection
          if(!muonSelect_ || (*muonSelect_)(*muon)){
            if(!muonIso_ || (*muonIso_)(*muon)) isoMuons.push_back(&(*muon));
          }
        }
      }

      /* 
         ------------------------------------------------------------

         Electron Selection

         ------------------------------------------------------------
         */

      // buffer isolated electronss
      std::vector<const reco::GsfElectron*> isoElecs;
      edm::Handle<edm::ValueMap<float> > electronId; 
      if(!electronId_.isUninitialized()) {
        if( !event.getByToken(electronId_, electronId) ) return;
      }

      edm::Handle<edm::View<reco::GsfElectron> > elecs;
      if( !event.getByToken(elecs_, elecs) ) {
        edm::LogWarning( "TopSingleLeptonHLTOfflineDQM" ) 
            << "Electron collection not found \n";
        return;
      }

      for(edm::View<reco::GsfElectron>::const_iterator elec=elecs->begin(); elec!=elecs->end(); ++elec){
        // restrict to electrons with good electronId
        int idx = elec-elecs->begin();
        if( electronId_.isUninitialized() ? true : ((int)(*electronId)[elecs->refAt(idx)] & eidPattern_) ){
          // apply preselection
          if(!elecSelect_ || (*elecSelect_)(*elec)){
            if(!elecIso_ || (*elecIso_)(*elec)) isoElecs.push_back(&(*elec));
          }
        }
      }

      /* 
         ------------------------------------------------------------

         Jet Selection

         ------------------------------------------------------------
         */

      const JetCorrector* corrector=0;
      if(!jetCorrector_.empty()){
        // check whether a jet correcto is in the event setup or not
        if(setup.find( edm::eventsetup::EventSetupRecordKey::makeKey<JetCorrectionsRecord>() )){
          corrector = JetCorrector::getJetCorrector(jetCorrector_, setup);
        }
        else{
          edm::LogVerbatim( "TopDiLeptonHLTOfflineDQM" ) 
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

      unsigned int mult=0;
      // buffer leadingJets
      std::vector<reco::Jet> leadingJets;
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

      for(edm::View<reco::Jet>::const_iterator jet=jets->begin(); jet!=jets->end(); ++jet){
        unsigned int idx=jet-jets->begin();
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
        // check for overlaps
        bool overlap=false;
        for(std::vector<const reco::GsfElectron*>::const_iterator elec=isoElecs.begin(); elec!=isoElecs.end(); ++elec){
          if(reco::deltaR((*elec)->eta(), (*elec)->phi(), jet->eta(), jet->phi())<0.4){overlap=true; break;}
        } if(overlap){continue;}
        // prepare jet to fill monitor histograms
        reco::Jet monitorJet=*jet; monitorJet.scaleEnergy(corrector ?  corrector->correction(*jet) : 1.);
        ++mult; // determine jet multiplicity
        if(idx==0) {
          leadingJets.push_back(monitorJet);
        }
        if(idx==1) {
          leadingJets.push_back(monitorJet);
        }
      }
      fill("jetMult_", mult);

      /* 
         ------------------------------------------------------------

         MET Selection

         ------------------------------------------------------------
         */

      // buffer for event logging 
      reco::MET caloMET;
      for(std::vector< edm::EDGetTokenT< edm::View<reco::MET> > >::const_iterator met_=mets_.begin(); met_!=mets_.end(); ++met_){

        edm::Handle<edm::View<reco::MET> > met;
        if( !event.getByToken(*met_, met) ) continue;

        if(met->begin()!=met->end()){
          unsigned int idx=met_-mets_.begin();
          if(idx==0){
            caloMET=*met->begin(); 
          }
        }
      }


      /* 
         ------------------------------------------------------------

         Event Monitoring

         ------------------------------------------------------------
         */
      const edm::TriggerNames& triggerNames = event.triggerNames(*triggerTable);
      // loop over trigger paths 
      for(unsigned int i=0; i<triggerNames.triggerNames().size(); ++i){
        bool elecmu = false;
        bool dielec = false;
        bool dimuon = false;
        // consider only path from triggerPaths
        string name = triggerNames.triggerNames()[i];
        for (unsigned int j=0; j<triggerPaths.size(); j++) {
          if (TString(name.c_str()).Contains(TString(triggerPaths[j]), TString::kIgnoreCase) && TString(name.c_str()).Contains(TString("ele"), TString::kIgnoreCase) && TString(name.c_str()).Contains(TString("mu"), TString::kIgnoreCase)) elecmu = true;
          else {
            if (TString(name.c_str()).Contains(TString(triggerPaths[j]), TString::kIgnoreCase) && TString(name.c_str()).Contains(TString("ele"), TString::kIgnoreCase)) dielec = true;
            if (TString(name.c_str()).Contains(TString(triggerPaths[j]), TString::kIgnoreCase) && TString(name.c_str()).Contains(TString("mu"), TString::kIgnoreCase)) dimuon = true;
          }
        }

        // ELECMU channel
        if( elecmu ){
          fill("decayChannel_", 0.5);
          if( isoElecs.size()>0 && isoMuons.size()>0 ) {
            double mass = (isoElecs[0]->p4()+isoMuons[0]->p4()).mass();
            if( (lowerEdge_==-1. && upperEdge_==-1.) || (lowerEdge_<mass && mass<upperEdge_) ){
              // fill plots for trigger monitoring
              if(!triggerTable_.isUninitialized()) fill(event, *triggerTable, "elecMu", elecMuPaths_);
              if(elecMuLogged_<=hists_.find("elecMuLogger_")->second->getNbinsY()){
                // log runnumber, lumi block, event number & some
                // more pysics infomation for interesting events
                fill("elecMuLogger_", 0.5, elecMuLogged_+0.5, event.eventAuxiliary().run()); 
                fill("elecMuLogger_", 1.5, elecMuLogged_+0.5, event.eventAuxiliary().luminosityBlock()); 
                fill("elecMuLogger_", 2.5, elecMuLogged_+0.5, event.eventAuxiliary().event()); 
                fill("elecMuLogger_", 3.5, elecMuLogged_+0.5, isoMuons[0]->pt()); 
                fill("elecMuLogger_", 4.5, elecMuLogged_+0.5, isoElecs[0]->pt()); 
                if(leadingJets.size()>0) fill("elecMuLogger_", 5.5, elecMuLogged_+0.5, leadingJets[0].pt()); 
                if(leadingJets.size()>1) fill("elecMuLogger_", 6.5, elecMuLogged_+0.5, leadingJets[1].pt()); 
                fill("elecMuLogger_", 7.5, elecMuLogged_+0.5, caloMET.et()); 
                ++elecMuLogged_; 
              }
            }
          }
        }

        // DIMUON channel
        if( dimuon ){
          fill("decayChannel_", 1.5);
          if (isoMuons.size()>1) {
            int charge = isoMuons[0]->charge()*isoMuons[1]->charge();
            double mass = (isoMuons[0]->p4()+isoMuons[1]->p4()).mass();
            fill(charge<0 ? "invMass_"    : "invMassWC_"    , mass       );
            if((lowerEdge_==-1. && upperEdge_==-1.) || (lowerEdge_<mass && mass<upperEdge_) ){
              // fill plots for trigger monitoring
              if(!triggerTable_.isUninitialized()) fill(event, *triggerTable, "diMuon", diMuonPaths_);
              if(diMuonLogged_<=hists_.find("diMuonLogger_")->second->getNbinsY()){
                // log runnumber, lumi block, event number & some
                // more pysics infomation for interesting events
                fill("diMuonLogger_", 0.5, diMuonLogged_+0.5, event.eventAuxiliary().run()); 
                fill("diMuonLogger_", 1.5, diMuonLogged_+0.5, event.eventAuxiliary().luminosityBlock()); 
                fill("diMuonLogger_", 2.5, diMuonLogged_+0.5, event.eventAuxiliary().event()); 
                fill("diMuonLogger_", 3.5, diMuonLogged_+0.5, isoMuons[0]->pt()); 
                fill("diMuonLogger_", 4.5, diMuonLogged_+0.5, isoMuons[1]->pt()); 
                if(leadingJets.size()>0) fill("diMuonLogger_", 5.5, diMuonLogged_+0.5, leadingJets[0].pt()); 
                if(leadingJets.size()>1) fill("diMuonLogger_", 6.5, diMuonLogged_+0.5, leadingJets[1].pt()); 
                fill("diMuonLogger_", 7.5, diMuonLogged_+0.5, caloMET.et()); 
                ++diMuonLogged_; 
              }
            }
          }
        }

        // DIELEC channel
        if( dielec ){
          fill("decayChannel_", 2.5);
          if( dielec  && isoElecs.size()>1 ){
            int charge = isoElecs[0]->charge()*isoElecs[1]->charge();
            double mass = (isoElecs[0]->p4()+isoElecs[1]->p4()).mass();
            fill(charge<0 ? "invMass_"    : "invMassWC_"    , mass       );
            if((lowerEdge_==-1. && upperEdge_==-1.) || (lowerEdge_<mass && mass<upperEdge_) ){
              // fill plots for trigger monitoring
              if(!triggerTable_.isUninitialized()) fill(event, *triggerTable, "diElec", diElecPaths_);
              if(diElecLogged_<=hists_.find("diElecLogger_")->second->getNbinsY()){
                // log runnumber, lumi block, event number & some
                // more pysics infomation for interesting events
                fill("diElecLogger_", 0.5, diElecLogged_+0.5, event.eventAuxiliary().run()); 
                fill("diElecLogger_", 1.5, diElecLogged_+0.5, event.eventAuxiliary().luminosityBlock()); 
                fill("diElecLogger_", 2.5, diElecLogged_+0.5, event.eventAuxiliary().event()); 
                fill("diElecLogger_", 3.5, diElecLogged_+0.5, isoElecs[0]->pt()); 
                fill("diElecLogger_", 4.5, diElecLogged_+0.5, isoElecs[1]->pt()); 
                if(leadingJets.size()>0) fill("diElecLogger_", 5.5, diElecLogged_+0.5, leadingJets[0].pt()); 
                if(leadingJets.size()>1) fill("diElecLogger_", 6.5, diElecLogged_+0.5, leadingJets[1].pt()); 
                fill("diElecLogger_", 7.5, diElecLogged_+0.5, caloMET.et()); 
                ++diElecLogged_; 
              }
            }
          }
        }
      }

      /* 
         ------------------------------------------------------------

         HLT Objects Monitoring

         ------------------------------------------------------------
         */

      // loop over trigger paths 
      for(unsigned int i=0; i<triggerNames.triggerNames().size(); ++i){
        // consider only path from triggerPaths
        string name = triggerNames.triggerNames()[i].c_str();
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
        // look only for modules actually run in this path
        unsigned int kElec=0;
        unsigned int kMuon=0;
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

          }
        }
        bool l1Matched = false;
        bool l2Matched = false;
        double l1DeltaRMin = 500.;
        double l2DeltaRMin = 500.;
        unsigned int l1IndMatched = 500;
        unsigned int l2IndMatched = 500;
        // access to hlt dielecs
        electronIds_.clear(); electronRefs_.clear();
        if (kElec > 0 && kMuon < 1 && isoElecs.size()>0) {
          const string& moduleLabelElec(moduleLabels[kElec]);
          const string  moduleTypeElec(hltConfig.moduleType(moduleLabelElec));
          const unsigned int filterIndexElec(triggerEventWithRefsHandle->filterIndex(edm::InputTag(moduleLabelElec,"",processName_)));
          triggerEventWithRefsHandle->getObjects(filterIndexElec,electronIds_,electronRefs_);
          const unsigned int nElectrons(electronIds_.size());
          double deltar1 = 600.;
          double deltar2 = 600.;
          for (unsigned int inde = 0; inde < isoElecs.size(); inde++) {
            if (nElectrons > 0) deltar1 = deltaR(*electronRefs_[0],*isoElecs[inde]); 
            if (nElectrons > 1) deltar2 = deltaR(*electronRefs_[1],*isoElecs[inde]); 
            if (deltar1 < deltar2 && deltar1 < l1DeltaRMin) {
              l1DeltaRMin = deltar1;
              l1IndMatched = inde;
            }
            if (deltar2 < deltar1 && deltar2 < l2DeltaRMin) {
              l2DeltaRMin = deltar2;
              l2IndMatched = inde;
            }
          }
          if (nElectrons > 0 && l1IndMatched < 500) fill("leptDeltaREta_", isoElecs[l1IndMatched]->eta(), l1DeltaRMin);   
          if (nElectrons > 1 && l2IndMatched < 500) fill("leptDeltaREta_", isoElecs[l2IndMatched]->eta(), l2DeltaRMin);   
          if (l1DeltaRMin < DRMIN) {
            l1Matched = true;
            fill("matchingMon_", 0.5 );
            fill("leptResolution_", fabs(isoElecs[l1IndMatched]->pt()-electronRefs_[0]->pt())/isoElecs[l1IndMatched]->pt() );   
          }
          if (l2DeltaRMin < DRMIN) {
            l2Matched = true;
            fill("matchingMon_", 1.5 );
            fill("leptResolution_", fabs(isoElecs[l2IndMatched]->pt()-electronRefs_[1]->pt())/isoElecs[l2IndMatched]->pt() );   
          }
        }
        // access to hlt dimuons
        muonIds_.clear(); muonRefs_.clear();
        l1DeltaRMin = 500.; l2DeltaRMin = 500.; double l3DeltaRMin = 500.;
        l1IndMatched = 500; l2IndMatched = 500; double l3IndMatched = 500;
        if (kMuon > 0 && kElec < 1 && isoMuons.size()>0) {
          const string& moduleLabelMuon(moduleLabels[kMuon]);
          const string  moduleTypeMuon(hltConfig.moduleType(moduleLabelMuon));
          const unsigned int filterIndexMuon(triggerEventWithRefsHandle->filterIndex(edm::InputTag(moduleLabelMuon,"",processName_)));
          triggerEventWithRefsHandle->getObjects(filterIndexMuon,muonIds_,muonRefs_);
          trigger::VRmuon myMuonRefs;
          const unsigned int nMuons(muonIds_.size());
          for (unsigned int l=0; l<nMuons; l++) {
            bool isNew = true;
            for (unsigned int ll=0; ll<myMuonRefs.size(); ll++) {
              if (fabs((myMuonRefs[ll]->pt()-muonRefs_[l]->pt())/muonRefs_[l]->pt()) < 1e-5) isNew = false;
            }
            if (isNew) myMuonRefs.push_back(muonRefs_[l]);
          }
          const unsigned int nMyMuons(myMuonRefs.size());
          double deltar1 = 600.;
          double deltar2 = 600.;
          double deltar3 = 600.;
          for (unsigned int indm = 0; indm < isoMuons.size(); indm++) {
            if (nMyMuons > 0) deltar1 = deltaR(*myMuonRefs[0],*isoMuons[indm]); 
            if (nMyMuons > 1) deltar2 = deltaR(*myMuonRefs[1],*isoMuons[indm]); 
            if (nMyMuons > 2) deltar3 = deltaR(*myMuonRefs[2],*isoMuons[indm]); 
            if (nMyMuons > 0 && (nMyMuons<1 || deltar1 < deltar2) && (nMyMuons<2 || deltar1<deltar3) && deltar1 < l1DeltaRMin) {
              l1DeltaRMin = deltar1;
              l1IndMatched = indm;
            }
            if (nMyMuons > 1 && deltar2 < deltar1 && (nMyMuons<3 || deltar2<deltar3) && deltar2 < l2DeltaRMin) {
              l2DeltaRMin = deltar2;
              l2IndMatched = indm;
            }
            if (nMyMuons > 2 && deltar3 < deltar1 && deltar3 < deltar2 && deltar3 < l3DeltaRMin) {
              l3DeltaRMin = deltar3;
              l3IndMatched = indm;
            }
          }
          if (nMyMuons > 0 && l1IndMatched < 500) fill("leptDeltaREta_", isoMuons[l1IndMatched]->eta(), l1DeltaRMin);   
          if (nMyMuons > 1 && l2IndMatched < 500) fill("leptDeltaREta_", isoMuons[l2IndMatched]->eta(), l2DeltaRMin);   
          if (nMyMuons > 2 && l3IndMatched < 500) fill("leptDeltaREta_", isoMuons[l3IndMatched]->eta(), l3DeltaRMin);   
          if (l1DeltaRMin < DRMIN) {
            l1Matched = true;
            fill("matchingMon_", 0.5 );
            fill("leptResolution_", fabs(isoMuons[l1IndMatched]->pt()-myMuonRefs[0]->pt())/isoMuons[l1IndMatched]->pt() );   
            if (l2DeltaRMin < DRMIN) {
              l2Matched = true;
              fill("matchingMon_", 1.5 );
              fill("leptResolution_", fabs(isoMuons[l2IndMatched]->pt()-myMuonRefs[1]->pt())/isoMuons[l2IndMatched]->pt() );
            } else if (l3DeltaRMin < DRMIN) {
              l2Matched = true;
              fill("matchingMon_", 1.5 );
              fill("leptResolution_", fabs(isoMuons[l3IndMatched]->pt()-myMuonRefs[2]->pt())/isoMuons[l3IndMatched]->pt() );
            } 
          } else {
            if (l2DeltaRMin < DRMIN) {
              l1Matched = true;
              fill("matchingMon_", 0.5 );
              fill("leptResolution_", fabs(isoMuons[l2IndMatched]->pt()-myMuonRefs[1]->pt())/isoMuons[l2IndMatched]->pt() );
              if (l3DeltaRMin < DRMIN) {
                l2Matched = true;
                fill("matchingMon_", 1.5 );
                fill("leptResolution_", fabs(isoMuons[l3IndMatched]->pt()-myMuonRefs[2]->pt())/isoMuons[l3IndMatched]->pt() );
              } 
            }
            if (l3DeltaRMin < DRMIN) {
              l1Matched = true;
              fill("matchingMon_", 0.5 );
              fill("leptResolution_", fabs(isoMuons[l3IndMatched]->pt()-myMuonRefs[2]->pt())/isoMuons[l3IndMatched]->pt() );
            }
          } 
        }
        // access to hlt elec-muon
        electronIds_.clear(); electronRefs_.clear();
        muonIds_.clear(); muonRefs_.clear();
        l1DeltaRMin = 500.; l2DeltaRMin = 500.; 
        l1IndMatched = 500; l2IndMatched = 500; 
        if (kElec > 0 && kMuon > 0 && isoElecs.size()>0) {
          const string& moduleLabelElec(moduleLabels[kElec]);
          const string  moduleTypeElec(hltConfig.moduleType(moduleLabelElec));
          const unsigned int filterIndexElec(triggerEventWithRefsHandle->filterIndex(edm::InputTag(moduleLabelElec,"",processName_)));
          triggerEventWithRefsHandle->getObjects(filterIndexElec,electronIds_,electronRefs_);
          const unsigned int nElectrons(electronIds_.size());
          double deltar = 600.;
          for (unsigned int inde = 0; inde < isoElecs.size(); inde++) {
            if (nElectrons > 0) deltar = deltaR(*electronRefs_[0],*isoElecs[inde]); 
            if (deltar < l1DeltaRMin) {
              l1DeltaRMin = deltar;
              l1IndMatched = inde;
            }
          }
          if (nElectrons > 0 && l1IndMatched < 500) fill("leptDeltaREta_", isoElecs[l1IndMatched]->eta(), l1DeltaRMin);   
          if (l1DeltaRMin < DRMIN) {
            l1Matched = true;
            fill("matchingMon_", 0.5 );
            fill("leptResolution_", fabs(isoElecs[l1IndMatched]->pt()-electronRefs_[0]->pt())/isoElecs[l1IndMatched]->pt() );   
          }
        }
        if (kElec > 0 && kMuon > 0 && isoMuons.size()>0) {
          const string& moduleLabelMuon(moduleLabels[kMuon]);
          const string  moduleTypeMuon(hltConfig.moduleType(moduleLabelMuon));
          const unsigned int filterIndexMuon(triggerEventWithRefsHandle->filterIndex(edm::InputTag(moduleLabelMuon,"",processName_)));
          triggerEventWithRefsHandle->getObjects(filterIndexMuon,muonIds_,muonRefs_);
          const unsigned int nMuons(muonIds_.size());
          if (isoMuons.size()<1) continue;
          double deltar = 600.;
          for (unsigned int indm = 0; indm < isoMuons.size(); indm++) {
            if (nMuons > 0) deltar = deltaR(*muonRefs_[0],*isoMuons[indm]); 
            if (deltar < l2DeltaRMin) {
              l2DeltaRMin = deltar;
              l2IndMatched = indm;
            }
          }
          if (nMuons > 0 && l2IndMatched < 500) fill("leptDeltaREta_", isoMuons[l2IndMatched]->eta(), l2DeltaRMin);   
          if (l2DeltaRMin < DRMIN) {
            l2Matched = true;
            fill("matchingMon_", 1.5 );
            fill("leptResolution_", fabs(isoMuons[l2IndMatched]->pt()-muonRefs_[0]->pt())/isoMuons[l2IndMatched]->pt() );   
          }
        }
        if (l1Matched && l2Matched) fill("matchingMon_", 2.5 );
      }

    }

}

TopDiLeptonHLTOfflineDQM::TopDiLeptonHLTOfflineDQM(const edm::ParameterSet& cfg): vertexSelect_(0), beamspotSelect_(0)
{
  // configure the preselection
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
    selection_[selectionStep(selectionOrder_.back())] = std::make_pair(sel.at(i), new HLTOfflineDQMTopDiLepton::MonitorDiLepton(selectionStep(selectionOrder_.back()).c_str(), cfg.getParameter<edm::ParameterSet>("setup"), consumesCollector()));
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
TopDiLeptonHLTOfflineDQM::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
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
TopDiLeptonHLTOfflineDQM::analyze(const edm::Event& event, const edm::EventSetup& setup)
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
TopDiLeptonHLTOfflineDQM::bookHistograms(DQMStore::IBooker &i, edm::Run const&, edm::EventSetup const&)
{
  for (auto& sel: selection_) {
    sel.second.second->book(i);
  }
}
