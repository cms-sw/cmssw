//#include <algorithm>
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DQM/Physics/src/TopDiLeptonOfflineDQM.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DQM/Physics/interface/TopDQMHelpers.h"

namespace TopDiLeptonOffline {

  MonitorEnsemble::MonitorEnsemble(const char* label, const edm::ParameterSet& cfg) : label_(label), lowerEdge_(-1.), upperEdge_(-1.)
  {
    // sources have to be given; this PSet is not optional
    edm::ParameterSet sources=cfg.getParameter<edm::ParameterSet>("sources");
    muons_= sources.getParameter<edm::InputTag>("muons");
    elecs_= sources.getParameter<edm::InputTag>("elecs");
    jets_ = sources.getParameter<edm::InputTag>("jets" );
    mets_ = sources.getParameter<std::vector<edm::InputTag> >("mets" );

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
      if( elecExtras.existsAs<edm::InputTag>("electronId") ){
	electronId_= elecExtras.getParameter<edm::InputTag>("electronId");
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
    }
    // triggerExtras are optional; they may be omitted or empty
    if( cfg.existsAs<edm::ParameterSet>("triggerExtras") ){
      edm::ParameterSet triggerExtras=cfg.getParameter<edm::ParameterSet>("triggerExtras");
      triggerTable_=triggerExtras.getParameter<edm::InputTag>("src");
      elecMuPaths_ =triggerExtras.getParameter<std::vector<std::string> >("pathsELECMU");
      diMuonPaths_ =triggerExtras.getParameter<std::vector<std::string> >("pathsDIMUON");
    }
    // massExtras is optional; in case it's not found no mass
    // window cuts are applied for the same flavor monitor
    // histograms
    if( cfg.existsAs<std::string>("massExtras") ){
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
    unsigned int nElecMu=elecMuPaths_.size();
    unsigned int nDiMuon=diMuonPaths_.size();

    // --- [STANDARD] --- //
    // invariant mass of opposite charge lepton pair (only filled for same flavor)
    hists_["invMass_"     ] = store_->book1D("InvMass"     , "M(lep1, lep2)"           ,       80,   0.,     320.);
    // invariant mass of opposite charge lepton pair (only filled for same flavor)
    hists_["invMassLog_"  ] = store_->book1D("InvMassLog"  , "log_{10}(M(lep1, lep2))" ,       80,   .1,      2.5);
    // invariant mass of same charge lepton pair (log10 for low mass region, only filled for same flavor)
    hists_["invMassWC_"   ] = store_->book1D("InvMassWC"   , "M_{WC}(L1, L2)"          ,       80,   0.,     320.);
    // invariant mass of same charge lepton pair (log10 for low mass region, only filled for same flavor)
    hists_["invMassWCLog_"] = store_->book1D("InvMassLogWC", "log_{10}(M_{WC})"        ,       80,   .1,      2.5);
    // decay channel [1]: muon/muon, [2]:elec/elec, [3]:elec/muon 
    hists_["decayChannel_"] = store_->book1D("DecayChannel", "Decay Channel"           ,        3,    0,        3);
    // trigger efficiency estimates for the electron muon channel
    hists_["elecMuEff_"   ] = store_->book1D("ElecMuEff"   , "Eff(e/#mu paths)"        ,  nElecMu,   0.,  nElecMu);
    // monitored trigger occupancy for the electron muon channel
    hists_["elecMuMon_"   ] = store_->book1D("ElecMuMon"   , "Mon(e/#mu paths)"        ,  nElecMu,   0.,  nElecMu);
    // trigger efficiency estimates for the di muon channel
    hists_["diMuonEff_"   ] = store_->book1D("DiMuonEff"   , "Eff(#mu/#mu paths)"      ,  nDiMuon,   0.,  nDiMuon);
    // monitored trigger occupancy for the di muon channel
    hists_["diMuonMon_"   ] = store_->book1D("DiMuonMon"   , "Mon(#mu/#mu paths)"      ,  nDiMuon,   0.,  nDiMuon);
    // pt of the leading lepton
    hists_["lep1Pt_"      ] = store_->book1D("LeptPt"      , "pt(lep1)"                ,       50,   0.,     200.);
    // pt of the 2. leading lepton
    hists_["lep2Pt_"      ] = store_->book1D("Lep2Pt"      , "pt(lep2)"                ,       50,   0.,     200.);
    // multiplicity of jets with pt>30 (corrected to L2+L3)
    hists_["jetMult_"     ] = store_->book1D("JetMult"     , "N_{30}(jet)"             ,       10,   0.,      10.); 
    // MET (calo)
    hists_["metCalo_"     ] = store_->book1D("METCalo"     , "MET_{Calo}"              ,       50,   0.,     200.);

    // set bin labels for trigger monitoring
    triggerBinLabels(std::string("elecMu"), elecMuPaths_);
    triggerBinLabels(std::string("diMuon"), diMuonPaths_);
    // set bin labels for decayChannel_
    hists_["decayChannel_"]->setBinLabel( 1, "#mu e"  , 1);
    hists_["decayChannel_"]->setBinLabel( 2, "#mu #mu", 1);
    hists_["decayChannel_"]->setBinLabel( 3, "e e"    , 1);

    if( verbosity_==STANDARD) return;

    // --- [VERBOSE] --- //
    // mean eta of the candidate leptons
    hists_["sumEtaL1L2_"  ] = store_->book1D("SumEtaL1L2"  , "<#eta>(lep1, lep2)"      ,       30,  -5.,       5.); 
    // deltaEta between the 2 candidate leptons
    hists_["dEtaL1L2_"    ] = store_->book1D("DEtaL1L2"    , "#Delta#eta(lep1,lep2)"   ,       30,   0.,       3.);
    // deltaPhi between the 2 candidate leptons
    hists_["dPhiL1L2_"    ] = store_->book1D("DPhiL1L2"    , "#Delta#phi(lep1,lep2)"   ,       32,   0.,      3.2);
    // pt of the candidate electron (depending on the decay channel)
    hists_["elecPt_"      ] = store_->book1D("ElecPt"      , "pt(e)"                   ,       50,   0.,     200.);
    // relative isolation of the candidate electron (depending on the decay channel)
    hists_["elecRelIso_"  ] = store_->book1D("ElecRelIso"  , "Iso_{Rel}(e)"            ,       50,   0.,       1.);
    // pt of the canddiate muon (depending on the decay channel)
    hists_["muonPt_"      ] = store_->book1D("MuonPt"      , "pt(#mu)"                 ,       50,   0.,     200.);
    // relative isolation of the candidate muon (depending on the decay channel)
    hists_["muonRelIso_"  ] = store_->book1D("MuonRelIso"  , "Iso_{Rel}(#mu)"          ,       50,   0.,       1.);
    // pt of the 1. leading jet (corrected to L2+L3)
    hists_["jet1Pt_"      ] = store_->book1D("Jet1Pt"      , "pt_{L2L3}(jet1)"         ,       60,   0.,     300.);   
    // pt of the 2. leading jet (corrected to L2+L3)
    hists_["jet2Pt_"      ] = store_->book1D("Jet2Pt"      , "pt_{L2L3}(jet2)"         ,       60,   0.,     300.);
    // MET (PF)
    hists_["metPflow_"    ] = store_->book1D("METPflow"    , "MET_{Pflow}"             ,       50,   0.,     200.);
    // MET (TC)
    hists_["metTC_"       ] = store_->book1D("METTC"       , "MET_{TC}"                ,       50,   0.,     200.);
    // dz for muons (to suppress cosmis)
    hists_["muonDelZ_"    ] = store_->book1D("MuonDelZ"    , "d_{z}(#mu)"              ,       50, -25.,      25.);
    // dxy for muons (to suppress cosmics)
    hists_["muonDelXY_"   ] = store_->book2D("MuonDelXY"   , "d_{xy}(#mu)"                , 50,  -0.1,  0.1, 50, -0.1,  0.1 );
    // lepton multiplicity after std isolation
    hists_["lepMultIso_"  ] = store_->book2D("LepMultIso"  , "N_{Iso}(e) vs N_{Iso}(#mu)" ,  5,    0.,   5.,  5,   0.,    5.);

    // set axes titles for dxy for muons
    hists_["muonDelXY_"   ]->setAxisTitle( "x [cm]", 1); hists_["muonDelXY_"   ]->setAxisTitle( "y [cm]", 2);
    // set axes titles for lepton multiplicity after std isolation
    hists_["lepMultIso_"  ]->setAxisTitle( "N_{Iso}(#mu)", 1); hists_["lepMultIso_"   ]->setAxisTitle( "N_{Iso}(elec)", 2);

    if( verbosity_==VERBOSE) return;

    // --- [DEBUG] --- //
    // electron multiplicity after std isolation
    hists_["elecMultIso_" ] = store_->book1D("ElecMultIso" , "N_{Iso}(e)"              ,       10,   0.,      10.);
    // muon multiplicity after std isolation
    hists_["muonMultIso_" ] = store_->book1D("MuonMultIso" , "N_{Iso}(#mu)"            ,       10,   0.,      10.);
    // calo isolation of the candidate muon (depending on the decay channel)
    hists_["muonCalIso_"  ] = store_->book1D("MuonCalIso"  , "Iso_{Cal}(#mu)"          ,       50,   0.,       1.);
    // track isolation of the candidate muon (depending on the decay channel)
    hists_["muonTrkIso_"  ] = store_->book1D("MuonTrkIso"  , "Iso_{Trk}(#mu)"          ,       50,   0.,       1.);
    // calo isolation of the candidate electron (depending on the decay channel)
    hists_["elecCalIso_"  ] = store_->book1D("ElecCalIso"  , "Iso_{Cal}(e)"            ,       50,   0.,       1.);
    // track isolation of the candidate electron (depending on the decay channel)
    hists_["elecTrkIso_"  ] = store_->book1D("ElecTrkIso"  , "Iso_{Trk}(e)"            ,       50,   0.,       1.);
    // eta of the leading jet
    hists_["jet1Eta_"     ] = store_->book1D("Jet1Eta"     , "#eta(jet1)"              ,       30,  -5.,       5.); 
    // eta of the 2. leading jet
    hists_["jet2Eta_"     ] = store_->book1D("Jet2Eta"     , "#eta(jet2)"              ,       30,  -5.,       5.);
    // pt of the 1. leading jet (not corrected)
    hists_["jet1PtRaw_"   ] = store_->book1D("Jet1PtRaw"   , "pt_{Raw}(jet1)"          ,       60,   0.,     300.);   
    // pt of the 2. leading jet (not corrected)     
    hists_["jet2PtRaw_"   ] = store_->book1D("Jet2PtRaw"   , "pt_{Raw}(jet2)"          ,       60,   0.,     300.);
    // deltaEta between the 2 leading jets
    hists_["dEtaJet1Jet2_"] = store_->book1D("DEtaJet1Jet2", "#Delta#eta(jet1,jet2)"   ,       30,   0.,       3.);
    // deltaEta between the lepton and the leading jet
    hists_["dEtaJet1Lep1_"] = store_->book1D("DEtaJet1Lep1", "#Delta#eta(jet1,lep1)"   ,       30,   0.,       3.);
    // deltaEta between the lepton and MET
    hists_["dEtaLep1MET_" ] = store_->book1D("DEtaLep1MET" , "#Delta#eta(lep1,MET)"    ,       30,   0.,       3.);
    // deltaEta between leading jet and MET
    hists_["dEtaJet1MET_" ] = store_->book1D("DEtaJet1MET" , "#Delta#eta(jet1,MET)"    ,       30,   0.,       3.);
    // deltaPhi of 2 leading jets
    hists_["dPhiJet1Jet2_"] = store_->book1D("DPhiJet1Jet2", "#Delta#phi(jet1,jet2)"   ,       32,   0.,      3.2);
    // deltaPhi of 1. lepton and 1. jet
    hists_["dPhiJet1Lep1_"] = store_->book1D("DPhiJet1Lep1", "#Delta#phi(jet1,lep1)"   ,       32,   0.,      3.2);
    // deltaPhi of 1. lepton and MET
    hists_["dPhiLep1MET_" ] = store_->book1D("DPhiLep1MET" , "#Delta#phi(lep1,MET)"    ,       32,   0.,      3.2);
    // deltaPhi of 1. jet and MET
    hists_["dPhiJet1MET_" ] = store_->book1D("DPhiJet1MET" , "#Delta#phi(jet1,MET)"    ,       32,   0.,      3.2);
    return;
  }

  void 
  MonitorEnsemble::fill(const edm::Event& event, const edm::EventSetup& setup) const
  {
    // fetch trigger event if configured such 
    edm::Handle<edm::TriggerResults> triggerTable;
    if(!triggerTable_.label().empty()) event.getByLabel(triggerTable_, triggerTable);

    /* 
    ------------------------------------------------------------

    Muon Selection

    ------------------------------------------------------------
    */

    // buffer isolated muons
    std::vector<const reco::Muon*> isoMuons;
    edm::Handle<edm::View<reco::Muon> > muons; event.getByLabel(muons_, muons);
    for(edm::View<reco::Muon>::const_iterator muon=muons->begin(); muon!=muons->end(); ++muon){
      // restrict to globalMuons
      if( muon->isGlobalMuon() ){ 
	fill("muonDelZ_" , muon->globalTrack()->vz());
	fill("muonDelXY_", muon->globalTrack()->vx(), muon->globalTrack()->vy());
	// apply preselection
	if((*muonSelect_)(*muon)){
	  double isolationTrk = muon->pt()/(muon->pt()+muon->isolationR03().sumPt);
	  double isolationCal = muon->pt()/(muon->pt()+muon->isolationR03().emEt+muon->isolationR03().hadEt);
	  double isolationRel = (muon->isolationR03().sumPt+muon->isolationR03().emEt+muon->isolationR03().hadEt)/muon->pt();
	  fill("muonTrkIso_" , isolationTrk); fill("muonCalIso_" , isolationCal); fill("muonRelIso_" , isolationRel);
	  if((*muonIso_)(*muon)) isoMuons.push_back(&(*muon));
	}
      }
    }
    fill("muonMultIso_", isoMuons.size());

    /* 
    ------------------------------------------------------------

    Electron Selection

    ------------------------------------------------------------
    */

    // buffer isolated electronss
    std::vector<const reco::GsfElectron*> isoElecs;
    edm::Handle<edm::ValueMap<float> > electronId; 
    if(!electronId_.label().empty()) event.getByLabel(electronId_, electronId);
    edm::Handle<edm::View<reco::GsfElectron> > elecs; event.getByLabel(elecs_, elecs);
    for(edm::View<reco::GsfElectron>::const_iterator elec=elecs->begin(); elec!=elecs->end(); ++elec){
      // restrict to electrons with good electronId
      int index = elec-elecs->begin();
      if( electronId_.label().empty() ? true : (*electronId)[elecs->refAt(index)]>0.99 ){
	// apply preselection
	if((*elecSelect_)(*elec)){
	  double isolationTrk = elec->pt()/(elec->pt()+elec->dr03TkSumPt());
	  double isolationCal = elec->pt()/(elec->pt()+elec->dr04EcalRecHitSumEt()+elec->dr04HcalTowerSumEt());
	  double isolationRel = (elec->dr03TkSumPt()+elec->dr04EcalRecHitSumEt()+elec->dr04HcalTowerSumEt())/elec->pt();
	  fill("elecTrkIso_" , isolationTrk); fill("elecCalIso_" , isolationCal); fill("elecRelIso_" , isolationRel);
	  if((*elecIso_)(*elec)) isoElecs.push_back(&(*elec));
	}
      }
    }
    fill("elecMultIso_", isoElecs.size());

    /* 
    ------------------------------------------------------------

    Jet Selection

    ------------------------------------------------------------
    */

    unsigned int mult30=0;
    // buffer leadingJets
    std::vector<reco::Jet> leadingJets;
    const JetCorrector* corrector=0;
    if(!jetCorrector_.empty()){
      // check whether a jet correcto is in the event setup or not
      if(setup.find( edm::eventsetup::EventSetupRecordKey::makeKey<JetCorrectionsRecord>() )){
	corrector = JetCorrector::getJetCorrector(jetCorrector_, setup);
      }
      else{
	//edm::LogVerbatim( "TopDiLeptonOfflineDQM" ) 
	//  << "\n"
	//  << "------------------------------------------------------------------------------------- \n"
	//  << " No JetCorrectionsRecord available from EventSetup:                                   \n" 
	//  << "  - Jets will not be corrected.                                                       \n"
	//  << "  - If you want to change this add the following                                      \n"
	//  << "    lines to your cfg file:                                                           \n"
	//  << "                                                                                      \n"
	//  << "  ## load jet corrections                                                             \n"
	//  << "  process.load(\"JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff\") \n"
	//  << "  process.prefer(\"ak5CaloL2L3\")                                                     \n"
	//  << "                                                                                      \n"
	//  << "------------------------------------------------------------------------------------- \n";
      }
    }
    edm::Handle<edm::View<reco::Jet> > jets; event.getByLabel(jets_, jets);
    for(edm::View<reco::Jet>::const_iterator jet=jets->begin(); jet!=jets->end(); ++jet){
      //check for overlap with the selected electrons
      bool overlap=false;
      unsigned int index=jet-jets->begin();
      for(std::vector<const reco::GsfElectron*>::const_iterator elec=isoElecs.begin(); elec!=isoElecs.end(); ++elec){
	if(reco::deltaR((*elec)->eta(), (*elec)->phi(), jet->eta(), jet->phi())<0.4){overlap=true; break;}
      }
      if(!overlap){
	// determine jet correction scale
	reco::Jet correctedJet=*jet; correctedJet.scaleEnergy(corrector ?  corrector->correction(*jet) : 1.);
	if(correctedJet.pt()>30) {++mult30;} // determine jet multiplicity
	if(index==0) {
	  leadingJets.push_back(correctedJet);
	  fill("jet1Pt_"    , correctedJet.pt());
	  fill("jet1PtRaw_" , jet->pt() );
	  fill("jet1Eta_"   , jet->eta());
	}
	if(index==1) {
	  leadingJets.push_back(correctedJet);
	  fill("jet2Pt_"    , correctedJet.pt());
	  fill("jet2PtRaw_" , jet->pt() );
	  fill("jet2Eta_"   , jet->eta());
	}
      }
    }
    if(leadingJets.size()>1){
      fill("dEtaJet1Jet2_" , leadingJets[0].eta()-leadingJets[1].eta());
      fill("dPhiJet1Jet2_" , reco::deltaPhi(leadingJets[0].phi(), leadingJets[1].phi()));
      if( !isoMuons.empty() ){
	if( isoElecs.empty() || isoMuons[0]->pt()>isoElecs[0]->pt() ){
	  fill("dEtaJet1Lep1_" , isoMuons[0]->eta()-leadingJets[0].eta());
	  fill("dPhiJet1Lep1_" , reco::deltaPhi(isoMuons[0]->phi() , leadingJets[0].phi()));
	} 
      }
      if( !isoElecs.empty() ){
	if( isoMuons.empty() || isoElecs[0]->pt()>isoMuons[0]->pt() ){
	  fill("dEtaJet1Lep1_" , isoElecs[0]->eta()-leadingJets[0].eta());
	  fill("dPhiJet1Lep1_" , reco::deltaPhi(isoElecs[0]->phi() , leadingJets[0].phi()));
	}
      }
    }
    fill("jetMult_", mult30);
    
    /* 
    ------------------------------------------------------------

    MET Selection

    ------------------------------------------------------------
    */

    for(std::vector<edm::InputTag>::const_iterator met_=mets_.begin(); met_!=mets_.end(); ++met_){
      edm::Handle<edm::View<reco::MET> > met; event.getByLabel(*met_, met);
      if(met->begin()!=met->end()){
	unsigned int idx=met_-mets_.begin();
	if(idx==0){ 
	  fill("metCalo_"     , met->begin()->et());
	  if(!leadingJets.empty()){
	    fill("dEtaJet1MET_" , leadingJets[0].eta()-met->begin()->eta());
	    fill("dPhiJet1MET_" , reco::deltaPhi(leadingJets[0].phi(), met->begin()->phi()));
	  }
	  if( !isoMuons.empty() ){
	    if( isoElecs.empty() || isoMuons[0]->pt()>isoElecs[0]->pt() ){
	      fill("dEtaLep1MET_" , isoMuons[0]->eta()-met->begin()->eta());
	      fill("dPhiLep1MET_" , reco::deltaPhi(isoMuons[0]->phi(), met->begin()->phi()));
	    } 
	  }
	  if( !isoElecs.empty() ){
	    if( isoMuons.empty() || isoElecs[0]->pt()>isoMuons[0]->pt() ){
	      fill("dEtaLep1MET_" , isoElecs[0]->eta()-met->begin()->eta());
	      fill("dPhiLep1MET_" , reco::deltaPhi(isoElecs[0]->phi(), met->begin()->phi()));
	    }
	  }
	}
	if(idx==1){ fill("metTC_"   , met->begin()->et());}
	if(idx==2){ fill("metPflow_", met->begin()->et());}
      }
    }


    /* 
    ------------------------------------------------------------

    Event Monitoring

    ------------------------------------------------------------
    */

    // check number of isolated leptons
    fill("lepMultIso_", isoMuons.size(), isoElecs.size());
    // ELECMU channel
    if( decayChannel(isoMuons, isoElecs) == ELECMU ){
      fill("decayChannel_", 0.5);
      double mass = (isoElecs[0]->p4()+isoMuons[0]->p4()).mass();
      if( (lowerEdge_==-1. && upperEdge_==-1.) || (lowerEdge_<mass && mass>upperEdge_) ){
	fill("dEtaL1L2_"  , isoElecs[0]->eta()-isoMuons[0]->eta()); 
	fill("sumEtaL1L2_", (isoElecs[0]->eta()+isoMuons[0]->eta())/2); 
	fill("dPhiL1L2_"  , reco::deltaPhi(isoElecs[0]->phi(), isoMuons[0]->eta())); 
	fill("elecPt_", isoElecs[0]->pt()); fill("muonPt_", isoMuons[0]->pt()); 
	fill("lep1Pt_", isoElecs[0]->pt()>isoMuons[0]->pt() ? isoElecs[0]->pt() : isoMuons[0]->pt());
	fill("lep2Pt_", isoElecs[0]->pt()>isoMuons[0]->pt() ? isoMuons[0]->pt() : isoElecs[0]->pt());
	// fill plots for trigger monitoring
	if(!triggerTable_.label().empty()) fill(event, *triggerTable, "elecMu", elecMuPaths_);
      }
    }
    // DIMUON channel
    if( decayChannel(isoMuons, isoElecs) == DIMUON ){
      fill("decayChannel_", 1.5);
      int charge = isoMuons[0]->charge()*isoMuons[1]->charge();
      double mass = (isoMuons[0]->p4()+isoMuons[1]->p4()).mass();
      fill(charge<0 ? "invMass_"    : "invMassWC_"    , mass       );
      fill(charge<0 ? "invMassLog_" : "invMassWCLog_" , log10(mass));
      if((lowerEdge_==-1. && upperEdge_==-1.) || (lowerEdge_<mass && mass>upperEdge_) ){
	fill("dEtaL1L2"  , isoMuons[0]->eta()-isoMuons[1]->eta() );
	fill("sumEtaL1L2", (isoMuons[0]->eta()+isoMuons[1]->eta())/2);
	fill("dPhiL1L2"  , reco::deltaPhi(isoMuons[0]->phi(),isoMuons[1]->phi()) );
	fill("muonPt_", isoMuons[0]->pt()); fill("muonPt_", isoMuons[1]->pt()); 
	fill("lep1Pt_", isoMuons[0]->pt()); fill("lep2Pt_", isoMuons[1]->pt()); 
	// fill plots for trigger monitoring
	if(!triggerTable_.label().empty()) fill(event, *triggerTable, "diMuon", diMuonPaths_);
      }
    }
    // DIELEC channel
    if( decayChannel(isoMuons, isoElecs) == DIELEC ){
      int charge = isoElecs[0]->charge()*isoElecs[1]->charge();
      double mass = (isoElecs[0]->p4()+isoElecs[1]->p4()).mass();
      fill("decayChannel_", 2.5);
      fill(charge<0 ? "invMass_"    : "invMassWC_"    , mass       );
      fill(charge<0 ? "invMassLog_" : "invMassWCLog_" , log10(mass));
      if((lowerEdge_==-1. && upperEdge_==-1.) || (lowerEdge_<mass && mass>upperEdge_) ){
	fill("dEtaL1L2"  , isoElecs[0]->eta()-isoElecs[1]->eta() );
	fill("sumEtaL1L2", (isoElecs[0]->eta()+isoElecs[1]->eta())/2);
	fill("dPhiL1L2"  , reco::deltaPhi(isoElecs[0]->phi(),isoElecs[1]->phi()) );
	fill("elecPt_", isoElecs[0]->pt()); fill("elecPt_", isoElecs[1]->pt()); 
	fill("lep1Pt_", isoElecs[0]->pt()); fill("lep2Pt_", isoElecs[1]->pt()); 
      }
    }
  }
  
}

TopDiLeptonOfflineDQM::TopDiLeptonOfflineDQM(const edm::ParameterSet& cfg): triggerTable_(""), vertex_("") 
{
  // configure the preselection
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
    selection_[selectionStep(selectionOrder_.back())] = std::make_pair(sel.at(i), new TopDiLeptonOffline::MonitorEnsemble(selectionStep(selectionOrder_.back()).c_str(), cfg.getParameter<edm::ParameterSet>("setup")));
  }
}

void 
TopDiLeptonOfflineDQM::analyze(const edm::Event& event, const edm::EventSetup& setup)
{ 
  if(!triggerTable_.label().empty()){
    edm::Handle<edm::TriggerResults> triggerTable;
    event.getByLabel(triggerTable_, triggerTable);
    if(!accept(event, *triggerTable, triggerPaths_)) return;
  }
  if(!vertex_.label().empty()){
    edm::Handle<std::vector<reco::Vertex> > vertex;
    event.getByLabel(vertex_, vertex);
    if(vertex->empty() || !(*vertexSelect_)(vertex->front())) return;
  }
  if(!beamspot_.label().empty()){
    edm::Handle<reco::BeamSpot> beamspot;
    event.getByLabel(beamspot_, beamspot);
    if(!(*beamspotSelect_)(*beamspot)) return;
  }
  // apply selection steps
  for(std::vector<std::string>::const_iterator selIt=selectionOrder_.begin(); selIt!=selectionOrder_.end(); ++selIt){
    std::string key = selectionStep(*selIt), type = objectType(*selIt);
    if(selection_.find(key)!=selection_.end()){
      if(type=="empty"){
	selection_[key].second->fill(event, setup);
      }
      if(type=="muons"){
	SelectionStep<reco::Muon> step(selection_[key].first);
	if(step.select(event)){
	  selection_[key].second->fill(event, setup);
	} else break;
      }
      if(type=="elecs"){
	SelectionStep<reco::GsfElectron> step(selection_[key].first);
	if(step.select(event)){ 
	  selection_[key].second->fill(event, setup);
	} else break;
      }
      if(type=="jets" ){
	SelectionStep<reco::Jet> step(selection_[key].first);
	if(step.select(event, setup)){
	  selection_[key].second->fill(event, setup);
	} else break;
      }
      if(type=="jets/pf" ){
	SelectionStep<reco::PFJet> step(selection_[key].first);
	if(step.select(event, setup)){
	  selection_[key].second->fill(event, setup);
	} else break;
      }
      if(type=="jets/calo" ){
	SelectionStep<reco::CaloJet> step(selection_[key].first);
	if(step.select(event, setup)){
	  selection_[key].second->fill(event, setup);
	} else break;
      }
      if(type=="met" ){
	SelectionStep<reco::MET> step(selection_[key].first);
	if(step.select(event)){
	  selection_[key].second->fill(event, setup);
	} else break;
      }
    }
  }
}


