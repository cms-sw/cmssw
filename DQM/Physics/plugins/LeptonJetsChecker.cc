#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQM/Physics/plugins/LeptonJetsChecker.h"

LeptonJetsChecker::LeptonJetsChecker(const edm::ParameterSet& iConfig)

{
  //now do what ever initialization is needed
  dqmStore_ = edm::Service<DQMStore>().operator->();
  outputFileName_      = iConfig.getParameter<std::string>("outputFileName");
  saveDQMMEs_          = iConfig.getParameter<bool>("saveDQMMEs");

  labelTriggerResults_ = iConfig.getParameter<edm::InputTag>("labelTriggerResults");
  useTrigger_          = iConfig.getParameter<bool>("useTrigger");
  labelBeamSpot_       = iConfig.getParameter<edm::InputTag>( "labelBeamSpot" );
  labelMuons_          = iConfig.getParameter<edm::InputTag>( "labelMuons" );
  labelElectrons_      = iConfig.getParameter<edm::InputTag>( "labelElectrons" );
  labelJets_           = iConfig.getParameter<edm::InputTag>( "labelJets" );
  labelMETs_           = iConfig.getParameter<edm::InputTag>( "labelMETs" );
  VetoLooseLepton_       = iConfig.getParameter<bool>( "VetoLooseLepton" );
  ApplyMETCut_         = iConfig.getParameter<bool>( "ApplyMETCut" );
  
  useJES_              = iConfig.getParameter<bool>("useJES");
  jetCorrector_        = iConfig.getParameter<std::string>   ("jetCorrector");
  
  PerformOctoberXDeltaRStep_ = iConfig.getParameter<bool>("PerformOctoberXDeltaRStep");
  
  leptonType_	       = iConfig.getParameter<std::string>( "leptonType" ); 
  if ("electron" == leptonType_) {otherLeptonType_ = "muon";}
  if ("muon" == leptonType_) {otherLeptonType_ = "electron";} 
  if (leptonType_ != "electron" && leptonType_ != "muon") {
    throw cms::Exception("Configuration") << "'leptonType' must be specified in configuration (default: LeptonJetsChecker_cfi.py) as either 'electron' or 'muon'.";
  }
  edm::LogInfo("Debug|LeptonJetsChecker") << "Selecting " << leptonType_ << "s, and vetoing " << otherLeptonType_ << "s." << std::endl;
  relativePath_ = leptonType_;
  relativePath_ += "JetsChecker";
  //Common tools: modules
  //plots after selection
  std::string label("_afterSelection");
  jetChecker        = new JetChecker(iConfig,relativePath_,label);
  metChecker        = new MetChecker(iConfig,relativePath_,label);
  muonChecker       = new MuonChecker(iConfig,relativePath_,label);
  electronChecker   = new ElectronChecker(iConfig,relativePath_,label);
  kinematicsChecker = new KinematicsChecker(iConfig,relativePath_,label);
    
  std::string labelNoSel("_NoSelection");
  jetCheckerNoSel        = new JetChecker(iConfig,relativePath_,labelNoSel);
  metCheckerNoSel        = new MetChecker(iConfig,relativePath_,labelNoSel);
  muonCheckerNoSel       = new MuonChecker(iConfig,relativePath_,labelNoSel);
  electronCheckerNoSel   = new ElectronChecker(iConfig,relativePath_,labelNoSel);
  kinematicsCheckerNoSel = new KinematicsChecker(iConfig,relativePath_,labelNoSel);
  	
  std::string labelNonIso("_after");
  labelNonIso += leptonType_;
  labelNonIso += "NonIsoSelection"; 
  jetCheckerLeptonNonIso        = new JetChecker(iConfig,relativePath_,labelNonIso);
  metCheckerLeptonNonIso        = new MetChecker(iConfig,relativePath_,labelNonIso);
  muonCheckerLeptonNonIso       = new MuonChecker(iConfig,relativePath_,labelNonIso);
  electronCheckerLeptonNonIso   = new ElectronChecker(iConfig,relativePath_,labelNonIso);
  kinematicsCheckerLeptonNonIso = new KinematicsChecker(iConfig,relativePath_,labelNonIso);
  	
  std::string labelLeptonIso("_after");
  labelLeptonIso += leptonType_;
  labelLeptonIso += "IsoSelection";
  jetCheckerLeptonIso        = new JetChecker(iConfig,relativePath_,labelLeptonIso);
  metCheckerLeptonIso        = new MetChecker(iConfig,relativePath_,labelLeptonIso);
  muonCheckerLeptonIso       = new MuonChecker(iConfig,relativePath_,labelLeptonIso);
  electronCheckerLeptonIso   = new ElectronChecker(iConfig,relativePath_,labelLeptonIso);
  kinematicsCheckerLeptonIso = new KinematicsChecker(iConfig,relativePath_,labelLeptonIso);
  	
  std::string labelVetoOtherLeptonType("_afterVeto");
  labelVetoOtherLeptonType += otherLeptonType_;
  labelVetoOtherLeptonType += "Selection";  
  jetCheckerVetoOtherLeptonType        = new JetChecker(iConfig,relativePath_,labelVetoOtherLeptonType);
  metCheckerVetoOtherLeptonType        = new MetChecker(iConfig,relativePath_,labelVetoOtherLeptonType);
  muonCheckerVetoOtherLeptonType       = new MuonChecker(iConfig,relativePath_,labelVetoOtherLeptonType);
  electronCheckerVetoOtherLeptonType   = new ElectronChecker(iConfig,relativePath_,labelVetoOtherLeptonType);
  kinematicsCheckerVetoOtherLeptonType = new KinematicsChecker(iConfig,relativePath_,labelVetoOtherLeptonType);
  	
  if (VetoLooseLepton_) {
    std::string labelVetoLooseMuon("_afterVetoLooseMuon");
    labelVetoLooseMuon += "sSelection";  
    jetCheckerVetoLooseMuon        = new JetChecker(iConfig,relativePath_,labelVetoLooseMuon);
    metCheckerVetoLooseMuon        = new MetChecker(iConfig,relativePath_,labelVetoLooseMuon);
    muonCheckerVetoLooseMuon       = new MuonChecker(iConfig,relativePath_,labelVetoLooseMuon);
    electronCheckerVetoLooseMuon   = new ElectronChecker(iConfig,relativePath_,labelVetoLooseMuon);
    kinematicsCheckerVetoLooseMuon = new KinematicsChecker(iConfig,relativePath_,labelVetoLooseMuon);

    std::string labelVetoLooseElectron("_afterVetoLooseElectron");
    labelVetoLooseElectron += "sSelection";  
    jetCheckerVetoLooseElectron        = new JetChecker(iConfig,relativePath_,labelVetoLooseElectron);
    metCheckerVetoLooseElectron        = new MetChecker(iConfig,relativePath_,labelVetoLooseElectron);
    muonCheckerVetoLooseElectron       = new MuonChecker(iConfig,relativePath_,labelVetoLooseElectron);
    electronCheckerVetoLooseElectron   = new ElectronChecker(iConfig,relativePath_,labelVetoLooseElectron);
    kinematicsCheckerVetoLooseElectron = new KinematicsChecker(iConfig,relativePath_,labelVetoLooseElectron);
  }
  
  if (ApplyMETCut_) {
    std::string labelMETcut("_afterMETSelection");  
    jetCheckerMET        = new JetChecker(iConfig,relativePath_,labelMETcut);
    metCheckerMET        = new MetChecker(iConfig,relativePath_,labelMETcut);
    muonCheckerMET       = new MuonChecker(iConfig,relativePath_,labelMETcut);
    electronCheckerMET   = new ElectronChecker(iConfig,relativePath_,labelMETcut);
    kinematicsCheckerMET = new KinematicsChecker(iConfig,relativePath_,labelMETcut); 
  }
    
  std::string label1Jets("_after1JetsSelection");
  jetChecker1Jets        = new JetChecker(iConfig,relativePath_,label1Jets);
  metChecker1Jets        = new MetChecker(iConfig,relativePath_,label1Jets);
  muonChecker1Jets       = new MuonChecker(iConfig,relativePath_,label1Jets);
  electronChecker1Jets   = new ElectronChecker(iConfig,relativePath_,label1Jets);
  kinematicsChecker1Jets = new KinematicsChecker(iConfig,relativePath_,label1Jets);
  	
  std::string label2Jets("_after2JetsSelection");
  jetChecker2Jets        = new JetChecker(iConfig,relativePath_,label2Jets);
  metChecker2Jets        = new MetChecker(iConfig,relativePath_,label2Jets);
  muonChecker2Jets       = new MuonChecker(iConfig,relativePath_,label2Jets);
  electronChecker2Jets   = new ElectronChecker(iConfig,relativePath_,label2Jets);
  kinematicsChecker2Jets = new KinematicsChecker(iConfig,relativePath_,label2Jets);
  	
  std::string label3Jets("_after3JetsSelection");
  jetChecker3Jets        = new JetChecker(iConfig,relativePath_,label3Jets);
  metChecker3Jets        = new MetChecker(iConfig,relativePath_,label3Jets);
  muonChecker3Jets       = new MuonChecker(iConfig,relativePath_,label3Jets);
  electronChecker3Jets   = new ElectronChecker(iConfig,relativePath_,label3Jets);
  kinematicsChecker3Jets = new KinematicsChecker(iConfig,relativePath_,label3Jets);
  	
  std::string label4Jets("_after4JetsSelection");
  jetChecker4Jets        = new JetChecker(iConfig,relativePath_,label4Jets);
  metChecker4Jets        = new MetChecker(iConfig,relativePath_,label4Jets);
  muonChecker4Jets       = new MuonChecker(iConfig,relativePath_,label4Jets);
  electronChecker4Jets   = new ElectronChecker(iConfig,relativePath_,label4Jets);
  kinematicsChecker4Jets = new KinematicsChecker(iConfig,relativePath_,label4Jets);

  std::string labelLeptonJets(TString("muon" == leptonType_ ? "Muon" : "Electron")+"PlusJets_");
  semiLeptonChecker      = new SemiLeptonChecker(iConfig,relativePath_,labelLeptonJets);
 
  if (PerformOctoberXDeltaRStep_) {
    std::string labelDeltaR("_afterDeltaRSelection");
    jetCheckerDeltaR        = new JetChecker(iConfig,relativePath_,labelDeltaR);
    metCheckerDeltaR        = new MetChecker(iConfig,relativePath_,labelDeltaR);
    muonCheckerDeltaR       = new MuonChecker(iConfig,relativePath_,labelDeltaR);
    electronCheckerDeltaR   = new ElectronChecker(iConfig,relativePath_,labelDeltaR);
    kinematicsCheckerDeltaR = new KinematicsChecker(iConfig,relativePath_,labelDeltaR);
  }

  //Can declare many time the same module and use them for different selection
  //other plots ??

  //Configuration
  NofJets                     = iConfig.getParameter<int>( "NofJets" );
  PtThrJets                   = iConfig.getParameter<double>( "PtThrJets" );
  EtaThrJets                  = iConfig.getParameter<double>( "EtaThrJets" );
  EHThrJets                   = iConfig.getParameter<double>( "EHThrJets" );
  JetDeltaRLeptonJetThreshold = iConfig.getParameter<double>( "JetDeltaRLeptonJetThreshold" );
  applyLeptonJetDeltaRCut     = iConfig.getParameter<bool>("applyLeptonJetDeltaRCut");
  PtThrMuons                  = iConfig.getParameter<double>( "PtThrMuons" );
  EtaThrMuons                 = iConfig.getParameter<double>( "EtaThrMuons" );
  MuonRelIso                  = iConfig.getParameter<double>( "MuonRelIso" );
  MuonVetoEM                  = iConfig.getParameter<double>( "MuonVetoEM" );
  MuonVetoHad                 = iConfig.getParameter<double>( "MuonVetoHad" );
  MuonD0Cut                   = iConfig.getParameter<double>( "MuonD0Cut" );
  useElectronID_              = iConfig.getParameter<bool>("useElectronID");
  electronIDLabel             = iConfig.getParameter<std::string>( "electronIDLabel" );
  PtThrElectrons              = iConfig.getParameter<double>( "PtThrElectrons" );
  EtaThrElectrons             = iConfig.getParameter<double>( "EtaThrElectrons" );
  ElectronRelIso              = iConfig.getParameter<double>( "ElectronRelIso" );
  ElectronD0Cut               = iConfig.getParameter<double>( "ElectronD0Cut" ); 
  Chi2Cut                     = iConfig.getParameter<int>("Chi2Cut");
  NofValidHits                = iConfig.getParameter<int>("NofValidHits");
  METThreshold                = iConfig.getParameter<double>( "METThreshold" );
  triggerPath                 = iConfig.getParameter<std::string>("triggerPath");
  Luminosity                  = iConfig.getParameter<int>("Luminosity");
  Xsection                    = iConfig.getParameter<double>("Xsection");
  vetoEBEETransitionRegion    = iConfig.getParameter<bool>("vetoEBEETransitionRegion");
  
  PtThrMuonLoose              = 0.0;
  EtaThrMuonLoose             = 99.9;
  RelIsoThrMuonLoose          = 99.9;
  PtThrElectronLoose          = 0.0;
  EtaThrElectronLoose         = 99.9; 
  RelIsoThrElectronLoose      = 99.9;
  electronIDLabelLoose        = std::string("eidLoose");
  
  if (VetoLooseLepton_) {
    PtThrMuonLoose            = iConfig.getParameter<double>( "PtThrMuonLoose" );
    EtaThrMuonLoose           = iConfig.getParameter<double>( "EtaThrMuonLoose" );
    RelIsoThrMuonLoose        = iConfig.getParameter<double>( "RelIsoThrMuonLoose" );
    if (PtThrMuonLoose > PtThrMuons) {throw cms::Exception("Configuration") << "Loose Muon pT threshold is higher than standard pT threshold" << std::endl;}
    if (EtaThrMuonLoose < EtaThrMuons) {throw cms::Exception("Configuration") << "Loose Muon eta threshold is lower than standard eta threshold" << std::endl;}
    if (RelIsoThrMuonLoose < MuonRelIso) {throw cms::Exception("Configuration") << "Loose Muon relative isolation  threshold is higher than standard relative isolation threshold" << std::endl;}
    PtThrElectronLoose        = iConfig.getParameter<double>( "PtThrElectronLoose" );  
    EtaThrElectronLoose       = iConfig.getParameter<double>( "EtaThrElectronLoose" );  
    RelIsoThrElectronLoose    = iConfig.getParameter<double>( "RelIsoThrElectronLoose" ); 
    electronIDLabelLoose      = iConfig.getParameter<std::string>( "electronIDLabelLoose" );
    if (PtThrElectronLoose > PtThrElectrons) {throw cms::Exception("Configuration") << "Loose Electron pT threshold is higher than standard pT threshold" << std::endl;}
    if (EtaThrElectronLoose < EtaThrElectrons) {throw cms::Exception("Configuration") << "Loose Electron eta threshold is lower than standard eta threshold" << std::endl;}
    if (RelIsoThrElectronLoose < ElectronRelIso) {throw cms::Exception("Configuration") << "Loose Electron relative isolation  threshold is higher than standard relative isolation threshold" << std::endl;} 
  }
}

LeptonJetsChecker::~LeptonJetsChecker()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  //delete dqmStore_;
  //delete jetChecker;
  //delete metChecker;
  //delete muonChecker;
  //delete kinematicsChecker;
}

double LeptonJetsChecker::ComputeNbEvent(MonitorElement* h, int bin){
  if(h->getBinContent(1)>0)return(h->getBinContent(bin+1)*Xsection*Luminosity/(h->getBinContent(1)));
  else return (0);
}

void
LeptonJetsChecker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //using namespace edm;

  //Here you handle the collection you want to access
  edm::Handle<edm::View<reco::GsfElectron> >  electronsHandle; 
  iEvent.getByLabel(labelElectrons_,electronsHandle);
  edm::View<reco::GsfElectron> electrons = *electronsHandle;

  edm::Handle<edm::View<reco::Muon> >  muonsHandle; 
  iEvent.getByLabel(labelMuons_,muonsHandle);
  edm::View<reco::Muon> muons = *muonsHandle;
   
  edm::Handle<edm::View<reco::CaloJet> >  jetsHandle; 
  iEvent.getByLabel(labelJets_,jetsHandle);
  edm::View<reco::CaloJet> jets = *jetsHandle;
   
  edm::Handle<edm::View<reco::CaloMET> >  metsHandle; 
  iEvent.getByLabel(labelMETs_,metsHandle);
  edm::View<reco::CaloMET> mets = *metsHandle;
  std::vector<reco::CaloMET> vmets;
  for(unsigned int i=0;i<mets.size();i++) vmets.push_back(mets.at(i));

  edm::Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByLabel(labelBeamSpot_, beamSpotHandle);
  reco::BeamSpot beamSpot = * beamSpotHandle;

  edm::Handle<edm::TriggerResults> trigResults;
  iEvent.getByLabel(labelTriggerResults_,trigResults);

  ///  Read eID results:
  edm::Handle<edm::ValueMap<float> >  eIDValueMap; 
  iEvent.getByLabel( electronIDLabel , eIDValueMap ); 
  //const edm::ValueMap<float> & eIDmap = * eIDValueMap ;
  
  ////////////////////////////////////////
  //Check if branches are available
  ////////////////////////////////////////
  if (!jetsHandle.isValid())      throw cms::Exception("ProductNotFound") << "Jet collection not found"       <<std::endl;
  if (!electronsHandle.isValid()) throw cms::Exception("ProductNotFound") << "Electron collection not found"  <<std::endl;
  if (!muonsHandle.isValid())     throw cms::Exception("ProductNotFound") << "Muon collection not found"      <<std::endl;
  if (!metsHandle.isValid())      throw cms::Exception("ProductNotFound") << "MET collection not found"       <<std::endl;
  if (!beamSpotHandle.isValid())  throw cms::Exception("ProductNotFound") << "BeamSpot not found"             <<std::endl;
  if (!trigResults.isValid())     throw cms::Exception("ProductNotFound") << "Trigger results not found"      <<std::endl;
  if (!eIDValueMap.isValid())     throw cms::Exception("ProductNotFound") << "electronID value map not found" <<std::endl;
  ////////////////////////////////////////
   
  edm::LogInfo("Debug|LeptonJetsChecker") << "Analyze event with LeptonJetChecker" << std::endl;
  Selection* selection = new Selection();
  selection->Set(beamSpot, jets, muons, electrons, mets); 
  selection->SetConfiguration(PtThrJets, EtaThrJets, EHThrJets, PtThrMuons, EtaThrMuons, MuonRelIso, MuonVetoEM, MuonVetoHad, PtThrElectrons, EtaThrElectrons, ElectronRelIso);
  selection->SetLeptonType(leptonType_);
  selection->SetMuonConfig(MuonD0Cut, Chi2Cut, NofValidHits);
  selection->SetElectronConfig(ElectronD0Cut, vetoEBEETransitionRegion, useElectronID_);
  selection->SetJetConfig(JetDeltaRLeptonJetThreshold, applyLeptonJetDeltaRCut);
  selection->SetMETConfig(METThreshold);
  selection->SetMuonLooseConfig(PtThrMuonLoose, EtaThrMuonLoose, RelIsoThrMuonLoose);
  selection->SetElectronLooseConfig(PtThrElectronLoose, EtaThrElectronLoose, RelIsoThrElectronLoose);
  const edm::ValueMap<float> & eIDmap = * eIDValueMap ;
  selection->SeteID( electrons, eIDmap );
  
  if (PerformOctoberXDeltaRStep_) {selection->SetJetConfig(JetDeltaRLeptonJetThreshold, false);}

  int nJets = selection->GetJets().size(); // number of jets before JES and Selection
  const JetCorrector *acorrector = JetCorrector::getJetCorrector(jetCorrector_,iSetup);
  selection->SelectJets(iEvent, iSetup, acorrector);

  bool selected = selection->isSelected(NofJets, leptonType_, 1, true);
  //select events have at least NofJets uncorrected jets, exact 1 leptonType_ lepton and veto lepton of other lepton type
  //need to expend selection to veto loose leptons as well
  
  edm::LogInfo("Debug|LeptonJetsChecker") << "Jets before JES and Selection: " << nJets << std::endl;
  edm::LogInfo("Debug|LeptonJetsChecker") << "Jets selected before JES: " << selection->GetSelectedJets().size() << std::endl;
  edm::LogInfo("Debug|LeptonJetsChecker") << "Jets selected after JES: " << selection->GetJets().size() << std::endl;
  edm::LogInfo("Debug|LeptonJetsChecker") << "Muons selected: " << selection->GetSelectedMuons().size() << std::endl;
  edm::LogInfo("Debug|LeptonJetsChecker") << "Electrons selected: " << selection->GetSelectedElectrons().size() << std::endl;
  edm::LogInfo("Debug|LeptonJetsChecker") << "Selection->isSelected " << selected << std::endl;
   
  //modules
  if(selected){
    edm::LogInfo("Debug|LeptonJetsChecker") << "Event selected" << std::endl;
    edm::LogInfo("Debug|LeptonJetsChecker") << "Module MetChecker" << std::endl;
    metChecker->analyze(vmets);
    edm::LogInfo("Debug|LeptonJetsChecker") << "Module JetChecker" << std::endl;
    jetChecker->analyze(selection->GetSelectedJets(), useJES_, iEvent, iSetup);
    edm::LogInfo("Debug|LeptonJetsChecker") << "Module MuonChecker" << std::endl;
    muonChecker->analyze(selection->GetSelectedMuons());
    edm::LogInfo("Debug|LeptonJetsChecker") << "Module ElectronChecker" << std::endl;
    electronChecker->analyze(selection->GetSelectedElectrons(), selection->getBeamSpot());
    edm::LogInfo("Debug|LeptonJetsChecker") << "Module KinematicsChecker" << std::endl;
    kinematicsChecker->analyze(selection->GetSelectedJets(), vmets, selection->GetSelectedMuons(), selection->GetSelectedElectrons());
  }

  //could run the same modules with different selection

  jetCheckerNoSel->analyze(selection->GetSelectedJets(0.,2.4,0.), useJES_, iEvent, iSetup);
  metCheckerNoSel->analyze(vmets);
  muonCheckerNoSel->analyze(selection->GetSelectedMuonsNonIso());
  electronCheckerNoSel->analyze(selection->GetSelectedElectronsNonIso(), selection->getBeamSpot());
  kinematicsCheckerNoSel->analyze(selection->GetSelectedJets(0.,2.4,0.), vmets, selection->GetSelectedMuonsNonIso(), selection->GetSelectedElectronsNonIso());
   
  //////////////////////////////
  //  Selection Table
  //////////////////////////////

  // Jets selected with JES from now on
  int nJetBinToFill = selection->GetJets().size();
  if (nJetBinToFill > 5) {nJetBinToFill = 5;}
  //all
  histocontainer_["Selection"]->Fill(0);
  histocontainer_["Selection_Vs_Multiplicity"]->Fill(nJetBinToFill, 0);
  //trigger
  bool triggered = false;
  edm::TriggerNames triggerNames_;
  triggerNames_.init(*trigResults);
  for(unsigned int i=0; i<triggerNames_.triggerNames().size();i++){
    if(triggerNames_.triggerNames()[i] == triggerPath) {
      if(trigResults->accept(i)){
	triggered = true;
	break;
      }
    }
  }
 
  if(triggered || !useTrigger_) {
    histocontainer_["Selection"]->Fill(1);
    histocontainer_["Selection_Vs_Multiplicity"]->Fill(nJetBinToFill, 1);

    //1 lepton (of specified type)
    bool OneLepton = false;
    if ("electron" == leptonType_) {
      for(unsigned int i=0;i<electrons.size();++i){
	if(electrons[i].pt()> PtThrElectrons && fabs(electrons[i].eta())< EtaThrElectrons){
	  OneLepton = true;
	  break;
	}
      } 	
    }
    if ("muon" == leptonType_) {
      for(unsigned int i=0;i<muons.size();++i){
// 	if(muons[i].pt()> PtThrMuons && fabs(muons[i].eta())< EtaThrMuons){
	if(muons[i].isGlobalMuon()){ // To compare with [TOPANA] sync exercise
	  OneLepton = true;
	  break;
	}
      }
    }
    if (OneLepton) {
      histocontainer_["Selection"]->Fill(2);
      histocontainer_["Selection_Vs_Multiplicity"]->Fill(nJetBinToFill, 2);
    }
    //1 good lepton non iso
    if ( ("muon" == leptonType_ && selection->GetSelectedMuonsNonIso().size()>0) || ("electron" == leptonType_ && selection->GetSelectedElectronsNonIso().size()>0) ) {
      histocontainer_["Selection"]->Fill(3);
      histocontainer_["Selection_Vs_Multiplicity"]->Fill(nJetBinToFill, 3);
      
      jetCheckerLeptonNonIso->analyze(selection->GetJets(), useJES_, iEvent, iSetup);
      metCheckerLeptonNonIso->analyze(vmets);
      muonCheckerLeptonNonIso->analyze(selection->GetSelectedMuonsNonIso());
      electronCheckerLeptonNonIso->analyze(selection->GetSelectedElectronsNonIso(), selection->getBeamSpot());
      kinematicsCheckerLeptonNonIso->analyze(selection->GetJets(), vmets, selection->GetSelectedMuonsNonIso(), selection->GetSelectedElectronsNonIso());
    }
    //>=1 good isolated lepton
    if ( ("muon" == leptonType_ && selection->GetSelectedMuons().size()>0) || ("electron" == leptonType_ && selection->GetSelectedElectrons().size()>0) ) {
      histocontainer_["Selection"]->Fill(4);
      histocontainer_["Selection_Vs_Multiplicity"]->Fill(nJetBinToFill, 4);
    }
    //exactly 1 good isolated lepton
    if ( ("muon" == leptonType_ && selection->GetSelectedMuons().size()==1) || ("electron" == leptonType_ && selection->GetSelectedElectrons().size()==1) ) {
      histocontainer_["Selection"]->Fill(5);
      histocontainer_["Selection_Vs_Multiplicity"]->Fill(nJetBinToFill, 5);
      jetCheckerLeptonIso->analyze(selection->GetJets(), useJES_, iEvent, iSetup);
      metCheckerLeptonIso->analyze(vmets);
      muonCheckerLeptonIso->analyze(selection->GetSelectedMuons());
      electronCheckerLeptonIso->analyze(selection->GetSelectedElectrons(), selection->getBeamSpot());
      kinematicsCheckerLeptonIso->analyze(selection->GetJets(), vmets, selection->GetSelectedMuons(), selection->GetSelectedElectronsNonIso());
      //veto muons for e+jets channel, but not veto electron for muon+jets channel at this step
      if ( ("muon" == leptonType_) || ("electron" == leptonType_ && selection->GetSelectedMuons().size()==0) ) {
	histocontainer_["Selection"]->Fill(6);
	histocontainer_["Selection_Vs_Multiplicity"]->Fill(nJetBinToFill, 6);
	jetCheckerVetoOtherLeptonType->analyze(selection->GetJets(), useJES_, iEvent, iSetup);
	metCheckerVetoOtherLeptonType->analyze(vmets);
	muonCheckerVetoOtherLeptonType->analyze(selection->GetSelectedMuonsNonIso());
	electronCheckerVetoOtherLeptonType->analyze(selection->GetSelectedElectronsNonIso(), selection->getBeamSpot());
	kinematicsCheckerVetoOtherLeptonType->analyze(selection->GetJets(), vmets, selection->GetSelectedMuonsNonIso(), selection->GetSelectedElectronsNonIso());
	
	//Apply MET cut
	if (selection->METpass() || !ApplyMETCut_) {
	  if (!PerformOctoberXDeltaRStep_) {
	    histocontainer_["Selection"]->Fill(7);
	    histocontainer_["Selection_Vs_Multiplicity"]->Fill(nJetBinToFill, 7);
	  }
	  
	  if (ApplyMETCut_) {
	    jetCheckerMET->analyze(selection->GetJets(), useJES_, iEvent, iSetup);
	    metCheckerMET->analyze(vmets);
	    muonCheckerMET->analyze(selection->GetSelectedMuonsNonIso());
	    electronCheckerMET->analyze(selection->GetSelectedElectronsNonIso(), selection->getBeamSpot());
	    kinematicsCheckerMET->analyze(selection->GetJets(), vmets, selection->GetSelectedMuonsNonIso(), selection->GetSelectedElectronsNonIso());
	  }
	  
	  if (PerformOctoberXDeltaRStep_) {
	    selection->SetJetConfig(JetDeltaRLeptonJetThreshold, true);
	    histocontainer_["Selection"]->Fill(7);
	    histocontainer_["Selection_Vs_Multiplicity"]->Fill(nJetBinToFill, 7);
	    
	    jetCheckerDeltaR->analyze(selection->GetJets(), useJES_, iEvent, iSetup);	    
	    metCheckerDeltaR->analyze(vmets);	    
	    muonCheckerDeltaR->analyze(selection->GetSelectedMuonsNonIso());	   
	    electronCheckerDeltaR->analyze(selection->GetSelectedElectronsNonIso(), selection->getBeamSpot());	    
	    kinematicsCheckerDeltaR->analyze(selection->GetJets(), vmets, selection->GetSelectedMuonsNonIso(), selection->GetSelectedElectronsNonIso());	    
	  }  
	  
	  
	  if(selection->GetJets().size()>0){
	    histocontainer_["Selection"]->Fill(8);
	    histocontainer_["Selection_Vs_Multiplicity"]->Fill(nJetBinToFill, 8);
	    jetChecker1Jets->analyze(selection->GetJets(), useJES_, iEvent, iSetup);
	    metChecker1Jets->analyze(vmets);
	    muonChecker1Jets->analyze(selection->GetSelectedMuonsNonIso());
	    electronChecker1Jets->analyze(selection->GetSelectedElectronsNonIso(), selection->getBeamSpot());
	    kinematicsChecker1Jets->analyze(selection->GetJets(), 
					    vmets, selection->GetSelectedMuonsNonIso(), selection->GetSelectedElectronsNonIso());
	  }
	  if(selection->GetJets().size()>1){
	    histocontainer_["Selection"]->Fill(9);
	    histocontainer_["Selection_Vs_Multiplicity"]->Fill(nJetBinToFill, 9);
	    jetChecker2Jets->analyze(selection->GetJets(), useJES_, iEvent, iSetup);
	    metChecker2Jets->analyze(vmets);
	    muonChecker2Jets->analyze(selection->GetSelectedMuonsNonIso());
	    electronChecker2Jets->analyze(selection->GetSelectedElectronsNonIso(), selection->getBeamSpot());
	    kinematicsChecker2Jets->analyze(selection->GetJets(), 
					    vmets, selection->GetSelectedMuonsNonIso(), selection->GetSelectedElectronsNonIso());
	  }
	  if(selection->GetJets().size()>2){
	    histocontainer_["Selection"]->Fill(10);
	    histocontainer_["Selection_Vs_Multiplicity"]->Fill(nJetBinToFill, 10);
	    jetChecker3Jets->analyze(selection->GetJets(), useJES_, iEvent, iSetup);
	    metChecker3Jets->analyze(vmets);
	    muonChecker3Jets->analyze(selection->GetSelectedMuonsNonIso());
	    electronChecker3Jets->analyze(selection->GetSelectedElectronsNonIso(), selection->getBeamSpot());
	    kinematicsChecker3Jets->analyze(selection->GetJets(), 
					    vmets, selection->GetSelectedMuonsNonIso(), selection->GetSelectedElectronsNonIso());
	  }
	  if(selection->GetJets().size()>3){
	    histocontainer_["Selection"]->Fill(11);
	    histocontainer_["Selection_Vs_Multiplicity"]->Fill(nJetBinToFill, 11);
	    jetChecker4Jets->analyze(selection->GetJets(), useJES_, iEvent, iSetup);
	    metChecker4Jets->analyze(vmets);
	    muonChecker4Jets->analyze(selection->GetSelectedMuonsNonIso());
	    electronChecker4Jets->analyze(selection->GetSelectedElectronsNonIso(), selection->getBeamSpot());
	    kinematicsChecker4Jets->analyze(selection->GetJets(),
					    vmets, selection->GetSelectedMuonsNonIso(), selection->GetSelectedElectronsNonIso());
	    if ( VetoLooseLepton_ ) {
	      if ( "muon" == leptonType_ ) {
	      //veto events with second lepton passing looser selection criteria (i.e. only one lepton (the selected one)
	      // should pass the loose selection criteria).
		if ( selection->GetSelectedMuonsLoose().size()==1 ) {
		  histocontainer_["Selection"]->Fill(12);
		  histocontainer_["Selection_Vs_Multiplicity"]->Fill(nJetBinToFill, 12);
		  
		  jetCheckerVetoLooseMuon->analyze(selection->GetJets(), useJES_, iEvent, iSetup);
		  metCheckerVetoLooseMuon->analyze(vmets);
		  muonCheckerVetoLooseMuon->analyze(selection->GetSelectedMuonsNonIso());
		  electronCheckerVetoLooseMuon->analyze(selection->GetSelectedElectronsNonIso(), selection->getBeamSpot());
		  kinematicsCheckerVetoLooseMuon->analyze(selection->GetJets(), vmets, selection->GetSelectedMuonsNonIso(), selection->GetSelectedElectronsNonIso());
		  
		  if ( selection->GetSelectedElectronsLoose().size()==0 ) {
		    histocontainer_["Selection"]->Fill(13);
		    histocontainer_["Selection_Vs_Multiplicity"]->Fill(nJetBinToFill, 13);
		    
		    jetCheckerVetoLooseElectron->analyze(selection->GetJets(), useJES_, iEvent, iSetup);
		    metCheckerVetoLooseElectron->analyze(vmets);
		    muonCheckerVetoLooseElectron->analyze(selection->GetSelectedMuonsNonIso());
		    electronCheckerVetoLooseElectron->analyze(selection->GetSelectedElectronsNonIso(), selection->getBeamSpot());
		    kinematicsCheckerVetoLooseElectron->analyze(selection->GetJets(), vmets, selection->GetSelectedMuonsNonIso(), selection->GetSelectedElectronsNonIso());
		    
		    semiLeptonChecker->analyze(selection->GetJets(), useJES_, vmets,
					       selection->GetSelectedMuons(), selection->GetSelectedElectrons(), iEvent, iSetup);
		    if(semiLeptonChecker->goodMET()) {
		      histocontainer_["Selection"]->Fill(14);
		      histocontainer_["Selection_Vs_Multiplicity"]->Fill(nJetBinToFill, 14);
		    }
		  }
		}
	      }
	      if ( "electron" == leptonType_ ) {
		if ( selection->GetSelectedElectronsLoose().size()==1 ) {
		  histocontainer_["Selection"]->Fill(12);
		  histocontainer_["Selection_Vs_Multiplicity"]->Fill(nJetBinToFill, 12);
		
		  jetCheckerVetoLooseElectron->analyze(selection->GetJets(), useJES_, iEvent, iSetup);
		  metCheckerVetoLooseElectron->analyze(vmets);
		  muonCheckerVetoLooseElectron->analyze(selection->GetSelectedMuonsNonIso());
		  electronCheckerVetoLooseElectron->analyze(selection->GetSelectedElectronsNonIso(), selection->getBeamSpot());
		  kinematicsCheckerVetoLooseElectron->analyze(selection->GetJets(), vmets, selection->GetSelectedMuonsNonIso(), selection->GetSelectedElectronsNonIso());
		  
		  if ( selection->GetSelectedMuonsLoose().size()==0 ) {
		    histocontainer_["Selection"]->Fill(13);
		    histocontainer_["Selection_Vs_Multiplicity"]->Fill(nJetBinToFill, 13);
		  
		    jetCheckerVetoLooseMuon->analyze(selection->GetJets(), useJES_, iEvent, iSetup);
		    metCheckerVetoLooseMuon->analyze(vmets);
		    muonCheckerVetoLooseMuon->analyze(selection->GetSelectedMuonsNonIso());
		    electronCheckerVetoLooseMuon->analyze(selection->GetSelectedElectronsNonIso(), selection->getBeamSpot());
		    kinematicsCheckerVetoLooseMuon->analyze(selection->GetJets(), vmets, selection->GetSelectedMuonsNonIso(), selection->GetSelectedElectronsNonIso());
		    
		    semiLeptonChecker->analyze(selection->GetJets(), useJES_, vmets,
					       selection->GetSelectedMuons(), selection->GetSelectedElectrons(), iEvent, iSetup);
		    if(semiLeptonChecker->goodMET()) {
		      histocontainer_["Selection"]->Fill(14);
		      histocontainer_["Selection_Vs_Multiplicity"]->Fill(nJetBinToFill, 14);
		    }
		  }
		}
	      }
	    }//close if VetoLooseLepton
	    else {
	      semiLeptonChecker->analyze(selection->GetJets(), useJES_, vmets,
					 selection->GetSelectedMuons(), selection->GetSelectedElectrons(), iEvent, iSetup);
	      if(semiLeptonChecker->goodMET()) {
		histocontainer_["Selection"]->Fill(12);
		histocontainer_["Selection_Vs_Multiplicity"]->Fill(nJetBinToFill, 12);
	      }
	    }
	  }//close if njets >3
	  if(selection->GetJets().size()==1) histocontainer_["JetMultiplicity"]->Fill(0);
	  if(selection->GetJets().size()==2) histocontainer_["JetMultiplicity"]->Fill(1);
	  if(selection->GetJets().size()==3) histocontainer_["JetMultiplicity"]->Fill(2);
	  if(selection->GetJets().size()==4) histocontainer_["JetMultiplicity"]->Fill(3);
	  if(selection->GetJets().size()>=5) histocontainer_["JetMultiplicity"]->Fill(4);
	}//close if METpass
      }//close if veto other lepton
    }//close if exactly 1 good isolated lepton
  }
  delete selection;
}

void 
LeptonJetsChecker::beginJob(const edm::EventSetup& es)
{
  edm::LogInfo("Debug|LeptonJetsChecker") << "[LeptonJetsChecker]: beginJob";
  
  dqmStore_->setCurrentFolder( relativePath_ );
 
  //define the histograms booked
  //TH1D
  int nbins = (VetoLooseLepton_) ? 15 : 13;
  histocontainer_["Selection"] = dqmStore_->book1D("Selection" ,"Nof events selected ",nbins,0, nbins);
  histocontainer_["Selection"]->getTH1()->SetOption("TEXT");
  histocontainer_["Selection"]->getTH1()->GetXaxis()->SetBinLabel(1,"All events");
  histocontainer_["Selection"]->getTH1()->GetXaxis()->SetBinLabel(2,TString("HLT: ")+triggerPath);
  histocontainer_["Selection"]->getTH1()->GetXaxis()->SetBinLabel(3,">= 1 "+TString(leptonType_));
  histocontainer_["Selection"]->getTH1()->GetXaxis()->SetBinLabel(4,">= 1 good "+TString(leptonType_));
  histocontainer_["Selection"]->getTH1()->GetXaxis()->SetBinLabel(5,">= 1 isolated "+TString(leptonType_));
  histocontainer_["Selection"]->getTH1()->GetXaxis()->SetBinLabel(6,"= 1 isolated "+TString(leptonType_));
  histocontainer_["Selection"]->getTH1()->GetXaxis()->SetBinLabel(7,"Veto on "+TString(otherLeptonType_));
  histocontainer_["Selection"]->getTH1()->GetXaxis()->SetBinLabel(8,"MET");    
  histocontainer_["Selection"]->getTH1()->GetXaxis()->SetBinLabel(9,">= 1 jet");
  histocontainer_["Selection"]->getTH1()->GetXaxis()->SetBinLabel(10,">= 2 jets");
  histocontainer_["Selection"]->getTH1()->GetXaxis()->SetBinLabel(11,">= 3 jets");
  if (VetoLooseLepton_) {
    histocontainer_["Selection"]->getTH1()->GetXaxis()->SetBinLabel(12,">= 4 jets");
    histocontainer_["Selection"]->getTH1()->GetXaxis()->SetBinLabel(13,"Veto Loose "+TString(leptonType_));
    histocontainer_["Selection"]->getTH1()->GetXaxis()->SetBinLabel(14,"Veto Loose "+TString(otherLeptonType_)+" (M3)");
    histocontainer_["Selection"]->getTH1()->GetXaxis()->SetBinLabel(15,"M3 prime");
  }
  else{
    histocontainer_["Selection"]->getTH1()->GetXaxis()->SetBinLabel(12,">= 4 jets (M3)");
    histocontainer_["Selection"]->getTH1()->GetXaxis()->SetBinLabel(13,"M3 prime");
  }
  
  histocontainer_["JetMultiplicity"] = dqmStore_->book1D("JetMultiplicity","Jet multiplicity",5,0,5);
  histocontainer_["JetMultiplicity"]->getTH1()->GetXaxis()->SetBinLabel(1,"1 jet");
  histocontainer_["JetMultiplicity"]->getTH1()->GetXaxis()->SetBinLabel(2,"2 jet");
  histocontainer_["JetMultiplicity"]->getTH1()->GetXaxis()->SetBinLabel(3,"3 jet");
  histocontainer_["JetMultiplicity"]->getTH1()->GetXaxis()->SetBinLabel(4,"4 jet");
  histocontainer_["JetMultiplicity"]->getTH1()->GetXaxis()->SetBinLabel(5,">=5 jet");

  histocontainer_["Selection_Vs_Multiplicity"] = dqmStore_->book2D("Selection_Vs_Multiplicity", "Selection versus Jet Multiplicity", 6, 0, 6, nbins, 0, nbins);
  histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->SetOption("TEXT");
  histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetXaxis()->SetBinLabel(1,"0 jets");
  histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetXaxis()->SetBinLabel(2,"1 jet");
  histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetXaxis()->SetBinLabel(3,"2 jets");
  histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetXaxis()->SetBinLabel(4,"3 jets");
  histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetXaxis()->SetBinLabel(5,"4 jets");
  histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetXaxis()->SetBinLabel(6,">=5 jets");

  //histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetYaxis()->SetTicks("-");
  histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetYaxis()->SetBinLabel(1,"All events");
  histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetYaxis()->SetBinLabel(2,TString("HLT: ")+triggerPath);
  histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetYaxis()->SetBinLabel(3,">= 1 "+TString(leptonType_));
  histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetYaxis()->SetBinLabel(4,">= 1 good "+TString(leptonType_));
  histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetYaxis()->SetBinLabel(5,">= 1 isolated "+TString(leptonType_));
  histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetYaxis()->SetBinLabel(6,"= 1 isolated "+TString(leptonType_));
  histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetYaxis()->SetBinLabel(7,"Veto on "+TString(otherLeptonType_));
  histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetYaxis()->SetBinLabel(8,"MET");    
  histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetYaxis()->SetBinLabel(9,">= 1 jet");
  histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetYaxis()->SetBinLabel(10,">= 2 jets");
  histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetYaxis()->SetBinLabel(11,">= 3 jets");
  if (VetoLooseLepton_) {
    histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetYaxis()->SetBinLabel(12,">= 4 jets");
    histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetYaxis()->SetBinLabel(13,"Veto Loose "+TString(leptonType_));
    histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetYaxis()->SetBinLabel(14,"Veto Loose "+TString(otherLeptonType_)+" (M3)");
    histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetYaxis()->SetBinLabel(15,"M3 prime");
  }
  else {
    histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetYaxis()->SetBinLabel(12,">= 4 jets (M3)");
    histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetYaxis()->SetBinLabel(13,"M3 prime");
  }
  
  if (PerformOctoberXDeltaRStep_) {
    histocontainer_["Selection"]->getTH1()->GetXaxis()->SetBinLabel(8,"#Delta R");
    histocontainer_["Selection_Vs_Multiplicity"]->getTH2F()->GetYaxis()->SetBinLabel(8,"#Delta R");
  }
  
  edm::LogInfo("Debug|LeptonJetsChecker") << "Call begin for modules ..." << std::endl;
  metChecker->begin(es);
  jetChecker->begin(es, jetCorrector_);
  muonChecker->begin(es);
  electronChecker->begin(es);
  kinematicsChecker->begin(es);
   
  jetCheckerNoSel->begin(es, jetCorrector_);
  metCheckerNoSel->begin(es);
  muonCheckerNoSel->begin(es);
  electronCheckerNoSel->begin(es);
  kinematicsCheckerNoSel->begin(es);
  	
  jetCheckerLeptonNonIso->begin(es, jetCorrector_);
  metCheckerLeptonNonIso->begin(es);
  muonCheckerLeptonNonIso->begin(es);
  electronCheckerLeptonNonIso->begin(es);
  kinematicsCheckerLeptonNonIso->begin(es);
  	
  jetCheckerLeptonIso->begin(es, jetCorrector_);
  metCheckerLeptonIso->begin(es);
  muonCheckerLeptonIso->begin(es);
  electronCheckerLeptonIso->begin(es);
  kinematicsCheckerLeptonIso->begin(es);
  	
  jetCheckerVetoOtherLeptonType->begin(es, jetCorrector_);
  metCheckerVetoOtherLeptonType->begin(es);
  muonCheckerVetoOtherLeptonType->begin(es);
  electronCheckerVetoOtherLeptonType->begin(es);
  kinematicsCheckerVetoOtherLeptonType->begin(es);
  
  if (VetoLooseLepton_) {
    jetCheckerVetoLooseMuon->begin(es, jetCorrector_);
    metCheckerVetoLooseMuon->begin(es);
    muonCheckerVetoLooseMuon->begin(es);
    electronCheckerVetoLooseMuon->begin(es);
    kinematicsCheckerVetoLooseMuon->begin(es);	
    jetCheckerVetoLooseElectron->begin(es, jetCorrector_);
    metCheckerVetoLooseElectron->begin(es);
    muonCheckerVetoLooseElectron->begin(es);
    electronCheckerVetoLooseElectron->begin(es);
    kinematicsCheckerVetoLooseElectron->begin(es);	
  }
 
  if (ApplyMETCut_) {
    jetCheckerMET->begin(es, jetCorrector_);
    metCheckerMET->begin(es);
    muonCheckerMET->begin(es);
    electronCheckerMET->begin(es);
    kinematicsCheckerMET->begin(es);
  }
  
  jetChecker1Jets->begin(es, jetCorrector_);
  metChecker1Jets->begin(es);
  muonChecker1Jets->begin(es);
  electronChecker1Jets->begin(es);
  kinematicsChecker1Jets->begin(es);
  	
  jetChecker2Jets->begin(es, jetCorrector_);
  metChecker2Jets->begin(es);
  muonChecker2Jets->begin(es);
  electronChecker2Jets->begin(es);
  kinematicsChecker2Jets->begin(es);
  	
  jetChecker3Jets->begin(es, jetCorrector_);
  metChecker3Jets->begin(es);
  muonChecker3Jets->begin(es);
  electronChecker3Jets->begin(es);
  kinematicsChecker3Jets->begin(es);
  	
  jetChecker4Jets->begin(es, jetCorrector_);
  metChecker4Jets->begin(es);
  muonChecker4Jets->begin(es);
  electronChecker4Jets->begin(es);
  kinematicsChecker4Jets->begin(es);

  semiLeptonChecker->beginJob(es, jetCorrector_);

  if (PerformOctoberXDeltaRStep_) {
    jetCheckerDeltaR->begin(es, jetCorrector_);
    metCheckerDeltaR->begin(es);
    muonCheckerDeltaR->begin(es);
    electronCheckerDeltaR->begin(es);
    kinematicsCheckerDeltaR->begin(es);
  }
  //Can declare many time the same module and use them for different selectition
}

void 
LeptonJetsChecker::endJob() {
  edm::LogInfo("Debug|LeptonJetsChecker") << "[LeptonJetsChecker]: endJob" << std::endl;
  edm::LogVerbatim ("MainResults") << " -------------------------------------------";
  edm::LogVerbatim ("MainResults") << " -------------------------------------------";
  edm::LogVerbatim ("MainResults") << " --     Report from Lepton+jet Checker    -- ";
  edm::LogVerbatim ("MainResults") << " -- Selecting " << leptonType_ << "s vetoing " << otherLeptonType_ << "s     --"; 
  edm::LogVerbatim ("MainResults") << " -------------------------------------------";
  edm::LogVerbatim ("MainResults") << " -------------------------------------------";

  //Write the main important numbers here ...
  //Selection table by example

  //modules
  edm::LogInfo("Debug|LeptonJetsChecker") << "[LeptonJetsChecker]: Call endJob for modules ...";
  metChecker->end();
  jetChecker->end();
  muonChecker->end();
  electronChecker->end();
  kinematicsChecker->end();

  
  //////////////////////////////
  //  Selection Table
  //////////////////////////////
  edm::LogVerbatim ("MainResults") << histocontainer_["Selection"]->getEntries() << " entries";
  edm::LogVerbatim ("MainResults") << std::left << std::setw(32) << " All events" << ComputeNbEvent( histocontainer_["Selection"] , 0);
  edm::LogVerbatim ("MainResults") << std::left << std::setw(32) << " Trigger" << ComputeNbEvent( histocontainer_["Selection"] , 1);
  edm::LogVerbatim ("MainResults") << " >= 1 " << std::left << std::setw(26) << leptonType_ << ComputeNbEvent( histocontainer_["Selection"] , 2);
  edm::LogVerbatim ("MainResults") << " >= 1 'good' " << std::left << std::setw(19) << leptonType_ << ComputeNbEvent( histocontainer_["Selection"] , 3);
  edm::LogVerbatim ("MainResults") << " >= 1 'good' isolated " << std::left << std::setw(10) << leptonType_  << ComputeNbEvent( histocontainer_["Selection"] , 4);
  edm::LogVerbatim ("MainResults") << "  = 1 'good' isolated " << std::left << std::setw(10) << leptonType_ << ComputeNbEvent( histocontainer_["Selection"] , 5);
  edm::LogVerbatim ("MainResults") << " veto on " << std::left << std::setw(23) << otherLeptonType_ << ComputeNbEvent( histocontainer_["Selection"] , 6);
  edm::LogVerbatim ("MainResults") << std::left << std::setw(32) << " MET" << ComputeNbEvent( histocontainer_["Selection"] , 7);
  edm::LogVerbatim ("MainResults") << std::left << std::setw(32) << " >0 jets" << ComputeNbEvent( histocontainer_["Selection"] , 8);
  edm::LogVerbatim ("MainResults") << std::left << std::setw(32) << " >1 jets" << ComputeNbEvent( histocontainer_["Selection"] , 9);
  edm::LogVerbatim ("MainResults") << std::left << std::setw(32) << " >2 jets" << ComputeNbEvent( histocontainer_["Selection"] , 10);
  if (VetoLooseLepton_) {
    edm::LogVerbatim ("MainResults") << std::left << std::setw(32) << " >3 jets" << ComputeNbEvent( histocontainer_["Selection"] , 11);
    edm::LogVerbatim ("MainResults") << " veto loose " << std::left << std::setw(20) << leptonType_ << ComputeNbEvent( histocontainer_["Selection"] , 12);
    edm::LogVerbatim ("MainResults") << " veto loose " << std::left << std::setw(20) << otherLeptonType_ << ComputeNbEvent( histocontainer_["Selection"] , 13);
    edm::LogVerbatim ("MainResults") << std::left << std::setw(32) << " M3" << ComputeNbEvent( histocontainer_["Selection"] , 13);
    edm::LogVerbatim ("MainResults") << std::left << std::setw(32) << " M3 prime" << ComputeNbEvent( histocontainer_["Selection"] , 14);
  }
  else {
    edm::LogVerbatim ("MainResults") << std::left << std::setw(32) << " >3 jets (M3)" << ComputeNbEvent( histocontainer_["Selection"] , 11);
    edm::LogVerbatim ("MainResults") << std::left << std::setw(32) << " M3 prime" << ComputeNbEvent( histocontainer_["Selection"] , 12);
  }

  if (PerformOctoberXDeltaRStep_) {
    edm::LogVerbatim ("MainResults") << std::endl << std::endl << "October exercise summary table:" << std::endl;
     edm::LogVerbatim ("MainResults") << "Step 1:   " << histocontainer_["Selection"]->getBinContent(2+1) << std::endl;
     edm::LogVerbatim ("MainResults") << "Step 2:   " << histocontainer_["Selection"]->getBinContent(5+1) << std::endl;
     edm::LogVerbatim ("MainResults") << "Step 3:   " << histocontainer_["Selection"]->getBinContent(6+1) << std::endl;
     edm::LogVerbatim ("MainResults") << "Step 4:   " << histocontainer_["Selection"]->getBinContent(7+1) << std::endl;
     edm::LogVerbatim ("MainResults") << "Step 5:   " << histocontainer_["Selection"]->getBinContent(11+1) << std::endl;
  }

  if(saveDQMMEs_)
    dqmStore_->save(outputFileName_);
}

//define this as a plug-in
DEFINE_FWK_MODULE(LeptonJetsChecker);
