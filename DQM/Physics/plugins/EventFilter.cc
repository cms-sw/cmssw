#include "DQM/Physics/plugins/EventFilter.h"

EventFilter::EventFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  labelTriggerResults_ = iConfig.getParameter<edm::InputTag>("labelTriggerResults");
  labelBeamSpot_        =   iConfig.getParameter<edm::InputTag>( "labelBeamSpot" );
  labelMuons_        =   iConfig.getParameter<edm::InputTag>( "labelMuons" );
  labelElectrons_        =   iConfig.getParameter<edm::InputTag>( "labelElectrons" );
  labelJets_        =   iConfig.getParameter<edm::InputTag>( "labelJets" );
  labelMETs_        =   iConfig.getParameter<edm::InputTag>( "labelMETs" );
  verbose_          =   iConfig.getParameter<bool>( "verbose" );

  //Configuration
  //MET
  METCut = iConfig.getParameter<double>( "METCut" );
  //Jets
  NofJets = iConfig.getParameter<int>( "NofJets" );
  PtThrJets = iConfig.getParameter<double>( "PtThrJets" );
  EtaThrJets = iConfig.getParameter<double>( "EtaThrJets" );
  EHThrJets = iConfig.getParameter<double>( "EHThrJets" );
  //Muons
  NofMuons = iConfig.getParameter<int>( "NofMuons" );
  PtThrMuons = iConfig.getParameter<double>( "PtThrMuons" );
  EtaThrMuons = iConfig.getParameter<double>( "EtaThrMuons" );
  MuonRelIso = iConfig.getParameter<double>( "MuonRelIso" );
  MuonVetoEM = iConfig.getParameter<double>( "MuonVetoEM" );
  MuonVetoHad = iConfig.getParameter<double>( "MuonVetoHad" );
  MuonD0Cut = iConfig.getParameter<double>("MuonD0Cut");
  Chi2Cut = iConfig.getParameter<int>("Chi2Cut");
  NofValidHits = iConfig.getParameter<int>("NofValidHits");
  //Electrons
  NofElectrons = iConfig.getParameter<int>( "NofElectrons" );
  PtThrElectrons = iConfig.getParameter<double>( "PtThrElectrons" );
  EtaThrElectrons = iConfig.getParameter<double>( "EtaThrElectrons" );
  ElectronRelIso = iConfig.getParameter<double>( "ElectronRelIso" );
  ElectronD0Cut = iConfig.getParameter<double>("ElectronD0Cut");
  //
  Veto2ndLepton = iConfig.getParameter<bool>( "Veto2ndLepton" );
  //HLT
  triggerPath = iConfig.getParameter<std::string>("triggerPath");
}


EventFilter::~EventFilter()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

bool
EventFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
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
  
  edm::Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByLabel(labelBeamSpot_, beamSpotHandle);
  reco::BeamSpot beamSpot = * beamSpotHandle;
  
  edm::Handle<edm::TriggerResults> trigResults;
  iEvent.getByLabel(labelTriggerResults_,trigResults);
  
  ////////////////////////////////////////
  //Check if branches are available
  ////////////////////////////////////////
  if (!jetsHandle.isValid()) throw cms::Exception("ProductNotFound") <<"Jet collection not found"<<std::endl;
  if (!electronsHandle.isValid()) throw cms::Exception("ProductNotFound") <<"Electron collection not found"<<std::endl;
  if (!muonsHandle.isValid()) throw cms::Exception("ProductNotFound") <<"Muon collection not found"<<std::endl;
  if (!metsHandle.isValid()) throw cms::Exception("ProductNotFound") <<"MET collection not found"<<std::endl;
  if (!beamSpotHandle.isValid()) throw cms::Exception("ProductNotFound") <<"BeamSpot not found"<<std::endl;
  if (!trigResults.isValid()) throw cms::Exception("ProductNotFound") <<"Trigger results not found"<<std::endl;
  
  edm::LogInfo("Debug") <<"Analyze event with LeptonJetChecker"<<std::endl;
  Selection* selection = new Selection();
  // give objects as arguments
  selection->Set(beamSpot, jets, muons, mets);
  // Enter the configuration
  selection->SetConfiguration(PtThrJets, EtaThrJets, EHThrJets, PtThrMuons, EtaThrMuons, MuonRelIso, MuonVetoEM, MuonVetoHad, PtThrElectrons, EtaThrElectrons, ElectronRelIso);
  selection->SetMuonConfig( MuonD0Cut, Chi2Cut, NofValidHits);
  selection->SetElectronConfig( ElectronD0Cut);

  bool selected = false;
  // Check if the event is selected or not !!
  // no leptons required in the event
  if(NofMuons==0 && NofElectrons==0)selected = selection->isSelected(NofJets, std::string("muon"), NofMuons); // no muons or electrons
  //semi-leptonic channel
  if(NofMuons>0 && NofElectrons==0) selected = selection->isSelected(NofJets, std::string("muon"), NofMuons, Veto2ndLepton);
  if(NofMuons==0 && NofElectrons>0) selected = selection->isSelected(NofJets, std::string("electron"), NofElectrons, Veto2ndLepton);
  //di-leptonic channel
  if(NofMuons>0 && NofElectrons>0)  selected = selection->isSelected(NofJets, NofMuons, NofElectrons);
  
  return selected;
}

void 
EventFilter::beginJob(const edm::EventSetup&)
{
}

void 
EventFilter::endJob() 
{
}

//define this as a plug-in
DEFINE_FWK_MODULE(EventFilter);
