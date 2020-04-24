#include "DQMOffline/Trigger/plugins/ObjMonitor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

// -----------------------------
//  constructors and destructor
// -----------------------------

ObjMonitor::ObjMonitor( const edm::ParameterSet& iConfig ) : 
  folderName_             ( iConfig.getParameter<std::string>("FolderName") )
  , metToken_             ( consumes<reco::PFMETCollection>      (iConfig.getParameter<edm::InputTag>("met")       ) )   
  , jetToken_             ( mayConsume<reco::PFJetCollection>      (iConfig.getParameter<edm::InputTag>("jets")      ) )   
  , eleToken_             ( mayConsume<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electrons") ) )   
  , muoToken_             ( mayConsume<reco::MuonCollection>       (iConfig.getParameter<edm::InputTag>("muons")     ) )
  , do_met_ (iConfig.getParameter<bool>("doMETHistos") )
  , do_jet_ (iConfig.getParameter<bool>("doJetHistos") )
  , do_ht_  (iConfig.getParameter<bool>("doHTHistos")  )
  , num_genTriggerEventFlag_(std::make_unique<GenericTriggerEventFlag>(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this))
  , den_genTriggerEventFlag_(std::make_unique<GenericTriggerEventFlag>(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"),consumesCollector(), *this))
  , metSelection_ ( iConfig.getParameter<std::string>("metSelection") )
  , jetSelection_ ( iConfig.getParameter<std::string>("jetSelection") )
  , jetId_        ( iConfig.getParameter<std::string>("jetId")        )
  , htjetSelection_ ( iConfig.getParameter<std::string>("htjetSelection"))
  , eleSelection_ ( iConfig.getParameter<std::string>("eleSelection") )
  , muoSelection_ ( iConfig.getParameter<std::string>("muoSelection") )
  , njets_      ( iConfig.getParameter<int>("njets" )      )
  , nelectrons_ ( iConfig.getParameter<int>("nelectrons" ) )
  , nmuons_     ( iConfig.getParameter<int>("nmuons" )     )
{
  if (do_met_){
    metDQM_.initialise(iConfig);
  }
  if (do_jet_){
    jetDQM_.initialise(iConfig);
  }
  if (do_ht_ ){
    htDQM_.initialise(iConfig);
  }
}

ObjMonitor::~ObjMonitor() = default;

void ObjMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
				 edm::Run const        & iRun,
				 edm::EventSetup const & iSetup) 
{  

  std::string currentFolder = folderName_ ;
  ibooker.setCurrentFolder(currentFolder.c_str());

  if (do_met_) metDQM_.bookHistograms(ibooker);
  if (do_jet_) jetDQM_.bookHistograms(ibooker);
  if (do_ht_ ) htDQM_.bookHistograms(ibooker);

  // Initialize the GenericTriggerEventFlag
  if ( num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() ) num_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on() ) den_genTriggerEventFlag_->initRun( iRun, iSetup );

}

void ObjMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && ! den_genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

  edm::Handle<reco::PFMETCollection> metHandle;
  iEvent.getByToken( metToken_, metHandle );
  reco::PFMET pfmet = metHandle->front();
  if ( ! metSelection_( pfmet ) ) return;
  
  float met = pfmet.pt();
  float phi = pfmet.phi();

  edm::Handle<reco::PFJetCollection> jetHandle;
  iEvent.getByToken( jetToken_, jetHandle );
  std::vector<reco::PFJet> jets;
  std::vector<reco::PFJet> htjets;
  if ( int(jetHandle->size()) < njets_ ) return;
  for ( auto const & j : *jetHandle ) {
    if ( jetSelection_( j ) ) {
      if (jetId_=="loose" || jetId_ =="tight"){
	double abseta = abs(j.eta());
	double NHF  = j.neutralHadronEnergyFraction();
	double NEMF = j.neutralEmEnergyFraction();
	double CHF  = j.chargedHadronEnergyFraction();
	double CEMF = j.chargedEmEnergyFraction();
	unsigned NumNeutralParticles =j.neutralMultiplicity();
	unsigned CHM      = j.chargedMultiplicity();
	bool passId = (jetId_=="loose" && looseJetId(abseta,NHF,NEMF,CHF,CEMF,NumNeutralParticles,CHM)) || (jetId_=="tight" && tightJetId(abseta,NHF,NEMF,CHF,CEMF,NumNeutralParticles,CHM));
	if (passId) jets.push_back(j);
      }
      else jets.push_back(j);
    }
    if ( htjetSelection_( j ) ) htjets.push_back(j);
  }
  if ( int(jets.size()) < njets_ ) return;
  
  edm::Handle<reco::GsfElectronCollection> eleHandle;
  iEvent.getByToken( eleToken_, eleHandle );
  std::vector<reco::GsfElectron> electrons;
  if ( int(eleHandle->size()) < nelectrons_ ) return;
  for ( auto const & e : *eleHandle ) {
    if ( eleSelection_( e ) ) electrons.push_back(e);
  }
  if ( int(electrons.size()) < nelectrons_ ) return;
  
  edm::Handle<reco::MuonCollection> muoHandle;
  iEvent.getByToken( muoToken_, muoHandle );
  if ( int(muoHandle->size()) < nmuons_ ) return;
  std::vector<reco::Muon> muons;
  for ( auto const & m : *muoHandle ) {
    if ( muoSelection_( m ) ) muons.push_back(m);
  }
  if ( int(muons.size()) < nmuons_ ) return;


  bool passNumCond = num_genTriggerEventFlag_->off() || num_genTriggerEventFlag_->accept( iEvent, iSetup);
  int ls = iEvent.id().luminosityBlock();

  if (do_met_) metDQM_.fillHistograms(met,phi,ls,passNumCond);
  if (do_jet_) jetDQM_.fillHistograms(jets,pfmet,ls,passNumCond);
  if (do_ht_ ) htDQM_.fillHistograms(htjets,met,ls,passNumCond);

}

bool ObjMonitor::looseJetId(const double & abseta,
			    const double & NHF,
			    const double & NEMF,
			    const double & CHF,
			    const double & CEMF,
			    const unsigned & NumNeutralParticles,
			    const unsigned & CHM)
{
  if (abseta<=2.7){
    unsigned NumConst = CHM+NumNeutralParticles;

    return ((NumConst>1 && NHF<0.99 && NEMF<0.99) && ((abseta<=2.4 && CHF>0 && CHM>0 && CEMF<0.99) || abseta>2.4));
  }
  else if (abseta<=3){
    return (NumNeutralParticles>2 && NEMF>0.01 && NHF<0.98);
  }
  else {
    return NumNeutralParticles>10 && NEMF<0.90;
  }
}
bool ObjMonitor::tightJetId(const double & abseta,
			    const double & NHF,
			    const double & NEMF,
			    const double & CHF,
			    const double & CEMF,
			    const unsigned & NumNeutralParticles,
			    const unsigned & CHM)
{
  if (abseta<=2.7){
    unsigned NumConst = CHM+NumNeutralParticles;
    return (NumConst>1 && NHF<0.90 && NEMF<0.90 ) && ((abseta<=2.4 && CHF>0 && CHM>0 && CEMF<0.99) || abseta>2.4);
  }
  else if (abseta<=3){
    return (NHF<0.98 && NEMF>0.01 && NumNeutralParticles>2);
  }
  else {
    return (NEMF<0.90 && NumNeutralParticles>10);
  }
}

void ObjMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>  ( "FolderName", "HLT/OBJ" );

  desc.add<edm::InputTag>( "met",      edm::InputTag("pfMet") );
  desc.add<edm::InputTag>( "jets",     edm::InputTag("ak4PFJetsCHS") );
  desc.add<edm::InputTag>( "electrons",edm::InputTag("gedGsfElectrons") );
  desc.add<edm::InputTag>( "muons",    edm::InputTag("muons") );
  desc.add<std::string>("metSelection", "pt > 0");
  desc.add<std::string>("jetSelection", "pt > 0");
  desc.add<std::string>("jetId", "");
  desc.add<std::string>("htjetSelection", "pt > 30");
  desc.add<std::string>("eleSelection", "pt > 0");
  desc.add<std::string>("muoSelection", "pt > 0");
  desc.add<int>("njets",      0);
  desc.add<int>("nelectrons", 0);
  desc.add<int>("nmuons",     0);

  edm::ParameterSetDescription genericTriggerEventPSet;
  genericTriggerEventPSet.add<bool>("andOr");
  genericTriggerEventPSet.add<edm::InputTag>("dcsInputTag", edm::InputTag("scalersRawToDigi") );
  genericTriggerEventPSet.add<std::vector<int> >("dcsPartitions",{});
  genericTriggerEventPSet.add<bool>("andOrDcs", false);
  genericTriggerEventPSet.add<bool>("errorReplyDcs", true);
  genericTriggerEventPSet.add<std::string>("dbLabel","");
  genericTriggerEventPSet.add<bool>("andOrHlt", true);
  genericTriggerEventPSet.add<edm::InputTag>("hltInputTag", edm::InputTag("TriggerResults::HLT") );
  genericTriggerEventPSet.add<std::vector<std::string> >("hltPaths",{});
  genericTriggerEventPSet.add<std::string>("hltDBKey","");
  genericTriggerEventPSet.add<bool>("errorReplyHlt",false);
  genericTriggerEventPSet.add<unsigned int>("verbosityLevel",1);

  desc.add<edm::ParameterSetDescription>("numGenericTriggerEventPSet", genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("denGenericTriggerEventPSet", genericTriggerEventPSet);

  desc.add<bool>("doMETHistos", true);
  edm::ParameterSetDescription histoPSet;
  METDQM::fillMetDescription(histoPSet);
  desc.add<bool>("doJetHistos", true);
  JetDQM::fillJetDescription(histoPSet);
  desc.add<bool>("doHTHistos", true);
  HTDQM::fillHtDescription(histoPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet",histoPSet);

  descriptions.add("objMonitoring", desc);
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ObjMonitor);
