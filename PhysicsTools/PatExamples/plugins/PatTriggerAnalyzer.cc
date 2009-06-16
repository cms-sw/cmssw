
#include "PhysicsTools/PatExamples/plugins/PatTriggerAnalyzer.h"
#include "PhysicsTools/PatUtils/interface/TriggerHelper.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

using namespace pat;
using namespace pat::helper;

PatTriggerAnalyzer::PatTriggerAnalyzer( const edm::ParameterSet & iConfig ) :
  hltProcessName_( iConfig.getParameter< std::string >( "hltProcessName" ) ),
  processName_( iConfig.getParameter< std::string >( "processName" ) ),
  trigger_( iConfig.getParameter< edm::InputTag >( "trigger" ) ),
  triggerEvent_( iConfig.getParameter< edm::InputTag >( "triggerEvent" ) ),
  photons_( iConfig.getParameter< edm::InputTag >( "photons" ) ),
  electrons_( iConfig.getParameter< edm::InputTag >( "electrons" ) ),
  muons_( iConfig.getParameter< edm::InputTag >( "muons" ) ),
  taus_( iConfig.getParameter< edm::InputTag >( "taus" ) ),
  jets_( iConfig.getParameter< edm::InputTag >( "jets" ) ),
  mets_( iConfig.getParameter< edm::InputTag >( "mets" ) ),
  histos1D_(),
  histos2D_()
{
}

PatTriggerAnalyzer::~PatTriggerAnalyzer()
{
}

void PatTriggerAnalyzer::beginJob( const edm::EventSetup & iSetup )
{
  edm::Service< TFileService > fileService;
  // histogram definitions
//   YOUR HISTOGRAM DEFINITIONS GO HERE!
//   histos1D_[ "histoName" ] = fileService->make< TH1D >( "[normal TH1D constructor]" ); // EXAMPLE CODE
//   histos1D_[ "histoName" ]->SetXTitle( "x-axis label" );                               // EXAMPLE CODE
//   histos1D_[ "histoName" ]->SetYTitle( "y-axis label" );                               // EXAMPLE CODE

  // This should normally happen in beginRun()
  if ( ! hltConfig_.init( hltProcessName_ ) ) {
    edm::LogError( "hltConfigExtraction" ) << "HLT config extraction error with process name " << hltProcessName_;
  }
}

void PatTriggerAnalyzer::beginRun( edm::Run & iRun, const edm::EventSetup & iSetup )
{
//   if ( ! hltConfig_.init( nameHLTProcess_ ) ) {
//     edm::LogError( "hltConfigExtraction" ) << "HLT config extraction error with process name " << nameHLTProcess_;
//   }
}

void PatTriggerAnalyzer::analyze( const edm::Event & iEvent, const edm::EventSetup & iSetup )
{
  // PAT trigger information
  edm::Handle< TriggerEvent > triggerEvent;
  iEvent.getByLabel( triggerEvent_, triggerEvent );
  edm::Handle< TriggerPathCollection > triggerPaths;
  iEvent.getByLabel( trigger_, triggerPaths );
  edm::Handle< TriggerFilterCollection > triggerFilters;
  iEvent.getByLabel( trigger_, triggerFilters );
  edm::Handle< TriggerObjectCollection > triggerObjects;
  iEvent.getByLabel( trigger_, triggerObjects );

  // PAT object collections
  edm::Handle< PhotonCollection > photons;
  iEvent.getByLabel( photons_, photons );
  edm::Handle< ElectronCollection > electrons;
  iEvent.getByLabel( electrons_, electrons );
  edm::Handle< MuonCollection > muons;
  iEvent.getByLabel( muons_, muons );
  edm::Handle< TauCollection > taus;
  iEvent.getByLabel( taus_, taus );
  edm::Handle< JetCollection > jets;
  iEvent.getByLabel( jets_, jets );
  edm::Handle< METCollection > mets;
  iEvent.getByLabel( mets_, mets );

  // PAT trigger collections from PAT trigger event
  const TriggerPathCollection   * eventPaths( triggerEvent->paths() );
  const TriggerFilterCollection * eventFilters( triggerEvent->filters() );
  const TriggerObjectCollection * eventObjects( triggerEvent->objects() );

  // PAT trigger helper for trigger matching information
  const TriggerMatchHelper matchHelper;

//   YOUR ANALYSIS CODE GOES HERE!
  
}

void PatTriggerAnalyzer::endRun()
{
}

void PatTriggerAnalyzer::endJob()
{
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PatTriggerAnalyzer );
