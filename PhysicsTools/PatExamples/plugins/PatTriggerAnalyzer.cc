
#include "PhysicsTools/PatExamples/plugins/PatTriggerAnalyzer.h"

#include "PhysicsTools/PatUtils/interface/TriggerHelper.h"

#include "TMath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

using namespace pat;
using namespace pat::helper;
using namespace TMath;

PatTriggerAnalyzer::PatTriggerAnalyzer( const edm::ParameterSet & iConfig ) :
  trigger_( iConfig.getParameter< edm::InputTag >( "trigger" ) ),
  triggerEvent_( iConfig.getParameter< edm::InputTag >( "triggerEvent" ) ),
  muons_( iConfig.getParameter< edm::InputTag >( "muons" ) ),
  muonMatch_( iConfig.getParameter< std::string >( "muonMatch" ) ),
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

  // PAT object collection
  edm::Handle< MuonCollection > muons;
  iEvent.getByLabel( muons_, muons );

  // PAT trigger helper for trigger matching information
  const TriggerMatchHelper matchHelper;

//   YOUR ANALYSIS CODE GOES HERE!

}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PatTriggerAnalyzer );
