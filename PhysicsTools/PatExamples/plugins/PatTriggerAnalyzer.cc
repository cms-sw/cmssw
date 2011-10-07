#include "TMath.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "PhysicsTools/PatUtils/interface/TriggerHelper.h"
#include "PhysicsTools/PatExamples/plugins/PatTriggerAnalyzer.h"
#include <iostream> // DEBUG


using namespace pat;


PatTriggerAnalyzer::PatTriggerAnalyzer( const edm::ParameterSet & iConfig ) :
  // pat::Trigger
  trigger_( iConfig.getParameter< edm::InputTag >( "trigger" ) ),
  // pat::TriggerEvent
  triggerEvent_( iConfig.getParameter< edm::InputTag >( "triggerEvent" ) ),
  // muon input collection
  muons_( iConfig.getParameter< edm::InputTag >( "muons" ) ),
  // muon match objects
  muonMatch_( iConfig.getParameter< std::string >( "muonMatch" ) ),
  // minimal id for of all trigger objects
  minID_( iConfig.getParameter< unsigned >( "minID" ) ),
  // maximal id for of all trigger objects
  maxID_( iConfig.getParameter< unsigned >( "maxID" ) ),
  histos1D_(), histos2D_()
{
}

PatTriggerAnalyzer::~PatTriggerAnalyzer()
{
}

void PatTriggerAnalyzer::beginJob()
{
  edm::Service< TFileService > fileService;

  // pt correlation plot
  histos2D_[ "ptTrigCand" ] = fileService->make< TH2D >( "ptTrigCand", "Object vs. candidate p_{T} (GeV)", 60, 0., 300., 60, 0., 300. );
  histos2D_[ "ptTrigCand" ]->SetXTitle( "candidate p_{T} (GeV)" );
  histos2D_[ "ptTrigCand" ]->SetYTitle( "object p_{T} (GeV)" );
  // eta correlation plot
  histos2D_[ "etaTrigCand" ] = fileService->make< TH2D >( "etaTrigCand", "Object vs. candidate #eta", 50, -2.5, 2.5, 50, -2.5, 2.5 );
  histos2D_[ "etaTrigCand" ]->SetXTitle( "candidate #eta" );
  histos2D_[ "etaTrigCand" ]->SetYTitle( "object #eta" );
  // phi correlation plot
  histos2D_[ "phiTrigCand" ] = fileService->make< TH2D >( "phiTrigCand", "Object vs. candidate #phi", 60, -TMath::Pi(), TMath::Pi(), 60, -TMath::Pi(), TMath::Pi() );
  histos2D_[ "phiTrigCand" ]->SetXTitle( "candidate #phi" );
  histos2D_[ "phiTrigCand" ]->SetYTitle( "object #phi" );
  // turn-on curves
  histos1D_[ "turnOn" ] = fileService->make< TH1D >( "turnOn", "p_{T} (GeV) of matched candidate", 10, 0., 50.);
  histos1D_[ "turnOn" ]->SetXTitle( "candidate p_{T} (GeV)" );
  histos1D_[ "turnOn" ]->SetYTitle( "# of objects" );
  // mean pt for all trigger objects
  histos1D_[ "ptMean" ] = fileService->make< TH1D >( "ptMean", "Mean p_{T} (GeV) per filter ID", maxID_ - minID_ + 1, minID_ - 0.5, maxID_ + 0.5);
  histos1D_[ "ptMean" ]->SetXTitle( "filter ID" );
  histos1D_[ "ptMean" ]->SetYTitle( "mean p_{T} (GeV)" );

  // initialize counters for mean pt calculation
  for( unsigned id = minID_; id <= maxID_; ++id ){
    sumN_ [ id ] = 0;
    sumPt_[ id ] = 0.;
  }
}

void PatTriggerAnalyzer::analyze( const edm::Event & iEvent, const edm::EventSetup & iSetup )
{
  // PAT trigger event
  edm::Handle< TriggerEvent > triggerEvent;
  iEvent.getByLabel( triggerEvent_, triggerEvent );

  // PAT object collection
  edm::Handle< MuonCollection > muons;
  iEvent.getByLabel( muons_, muons );

  // PAT trigger helper for trigger matching information
  const helper::TriggerMatchHelper matchHelper;

  /*
    kinematics comparison
  */

  // loop over muon references (PAT muons have been used in the matcher in task 3)
  for( size_t iMuon = 0; iMuon < muons->size(); ++iMuon ) {
    // we need all these ingedients to recieve matched trigger object from the matchHelper
    const TriggerObjectRef trigRef( matchHelper.triggerMatchObject( muons, iMuon, muonMatch_, iEvent, *triggerEvent ) );
    // finally we can fill the histograms
    if ( trigRef.isAvailable() && trigRef.isNonnull() ) { // check references (necessary!)
      histos2D_[ "ptTrigCand" ]->Fill( muons->at( iMuon ).pt(), trigRef->pt() );
      histos2D_[ "etaTrigCand" ]->Fill( muons->at( iMuon ).eta(), trigRef->eta() );
      histos2D_[ "phiTrigCand" ]->Fill( muons->at( iMuon ).phi(), trigRef->phi() );
    }
  }

  /*
    turn-on curve
  */

  // get the trigger objects corresponding to the used matching (HLT muons)
  const TriggerObjectRefVector trigRefs( triggerEvent->objects( trigger::TriggerMuon ) );
  // loop over selected trigger objects
  for ( TriggerObjectRefVector::const_iterator iTrig = trigRefs.begin(); iTrig != trigRefs.end(); ++iTrig ) {
    // get all matched candidates for the trigger object
    const reco::CandidateBaseRefVector candRefs( matchHelper.triggerMatchCandidates( ( *iTrig), muonMatch_, iEvent, *triggerEvent ) );
    if ( candRefs.empty() ) continue;
    // fill the histogram...
    // (only for the first match, since we resolved ambiguities in the matching configuration,
    // so that we have one at maximum per trigger object)
    reco::CandidateBaseRef muonRef( candRefs.at( 0 ) );
    if ( muonRef.isAvailable() && muonRef.isNonnull() ) {
      histos1D_[ "turnOn" ]->Fill( muonRef->pt() );
    }
  }

  /*
    mean pt
  */

  // loop over all trigger match objects from minID to maxID; have
  // a look to DataFormats/HLTReco/interface/TriggerTypeDefs.h to
  // know more about the available trrigger object id's
  for ( unsigned id=minID_; id<=maxID_; ++id ) {
    // vector of all objects for a given object id
    const TriggerObjectRefVector objRefs( triggerEvent->objects( id ) );
    // buffer the number of objects
    sumN_[ id ] += objRefs.size();
    // iterate the objects and buffer the pt of the objects
    for ( TriggerObjectRefVector::const_iterator iRef = objRefs.begin(); iRef != objRefs.end(); ++iRef ) {
      sumPt_[ id ] += ( *iRef )->pt();
    }
  }
}

void PatTriggerAnalyzer::endJob()
{
  // normalize the entries for the mean pt plot
  for(unsigned id=minID_; id<=maxID_; ++id){
    if( sumN_[ id ]!=0 ) histos1D_[ "ptMean" ]->Fill( id, sumPt_[ id ]/sumN_[ id ] );
  }
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PatTriggerAnalyzer );
