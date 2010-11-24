#include "TMath.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "PhysicsTools/PatUtils/interface/TriggerHelper.h"
#include "PhysicsTools/PatExamples/plugins/PatTriggerAnalyzer.h"


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

  /*   YOUR HISTOGRAM DEFINITIONS GO HERE!
  histos1D_[ "histoName" ] = fileService->make< TH1D >( "[normal TH1D constructor]" ); // EXAMPLE CODE
  histos1D_[ "histoName" ]->SetXTitle( "x-axis label" );                               // EXAMPLE CODE
  histos1D_[ "histoName" ]->SetYTitle( "y-axis label" );                               // EXAMPLE CODE
  */

  // pt correlation plot
  histos2D_[ "ptTrigCand"  ] = fileService->make< TH2D >( "ptTrigCand", "Object vs. candidate p_{T} (GeV)", 60, 0., 300., 60, 0., 300. );
  histos2D_[ "ptTrigCand"  ]->SetXTitle( "candidate p_{T} (GeV)" );
  histos2D_[ "ptTrigCand"  ]->SetYTitle( "object p_{T} (GeV)" );
  // eta correlation plot
  histos2D_[ "etaTrigCand" ] = fileService->make< TH2D >( "etaTrigCand", "Object vs. candidate #eta", 50, -2.5, 2.5, 50, -2.5, 2.5 );
  histos2D_[ "etaTrigCand" ]->SetXTitle( "candidate #eta" );
  histos2D_[ "etaTrigCand" ]->SetYTitle( "object #eta" );
  // phi correlation plot
  histos2D_[ "phiTrigCand" ] = fileService->make< TH2D >( "phiTrigCand", "Object vs. candidate #phi", 60, -TMath::Pi(), TMath::Pi(), 60, -TMath::Pi(), TMath::Pi() );
  histos2D_[ "phiTrigCand" ]->SetXTitle( "candidate #phi" );
  histos2D_[ "phiTrigCand" ]->SetYTitle( "object #phi" );
  // mean pt for all trigger objects
  histos1D_[ "ptMean"      ] = fileService->make< TH1D >( "ptMean", "Mean p_{T} (GeV) per filter ID", maxID_ - minID_ + 1, minID_ - 0.5, maxID_ + 0.5);
  histos1D_[ "ptMean"      ]->SetXTitle( "filter ID" );
  histos1D_[ "ptMean"      ]->SetYTitle( "mean p_{T} (GeV)" );

  // initialize counters for mean pt calculation
  for( unsigned id = minID_; id <= maxID_; ++id ){
    sumN_ [ id ] = 0;
    sumPt_[ id ] = 0.;
  }
}

void PatTriggerAnalyzer::analyze( const edm::Event & iEvent, const edm::EventSetup & iSetup )
{
  // trigger event
  edm::Handle< pat::TriggerEvent > triggerEvent;
  iEvent.getByLabel( triggerEvent_, triggerEvent );
  // trigger paths from patTrigger
  edm::Handle< pat::TriggerPathCollection > triggerPaths;
  iEvent.getByLabel( trigger_, triggerPaths );
  // trigger filters from patTrigger
  edm::Handle< pat::TriggerFilterCollection > triggerFilters;
  iEvent.getByLabel( trigger_, triggerFilters );
  // trigger objects from patTrigger
  edm::Handle< pat::TriggerObjectCollection > triggerObjects;
  iEvent.getByLabel( trigger_, triggerObjects );

  // PAT object collection
  edm::Handle< pat::MuonCollection > muons;
  iEvent.getByLabel( muons_, muons );

  // PAT trigger helper for trigger matching information
  const pat::helper::TriggerMatchHelper matchHelper;

  /*
    YOUR ANALYSIS CODE GOES HERE!
  */


  /*
    kinematics comparison
  */

  // loop over muon references (PAT muons have been used in the matcher in task 3)
  for( size_t iMuon=0; iMuon<muons->size(); ++iMuon){
    // we need all these ingedients to recieve matched trigger object from the matchHelper
    const pat::TriggerObjectRef trigRef( matchHelper.triggerMatchObject( muons, iMuon, muonMatch_, iEvent, *triggerEvent ) );
    // finally we can fill the histograms
    if ( trigRef.isAvailable() ) { // check references (necessary!)
      histos2D_[ "ptTrigCand" ]->Fill( muons->at( iMuon ).pt(), trigRef->pt() );
      histos2D_[ "etaTrigCand" ]->Fill( muons->at( iMuon ).eta(), trigRef->eta() );
      histos2D_[ "phiTrigCand" ]->Fill( muons->at( iMuon ).phi(), trigRef->phi() );
    }
  }

  /*
    mean pt
  */

  // loop over all trigger match objects from minID to maxID; have
  // a look to DataFormats/HLTReco/interface/TriggerTypeDefs.h to
  // know more about the available trrigger object id's
  for(unsigned id=minID_; id<=maxID_; ++id){
    // vector of all objects for a given object id
    const pat::TriggerObjectRefVector objRefs( triggerEvent->objects( id ) );
    // buffer the number of objects
    sumN_[ id ] += objRefs.size();
    // iterate the objects and buffer the pt of the objects
    for(pat::TriggerObjectRefVector::const_iterator iRef=objRefs.begin(); iRef!=objRefs.end(); ++iRef){
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
