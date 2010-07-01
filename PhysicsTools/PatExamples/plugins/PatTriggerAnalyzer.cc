#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TH1D.h"
#include "TH2D.h"

#include <string>
#include <map>


namespace pat {

  class PatTriggerAnalyzer : public edm::EDAnalyzer {

    public:

      explicit PatTriggerAnalyzer( const edm::ParameterSet & iConfig );
      ~PatTriggerAnalyzer() {};

    private:

      virtual void beginJob() ;
      virtual void analyze( const edm::Event & iEvent, const edm::EventSetup & iSetup );
      virtual void endJob();

      edm::InputTag trigger_;
      edm::InputTag triggerEvent_;
      edm::InputTag muons_;
      std::string   muonMatch_;

      unsigned minID_;
      unsigned maxID_;

      std::map< std::string, TH1D* > histos1D_;
      std::map< std::string, TH2D* > histos2D_;

      std::map< unsigned, unsigned > sumN_;
      std::map< unsigned, double >   sumPt_;
  };

}


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "PhysicsTools/PatUtils/interface/TriggerHelper.h"
#include "TMath.h"

#include "DataFormats/PatCandidates/interface/TriggerEvent.h"
#include "DataFormats/PatCandidates/interface/Muon.h"


using namespace pat;
using namespace pat::helper;
using namespace TMath;


PatTriggerAnalyzer::PatTriggerAnalyzer( const edm::ParameterSet & iConfig )
  : trigger_( iConfig.getParameter< edm::InputTag >( "trigger" ) )
  , triggerEvent_( iConfig.getParameter< edm::InputTag >( "triggerEvent" ) )
  , muons_( iConfig.getParameter< edm::InputTag >( "muons" ) )
  , muonMatch_( iConfig.getParameter< std::string >( "muonMatch" ) )
  , minID_( iConfig.getParameter< unsigned >( "minID" ) )
  , maxID_( iConfig.getParameter< unsigned >( "maxID" ) )
  , histos1D_()
  , histos2D_()
{
}


void PatTriggerAnalyzer::beginJob()
{

  edm::Service< TFileService > fileService;
  // histogram definitions
//   YOUR HISTOGRAM DEFINITIONS GO HERE!
//   histos1D_[ "histoName" ] = fileService->make< TH1D >( "[normal TH1D constructor]" ); // EXAMPLE CODE
//   histos1D_[ "histoName" ]->SetXTitle( "x-axis label" );                               // EXAMPLE CODE
//   histos1D_[ "histoName" ]->SetYTitle( "y-axis label" );                               // EXAMPLE CODE
  histos2D_[ "ptTrigCand" ] = fileService->make< TH2D >( "ptTrigCand", "Object vs. candidate p_{T} (GeV)", 60, 0., 300., 60, 0., 300. );
  histos2D_[ "ptTrigCand" ]->SetXTitle( "candidate p_{T} (GeV)" );
  histos2D_[ "ptTrigCand" ]->SetYTitle( "object p_{T} (GeV)" );
  histos2D_[ "etaTrigCand" ] = fileService->make< TH2D >( "etaTrigCand", "Object vs. candidate #eta", 50, -2.5, 2.5, 50, -2.5, 2.5 );
  histos2D_[ "etaTrigCand" ]->SetXTitle( "candidate #eta" );
  histos2D_[ "etaTrigCand" ]->SetYTitle( "object #eta" );
  histos2D_[ "phiTrigCand" ] = fileService->make< TH2D >( "phiTrigCand", "Object vs. candidate #phi", 60, -Pi(), Pi(), 60, -Pi(), Pi() );
  histos2D_[ "phiTrigCand" ]->SetXTitle( "candidate #phi" );
  histos2D_[ "phiTrigCand" ]->SetYTitle( "object #phi" );
  histos1D_[ "ptMean" ] = fileService->make< TH1D >( "ptMean", "Mean p_{T} (GeV) per filter ID", maxID_ - minID_ + 1, minID_ - 0.5, maxID_ + 0.5);
  histos1D_[ "ptMean" ]->SetXTitle( "filter ID" );
  histos1D_[ "ptMean" ]->SetYTitle( "mean p_{T} (GeV)" );

  for ( unsigned id = minID_; id <= maxID_; ++id ) {
    sumN_[ id ] = 0;
    sumPt_[ id ] = 0.;
  }

}


void PatTriggerAnalyzer::analyze( const edm::Event & iEvent, const edm::EventSetup & iSetup )
{

  // PAT trigger information
  edm::Handle< TriggerEvent > triggerEvent;
  iEvent.getByLabel( triggerEvent_, triggerEvent );

  // PAT object collection
  edm::Handle< MuonCollection > muons;
  iEvent.getByLabel( muons_, muons );

  // PAT trigger helper for trigger matching information
  const TriggerMatchHelper matchHelper;

//   YOUR ANALYSIS CODE GOES HERE!

  // kinematics comparison
  for ( size_t iMuon = 0; iMuon < muons->size(); ++iMuon ) { // loop over muon references (PAT muons have been used in the matcher in task 3)
    const TriggerObjectRef trigRef(matchHelper.triggerMatchObject( muons, iMuon, muonMatch_, iEvent, *triggerEvent ) );
    // fill histograms
    if ( trigRef.isAvailable() ) { // check references (necessary!)
      histos2D_[ "ptTrigCand" ]->Fill( muons->at( iMuon ).pt(), trigRef->pt() );
      histos2D_[ "etaTrigCand" ]->Fill( muons->at( iMuon ).eta(), trigRef->eta() );
      histos2D_[ "phiTrigCand" ]->Fill( muons->at( iMuon ).phi(), trigRef->phi() );
    }
  } // iMuon

  // mean pt
  for ( unsigned id = minID_; id <= maxID_; ++id ) {
    const TriggerObjectRefVector objRefs( triggerEvent->objects( id ) );
    sumN_[ id ] += objRefs.size();
    for ( TriggerObjectRefVector::const_iterator iRef = objRefs.begin(); iRef != objRefs.end(); ++iRef ) {
      sumPt_[ id ] += ( *iRef )->pt();
    }
  }

}

void PatTriggerAnalyzer::endJob()
{

  for ( unsigned id = minID_; id <= maxID_; ++id ) {
    if ( sumN_[ id ] != 0 ) histos1D_[ "ptMean" ]->Fill( id, sumPt_[ id ]/sumN_[ id ] );
  }

}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PatTriggerAnalyzer );
