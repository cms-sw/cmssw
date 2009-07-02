
#include <string>
#include <map>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/PatCandidates/interface/TriggerEvent.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "PhysicsTools/PatUtils/interface/TriggerHelper.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "TH1D.h"
#include "TH2D.h"
#include "TMath.h"


namespace pat {

  class TriggerAnalyzer : public edm::EDAnalyzer {
  
    public:
    
      // constructor(s) and destructor
      explicit TriggerAnalyzer( const edm::ParameterSet & iConfig );
      ~TriggerAnalyzer();
    
    private:
  
      // methods
      virtual void beginJob( const edm::EventSetup & iSetup ) ;
      virtual void analyze( const edm::Event & iEvent, const edm::EventSetup & iSetup );
      
      // configuration parameters: input sources
      edm::InputTag triggerEvent_;
      edm::InputTag muons_;
      std::string   muonMatch_;
      unsigned      muonID_;

      // histograms
      std::map< std::string, TH1D* > histos1D_;
      std::map< std::string, TH2D* > histos2D_;
  };

}


using namespace pat;
using namespace pat::helper;
using namespace TMath;


TriggerAnalyzer::TriggerAnalyzer( const edm::ParameterSet & iConfig ) :
  triggerEvent_( iConfig.getParameter< edm::InputTag >( "triggerEvent" ) ),
  muons_( iConfig.getParameter< edm::InputTag >( "muons" ) ),
  muonMatch_( iConfig.getParameter< std::string >( "muonMatch" ) ),
  muonID_( iConfig.getParameter< unsigned >( "muonID" ) ),
  histos1D_(),
  histos2D_()
{
}


TriggerAnalyzer::~TriggerAnalyzer()
{
}


void TriggerAnalyzer::beginJob( const edm::EventSetup & iSetup )
{
  // histogram definitions
  edm::Service< TFileService > fileService;
  histos2D_[ "ptMatch" ] = fileService->make< TH2D >( "ptMatch", "trigger object vs. candidate p_{T} (GeV)", 20, 0., 60., 20, 0., 60. );
  histos2D_[ "ptMatch" ]->SetXTitle( "candidate p_{T} (GeV)" );
  histos2D_[ "ptMatch" ]->SetYTitle( "object p_{T} (GeV)" );
  histos1D_[ "ptCand" ] = fileService->make< TH1D >( "ptCand", "candidate p_{T} (GeV)", 20, 0., 20. );
  histos1D_[ "ptCand" ]->SetXTitle( "p_{T} (GeV)" );
  histos1D_[ "ptCand" ]->SetYTitle( "candidates" );
  histos1D_[ "ptTrig" ] = fileService->make< TH1D >( "ptTrig", "trigger object p_{T} (GeV)", 20, 0., 20. );
  histos1D_[ "ptTrig" ]->SetXTitle( "p_{T} (GeV)" );
  histos1D_[ "ptTrig" ]->SetYTitle( "trigger objects" );
  histos1D_[ "ptTrig" ]->SetLineColor( kRed );
  histos1D_[ "ptTrigAll" ] = fileService->make< TH1D >( "ptTrigAll", "all trigger object p_{T} (GeV)", 20, 0., 20. );
  histos1D_[ "ptTrigAll" ]->SetXTitle( "p_{T} (GeV)" );
  histos1D_[ "ptTrigAll" ]->SetYTitle( "all trigger objects" );
  histos1D_[ "ptTrigAll" ]->SetLineColor( kBlue );
}


void TriggerAnalyzer::analyze( const edm::Event & iEvent, const edm::EventSetup & iSetup )
{
  // load PAT trigger event
  edm::Handle< TriggerEvent > triggerEvent;
  iEvent.getByLabel( triggerEvent_, triggerEvent );

  // load PAT object collection
  edm::Handle< MuonCollection > muons;
  iEvent.getByLabel( muons_, muons );

  // activate PAT trigger helper for trigger matching information
  const TriggerMatchHelper matchHelper;

  // filling histograms
  const TriggerObjectMatch * triggerMatch( triggerEvent->triggerObjectMatchResult( muonMatch_ ) );                           // <== missing piece
  for ( size_t iMuon = 0; iMuon < muons->size(); ++iMuon ) {
    const reco::CandidateBaseRef candBaseRef( MuonRef( muons, iMuon ) );
    const TriggerObjectRef trigRef( matchHelper.triggerMatchObject( candBaseRef, triggerMatch, iEvent, *triggerEvent ) );                                  // <== missing piece
    // fill histograms
    if ( trigRef.isAvailable() ) { // check references (necessary!)
      histos2D_[ "ptMatch" ]->Fill( candBaseRef->pt(), trigRef->pt() );                                  // <== missing piece
      histos1D_[ "ptCand" ]->Fill( candBaseRef->pt() );                                   // <== missing piece
      histos1D_[ "ptTrig" ]->Fill( trigRef->pt() );                                   // <== missing piece
    }
  } // iMuon
  const TriggerObjectRefVector trigRefVector( triggerEvent->objects( muonID_ ) );                        // <== missing piece
  for ( TriggerObjectRefVector::const_iterator iTrig = trigRefVector.begin(); iTrig != trigRefVector.end(); ++iTrig ) {
    histos1D_[ "ptTrigAll" ]->Fill( ( *iTrig )->pt() );                                  // <== missing piece
  }
  
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( TriggerAnalyzer );
