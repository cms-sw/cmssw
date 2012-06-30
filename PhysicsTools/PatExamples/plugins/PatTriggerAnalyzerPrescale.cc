#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TH1D.h"


class PatTriggerAnalyzerPrescale : public edm::EDAnalyzer {

 public:
  /// default constructor
  explicit PatTriggerAnalyzerPrescale( const edm::ParameterSet & iConfig );
  /// default destructor
  ~PatTriggerAnalyzerPrescale(){};

 private:
  /// everything that needs to be done before the event loop
  virtual void beginJob();
  /// everything that needs to be done during the event loop
  virtual void analyze( const edm::Event & iEvent, const edm::EventSetup & iSetup );
  /// everything that needs to be done after the event loop
  virtual void endJob(){};

  /// histogram
  TH1D * histo_;

  /// event counter
  Int_t bin_;

  /// HLT path name configuration parameter
  std::string pathName_;

};

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/PatCandidates/interface/TriggerEvent.h"


using namespace pat;


PatTriggerAnalyzerPrescale::PatTriggerAnalyzerPrescale( const edm::ParameterSet & iConfig )
: bin_( 0 )
, pathName_( iConfig.getParameter< std::string >( "pathName" ) )
{
}

void PatTriggerAnalyzerPrescale::beginJob()
{
  edm::Service< TFileService > fileService;

  // Histogram definition for 100 events on the x-axis
  histo_ = fileService->make< TH1D >( "histo_", std::string( "Prescale values of " + pathName_ ).c_str(), 100, 0., 100.);
  histo_->SetXTitle( "event" );
  histo_->SetYTitle( "prescale" );
  histo_->SetMinimum( 0. );
}

void PatTriggerAnalyzerPrescale::analyze( const edm::Event & iEvent, const edm::EventSetup & iSetup )
{
  // PAT trigger event
  edm::Handle< TriggerEvent > triggerEvent;
  iEvent.getByLabel( "patTriggerEvent", triggerEvent );

  // Get the HLT path
  const TriggerPath * path( triggerEvent->path( pathName_ ) );

  // Fill prescale factor into histogram
  ++bin_;
  if ( path ) histo_->SetBinContent( bin_, path->prescale() );
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PatTriggerAnalyzerPrescale );
