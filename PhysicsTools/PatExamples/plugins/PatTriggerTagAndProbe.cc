#include <map>
#include <string>

#include "TH1D.h"
#include "TH2D.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/TriggerEvent.h"


class PatTriggerTagAndProbe : public edm::EDAnalyzer {

 public:
  /// default constructor
  explicit PatTriggerTagAndProbe( const edm::ParameterSet & iConfig );
  /// default destructor
  ~PatTriggerTagAndProbe();

 private:
  /// everythin that needs to be done before the event loop
  virtual void beginJob() ;
  /// everythin that needs to be done during the event loop
  virtual void analyze( const edm::Event & iEvent, const edm::EventSetup & iSetup );
  /// everythin that needs to be done after the event loop
  virtual void endJob();

  /// helper function to set proper bin errors
  void setErrors(TH1D& h, const TH1D& ref);

  /// input for patTriggerEvent
  edm::EDGetTokenT< pat::TriggerEvent > triggerEventToken_;
  /// input for muons
  edm::EDGetTokenT< pat::MuonCollection > muonsToken_;
  /// input for trigger match objects
  std::string   muonMatch_;
  /// management of 1d histograms
  std::map< std::string, TH1D* > histos1D_;
};


#include "TMath.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "PhysicsTools/PatUtils/interface/TriggerHelper.h"


PatTriggerTagAndProbe::PatTriggerTagAndProbe( const edm::ParameterSet & iConfig ) :
  // pat::TriggerEvent
  triggerEventToken_( consumes< pat::TriggerEvent >( iConfig.getParameter< edm::InputTag >( "triggerEvent" ) ) ),
  // muon input collection
  muonsToken_( consumes< pat::MuonCollection >( iConfig.getParameter< edm::InputTag >( "muons" ) ) ),
  // muon match objects
  muonMatch_( iConfig.getParameter< std::string >( "muonMatch" ) ),
  // histogram management
  histos1D_()
{
}

PatTriggerTagAndProbe::~PatTriggerTagAndProbe()
{
}

void PatTriggerTagAndProbe::beginJob()
{
  edm::Service< TFileService > fileService;

  // mass plot around Z peak
  histos1D_[ "mass"    ] = fileService->make< TH1D >( "mass"  , "Mass_{Z} (GeV)",  90,   30., 120.);
  // pt for test candidate
  histos1D_[ "testPt"  ] = fileService->make< TH1D >( "testPt"  , "p_{T} (GeV)" , 100,    0., 100.);
  // pt for probe candidate
  histos1D_[ "probePt" ] = fileService->make< TH1D >( "probePt" , "p_{T} (GeV)" , 100,    0., 100.);
  // eta for test candidate
  histos1D_[ "testEta" ] = fileService->make< TH1D >( "testEta" , "#eta"        ,  48,  -2.4,  2.4);
  // eta for probe candidate
  histos1D_[ "probeEta"] = fileService->make< TH1D >( "probeEta", "#eta"        ,  48,  -2.4,  2.4);
}

void PatTriggerTagAndProbe::analyze( const edm::Event & iEvent, const edm::EventSetup & iSetup )
{
  // trigger event
  edm::Handle< pat::TriggerEvent > triggerEvent;
  iEvent.getByToken( triggerEventToken_, triggerEvent );
  // pat candidate collection
  edm::Handle< pat::MuonCollection > muons;
  iEvent.getByToken( muonsToken_, muons );

  // pat trigger helper to recieve for trigger
  // matching information
  const pat::helper::TriggerMatchHelper matchHelper;

  // ask for trigger accept of HLT_Mu9; otherwise we don't even start
  if(!(triggerEvent->path("HLT_IsoMu17_v5")->wasRun() && triggerEvent->path("HLT_IsoMu17_v5")->wasAccept())){
    return;
  }

  // loop over muon references for the tag muon
  for( size_t idxTag=0; idxTag<muons->size(); ++idxTag){
    const pat::TriggerObjectRef trigRefTag( matchHelper.triggerMatchObject( muons, idxTag, muonMatch_, iEvent, *triggerEvent ) );
    if( trigRefTag.isAvailable() ){
      // loop over muon references for the probe/test muon
      for( size_t idxProbe=0; idxProbe<muons->size() && idxProbe!=idxTag; ++idxProbe){
	histos1D_[ "mass" ]->Fill( (muons->at(idxTag).p4()+muons->at(idxProbe).p4()).mass() );
	if(fabs((muons->at(idxTag).p4()+muons->at(idxProbe).p4()).mass()-90)<5){
	  const pat::TriggerObjectRef trigRefProbe( matchHelper.triggerMatchObject( muons, idxProbe, muonMatch_, iEvent, *triggerEvent ) );
	  histos1D_[ "probePt"  ]->Fill( muons->at(idxProbe).pt () );
	  histos1D_[ "probeEta" ]->Fill( muons->at(idxProbe).eta() );
	  if( trigRefProbe.isAvailable() ){
	    histos1D_[ "testPt" ]->Fill( muons->at(idxProbe).pt () );
	    histos1D_[ "testEta"]->Fill( muons->at(idxProbe).eta() );
	  }
	}
      }
    }
  }
}

void PatTriggerTagAndProbe::endJob()
{
  // normalize the entries of the histograms
  histos1D_[ "testPt"  ]->Divide(histos1D_  [ "probePt"  ]);
  setErrors(*histos1D_["testPt" ],*histos1D_[ "probePt"  ]);
  histos1D_[ "testEta" ]->Divide(histos1D_  [ "probeEta" ]);
  setErrors(*histos1D_["testEta"],*histos1D_[ "probeEta" ]);
}

void PatTriggerTagAndProbe::setErrors(TH1D& h, const TH1D& ref)
{
  for(int bin=0; bin<h.GetNbinsX(); ++bin){
    if(ref.GetBinContent(bin+1)>0){
      h.SetBinError(bin+1, sqrt((h.GetBinContent(bin+1)*(1.-h.GetBinContent(bin+1)))/ref.GetBinContent(bin+1)));
    } else{ h.SetBinError(bin+1, 0.); }
  }
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PatTriggerTagAndProbe );
