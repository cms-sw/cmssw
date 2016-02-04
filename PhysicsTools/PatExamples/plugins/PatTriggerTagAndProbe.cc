#include "TMath.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "PhysicsTools/PatUtils/interface/TriggerHelper.h"
#include "PhysicsTools/PatExamples/plugins/PatTriggerTagAndProbe.h"


PatTriggerTagAndProbe::PatTriggerTagAndProbe( const edm::ParameterSet & iConfig ) :
  // pat::TriggerEvent
  triggerEvent_( iConfig.getParameter< edm::InputTag >( "triggerEvent" ) ),
  // muon input collection
  muons_( iConfig.getParameter< edm::InputTag >( "muons" ) ),
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
  iEvent.getByLabel( triggerEvent_, triggerEvent );
  // pat candidate collection
  edm::Handle< pat::MuonCollection > muons;
  iEvent.getByLabel( muons_, muons );

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
