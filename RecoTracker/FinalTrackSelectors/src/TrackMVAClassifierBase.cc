#include "RecoTracker/FinalTrackSelectors/interface/TrackMVAClassifier.h"

#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h" 

#include "DataFormats/TrackReco/interface/Track.h"


#include<cassert>


void TrackMVAClassifierBase::fill( edm::ParameterSetDescription& desc) {
  desc.add<edm::InputTag>("src",edm::InputTag());
  desc.add<edm::InputTag>("beamspot",edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("vertices",edm::InputTag("firstStepPrimaryVertices"));
  desc.add<std::string>("GBRForestLabel",std::string());
  desc.add<std::string>("GBRForestFileName",std::string());
  // default cuts for "cut based classification"
  std::vector<double> cuts = {-.7, 0.1, .7};
  desc.add<std::vector<double>>("qualityCuts", cuts);
}


TrackMVAClassifierBase::~TrackMVAClassifierBase(){}

TrackMVAClassifierBase::TrackMVAClassifierBase( const edm::ParameterSet & cfg ) :
  src_( consumes<reco::TrackCollection>( cfg.getParameter<edm::InputTag>( "src" ) ) ),
  beamspot_( consumes<reco::BeamSpot>( cfg.getParameter<edm::InputTag>( "beamspot" ) ) ),
  vertices_(consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>( "vertices" ))),
  forestLabel_(cfg.getParameter<std::string>("GBRForestLabel")),
  dbFileName_(cfg.getParameter<std::string>("GBRForestFileName")),
  useForestFromDB_( (!forestLabel_.empty()) & dbFileName_.empty()) {

  auto const & qv  = cfg.getParameter<std::vector<double>>("qualityCuts");
  assert(qv.size()==3);
  std::copy(std::begin(qv),std::end(qv),std::begin(qualityCuts));
  
  produces<MVACollection>("MVAValues");
  produces<QualityMaskCollection>("QualityMasks");

}

void TrackMVAClassifierBase::produce(edm::Event& evt, const edm::EventSetup& es ) {

  // Get tracks 
  edm::Handle<reco::TrackCollection> hSrcTrack;
  evt.getByToken(src_, hSrcTrack );
  auto const & tracks(*hSrcTrack);

    // looking for the beam spot
  edm::Handle<reco::BeamSpot> hBsp;
  evt.getByToken(beamspot_, hBsp);

	
  // Select good primary vertices for use in subsequent track selection
  edm::Handle<reco::VertexCollection> hVtx;
  evt.getByToken(vertices_, hVtx);

  GBRForest const * forest = forest_.get();
  if(useForestFromDB_){
    edm::ESHandle<GBRForest> forestHandle;
    es.get<GBRWrapperRcd>().get(forestLabel_,forestHandle);
    forest = forestHandle.product();
  }

  // products
  auto mvas = std::make_unique<MVACollection>(tracks.size(),-99.f);
  auto quals = std::make_unique<QualityMaskCollection>(tracks.size(),0);

  
  
  computeMVA(tracks,*hBsp,*hVtx,forest,*mvas);
  assert((*mvas).size()==tracks.size());

  unsigned int k=0;
  for (auto mva : *mvas) {
    (*quals)[k++]
      =  (mva>qualityCuts[0]) << reco::TrackBase::loose
      |  (mva>qualityCuts[1]) << reco::TrackBase::tight
      |  (mva>qualityCuts[2]) << reco::TrackBase::highPurity
     ;

  }
  

  evt.put(std::move(mvas),"MVAValues");
  evt.put(std::move(quals),"QualityMasks");
  
}


#include <TFile.h>
void TrackMVAClassifierBase::beginStream(edm::StreamID) {
  if(!dbFileName_.empty()){
     TFile gbrfile(dbFileName_.c_str());
     forest_.reset((GBRForest*)gbrfile.Get(forestLabel_.c_str()));
  }
}

