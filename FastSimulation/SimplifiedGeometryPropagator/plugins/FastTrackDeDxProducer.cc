// -*- C++ -*-
//
// Package:    FastTrackDeDxProducer
// Class:      FastTrackDeDxProducer
// 
/**\class FastTrackDeDxProducer FastTrackDeDxProducer.cc RecoTracker/FastTrackDeDxProducer/src/FastTrackDeDxProducer.cc

   Description: <one line class summary>

   Implementation:
   <Notes on implementation>
*/
//
// Original Author:  andrea
//         Created:  Thu May 31 14:09:02 CEST 2007
//    Code Updates:  loic Quertenmont (querten)
//         Created:  Thu May 10 14:09:02 CEST 2008
//
//


// system include files
//#include "RecoTracker/DeDx/plugins/DeDxEstimatorProducer.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/FastTrackDeDxProducer.h"

using namespace reco;
using namespace std;
using namespace edm;


//void yuval(const std::string& a);

void FastTrackDeDxProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<string>("estimator","generic");
  desc.add<edm::InputTag>("tracks",edm::InputTag("generalTracks"));
  desc.add<bool>("UsePixel",false); 
  desc.add<bool>("UseStrip",true); 
  desc.add<double>("MeVperADCPixel",3.61e-06*265);
  desc.add<double>("MeVperADCStrip",3.61e-06);
  desc.add<bool>("ShapeTest",true);      
  desc.add<bool>("UseCalibration",false);  
  desc.add<string>("calibrationPath", "");
  desc.add<string>("Reccord", "SiStripDeDxMip_3D_Rcd");
  desc.add<string>("ProbabilityMode", "Accumulation");
  desc.add<double>("fraction", 0.4);
  desc.add<double>("exponent",-2.0);
  desc.add<bool>("convertFromGeV2MeV",true);
  desc.add<bool>("nothick",false);  
  desc.add<edm::InputTag>("simHits");
  desc.add<edm::InputTag>("simHit2RecHitMap");
  descriptions.add("FastTrackDeDxProducer",desc);
}


FastTrackDeDxProducer::FastTrackDeDxProducer(const edm::ParameterSet& iConfig)
: simHitsToken(consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("simHits")))
, simHit2RecHitMapToken(consumes<FastTrackerRecHitRefCollection>(iConfig.getParameter<edm::InputTag>("simHit2RecHitMap")))
{
  produces<ValueMap<DeDxData> >();

  string estimatorName = iConfig.getParameter<string>("estimator");
  if     (estimatorName == "median")              m_estimator = new MedianDeDxEstimator(iConfig);
  else if(estimatorName == "generic")             m_estimator = new GenericAverageDeDxEstimator  (iConfig);
  else if(estimatorName == "truncated")           m_estimator = new TruncatedAverageDeDxEstimator(iConfig);
  //else if(estimatorName == "unbinnedFit")         m_estimator = new UnbinnedFitDeDxEstimator(iConfig);
  else if(estimatorName == "productDiscrim")      m_estimator = new ProductDeDxDiscriminator(iConfig);
  else if(estimatorName == "btagDiscrim")         m_estimator = new BTagLikeDeDxDiscriminator(iConfig);
  else if(estimatorName == "smirnovDiscrim")      m_estimator = new SmirnovDeDxDiscriminator(iConfig);
  else if(estimatorName == "asmirnovDiscrim")     m_estimator = new ASmirnovDeDxDiscriminator(iConfig);
  else throw cms::Exception("fastsim::SimplifiedGeometry::FastTrackDeDxProducer.cc") << " estimator name does not exist";

  //Commented for now, might be used in the future
  //   MaxNrStrips         = iConfig.getUntrackedParameter<unsigned>("maxNrStrips"        ,  255);

  m_tracksTag = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"));

  //simHitsToken = consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("simHits"));
  //simHit2RecHitMapToken = consumes<FastTrackerRecHitRefCollection>(iConfig.getParameter<edm::InputTag>("simHit2RecHitMap"));

  usePixel = iConfig.getParameter<bool>("UsePixel"); 
  useStrip = iConfig.getParameter<bool>("UseStrip");
  meVperADCPixel = iConfig.getParameter<double>("MeVperADCPixel"); 
  meVperADCStrip = iConfig.getParameter<double>("MeVperADCStrip"); 

  shapetest = iConfig.getParameter<bool>("ShapeTest");
  useCalibration = iConfig.getParameter<bool>("UseCalibration");
  m_calibrationPath = iConfig.getParameter<string>("calibrationPath");

  convertFromGeV2MeV = iConfig.getParameter<bool>("convertFromGeV2MeV");
  nothick = iConfig.getParameter<bool>("nothick");  

  if(!usePixel && !useStrip)
  	throw cms::Exception("fastsim::SimplifiedGeometry::FastTrackDeDxProducer.cc") << " Pixel Hits AND Strip Hits will not be used to estimate dEdx --> BUG, Please Update the config file";


}


FastTrackDeDxProducer::~FastTrackDeDxProducer()
{
  delete m_estimator;
}

// ------------ method called once each job just before starting event loop  ------------
void  FastTrackDeDxProducer::beginRun(edm::Run const& run, const edm::EventSetup& iSetup)
{
  if(useCalibration && calibGains.empty()){
    edm::ESHandle<TrackerGeometry> tkGeom;
    iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );
    m_off = tkGeom->offsetDU(GeomDetEnumerators::PixelBarrel); //index start at the first pixel
	
    DeDxTools::makeCalibrationMap(m_calibrationPath, *tkGeom, calibGains, m_off);
  }

  m_estimator->beginRun(run, iSetup);
}



void FastTrackDeDxProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  auto trackDeDxEstimateAssociation = std::make_unique<ValueMap<DeDxData>>();  
  ValueMap<DeDxData>::Filler filler(*trackDeDxEstimateAssociation);

  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByToken(m_tracksTag,trackCollectionHandle);
  //std::vector<DeDxData> dedxEstimate( trackCollectionHandle->size() );
  const auto& trackCollection = *trackCollectionHandle;
  std::vector<DeDxData> dedxEstimate( trackCollection.size() );



  for(unsigned int j=0;j<trackCollectionHandle->size();j++){            
    const reco::TrackRef track = reco::TrackRef( trackCollectionHandle.product(), j );
    
    int NClusterSaturating = 0; 
    DeDxHitCollection dedxHits;
      
    auto const & trajParams = track->extra()->trajParams();
    assert(trajParams.size()==track->recHitsSize());
      
    auto hb = track->recHitsBegin();
    dedxHits.reserve(track->recHitsSize()/2);
    for(unsigned int h=0;h<track->recHitsSize();h++){
	const FastTrackerRecHit recHit = static_cast< const FastTrackerRecHit & >(*(*(hb+h)));
	if(!recHit.isValid()) continue;//FastTrackerRecHit recHit = *(hb+h);
	
	auto trackDirection = trajParams[h].direction();         
	float cosine = trackDirection.z()/trackDirection.mag();
	processHit(recHit, track->p(), cosine, dedxHits, NClusterSaturating);
    }
  
    sort(dedxHits.begin(),dedxHits.end(),less<DeDxHit>());   
    std::pair<float,float> val_and_error = m_estimator->dedx(dedxHits);
    //WARNING: Since the dEdX Error is not properly computed for the moment
    //It was decided to store the number of saturating cluster in that dataformat
    val_and_error.second = NClusterSaturating; 
    dedxEstimate[j] = DeDxData(val_and_error.first, val_and_error.second, dedxHits.size() );
    }


  filler.insert(trackCollectionHandle, dedxEstimate.begin(), dedxEstimate.end());
  // fill the association map and put it into the event
  filler.fill();
  iEvent.put(std::move(trackDeDxEstimateAssociation));
}


void FastTrackDeDxProducer::processHit(const FastTrackerRecHit &recHit, float trackMomentum, float& cosine, reco::DeDxHitCollection& dedxHits, int& NClusterSaturating){


  auto const & thit = static_cast<BaseTrackerRecHit const&>(recHit);
  //const TrackerSingleRecHit thit = static_cast<TrackerSingleRecHit const&>(recHit);//playing now


  if(!thit.isValid())return;

  if(recHit.isPixel()){
    //std::cout << "we got pixels" << std::endl;
    if(!usePixel) return;

    auto& detUnit     = *(recHit.detUnit());
    float pathLen     = detUnit.surface().bounds().thickness()/fabs(cosine);
    if (nothick) pathLen = 1.0;
    float charge = recHit.energyLoss()/pathLen;
    if (convertFromGeV2MeV) charge*=1000;
    dedxHits.push_back( DeDxHit( charge, trackMomentum, pathLen, thit.geographicalId()) );
  }
  else if(!recHit.isPixel()){// && !thit.isMatched()){//check what thit.isMatched is doing
    if(!useStrip) return;
    auto& detUnit     = *(recHit.detUnit());
    float pathLen     = detUnit.surface().bounds().thickness()/fabs(cosine);
    if (nothick) pathLen = 1.0;
    float dedxOfRecHit = recHit.energyLoss()/pathLen;
    if (convertFromGeV2MeV) dedxOfRecHit*=1000;
    if(!shapetest ){
      dedxHits.push_back( DeDxHit( dedxOfRecHit, trackMomentum, pathLen, thit.geographicalId()) );
    }
  }
}



//define this as a plug-in
DEFINE_FWK_MODULE(FastTrackDeDxProducer);
