// -*- C++ -*-
//
// Package:    DeDxHitInfoProducer
// Class:      DeDxHitInfoProducer
// 
/**\class DeDxHitInfoProducer DeDxHitInfoProducer.cc RecoTracker/DeDx/plugins/DeDxHitInfoProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  loic Quertenmont (querten)
//         Created:  Mon Nov 21 14:09:02 CEST 2014
//

#include "RecoTracker/DeDx/plugins/DeDxHitInfoProducer.h"

// system include files


using namespace reco;
using namespace std;
using namespace edm;

DeDxHitInfoProducer::DeDxHitInfoProducer(const edm::ParameterSet& iConfig):
   usePixel          ( iConfig.getParameter<bool>    ("usePixel")       ),
   useStrip          ( iConfig.getParameter<bool>    ("useStrip")       ),
   MeVperADCPixel    ( iConfig.getParameter<double>  ("MeVperADCPixel") ),
   MeVperADCStrip    ( iConfig.getParameter<double>  ("MeVperADCStrip") ),
   minTrackHits      ( iConfig.getParameter<unsigned>("minTrackHits")   ),
   minTrackPt        ( iConfig.getParameter<double>  ("minTrackPt"  )   ),
   minTrackPtPrescale( iConfig.getParameter<double>  ("minTrackPtPrescale") ),
   maxTrackEta       ( iConfig.getParameter<double>  ("maxTrackEta" )   ),
   m_calibrationPath ( iConfig.getParameter<string>  ("calibrationPath")),
   useCalibration    ( iConfig.getParameter<bool>    ("useCalibration") ),
   shapetest         ( iConfig.getParameter<bool>    ("shapeTest")      ),
   lowPtTracksPrescalePass( iConfig.getParameter<uint32_t>("lowPtTracksPrescalePass") ),
   lowPtTracksPrescaleFail( iConfig.getParameter<uint32_t>("lowPtTracksPrescaleFail") ),
   lowPtTracksEstimator(iConfig.getParameter<edm::ParameterSet>("lowPtTracksEstimatorParameters")),
   lowPtTracksDeDxThreshold(iConfig.getParameter<double>("lowPtTracksDeDxThreshold"))
{
   produces<reco::DeDxHitInfoCollection >();
   produces<reco::DeDxHitInfoAss >();
   produces<edm::ValueMap<int> >("prescale");

   m_tracksTag = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"));

   if(!usePixel && !useStrip)
   edm::LogError("DeDxHitsProducer") << "No Pixel Hits NOR Strip Hits will be saved.  Running this module is useless";
}


DeDxHitInfoProducer::~DeDxHitInfoProducer(){}

// ------------ method called once each job just before starting event loop  ------------
void  DeDxHitInfoProducer::beginRun(edm::Run const& run, const edm::EventSetup& iSetup)
{
   iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );
   if(useCalibration && calibGains.empty()){
      m_off = tkGeom->offsetDU(GeomDetEnumerators::PixelBarrel); //index start at the first pixel

      DeDxTools::makeCalibrationMap(m_calibrationPath, *tkGeom, calibGains, m_off);
   }
}



void DeDxHitInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{  
  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByToken(m_tracksTag,trackCollectionHandle);
  const TrackCollection& trackCollection(*trackCollectionHandle.product());

  // creates the output collection
  auto resultdedxHitColl = std::make_unique<reco::DeDxHitInfoCollection>();

  std::vector<int> indices; std::vector<int> prescales;
  uint64_t state[2] = { iEvent.id().event(), iEvent.id().luminosityBlock() };
  for(unsigned int j=0;j<trackCollection.size();j++){            
     const reco::Track& track = trackCollection[j];

     //track selection
     bool passPt = (track.pt() >= minTrackPt), passLowDeDx = false, passHighDeDx = false, pass = passPt;
     if (!pass && (track.pt() >= minTrackPtPrescale)) {
        if (lowPtTracksPrescalePass > 0) {
            passHighDeDx = ((xorshift128p(state) % lowPtTracksPrescalePass) == 0);
        } 
        if (lowPtTracksPrescaleFail > 0) {
            passLowDeDx = ((xorshift128p(state) % lowPtTracksPrescaleFail) == 0);
        }
        pass = passHighDeDx || passLowDeDx;
     }
     if(!pass || std::abs(track.eta())>maxTrackEta ||track.numberOfValidHits()<minTrackHits){
        indices.push_back(-1);
        continue;
     }

     reco::DeDxHitInfo hitDeDxInfo;
     auto const & trajParams = track.extra()->trajParams();
     auto hb = track.recHitsBegin();
        for(unsigned int h=0;h<track.recHitsSize();h++){
           auto recHit = *(hb+h);
           if(!recHit->isValid()) continue;

           auto trackDirection = trajParams[h].direction();
           float cosine = trackDirection.z()/trackDirection.mag();
 
           processHit(recHit, track.p(), cosine, hitDeDxInfo, trajParams[h].position());
     }

     if (!passPt) {
        std::vector<DeDxHit> hits; hits.reserve(hitDeDxInfo.size());
        for (unsigned int i = 0, n = hitDeDxInfo.size(); i < n; ++i) {
           if (hitDeDxInfo.detId(i).subdetId() <= 2) {
               hits.push_back( DeDxHit(hitDeDxInfo.charge(i)/hitDeDxInfo.pathlength(i) * MeVperADCPixel, 0,0,0) );
           } else {
               if (shapetest && !DeDxTools::shapeSelection(*hitDeDxInfo.stripCluster(i))) continue;
               hits.push_back( DeDxHit(hitDeDxInfo.charge(i)/hitDeDxInfo.pathlength(i) * MeVperADCStrip, 0,0,0) );
           }
        }
        std::sort(hits.begin(), hits.end(), std::less<DeDxHit>());
        if (lowPtTracksEstimator.dedx(hits).first < lowPtTracksDeDxThreshold) {
            if (passLowDeDx) {
                prescales.push_back(lowPtTracksPrescaleFail);
            } else {
                indices.push_back(-1);
                continue;
            }
        } else {
            if (passHighDeDx) {
                prescales.push_back(lowPtTracksPrescalePass);
            } else {
                indices.push_back(-1);
                continue;
            }

        }
     } else {
        prescales.push_back(1);
     }
     indices.push_back(resultdedxHitColl->size());
     resultdedxHitColl->push_back(hitDeDxInfo);
  }
  ///////////////////////////////////////
 

  edm::OrphanHandle<reco::DeDxHitInfoCollection> dedxHitCollHandle = iEvent.put(std::move(resultdedxHitColl));

  //create map passing the handle to the matched collection
  auto dedxMatch = std::make_unique<reco::DeDxHitInfoAss>(dedxHitCollHandle);
  reco::DeDxHitInfoAss::Filler filler(*dedxMatch);  
  filler.insert(trackCollectionHandle, indices.begin(), indices.end()); 
  filler.fill();
  iEvent.put(std::move(dedxMatch));

  auto dedxPrescale = std::make_unique<edm::ValueMap<int>>();
  edm::ValueMap<int>::Filler pfiller(*dedxPrescale);
  pfiller.insert(dedxHitCollHandle, prescales.begin(), prescales.end());
  pfiller.fill();
  iEvent.put(std::move(dedxPrescale), "prescale");
}

void DeDxHitInfoProducer::processHit(const TrackingRecHit* recHit, const float trackMomentum, const float cosine, reco::DeDxHitInfo& hitDeDxInfo,  const LocalPoint& hitLocalPos){
      auto const & thit = static_cast<BaseTrackerRecHit const&>(*recHit);
      if(!thit.isValid())return;

      float cosineAbs = std::max(0.00000001f,std::abs(cosine));//make sure cosine is not 0

      auto const & clus = thit.firstClusterRef();
      if(!clus.isValid())return;


      if(clus.isPixel()){
          if(!usePixel) return;

          const auto * detUnit = recHit->detUnit();
          if (detUnit == nullptr) detUnit = tkGeom->idToDet(thit.geographicalId());
          float pathLen     = detUnit->surface().bounds().thickness()/cosineAbs;
          float chargeAbs   = clus.pixelCluster().charge();
          hitDeDxInfo.addHit(chargeAbs, pathLen, thit.geographicalId(), hitLocalPos, clus.pixelCluster() );
       }else if(clus.isStrip() && !thit.isMatched()){
          if(!useStrip) return;

          const auto * detUnit = recHit->detUnit();
          if (detUnit == nullptr) detUnit = tkGeom->idToDet(thit.geographicalId());
          int   NSaturating = 0;
          float pathLen     = detUnit->surface().bounds().thickness()/cosineAbs;
          float chargeAbs   = DeDxTools::getCharge(&(clus.stripCluster()),NSaturating, *detUnit, calibGains, m_off);
          hitDeDxInfo.addHit(chargeAbs, pathLen, thit.geographicalId(), hitLocalPos, clus.stripCluster() );
       }else if(clus.isStrip() && thit.isMatched()){
          if(!useStrip) return;
          const SiStripMatchedRecHit2D* matchedHit=dynamic_cast<const SiStripMatchedRecHit2D*>(recHit);
          if(!matchedHit)return;

          const auto * detUnitM = matchedHit->monoHit().detUnit();
          if (detUnitM == nullptr) detUnitM = tkGeom->idToDet(matchedHit->monoHit().geographicalId());
          int   NSaturating = 0;
          float pathLen     = detUnitM->surface().bounds().thickness()/cosineAbs;
          float chargeAbs   = DeDxTools::getCharge(&(matchedHit->monoHit().stripCluster()),NSaturating, *detUnitM, calibGains, m_off);
          hitDeDxInfo.addHit(chargeAbs, pathLen, thit.geographicalId(), hitLocalPos, matchedHit->monoHit().stripCluster() );

          const auto * detUnitS = matchedHit->stereoHit().detUnit();
          if (detUnitS == nullptr) detUnitS = tkGeom->idToDet(matchedHit->stereoHit().geographicalId());
          NSaturating = 0;
          pathLen     = detUnitS->surface().bounds().thickness()/cosineAbs;
          chargeAbs   = DeDxTools::getCharge(&(matchedHit->stereoHit().stripCluster()),NSaturating, *detUnitS, calibGains, m_off);
          hitDeDxInfo.addHit(chargeAbs, pathLen, thit.geographicalId(), hitLocalPos, matchedHit->stereoHit().stripCluster() );          
       }
}



//define this as a plug-in
DEFINE_FWK_MODULE(DeDxHitInfoProducer);
