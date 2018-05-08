/** \class GEMCosmicMuon  
 * Produces a collection of tracks's in GEM cosmic ray stand. 
 *
 * \author Jason Lee
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/GEMGeometry/interface/GEMGeometry.h>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonSmoother.h"
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonSmoother.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "RecoTracker/TrackProducer/src/TrajectoryToResiduals.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"

using namespace std;

class GEMCosmicMuon : public edm::stream::EDProducer<> {
public:
  /// Constructor
  explicit GEMCosmicMuon(const edm::ParameterSet&);
  /// Destructor
  virtual ~GEMCosmicMuon() {}
  /// Produce the GEMSegment collection
  void produce(edm::Event&, const edm::EventSetup&) override;
  bool doInnerSeeding_;
  double trackChi2_, trackResX_, trackResY_;

  multimap<float,const GEMChamber*> detLayerMap_;
  
private:
  int iev; // events through
  edm::EDGetTokenT<GEMRecHitCollection> theGEMRecHitToken_;
  CosmicMuonSmoother* theSmoother_;
  MuonServiceProxy* theService_;
  KFUpdator* theUpdator_;

  MuonTransientTrackingRecHit::MuonRecHitContainer getHitsFromLayer(vector<const GEMChamber*> &chambers,
								    const GEMRecHitCollection* gemHits);
  
  unique_ptr<vector<TrajectorySeed> > findSeeds(MuonTransientTrackingRecHit::MuonRecHitContainer topSeeds,
						MuonTransientTrackingRecHit::MuonRecHitContainer bottomSeeds);
  Trajectory makeTrajectory(TrajectorySeed& seed, const GEMRecHitCollection* gemHits);
  TrackingRecHit::ConstRecHitContainer findMissingHits(Trajectory& track);
  
};

GEMCosmicMuon::GEMCosmicMuon(const edm::ParameterSet& ps) : iev(0) {
  doInnerSeeding_ = ps.getParameter<bool>("doInnerSeeding");
  trackChi2_ = ps.getParameter<double>("trackChi2");
  trackResX_ = ps.getParameter<double>("trackResX");
  trackResY_ = ps.getParameter<double>("trackResY");
  theGEMRecHitToken_ = consumes<GEMRecHitCollection>(ps.getParameter<edm::InputTag>("gemRecHitLabel"));
  // register what this produces
  edm::ParameterSet serviceParameters = ps.getParameter<edm::ParameterSet>("ServiceParameters");
  theService_ = new MuonServiceProxy(serviceParameters);
  edm::ParameterSet smootherPSet = ps.getParameter<edm::ParameterSet>("MuonSmootherParameters");
  theSmoother_ = new CosmicMuonSmoother(smootherPSet,theService_);
  theUpdator_ = new KFUpdator();
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  produces<vector<Trajectory> >();
  produces<vector<TrajectorySeed> >();
}

void GEMCosmicMuon::produce(edm::Event& ev, const edm::EventSetup& setup) {
  //  cout << "GEMCosmicMuon::start producing segments for " << ++iev << "th event with gem data" << endl;  
  unique_ptr<reco::TrackCollection >          trackCollection( new reco::TrackCollection() );
  unique_ptr<TrackingRecHitCollection >       trackingRecHitCollection( new TrackingRecHitCollection() );
  unique_ptr<reco::TrackExtraCollection >     trackExtraCollection( new reco::TrackExtraCollection() );
  unique_ptr<vector<Trajectory> >             trajectorys( new vector<Trajectory>() );
  unique_ptr<vector<TrajectorySeed> >         trajectorySeeds( new vector<TrajectorySeed>() );
  TrackingRecHitRef::key_type recHitsIndex = 0;
  TrackingRecHitRefProd recHitCollectionRefProd = ev.getRefBeforePut<TrackingRecHitCollection>();
  reco::TrackExtraRef::key_type trackExtraIndex = 0;
  reco::TrackExtraRefProd trackExtraCollectionRefProd = ev.getRefBeforePut<reco::TrackExtraCollection>();
  
  theService_->update(setup);

  edm::ESHandle<GEMGeometry> gemg;
  setup.get<MuonGeometryRecord>().get(gemg);
  const GEMGeometry* mgeom = &*gemg;
  
  // get the collection of GEMRecHit
  edm::Handle<GEMRecHitCollection> gemRecHits;
  ev.getByToken(theGEMRecHitToken_,gemRecHits);

  if (gemRecHits->size() <3){
    ev.put(move(trajectorySeeds));
    ev.put(move(trackCollection));
    ev.put(move(trackingRecHitCollection));
    ev.put(move(trackExtraCollection));
    ev.put(move(trajectorys));
    return;
  }

  detLayerMap_.clear();
  for (auto ch : mgeom->chambers()){    
    //cout << ch->id() << " y = " <<ch->position().y() <<endl;
    // save key as neg to sort from top to bottom
    detLayerMap_.insert( make_pair(-ch->position().y(), ch) );
  }

  // get top and bottom chambers
  vector<const GEMChamber*> topChamb(3), botChamb(3);
  for (auto map: detLayerMap_){
    //cout <<" y = " <<map.first << " det " << map.second->id()<<endl;
    int col = map.second->id().chamber()/10;
    if (!topChamb[col]){
      topChamb[col] = map.second;
    }
    botChamb[col] = map.second;
  }

  if (doInnerSeeding_){
    for (int i = 0; i<3; ++i){
      GEMDetId id = topChamb[i]->id();
      auto sc = mgeom->superChamber(id);
      int layer = 3-id.layer(); //2->1 and 1->2      
      topChamb[i] = sc->chamber(layer);      
      //cout <<"topChamb["<<i<<"]" << topChamb[i]->id()<<endl;;
    }
    for (int i = 0; i<3; ++i){
      GEMDetId id = botChamb[i]->id();
      auto sc = mgeom->superChamber(id);
      int layer = 3-id.layer(); //2->1 and 1->2      
      botChamb[i] = sc->chamber(layer);
      //cout <<"botChamb["<<i<<"]" << botChamb[i]->id()<< endl;
    }
  }
  
  auto topSeeds    = getHitsFromLayer(topChamb, gemRecHits.product());
  auto bottomSeeds = getHitsFromLayer(botChamb, gemRecHits.product());
    
  auto trajectorySeedCands = findSeeds(topSeeds, bottomSeeds);
  //cout << "GEMCosmicMuon::topSeeds->size() " << topSeeds.size() << endl;
  //  cout << "GEMCosmicMuon::trajectorySeeds->size() " << trajectorySeeds->size() << endl;

  // need to loop over seeds, make best track and save only best track
  //TrajectorySeed seed =trajectorySeeds->at(0);
  Trajectory bestTrajectory;
  TrajectorySeed bestSeed;
  float maxChi2 = trackChi2_;
  for (auto seed : *trajectorySeedCands){
    Trajectory smoothed = makeTrajectory(seed, gemRecHits.product());
    if (smoothed.isValid()){
      cout << "GEMCosmicMuon::Trajectory " << smoothed.foundHits() << endl;
      cout << "GEMCosmicMuon::Trajectory chiSquared/ndof " << smoothed.chiSquared()/float(smoothed.ndof()) << endl;
      //if (( maxChi2 > smoothed.chiSquared()/float(smoothed.ndof())) and ( smoothed.chiSquared()/float(smoothed.ndof()) > 7.0)){
      if (maxChi2 > smoothed.chiSquared()/float(smoothed.ndof())){
	maxChi2 = smoothed.chiSquared()/float(smoothed.ndof());
	bestTrajectory = smoothed;
        bestSeed = seed;
      }
    }
  }
  if (!bestTrajectory.isValid()){
    ev.put(move(trajectorySeeds));
    ev.put(move(trackCollection));
    ev.put(move(trackingRecHitCollection));
    ev.put(move(trackExtraCollection));
    ev.put(move(trajectorys));
    return;
  }
  //cout << maxChi2 << endl;
  cout << "GEMCosmicMuon::bestTrajectory " << bestTrajectory.foundHits() << endl;
  cout << "GEMCosmicMuon::bestTrajectory chiSquared/ ndof " << bestTrajectory.chiSquared()/float(bestTrajectory.ndof()) << endl;
  //cout << maxChi2 << endl;
  // make track
  const FreeTrajectoryState* ftsAtVtx = bestTrajectory.geometricalInnermostState().freeState();
  
  GlobalPoint pca = ftsAtVtx->position();
  math::XYZPoint persistentPCA(pca.x(),pca.y(),pca.z());
  GlobalVector p = ftsAtVtx->momentum();
  math::XYZVector persistentMomentum(p.x(),p.y(),p.z());
  
  reco::Track track(bestTrajectory.chiSquared(), 
		    bestTrajectory.ndof(true),
		    persistentPCA,
		    persistentMomentum,
		    ftsAtVtx->charge(),
		    ftsAtVtx->curvilinearError());
 
  reco::TrackExtra tx;
  //adding rec hits
  TrackingRecHit::ConstRecHitContainer transHits = findMissingHits(bestTrajectory);
  unsigned int nHitsAdded = 0;
  for (Trajectory::RecHitContainer::const_iterator recHit = transHits.begin(); recHit != transHits.end(); ++recHit) {
    TrackingRecHit *singleHit = (**recHit).hit()->clone();
    trackingRecHitCollection->push_back( singleHit );  
    ++nHitsAdded;
  }
  tx.setHits(recHitCollectionRefProd, recHitsIndex, nHitsAdded);
  tx.setResiduals(trajectoryToResiduals(bestTrajectory));
  tx.setSeedRef(bestTrajectory.seedRef());
  recHitsIndex +=nHitsAdded;

  trackExtraCollection->emplace_back(tx );
  reco::TrackExtraRef trackExtraRef(trackExtraCollectionRefProd, trackExtraIndex++ );
  track.setExtra(trackExtraRef);
  trackCollection->emplace_back(track);
  trajectorySeeds->emplace_back(bestSeed);
  trajectorys->emplace_back(bestTrajectory);      
  
  // fill the collection
  // put collection in event

  ev.put(move(trajectorySeeds));
  ev.put(move(trackCollection));
  ev.put(move(trackingRecHitCollection));
  ev.put(move(trackExtraCollection));
  ev.put(move(trajectorys));
  
}

unique_ptr<vector<TrajectorySeed> > GEMCosmicMuon::findSeeds(MuonTransientTrackingRecHit::MuonRecHitContainer topSeeds,
							     MuonTransientTrackingRecHit::MuonRecHitContainer bottomSeeds)
{
  unique_ptr<vector<TrajectorySeed> > trajectorySeeds( new vector<TrajectorySeed>());
  if (topSeeds.size() > 0 && bottomSeeds.size() > 0){
    
    for (auto tophit : topSeeds){
      //cout << "GEMCosmicMuon::tophit       " << tophit->globalPosition() << endl;
      for (auto bottomhit : bottomSeeds){
	//cout << "GEMCosmicMuon::bottomhit    " << bottomhit->globalPosition() << endl;
	
	GlobalVector segDirGV = tophit->globalPosition() - bottomhit->globalPosition();

	int charge= 1;
	AlgebraicSymMatrix mat(5,0);
	mat = tophit->parametersError().similarityT( tophit->projectionMatrix() );
	// mat[0][0] = 0.5;
	// mat[1][1] = 10;
	// mat[2][2] = 0.5;
	LocalTrajectoryError error(asSMatrix<5>(mat));
	// get first hit	
	LocalPoint segPos = tophit->localPosition();
	LocalVector segDir = tophit->det()->toLocal(segDirGV);
	LocalTrajectoryParameters param(segPos, segDir, charge);
	TrajectoryStateOnSurface tsos(param, error, tophit->det()->surface(), &*theService_->magneticField());

	//auto tsosBot = theService_->propagator("StraightLinePropagator")->propagate(tsos,bottomhit->det()->surface());
	//cout << "GEMCosmicMuon::tsos        " << tsos << endl;
	//tsos = theUpdator_->update(tsosBot, *bottomhit);
	cout << "GEMCosmicMuon::tsos update " << tsos << endl;
	  
	PTrajectoryStateOnDet seedTSOS = trajectoryStateTransform::persistentState(tsos, tophit->rawId());
	
	edm::OwnVector<TrackingRecHit> seedHits;
	seedHits.push_back(tophit->cloneHit());
	seedHits.push_back(bottomhit->cloneHit());

	TrajectorySeed seed(seedTSOS,seedHits,alongMomentum);
	trajectorySeeds->emplace_back(seed);
      }
    }
  }
  return trajectorySeeds;
}

Trajectory GEMCosmicMuon::makeTrajectory(TrajectorySeed& seed,
					 const GEMRecHitCollection* gemHits)
{
  PTrajectoryStateOnDet ptsd1(seed.startingState());
  DetId did(ptsd1.detId());
  const BoundPlane& bp = theService_->trackingGeometry()->idToDet(did)->surface();
  TrajectoryStateOnSurface tsos = trajectoryStateTransform::transientState(ptsd1,&bp,&*theService_->magneticField());

  TrackingRecHit::ConstRecHitContainer consRecHits;
  
  float previousLayer = -200;//skip first layer
  for (auto chmap : detLayerMap_){
    //// skip same layers
    if (chmap.first == previousLayer) continue;
    previousLayer = chmap.first;
    
    auto refChamber = chmap.second;
    shared_ptr<MuonTransientTrackingRecHit> tmpRecHit;

    tsos = theService_->propagator("StraightLinePropagator")->propagate(tsos,refChamber->surface());
    if (!tsos.isValid()){
      continue;
    }
    
    GlobalPoint tsosGP = tsos.globalPosition();
    cout << "tsos gp   "<< tsosGP << refChamber->id() <<endl;
    
    float maxR = 500;
    // find best in all layers
    for (auto col : detLayerMap_){
      // only look in same layer
      if (chmap.first != col.first) continue;      
      auto ch = col.second;
      for (auto etaPart : ch->etaPartitions()){
	GEMDetId etaPartID = etaPart->id();
	
	GEMRecHitCollection::range range = gemHits->get(etaPartID);
	//cout<< "Number of GEM rechits available , from chamber: "<< etaPartID<<endl;
	for (GEMRecHitCollection::const_iterator rechit = range.first; rechit!=range.second; ++rechit){

	  LocalPoint tsosLP = etaPart->toLocal(tsosGP);
	  LocalPoint rhLP = (*rechit).localPosition();
	  //double y_err = (*rechit).localPositionError().yy();	
	  //if (abs(rhLP.x() - tsosLP.x()) > trackResX_) continue;	
	  //if (abs(rhLP.y() - tsosLP.y()) > y_err*trackResY_) continue;
	  // need to find best hits per chamber
	  float deltaR = (rhLP - tsosLP).mag();
	  if (maxR > deltaR){
	  
	    cout << " found hit   "<< etaPartID << " pos = "<< rhLP << " R = "<<deltaR <<endl;
	    const GeomDet* geomDet(etaPart);	  
	    tmpRecHit = MuonTransientTrackingRecHit::specificBuild(geomDet,&*rechit);
	    maxR = deltaR;
	  }
	}
      }
    }
    
    if (tmpRecHit){      
      consRecHits.emplace_back(tmpRecHit);
    }
  }
  if (consRecHits.size() <3) return Trajectory();

  auto firstHit = consRecHits.front();
  tsos = theService_->propagator("StraightLinePropagator")->propagate(tsos,*(firstHit->surface()));

  vector<Trajectory> fitted = theSmoother_->trajectories(seed, consRecHits, tsos);
  if(fitted.size() == 0) return Trajectory();
  else return fitted.front();
}

TrackingRecHit::ConstRecHitContainer GEMCosmicMuon::findMissingHits(Trajectory& track)
{
  TrajectoryStateOnSurface tsos = track.geometricalInnermostState();
  TrackingRecHit::ConstRecHitContainer recHits = track.recHits();
  
  float previousLayer = -200;//skip first layer
  int nmissing=0;
  for (auto chmap : detLayerMap_){
    //// skip same layers
    if (chmap.first == previousLayer) continue;
    previousLayer = chmap.first;

    bool hasHit = false;
    for (auto hit : recHits){
      // cout <<" chmap.first "<< chmap.first
      // 	   << " -1*hit->globalPosition().y() "<< -1*hit->globalPosition().y()
      // 	   <<endl;
      if (abs(chmap.first + hit->globalPosition().y()) < 1. ){
	hasHit = true;
	break;
      }
    }
    if (hasHit) continue;
    
    auto refChamber = chmap.second;

    tsos = theService_->propagator("StraightLinePropagator")->propagate(tsos,refChamber->surface());
    if (!tsos.isValid()){
      continue;
    }
    
    GlobalPoint tsosGP = tsos.globalPosition();

    cout << "tsos gp   "<< tsosGP << refChamber->id() <<endl;
    //cout << "tsos error "<< tsos.localError().positionError() << endl;
    
    shared_ptr<MuonTransientTrackingRecHit> tmpRecHit;
    ////no rechit, make missing hit
    for (auto col : detLayerMap_){
      // only look in same layer
      if (chmap.first != col.first) continue;
      auto ch = col.second;
      for (auto etaPart : ch->etaPartitions()){	
	const LocalPoint pos = etaPart->toLocal(tsosGP);
	const LocalPoint pos2D(pos.x(), pos.y(), 0);
	const BoundPlane& bps(etaPart->surface());
	  
	if(abs(pos.y()) < 17 && abs(pos.x()) < 40 )
	  cout << " missing hit "<< etaPart->id() << " pos = "<<pos<< " R = "<<pos.mag() <<" inside "
	       <<  bps.bounds().inside(pos2D) <<endl;
	  
	if (bps.bounds().inside(pos2D)){
	    
	    
	  //if (!bp.bounds().inside(pos)) continue;
	  //cout << "made missing hit "<<etaPartID   <<endl;
	    
	  auto missingHit = make_unique<GEMRecHit>(etaPart->id(), -10, pos2D);
	  const GeomDet* geomDet(etaPart);	  
	  tmpRecHit = MuonTransientTrackingRecHit::specificBuild(geomDet,missingHit.get());
	  tmpRecHit->invalidateHit();
	  break;
	}
      }
    }
    
    if (tmpRecHit){      
      recHits.push_back(tmpRecHit);
      ++nmissing;
    }
  }
  cout << "found "<<nmissing <<" missing hits"<<endl;
  
  return recHits;
}

MuonTransientTrackingRecHit::MuonRecHitContainer GEMCosmicMuon::getHitsFromLayer(vector<const GEMChamber*> &chambers, const GEMRecHitCollection* gemHits)
{
  MuonTransientTrackingRecHit::MuonRecHitContainer hits;
  for (auto ch : chambers){
    if (!ch) continue;
    for (auto etaPart : ch->etaPartitions()){
      GEMDetId etaPartID = etaPart->id();
      
      GEMRecHitCollection::range range = gemHits->get(etaPartID);
      //cout<< "Number of GEM rechits available , from chamber: "<< etaPartID<<endl;
      for (GEMRecHitCollection::const_iterator rechit = range.first; rechit!=range.second; ++rechit){
	const GeomDet* geomDet(etaPart);
	//cout <<"getHitsFromLayer "<< etaPartID<< rechit->localPosition()<<endl;
	hits.push_back(MuonTransientTrackingRecHit::specificBuild(geomDet,&*rechit));
      }
    }
  }
  return hits;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEMCosmicMuon);
