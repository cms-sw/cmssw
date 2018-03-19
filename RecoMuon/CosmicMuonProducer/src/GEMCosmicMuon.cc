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
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonSmoother.h"
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonSmoother.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"

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
  
private:
  int iev; // events through
  edm::EDGetTokenT<GEMRecHitCollection> theGEMRecHitToken_;
  CosmicMuonSmoother* theSmoother_;
  MuonServiceProxy* theService_;
  KFUpdator* theUpdator_;
  unique_ptr<std::vector<TrajectorySeed> > findSeeds(MuonTransientTrackingRecHit::MuonRecHitContainer &topSeeds,
						     MuonTransientTrackingRecHit::MuonRecHitContainer &bottomSeeds);
  Trajectory makeTrajectory(TrajectorySeed& seed, const GEMRecHitCollection* gemHits, vector<const GEMChamber*>& gemChambers);
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
    ev.put(std::move(trajectorySeeds));
    ev.put(std::move(trackCollection));
    ev.put(std::move(trackingRecHitCollection));
    ev.put(std::move(trackExtraCollection));
    ev.put(std::move(trajectorys));
    return;
  }
  
  //  cout << "GEMCosmicMuon::gemRecHits " << gemRecHits->size() << endl;
  vector<const GEMChamber*> gemChambers;
  vector<int> topChambers, bottomChambers;
  // get all exisiting chamber and find top and bottom chambers
  for (int irow = 0; irow < 3; ++irow){
    int itopChamber = -1, ibottomChamber = -1;
    for (int ichamber = 1; ichamber < 10; ++ichamber){
      int chamberNo = ichamber+irow*10;
      for (int ilayer = 1; ilayer < 3; ++ilayer){
	GEMDetId chamberID(1,1,1, ilayer, chamberNo, 0);
	const GEMChamber* ch = mgeom->chamber(chamberID);
	if (!ch) continue;
	gemChambers.emplace_back(ch);
	if ( ibottomChamber < 0 ) ibottomChamber = chamberNo;
	itopChamber = chamberNo;
      }
    }
    if ( itopChamber > 0 )
      topChambers.emplace_back(itopChamber);
    if ( ibottomChamber > 0 )
      bottomChambers.emplace_back(ibottomChamber);
  }
  int topLayer = 1, bottomLayer = 2;
  if (doInnerSeeding_){
    topLayer = 2; bottomLayer = 1;
  }
  
  MuonTransientTrackingRecHit::MuonRecHitContainer topSeeds, bottomSeeds;  
  // get top seeds
  for (auto cham : topChambers){
    GEMDetId chamberID(1,1,1, topLayer, cham, 0);
    const GEMChamber* ch = mgeom->chamber(chamberID);
    for (auto etaPart : ch->etaPartitions()){
      GEMDetId etaPartID = etaPart->id();
      
      GEMRecHitCollection::range range = gemRecHits->get(etaPartID);
      //cout<< "Number of GEM rechits available , from chamber: "<< etaPartID<<endl;
      for (GEMRecHitCollection::const_iterator rechit = range.first; rechit!=range.second; ++rechit){
	const GeomDet* geomDet(etaPart);
	cout<< "top seed: chamber "<< etaPartID <<", clusterSize = "<< (*rechit).clusterSize() <<endl;
	topSeeds.emplace_back(MuonTransientTrackingRecHit::specificBuild(geomDet,&*rechit));
      }
    }
  }
  
  for (auto cham : bottomChambers){
    GEMDetId chamberID(1,1,1, bottomLayer, cham, 0);
    const GEMChamber* ch = mgeom->chamber(chamberID);
    for (auto etaPart : ch->etaPartitions()){
      GEMDetId etaPartID = etaPart->id();
      
      GEMRecHitCollection::range range = gemRecHits->get(etaPartID);
      //cout<< "Number of GEM rechits available , from chamber: "<< etaPartID<<endl;
      for (GEMRecHitCollection::const_iterator rechit = range.first; rechit!=range.second; ++rechit){
	const GeomDet* geomDet(etaPart);
	cout<< "bottom seed: chamber "<< etaPartID <<", clusterSize = "<< (*rechit).clusterSize() <<endl;
	bottomSeeds.emplace_back(MuonTransientTrackingRecHit::specificBuild(geomDet,&*rechit));
      }
    }
  }
  
  trajectorySeeds = findSeeds(topSeeds, bottomSeeds);
  //  cout << "GEMCosmicMuon::trajectorySeeds->size() " << trajectorySeeds->size() << endl;

  // need to loop over seeds, make best track and save only best track
  //TrajectorySeed seed =trajectorySeeds->at(0);
  Trajectory bestTrajectory;
  TrajectorySeed bestSeed;
  float maxChi2 = trackChi2_;
  for (auto seed : *trajectorySeeds){
    Trajectory smoothed = makeTrajectory(seed, gemRecHits.product(), gemChambers);
    if (smoothed.isValid()){
      trajectorys->emplace_back(smoothed);
      //      cout << "GEMCosmicMuon::Trajectory " << smoothed.foundHits() << endl;
      //      cout << "GEMCosmicMuon::Trajectory chiSquared/ ndof " << smoothed.chiSquared()/float(smoothed.ndof()) << endl;
      //if (( maxChi2 > smoothed.chiSquared()/float(smoothed.ndof())) and ( smoothed.chiSquared()/float(smoothed.ndof()) > 7.0)){
      if (maxChi2 > smoothed.chiSquared()/float(smoothed.ndof())){
	maxChi2 = smoothed.chiSquared()/float(smoothed.ndof());
	bestTrajectory = smoothed;
        bestSeed = seed;
      }
    }
  }
  if (!bestTrajectory.isValid()){
    ev.put(std::move(trajectorySeeds));
    ev.put(std::move(trackCollection));
    ev.put(std::move(trackingRecHitCollection));
    ev.put(std::move(trackExtraCollection));
    ev.put(std::move(trajectorys));
    return;
  }
  //cout << maxChi2 << endl;
  //cout << "GEMCosmicMuon::bestTrajectory " << bestTrajectory.foundHits() << endl;
  //cout << "GEMCosmicMuon::bestTrajectory chiSquared/ ndof " << bestTrajectory.chiSquared()/float(bestTrajectory.ndof()) << endl;
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
 
  // create empty collection of Segments
  //cout << "GEMCosmicMuon::track " << track.pt() << endl;
  
  // reco::TrackExtra tx(track.outerPosition(), track.outerMomentum(), track.outerOk(),
  // 		      track.innerPosition(), track.innerMomentum(), track.innerOk(),
  // 		      track.outerStateCovariance(), track.outerDetId(),
  // 		      track.innerStateCovariance(), track.innerDetId(),
  // 		      track.seedDirection(), edm::RefToBase<TrajectorySeed>());
  // 		      //, bestTrajectory.seedRef() );
  reco::TrackExtra tx;
  //tx.setResiduals(track.residuals());
  //adding rec hits
  Trajectory::RecHitContainer transHits = bestTrajectory.recHits();
  unsigned int nHitsAdded = 0;
  for (Trajectory::RecHitContainer::const_iterator recHit = transHits.begin(); recHit != transHits.end(); ++recHit) {
    TrackingRecHit *singleHit = (**recHit).hit()->clone();
    trackingRecHitCollection->push_back( singleHit );  
    ++nHitsAdded;
  }
  tx.setHits(recHitCollectionRefProd, recHitsIndex, nHitsAdded);
  recHitsIndex +=nHitsAdded;

  trackExtraCollection->emplace_back(tx );

  reco::TrackExtraRef trackExtraRef(trackExtraCollectionRefProd, trackExtraIndex++ );
  track.setExtra(trackExtraRef);
  trackCollection->emplace_back(track);
  
  // fill the collection
  // put collection in event
  trajectorySeeds->emplace_back(bestSeed);

  ev.put(std::move(trajectorySeeds));
  ev.put(std::move(trackCollection));
  ev.put(std::move(trackingRecHitCollection));
  ev.put(std::move(trackExtraCollection));
  ev.put(std::move(trajectorys));
  
}

unique_ptr<std::vector<TrajectorySeed> > GEMCosmicMuon::findSeeds(MuonTransientTrackingRecHit::MuonRecHitContainer &topSeeds,
								  MuonTransientTrackingRecHit::MuonRecHitContainer &bottomSeeds)
{
  unique_ptr<std::vector<TrajectorySeed> > trajectorySeeds( new vector<TrajectorySeed>());
  if (topSeeds.size() > 0 && bottomSeeds.size() > 0){
    
    for (auto bottomhit : bottomSeeds){
      for (auto tophit : topSeeds){
	if (bottomhit->globalPosition().y() < tophit->globalPosition().y()){
	  LocalPoint segPos = bottomhit->localPosition();
	  GlobalVector segDirGV(tophit->globalPosition().x() - bottomhit->globalPosition().x(),
				tophit->globalPosition().y() - bottomhit->globalPosition().y(),
				tophit->globalPosition().z() - bottomhit->globalPosition().z());

	  segDirGV *=10;
	  LocalVector segDir = bottomhit->det()->toLocal(segDirGV);
	  // cout << "GEMCosmicMuon::GlobalVector " << segDirGV << endl;
	  // cout << "GEMCosmicMuon::LocalVector  " << segDir << endl;
  
	  int charge= 1;
	  LocalTrajectoryParameters param(segPos, segDir, charge);
  
	  AlgebraicSymMatrix mat(5,0);
	  mat = bottomhit->parametersError().similarityT( bottomhit->projectionMatrix() );
	  //float p_err = 0.2;
	  //mat[0][0]= p_err;
	  LocalTrajectoryError error(asSMatrix<5>(mat));

	  // get first hit
	  TrajectoryStateOnSurface tsos(param, error, bottomhit->det()->surface(), &*theService_->magneticField());
	  //cout << "GEMCosmicMuon::tsos " << tsos << endl;
	  uint32_t id = bottomhit->rawId();
	  PTrajectoryStateOnDet const & seedTSOS = trajectoryStateTransform::persistentState(tsos, id);
	
	  edm::OwnVector<TrackingRecHit> seedHits;
	  seedHits.push_back(bottomhit->hit()->clone());
	  seedHits.push_back(tophit->hit()->clone());

	  TrajectorySeed seed(seedTSOS,seedHits,alongMomentum);
	  trajectorySeeds->emplace_back(seed);
	}
      }
    }
  }
  return trajectorySeeds;
}

Trajectory GEMCosmicMuon::makeTrajectory(TrajectorySeed& seed,
					 const GEMRecHitCollection* gemHits,
					 vector<const GEMChamber*>& gemChambers)
{
  PTrajectoryStateOnDet ptsd1(seed.startingState());
  DetId did(ptsd1.detId());
  const BoundPlane& bp = theService_->trackingGeometry()->idToDet(did)->surface();
  TrajectoryStateOnSurface tsos = trajectoryStateTransform::transientState(ptsd1,&bp,&*theService_->magneticField());

  TrajectoryStateOnSurface tsosCurrent = tsos;
  
  TransientTrackingRecHit::ConstRecHitContainer consRecHits;
  for (auto ch : gemChambers){
    //const DetLayer* layer = theService_->detLayerGeometry()->idToLayer( ch->id().rawId() );
    std::shared_ptr<MuonTransientTrackingRecHit> tmpRecHit;
    
    tsosCurrent = theService_->propagator("StraightLinePropagator")->propagate(tsosCurrent,ch->surface());
    if (!tsosCurrent.isValid()) return Trajectory();
    GlobalPoint tsosGP = tsosCurrent.freeTrajectoryState()->position();
    //TrackingRecHit *tmpRecHit = new TrackingRecHit(ch);
    //cout << "tsosGP "<< tsosGP <<endl;
    float maxR = 9999;

    for (auto etaPart : ch->etaPartitions()){
      GEMDetId etaPartID = etaPart->id();
      
      GEMRecHitCollection::range range = gemHits->get(etaPartID);
      //cout<< "Number of GEM rechits available , from chamber: "<< etaPartID<<endl;
      for (GEMRecHitCollection::const_iterator rechit = range.first; rechit!=range.second; ++rechit){

	LocalPoint tsosLP = etaPart->toLocal(tsosGP);
	LocalPoint rhLP = (*rechit).localPosition();
	double y_err = (*rechit).localPositionError().yy();
	
	if (abs(rhLP.x() - tsosLP.x()) > trackResX_) continue;
	
	if (abs(rhLP.z() - tsosLP.z()) > y_err*trackResY_) continue;
	// need to find best hits per chamber
	float deltaR = (rhLP - tsosLP).mag();
	if (maxR > deltaR){
	  const GeomDet* geomDet(etaPart);	  
	  tmpRecHit = MuonTransientTrackingRecHit::specificBuild(geomDet,&*rechit);
	  maxR = deltaR;
	}
      }
    }
    
    if (tmpRecHit){
      //cout << "hit gp "<< tmpRecHit->globalPosition() <<endl;
      tsosCurrent = theUpdator_->update(tsosCurrent, *tmpRecHit);
      consRecHits.emplace_back(tmpRecHit);
    }
  }
  if (consRecHits.size() <3) return Trajectory();
  vector<Trajectory> fitted = theSmoother_->trajectories(seed, consRecHits, tsos);
  return fitted.front();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEMCosmicMuon);
