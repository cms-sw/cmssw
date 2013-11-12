// Package:    EgammaElectronAlgos
// Class:      SeedFilter.

#include "RecoEgamma/EgammaElectronAlgos/interface/SeedFilter.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"


#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include <vector>

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromRegionHits.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/Point3D.h"

using namespace std;
using namespace reco;

SeedFilter::SeedFilter(const edm::ParameterSet& conf,
		       const SeedFilter::Tokens& tokens,
		       edm::ConsumesCollector& iC)
 {
  edm::LogInfo("EtaPhiRegionSeedFactory") << "Enter the EtaPhiRegionSeedFactory";
  edm::ParameterSet regionPSet = conf.getParameter<edm::ParameterSet>("RegionPSet");

  ptmin_        = regionPSet.getParameter<double>("ptMin");
  originradius_ = regionPSet.getParameter<double>("originRadius");
  halflength_   = regionPSet.getParameter<double>("originHalfLength");
  deltaEta_     = regionPSet.getParameter<double>("deltaEtaRegion");
  deltaPhi_     = regionPSet.getParameter<double>("deltaPhiRegion");
  useZvertex_   = regionPSet.getParameter<bool>("useZInVertex");
  vertexSrc_    = tokens.token_vtx;

  // setup orderedhits setup (in order to tell seed generator to use pairs/triplets, which layers)
  edm::ParameterSet hitsfactoryPSet = conf.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet");
  std::string hitsfactoryName = hitsfactoryPSet.getParameter<std::string>("ComponentName");

  // HIT FACTORY MODE (JRV)
  // -1 = look for existing hit collections for both pixels and strips
  // 0 = look for pixel hit collections and build strip hits on demand (regional unpacking)
  // 1 = build both pixel and strip hits on demand (regional unpacking)
  hitsfactoryMode_ = hitsfactoryPSet.getUntrackedParameter<int>("useOnDemandTracker");

  // get orderd hits generator from factory
  OrderedHitsGenerator*  hitsGenerator = OrderedHitsGeneratorFactory::get()->create(hitsfactoryName, hitsfactoryPSet, iC);

  // start seed generator
  // FIXME??
  edm::ParameterSet creatorPSet;
  creatorPSet.addParameter<std::string>("propagator","PropagatorWithMaterial");

  combinatorialSeedGenerator = new SeedGeneratorFromRegionHits(hitsGenerator,0,
                                    SeedCreatorFactory::get()->create("SeedFromConsecutiveHitsCreator", creatorPSet)
				                  	       );
  beamSpotTag_ = tokens.token_bs; ;
  measurementTrackerName_ = conf.getParameter<edm::InputTag>("measurementTrackerEvent").encode() ;

 }

SeedFilter::~SeedFilter() {
  delete combinatorialSeedGenerator;
}

void SeedFilter::seeds(edm::Event& e, const edm::EventSetup& setup, const reco::SuperClusterRef &scRef, TrajectorySeedCollection *output) {

  setup.get<IdealMagneticFieldRecord>().get(theMagField);
  std::auto_ptr<TrajectorySeedCollection> seedColl(new TrajectorySeedCollection());

  GlobalPoint vtxPos;
  double deltaZVertex;

  //edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  //e.getByType(recoBeamSpotHandle);

  // gets its position
  //const reco::BeamSpot::Point& BSPosition = recoBeamSpotHandle->position();
  //vtxPos = GlobalPoint(BSPosition.x(), BSPosition.y(), BSPosition.z());
  //double sigmaZ = recoBeamSpotHandle->sigmaZ();
  //double sigmaZ0Error = recoBeamSpotHandle->sigmaZ0Error();
  //double sq=sqrt(sigmaZ*sigmaZ+sigmaZ0Error*sigmaZ0Error);
  //deltaZVertex = 3*sq; //halflength_;

  // get the primary vertex (if any)
  reco::VertexCollection vertices;
  edm::Handle<reco::VertexCollection> h_vertices;
  if (e.getByToken(vertexSrc_, h_vertices)) {
    vertices = *(h_vertices.product());
  } else {
	  LogDebug("SeedFilter") << "SeedFilter::seeds"
				  << "No vertex collection found: using beam-spot";
  }

  if (!vertices.empty() && useZvertex_) {
    vtxPos = GlobalPoint(vertices.front().x(), vertices.front().y(), vertices.front().z());
    deltaZVertex = halflength_;

  } else {
    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
    //e.getByType(recoBeamSpotHandle);
    e.getByToken(beamSpotTag_,recoBeamSpotHandle);
    // gets its position
    const reco::BeamSpot::Point& BSPosition = recoBeamSpotHandle->position();
    double sigmaZ = recoBeamSpotHandle->sigmaZ();
    double sigmaZ0Error = recoBeamSpotHandle->sigmaZ0Error();
    double sq=sqrt(sigmaZ*sigmaZ+sigmaZ0Error*sigmaZ0Error);
    vtxPos = GlobalPoint(BSPosition.x(), BSPosition.y(), BSPosition.z());
    deltaZVertex = 3*sq;
  }

  //seeds selection
  float energy = scRef->energy();

  const GlobalPoint clusterPos(scRef->position().x(), scRef->position().y(), scRef->position().z());

  //===============================================
  // EtaPhiRegion and seeds for electron hypothesis
  //===============================================

  TrackCharge aCharge = -1 ;
  FreeTrajectoryState fts = FTSFromVertexToPointFactory::get(*theMagField, clusterPos, vtxPos, energy, aCharge);

  RectangularEtaPhiTrackingRegion etaphiRegionMinus(fts.momentum(),
                                                    vtxPos,
                                                    ptmin_,
                                                    originradius_,
                                                    deltaZVertex,
						    //                                                    deltaEta_,
						    //                                                    deltaPhi_,-1);
						    RectangularEtaPhiTrackingRegion::Margin(std::abs(deltaEta_),std::abs(deltaEta_)),
						    RectangularEtaPhiTrackingRegion::Margin(std::abs(deltaPhi_),std::abs(deltaPhi_))
						    ,hitsfactoryMode_,
						    true, /*default in header*/
						    measurementTrackerName_
						    );
  combinatorialSeedGenerator->run(*seedColl, etaphiRegionMinus, e, setup);

  for (unsigned int i = 0; i<seedColl->size(); ++i)
    output->push_back((*seedColl)[i]);

  seedColl->clear();

  //===============================================
  // EtaPhiRegion and seeds for positron hypothesis
  //===============================================

  TrackCharge aChargep = 1 ;
  fts = FTSFromVertexToPointFactory::get(*theMagField, clusterPos, vtxPos, energy, aChargep);

  RectangularEtaPhiTrackingRegion etaphiRegionPlus(fts.momentum(),
                                                   vtxPos,
                                                   ptmin_,
                                                   originradius_,
                                                   deltaZVertex,
						   //                                                   deltaEta_,
						   //                                                   deltaPhi_,-1);
						    RectangularEtaPhiTrackingRegion::Margin(std::abs(deltaEta_),std::abs(deltaEta_)),
						    RectangularEtaPhiTrackingRegion::Margin(std::abs(deltaPhi_),std::abs(deltaPhi_))
						    ,hitsfactoryMode_
						   ,true, /*default in header*/
						   measurementTrackerName_
						   );

  combinatorialSeedGenerator->run(*seedColl, etaphiRegionPlus, e, setup);

  for (unsigned int i = 0; i<seedColl->size(); ++i){
    TrajectorySeed::range r = (*seedColl)[i].recHits();
    // first Hit
    TrajectorySeed::const_iterator it = r.first;
    const TrackingRecHit* a1 = &(*it);
    // now second Hit
    it++;
    const TrackingRecHit* a2 = &(*it);
    // check if the seed is already in the collection
    bool isInCollection = false;
    for(unsigned int j=0; j<output->size(); ++j) {
      TrajectorySeed::range r = (*seedColl)[i].recHits();
      // first Hit
      TrajectorySeed::const_iterator it = r.first;
      const TrackingRecHit* b1 = &(*it);
      // now second Hit
      it++;
      const TrackingRecHit* b2 = &(*it);
      if ((b1->sharesInput(a1, TrackingRecHit::all)) && (b2->sharesInput(a2, TrackingRecHit::all))) {
	isInCollection = true;
	break;
      }
    }
    if (!isInCollection)
      output->push_back((*seedColl)[i]);
  }

  seedColl->clear();
}

