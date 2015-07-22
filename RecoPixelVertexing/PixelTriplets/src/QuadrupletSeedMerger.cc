#include "RecoPixelVertexing/PixelTriplets/interface/QuadrupletSeedMerger.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/KDTreeLinkerAlgo.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/KDTreeLinkerTools.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"

#include "DataFormats/GeometryVector/interface/Pi.h"

#include <algorithm>

namespace {
  template <typename T>
  constexpr T sqr(T x) {
    return x*x;
  }
}

/***

QuadrupletSeedMerger

* merge triplet into quadruplet seeds
  for either SeedingHitSets or TrajectorySeeds

* method: 
    QuadrupletSeedMerger::mergeTriplets( const OrderedSeedingHits&, const edm::EventSetup& ) const
  for use in HLT with: RecoPixelVertexing/PixelTrackFitting/src/PixelTrackReconstruction.cc
  contains the merging functionality

* method:
    QuadrupletSeedMerger::mergeTriplets( const TrajectorySeedCollection&, TrackingRegion&, EventSetup& ) const
  is a wrapper for the former, running on TrajectorySeeds
  for use in iterative tracking with: RecoTracker/TkSeedGenerator/plugins/SeedGeneratorFromRegionHitsEDProducer

***/


// Helper functions
namespace {
  bool areHitsOnLayers(const SeedMergerPixelLayer& layer1, const SeedMergerPixelLayer& layer2,
                       const std::pair<SeedingHitSet::ConstRecHitPointer,SeedingHitSet::ConstRecHitPointer>& hits,
                       const TrackerTopology *tTopo) {
    DetId firstHitId = hits.first->geographicalId();
    DetId secondHitId = hits.second->geographicalId();
    return ((layer1.isContainsDetector(firstHitId, tTopo) &&
             layer2.isContainsDetector(secondHitId, tTopo)) ||
            (layer1.isContainsDetector(secondHitId, tTopo) &&
             layer2.isContainsDetector(firstHitId, tTopo)));
  }

  bool areHitsEqual(const TrackingRecHit & hit1, const TrackingRecHit & hit2) {
    constexpr double epsilon = 0.00001;
    if(hit1.geographicalId() != hit2.geographicalId())
      return false;

    LocalPoint lp1 = hit1.localPosition(), lp2 = hit2.localPosition();
    return std::abs(lp1.x() - lp2.x()) < epsilon && std::abs(lp1.y() - lp2.y()) < epsilon;
  }
}



///
///
///
QuadrupletSeedMerger::QuadrupletSeedMerger(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC):
  QuadrupletSeedMerger(iConfig, nullptr, iC) {}
QuadrupletSeedMerger::QuadrupletSeedMerger(const edm::ParameterSet& iConfig, const edm::ParameterSet& seedCreatorConfig, edm::ConsumesCollector& iC):
  QuadrupletSeedMerger(iConfig,
                       SeedCreatorFactory::get()->create(seedCreatorConfig.getParameter<std::string>("ComponentName") , seedCreatorConfig), 
                       iC) {}
QuadrupletSeedMerger::QuadrupletSeedMerger(const edm::ParameterSet& iConfig, SeedCreator *seedCreator, edm::ConsumesCollector& iC):
  theLayerBuilder_(iConfig, iC),
  theSeedCreator_(seedCreator)
 {

  // by default, do not..
  // ..merge triplets
  isMergeTriplets_ = false;
  // ..add remaining triplets
  isAddRemainingTriplets_ = false;
}

void QuadrupletSeedMerger::update(const edm::EventSetup& es) {
  // copy geometry
  es.get<TrackerDigiGeometryRecord>().get( theTrackerGeometry_ );
}

///
///
///
QuadrupletSeedMerger::~QuadrupletSeedMerger() {

}



///
/// merge triplets into quadruplets
/// INPUT: OrderedSeedingHits
/// OUTPUT: SeedingHitSets
///
/// this method is used in RecoPixelVertexing/PixelTrackFitting/src/PixelTrackReconstruction.cc
/// and contains the basic merger functionality
///

//const std::vector<SeedingHitSet> QuadrupletSeedMerger::mergeTriplets( const OrderedSeedingHits& inputTriplets, const edm::EventSetup& es ) {
const OrderedSeedingHits& QuadrupletSeedMerger::mergeTriplets( const OrderedSeedingHits& inputTriplets, const edm::EventSetup& es ) {

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  es.get<TrackerTopologyRcd>().get(tTopoHand);
  const TrackerTopology *tTopo=tTopoHand.product();
  
  // the list of layers on which quadruplets should be formed
  if(theLayerBuilder_.check(es)) {
    theLayerSets_ = theLayerBuilder_.layers( es ); // this is a vector<vector<SeedingLayer> >
  }

  
  // make a working copy of the input triplets
  // to be able to remove merged triplets

  std::vector<std::pair<double,double> > phiEtaCache;
  std::vector<SeedingHitSet> tripletCache; //somethings strange about OrderedSeedingHits?

  const unsigned int nInputTriplets = inputTriplets.size();
  phiEtaCache.reserve(nInputTriplets);
  tripletCache.reserve(nInputTriplets);

  for( unsigned int it = 0; it < nInputTriplets; ++it ) {
    tripletCache.push_back((inputTriplets[it]));
    phiEtaCache.push_back(calculatePhiEta( (tripletCache[it]) ));

  }

  // the output
  quads_.clear();

  // check if the input is all triplets
  // (code is also used for pairs..)
  // if not, copy the input to the output & return
  bool isAllTriplets = true;
  for( unsigned int it = 0; it < nInputTriplets; ++it ) {
    if( tripletCache[it].size() != 3 ) {
      isAllTriplets = false;
      break;
    }
  }

  if( !isAllTriplets && isMergeTriplets_ )
    std::cout << "[QuadrupletSeedMerger::mergeTriplets] (in HLT) ** bailing out since non-triplets in input." << std::endl;

  if( !isAllTriplets || !isMergeTriplets_ ) {
    quads_.reserve(nInputTriplets);
    for( unsigned int it = 0; it < nInputTriplets; ++it ) {
      quads_.push_back( (tripletCache[it]));
    }

    return quads_;
  }


  quads_.reserve(0.2*nInputTriplets); //rough guess

  // loop all possible 4-layer combinations
  // as specified in python/quadrupletseedmerging_cff.py

  std::vector<bool> usedTriplets(nInputTriplets,false);
  std::pair<SeedingHitSet::ConstRecHitPointer,SeedingHitSet::ConstRecHitPointer> sharedHits;
  std::pair<SeedingHitSet::ConstRecHitPointer,SeedingHitSet::ConstRecHitPointer> nonSharedHits;

  // k-d tree, indices are (th)eta, phi
  // build the tree
  std::vector<KDTreeNodeInfo<unsigned int> > nodes;
  std::vector<unsigned int> foundNodes;
  nodes.reserve(2*nInputTriplets);
  foundNodes.reserve(100);
  KDTreeLinkerAlgo<unsigned int> kdtree;
  double minEta=1e10, maxEta=-1e10;
  for(unsigned int it=0; it < nInputTriplets; ++it) {
    double phi = phiEtaCache[it].first;
    double eta = phiEtaCache[it].second;
    nodes.push_back(KDTreeNodeInfo<unsigned int>(it, eta, phi));
    minEta = std::min(minEta, eta);
    maxEta = std::max(maxEta, eta);

    // to wrap all points in phi
    // if(phi < 0) phi += twoPi(); else phi -= twoPi();
    double twoPi = std::copysign(Geom::twoPi(), phi);
    nodes.push_back(KDTreeNodeInfo<unsigned int>(it, eta, phi-twoPi));
  }
  KDTreeBox kdEtaPhi(minEta-0.01, maxEta+0.01, -1*Geom::twoPi(), Geom::twoPi());
  kdtree.build(nodes, kdEtaPhi);
  nodes.clear();
  
  // loop over triplets, search for close-by triplets by using the k-d tree
  // also identify the hits which are shared by triplet pairs, and
  // store indices to hits which are not shared
  std::vector<unsigned int> t1List;
  std::vector<unsigned int> t2List;
  std::vector<short> t1NonSharedHitList;
  std::vector<short> t2NonSharedHitList;
  constexpr short sharedToNonShared[7] = {-1, -1, -1,
                                          2, // 011=3 shared, not shared = 2
                                          -1,
                                          1, // 101=5 shared, not shared = 1
                                          0}; // 110=6 shared, not shared = 0
  constexpr short nonSharedToShared[3][2] = {
    {1, 2},
    {0, 2},
    {0, 1}
  };

  typedef std::tuple<unsigned int, short, short> T2NonSharedTuple;
  std::vector<T2NonSharedTuple> t2Tmp; // temporary to sort t2's before insertion to t2List
  for(unsigned int t1=0; t1<nInputTriplets; ++t1) {
    double phi = phiEtaCache[t1].first;
    double eta = phiEtaCache[t1].second;

    KDTreeBox box(eta-0.05, eta+0.05, phi-0.15, phi+0.15);
    foundNodes.clear();
    kdtree.search(box, foundNodes);
    if(foundNodes.empty())
      continue;

    const SeedingHitSet& tr1 = tripletCache[t1];

    for(unsigned int t2: foundNodes) {
      if(t1 >= t2)
        continue;

      // Ensure here that the triplet pairs share two hits.
      const SeedingHitSet& tr2 = tripletCache[t2];

      // If neither of first two hits in tr1 are found from tr2, this
      // pair can be skipped
      int equalHits=0;
      // Find the indices of shared hits in both t1 and t2, use them
      // to obtain the index of non-shared hit for both. The index of
      // non-shared hit is then stored for later use (the indices of
      // shared hits can be easily obtained from them).
      unsigned int t1Shared = 0;
      unsigned int t2Shared = 0;
      for(unsigned int i=0; i<2; ++i) {
        for(unsigned int j=0; j<3; ++j) {
          if(areHitsEqual(*tr1[i], *tr2[j])) {
            t1Shared |= (1<<i);
            t2Shared |= (1<<j);
            ++equalHits;
            break;
          }
        }
      }
      if(equalHits == 0)
        continue;
      // If, after including the third hit of tr1, number of equal
      // hits is not 2, this pair can be skipped
      if(equalHits != 2) {
        for(unsigned int j=0; j<3; ++j) {
          if(areHitsEqual(*tr1[2], *tr2[j])) {
            t1Shared |= (1<<2);
            t2Shared |= (1<<j);
            ++equalHits;
            break;
          }
        }
        if(equalHits != 2)
          continue;
      }
      // valid values for the bitfields are 011=3, 101=5, 110=6
      assert(t1Shared <= 6 && t2Shared <= 6); // against out-of-bounds of sharedToNonShared[]
      short t1NonShared = sharedToNonShared[t1Shared];
      short t2NonShared = sharedToNonShared[t2Shared];
      assert(t1NonShared >= 0 && t2NonShared >= 0); // against invalid value from sharedToNonShared[]

      t2Tmp.emplace_back(t2, t1NonShared, t2NonShared);
    }

    // Sort to increasing order in t2 in order to get exactly same result as before
    std::sort(t2Tmp.begin(), t2Tmp.end(), [](const T2NonSharedTuple& a, const T2NonSharedTuple& b){
        return std::get<0>(a) < std::get<0>(b);
      });
    for(T2NonSharedTuple& t2tpl: t2Tmp) {
      t1List.push_back(t1);
      t2List.push_back(std::get<0>(t2tpl));
      t1NonSharedHitList.push_back(std::get<1>(t2tpl));
      t2NonSharedHitList.push_back(std::get<2>(t2tpl));
    }
    t2Tmp.clear();
  }
  nodes.clear();

  for( ctfseeding::SeedingLayerSets::const_iterator lsIt = theLayerSets_.begin(); lsIt < theLayerSets_.end(); ++lsIt ) {

    // fill a vector with the layers in this set
    std::vector<SeedMergerPixelLayer> currentLayers;
    currentLayers.reserve(lsIt->size());
    for( ctfseeding::SeedingLayers::const_iterator layIt = lsIt->begin(); layIt < lsIt->end(); ++layIt ) {
      currentLayers.push_back( SeedMergerPixelLayer( layIt->name() ) );
    }
    // loop all pair combinations of these 4 layers;
    // strategy is to look for shared hits on such a pair and
    // then merge the remaining hits in both triplets if feasible
    // (SeedingLayers is a vector<SeedingLayer>)

    for( unsigned int s1=0; s1<currentLayers.size()-1; s1++) {

      for( unsigned int s2=s1+1; s2<currentLayers.size(); s2++) {

	std::vector<unsigned int> nonSharedLayerNums;
	for ( unsigned int us1=0; us1<currentLayers.size(); us1++) {
	  if ( s1!=us1 && s2!=us1) nonSharedLayerNums.push_back(us1);
	}

	// loop all possible triplet pairs (which come as input)
	for (unsigned int t12=0; t12<t1List.size(); t12++) {
	  unsigned int t1=t1List[t12];
	  unsigned int t2=t2List[t12];

	    if (usedTriplets[t1] || usedTriplets[t2] ) continue; 

            const SeedingHitSet& firstTriplet = tripletCache[t1];

            short t1NonShared = t1NonSharedHitList[t12];
            sharedHits.first = firstTriplet[nonSharedToShared[t1NonShared][0]];
            sharedHits.second = firstTriplet[nonSharedToShared[t1NonShared][1]];

            // are the shared hits on these two layers?
            if(areHitsOnLayers(currentLayers[s1], currentLayers[s2], sharedHits, tTopo)) {
              short t2NonShared = t2NonSharedHitList[t12];
              const SeedingHitSet& secondTriplet = tripletCache[t2];
              nonSharedHits.first = firstTriplet[t1NonShared];
              nonSharedHits.second = secondTriplet[t2NonShared];

	      // are the remaining hits on different layers?
              if(areHitsOnLayers(currentLayers[nonSharedLayerNums[0]], currentLayers[nonSharedLayerNums[1]], nonSharedHits, tTopo)) {
                QuadrupletHits unsortedHits{ {sharedHits.first, sharedHits.second,
                      nonSharedHits.first, nonSharedHits.second} };

                mySort(unsortedHits);

		//start here with old addtoresult
		if( isValidQuadruplet( unsortedHits, currentLayers, tTopo ) ) {
		  // and create the quadruplet
		  SeedingHitSet quadruplet(unsortedHits[0],unsortedHits[1],unsortedHits[2],unsortedHits[3]);
		  
		  // insert this quadruplet
		  quads_.push_back( quadruplet );
		  // remove both triplets from the list,
		  // needs this 4-permutation since we're in a double loop
		  usedTriplets[t1]=true;
		  usedTriplets[t2]=true;
		}
		
	      } // isMergeableHitsInTriplets
 	    } // isTripletsShareHitsOnLayers
	    // } // triplet double loop
	}
      } // seeding layers double loop
    }
  
  } // seeding layer sets
  
  // add the remaining triplets
  if( isAddRemainingTriplets_ ) {
    for( unsigned int it = 0; it < nInputTriplets; ++it ) {
      if ( !usedTriplets[it] ) 
	quads_.push_back( tripletCache[it]);
    }
  }

  //calc for printout...
//  unsigned int nLeft=0;
//  for ( unsigned int i=0; i<nInputTriplets; i++)
//    if ( !usedTriplets[i] ) nLeft++;
  // some stats
//  std::cout << " [QuadrupletSeedMerger::mergeTriplets] -- Created: " << theResult.size()
//	    << " quadruplets from: " << nInputTriplets << " input triplets (" << nLeft
//	    << " remaining ";
//  std::cout << (isAddRemainingTriplets_?"added":"dropped");
//  std::cout << ")." << std::endl;

  return quads_;

}



///
/// merge triplets into quadruplets
/// INPUT: TrajectorySeedCollection
/// OUTPUT: TrajectorySeedCollection
///
/// this is a wrapper for:
/// vector<SeedingHitSet> mergeTriplets( const OrderedSeedingHits& )
/// for use in RecoTracker/TkSeedGenerator/plugins/SeedGeneratorFromRegionHitsEDProducer.cc
/// (iterative tracking)
///
const TrajectorySeedCollection QuadrupletSeedMerger::mergeTriplets( const TrajectorySeedCollection& seedCollection,
								    const TrackingRegion& region,
								    const edm::EventSetup& es) {

  // ttrh builder for HitSet -> TrajectorySeed conversion;
  // require this to be correctly configured, otherwise -> exception
  es.get<TransientRecHitRecord>().get( theTTRHBuilderLabel_, theTTRHBuilder_ );

  // output collection
  TrajectorySeedCollection theResult;

  // loop to see if we have triplets ONLY
  // if not, copy input -> output and return
  bool isAllTriplets = true;
  for( TrajectorySeedCollection::const_iterator aTrajectorySeed = seedCollection.begin();
       aTrajectorySeed < seedCollection.end(); ++aTrajectorySeed ) {
    if( 3 != aTrajectorySeed->nHits() ) isAllTriplets = false;
  }

  if( !isAllTriplets && isMergeTriplets_ )
    std::cout << " [QuadrupletSeedMerger::mergeTriplets] (in RECO) -- bailing out since non-triplets in input." << std::endl;

  if( !isAllTriplets || !isMergeTriplets_ ) {
    for( TrajectorySeedCollection::const_iterator aTrajectorySeed = seedCollection.begin();
       aTrajectorySeed < seedCollection.end(); ++aTrajectorySeed ) {
      theResult.push_back( *aTrajectorySeed );
    }

    return theResult;
  }


  // all this fiddling here is now about converting
  // TrajectorySeedCollection <-> OrderedSeedingHits

  // create OrderedSeedingHits first;
  OrderedHitTriplets inputTriplets;

  // loop seeds
  for( TrajectorySeedCollection::const_iterator aTrajectorySeed = seedCollection.begin();
       aTrajectorySeed < seedCollection.end(); ++aTrajectorySeed ) {

    std::vector<SeedingHitSet::ConstRecHitPointer> recHitPointers;

    // loop RecHits
    const TrajectorySeed::range theHitsRange = aTrajectorySeed->recHits();
    for( edm::OwnVector<TrackingRecHit>::const_iterator aHit = theHitsRange.first;
	 aHit < theHitsRange.second; ++aHit ) {
      
      // this is a collection of: ReferenceCountingPointer< SeedingHitSet> 
      recHitPointers.push_back( (SeedingHitSet::ConstRecHitPointer)(&*aHit ) );

    }
    
    // add to input collection
    inputTriplets.push_back( OrderedHitTriplet( recHitPointers.at( 0 ), recHitPointers.at( 1 ), recHitPointers.at( 2 ) ) );

  }

  // do the real merging..
  const OrderedSeedingHits &quadrupletHitSets = mergeTriplets( inputTriplets, es );
  
  // convert back to TrajectorySeedCollection

  // the idea here is to fetch the same SeedCreator and PSet
  // as those used by the plugin which is calling the merger
  // (at the moment that's SeedGeneratorFromRegionHitsEDProducer)
  theSeedCreator_->init(region, es, 0);
  for ( unsigned int i=0; i< quadrupletHitSets.size(); i++) {
    // add trajectory seed to result collection
    theSeedCreator_->makeSeed( theResult, quadrupletHitSets[i]);
  }

  return theResult;
  
}


///
///
///
std::pair<double,double> QuadrupletSeedMerger::calculatePhiEta( SeedingHitSet const& nTuplet ) const {

//   if( nTuplet.size() < 3 ) {
//     std::cerr << " [QuadrupletSeedMerger::calculatePhiEta] ** ERROR: nTuplet has less than 3 hits" << std::endl;
//     throw; // tbr.
//   }

  GlobalPoint p1 = nTuplet[0]->globalPosition();
  GlobalPoint p2 = nTuplet[1]->globalPosition();

  const double x1 = p1.x();
  const double x2 = p2.x();
  const double y1 = p1.y();
  const double y2 = p2.y();
  const double z1 = p1.z();
  const double z2 = p2.z();

  const double phi = atan2( x2 - x1, y2 -y1 );
  const double eta = acos( (z2 - z1) / sqrt( sqr( x2 - x1 ) + sqr( y2 - y1 ) + sqr( z2 - z1 ) ) ); // this is theta angle in reality

  std::pair<double,double> retVal;
  retVal=std::make_pair (phi,eta);
  return retVal;
  //return std::make_pair<double,double>( phi, eta );
  
}




///
///
///
void QuadrupletSeedMerger::printHit( const TrackingRecHit* aHit ) const {

  const GeomDet* geomDet = theTrackerGeometry_->idToDet( aHit->geographicalId() );
  const double r = geomDet->surface().position().perp();
  const double x = geomDet->toGlobal( aHit->localPosition() ).x();
  const double y = geomDet->toGlobal( aHit->localPosition() ).y();
  const double z = geomDet->toGlobal( aHit->localPosition() ).z();
  std::cout << "<RecHit> x: " << x << " y: " << y << " z: " << z << " r: " << r << std::endl;

}

///
///
///
void QuadrupletSeedMerger::printNtuplet( const SeedingHitSet& aNtuplet ) const {

  std::cout << "DUMPING NTUPLET OF SIZE:";
  std::cout << aNtuplet.size() << std::endl;

  for( unsigned int aHit = 0; aHit < aNtuplet.size(); ++aHit ) {

    const TrackingRecHit* theHit = aNtuplet[aHit]->hit();
    const GeomDet* geomDet = theTrackerGeometry_->idToDet( theHit->geographicalId() );
    const double x = geomDet->toGlobal( theHit->localPosition() ).x();
    const double y = geomDet->toGlobal( theHit->localPosition() ).y();
    const double z = geomDet->toGlobal( theHit->localPosition() ).z();
    const double r = sqrt( x*x + y*y );

    unsigned int layer;
    std::string detName;
    if( PixelSubdetector::PixelBarrel == theHit->geographicalId().subdetId() ) {
      detName = "BPIX ";
      PixelBarrelName pbn( aNtuplet[aHit]->hit()->geographicalId());
      layer = pbn.layerName();
    }
    else {
      detName = "FPIX";
      if( z > 0 ) detName += "+";
      else detName += "-";

      PixelEndcapName pen( theHit->geographicalId() );
      layer = pen.diskName();
    }

    std::cout << "<NtupletHit> D: " << detName << " L: " << layer << " x: " << x << " y: " << y << " z: " << z << " r: " << r << std::endl;
    
}

  std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;

}



///
///
///
void QuadrupletSeedMerger::setTTRHBuilderLabel( std::string label ) {

  theTTRHBuilderLabel_ = label;
  
}



///
///
///
void QuadrupletSeedMerger::setMergeTriplets( bool isMergeTriplets ) {
  isMergeTriplets_ = isMergeTriplets;
}



///
///
///
void QuadrupletSeedMerger::setAddRemainingTriplets( bool isAddTriplets ) {
  isAddRemainingTriplets_ = isAddTriplets;
}



///
/// check for validity of a (radius-) *sorted* quadruplet:
///  1. after sorting, hits must be on layers according to the 
///     order given in PixelSeedMergerQuadruplets (from cfg)
///
bool QuadrupletSeedMerger::isValidQuadruplet(const QuadrupletHits &quadruplet, const std::vector<SeedMergerPixelLayer>& layers, const TrackerTopology *tTopo) const {

  const unsigned int quadrupletSize = quadruplet.size();

  // basic size test..
  if( quadrupletSize != layers.size() ) {
    std::cout << " [QuadrupletSeedMerger::isValidQuadruplet] ** WARNING: size mismatch: "
	      << quadrupletSize << "/" << layers.size() << std::endl;
    return false;
  }

  // go along layers and check if all (parallel) quadruplet hits match
  for( unsigned int index = 0; index < quadrupletSize; ++index ) {
    if( ! layers[index].isContainsDetector( quadruplet[index]->geographicalId(), tTopo ) ) {
      return false;
    }
  }

  return true;

}


///
///
///
SeedMergerPixelLayer::SeedMergerPixelLayer( const std::string& name ) {
  
  if( ! isValidName( name ) ) { 
    std::cerr << " [SeedMergerPixelLayer::SeedMergerPixelLayer] ** ERROR: illegal name: \"" << name << "\"." << std::endl;
    isValid_ = false;
    return;
  }

  // bare name, can be done here
  name_ = name;

  // (output format -> see DataFormats/SiPixelDetId/interface/PixelSubdetector.h)
  if( std::string::npos != name_.find( "BPix" ) ) 
    { subdet_ = PixelSubdetector::PixelBarrel; side_ = Undefined;}
  else if( std::string::npos != name_.find( "FPix" ) ) 
    { subdet_ = PixelSubdetector::PixelEndcap;
      if( std::string::npos != name_.find( "pos", 6 ) ) side_ = Plus;
      else if( std::string::npos != name_.find( "neg", 6 ) ) side_ = Minus;
      else {
	std::cerr << " [PixelLayerNameParser::side] ** ERROR: something's wrong here.." << std::endl;
	side_ = SideError;
      }
    }
  else {
    std::cerr << " [PixelLayerNameParser::subdetector] ** ERROR: something's wrong here.." << std::endl;
  }

  // layer
  layer_ = atoi( name_.substr( 4, 1 ).c_str() );

}



///
/// check if we have a name string as expected
///
bool SeedMergerPixelLayer::isValidName( const std::string& name ) {

  const int layer = atoi( name.substr( 4, 1 ).c_str() );

  if( std::string::npos != name.find( "BPix" ) ) {
    if( layer > 0 && layer < 5 ) return true;
  }

  else if( std::string::npos != name.find( "FPix" ) ) {
    if( layer > 0 && layer < 10 ) {
      if( std::string::npos != name.find( "pos", 6 ) || std::string::npos != name.find( "neg", 6 ) ) return true;
    }

  }

  std::cerr << " [SeedMergerPixelLayer::isValidName] ** WARNING: invalid name: \"" << name << "\"." << std::endl;
  return false;

}



///
/// check if the layer or disk described by this object
/// is the one carrying the detector: detId
///
bool SeedMergerPixelLayer::isContainsDetector( const DetId& detId, const TrackerTopology *tTopo ) const {

  PixelSubdetector::SubDetector subdet = getSubdet();

  // same subdet?
  if( detId.subdetId() == subdet ) {
    
    // same barrel layer?
    if( PixelSubdetector::PixelBarrel == subdet ) {
      if (tTopo->pxbLayer(detId) == getLayerNumber()) {
	return true;
      }
    }
    
    // same endcap disk?
    else if( PixelSubdetector::PixelEndcap == subdet ) {
      
      if (tTopo->pxfDisk(detId) == getLayerNumber()) {
	if (tTopo->pxfSide(detId) == (unsigned)getSide()) {
	  return true;
	}
      }
    }
    
  }
  
  return false;
  
}


void QuadrupletSeedMerger::mySort(std::array<SeedingHitSet::ConstRecHitPointer, 4>& unsortedHits) {
  float radiiSq[4];
  for ( unsigned int iR=0; iR<4; iR++){
    GlobalPoint p1 = unsortedHits[iR]->globalPosition();
    radiiSq[iR]=( p1.x()*p1.x()+p1.y()*p1.y()); // no need to take the sqrt
  }
  SeedingHitSet::ConstRecHitPointer tempRHP;
  float tempFloat=0.;
  for ( unsigned int iR1=0; iR1<3; iR1++) {
    for ( unsigned int iR2=iR1+1; iR2<4; iR2++) {
      if (radiiSq[iR1]>radiiSq[iR2]) {
	tempRHP=unsortedHits[iR1];
	unsortedHits[iR1]=unsortedHits[iR2];
	unsortedHits[iR2]=tempRHP;
	tempFloat=radiiSq[iR1];
	radiiSq[iR1]=radiiSq[iR2];
	radiiSq[iR2]=tempFloat;
      }
    }
  }
}



