#include "RecoPixelVertexing/PixelTriplets/interface/QuadrupletSeedMerger.h"
#include "RecoPixelVertexing/PixelTriplets/interface/KDTreeLinkerAlgo.h"
#include "RecoPixelVertexing/PixelTriplets/interface/KDTreeLinkerTools.h"

#include "DataFormats/GeometryVector/interface/Pi.h"

#include <time.h>
#include <cmath>
#include <algorithm>

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
                       const std::pair<TransientTrackingRecHit::ConstRecHitPointer,TransientTrackingRecHit::ConstRecHitPointer>& hits) {
    DetId firstHitId = hits.first->geographicalId();
    DetId secondHitId = hits.second->geographicalId();
    return ((layer1.isContainsDetector(firstHitId) &&
             layer2.isContainsDetector(secondHitId)) ||
            (layer1.isContainsDetector(secondHitId) &&
             layer2.isContainsDetector(firstHitId)));
  }
}



///
///
///
QuadrupletSeedMerger::QuadrupletSeedMerger( ) {

  // by default, do not..
  // ..merge triplets
  isMergeTriplets_ = false;
  // ..add remaining triplets
  isAddRemainingTriplets_ = false;

  // default is the layer list from plain quadrupletseedmerging_cff
  // unless configured contrarily via setLayerListName()
  layerListName_ = std::string( "PixelSeedMergerQuadruplets" );
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
  
  // the list of layers on which quadruplets should be formed
  edm::ESHandle<SeedingLayerSetsBuilder> layerBuilder;
  es.get<TrackerDigiGeometryRecord>().get( layerListName_.c_str(), layerBuilder );
  theLayerSets_ = layerBuilder->layers( es ); // this is a vector<vector<SeedingLayer> >

  
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
  std::pair<TransientTrackingRecHit::ConstRecHitPointer,TransientTrackingRecHit::ConstRecHitPointer> sharedHits;
  std::pair<TransientTrackingRecHit::ConstRecHitPointer,TransientTrackingRecHit::ConstRecHitPointer> nonSharedHits;

  // k-d tree, indices are (th)eta, phi
  // build the tree
  std::vector<KDTreeNodeInfo<unsigned int> > nodes; // re-use for searching too
  nodes.reserve(2*nInputTriplets);
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
    nodes.clear();
    kdtree.search(box, nodes);
    if(nodes.empty())
      continue;

    const SeedingHitSet& tr1 = tripletCache[t1];
    const TrackingRecHit *tr1h[3] = {tr1[0]->hit(), tr1[1]->hit(), tr1[2]->hit()};

    for(size_t i=0; i<nodes.size(); ++i) {
      unsigned int t2 = nodes[i].data;
      if(t1 >= t2)
        continue;

      // Ensure here that the triplet pairs share two hits.
      const SeedingHitSet& tr2 = tripletCache[t2];
      const TrackingRecHit *tr2h[3] = {tr2[0]->hit(), tr2[1]->hit(), tr2[2]->hit()};

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
          if(isEqual(tr1h[i], tr2h[j])) {
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
          if(isEqual(tr1h[2], tr2h[j])) {
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

	    //    if ( !phiEtaClose[t1*nInputTriplets+t2] ) continue;

            // are the shared hits on these two layers?
            if(areHitsOnLayers(currentLayers[s1], currentLayers[s2], sharedHits)) {
              short t2NonShared = t2NonSharedHitList[t12];
              const SeedingHitSet& secondTriplet = tripletCache[t2];
              nonSharedHits.first = firstTriplet[t1NonShared];
              nonSharedHits.second = secondTriplet[t2NonShared];

	      // are the remaining hits on different layers?
              if(areHitsOnLayers(currentLayers[nonSharedLayerNums[0]], currentLayers[nonSharedLayerNums[1]], nonSharedHits)) {
		std::vector<TransientTrackingRecHit::ConstRecHitPointer> unsortedHits=mySort(sharedHits.first,
											     sharedHits.second,
											     nonSharedHits.first,
											     nonSharedHits.second);

		//start here with old addtoresult
		if( isValidQuadruplet( unsortedHits, currentLayers ) ) {
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
								    const edm::EventSetup& es,
                                                                    const edm::ParameterSet& cfg ) {

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

    std::vector<TransientTrackingRecHit::RecHitPointer> recHitPointers;

    // loop RecHits
    const TrajectorySeed::range theHitsRange = aTrajectorySeed->recHits();
    for( edm::OwnVector<TrackingRecHit>::const_iterator aHit = theHitsRange.first;
	 aHit < theHitsRange.second; ++aHit ) {
      
      // this is a collection of: ReferenceCountingPointer< TransientTrackingRecHit> 
      recHitPointers.push_back( theTTRHBuilder_->build( &(*aHit) ) );

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
  edm::ParameterSet creatorPSet = cfg.getParameter<edm::ParameterSet>("SeedCreatorPSet");
  std::string const& creatorName = creatorPSet.getParameter<std::string>( "ComponentName" );
  SeedCreator* seedCreator = SeedCreatorFactory::get()->create( creatorName, creatorPSet );

  for ( unsigned int i=0; i< quadrupletHitSets.size(); i++) {
    // add trajectory seed to result collection
    seedCreator->trajectorySeed( theResult, quadrupletHitSets[i], region, es, 0 );
  }

  return theResult;
  
}



///
///
///
bool QuadrupletSeedMerger::isEqual( const TrackingRecHit* hit1, const TrackingRecHit* hit2 ) const {

  const double epsilon = 0.00001;
  
  DetId det1 =  hit1->geographicalId(), det2 =  hit2->geographicalId();
  if (det1 == det2) { 
    LocalPoint lp1 = hit1->localPosition(), lp2 = hit2->localPosition();
    if( ( fabs( lp1.x() - lp2.x() ) < epsilon ) &&
	( fabs( lp1.y() - lp2.y() ) < epsilon ) ) {
      return true;
    }
    
  }
  return false;
  
}



///
///
///
std::pair<double,double> QuadrupletSeedMerger::calculatePhiEta( SeedingHitSet const& nTuplet ) const {

//   if( nTuplet.size() < 3 ) {
//     std::cerr << " [QuadrupletSeedMerger::calculatePhiEta] ** ERROR: nTuplet has less than 3 hits" << std::endl;
//     throw; // tbr.
//   }

  const TrackingRecHit* hit1 = nTuplet[0]->hit();
  const GeomDet* geomDet1 = theTrackerGeometry_->idToDet( hit1->geographicalId() );

  const TrackingRecHit* hit2 = nTuplet[1]->hit();
  const GeomDet* geomDet2 = theTrackerGeometry_->idToDet( hit2->geographicalId() );

  GlobalPoint p1=geomDet1->toGlobal( hit1->localPosition() );
  GlobalPoint p2=geomDet2->toGlobal( hit2->localPosition() );

  const double x1 = p1.x();
  const double x2 = p2.x();
  const double y1 = p1.y();
  const double y2 = p2.y();
  const double z1 = p1.z();
  const double z2 = p2.z();

  const double phi = atan2( x2 - x1, y2 -y1 );
  const double eta = acos( (z2 - z1) / sqrt( pow( x2 - x1, 2. ) + pow( y2 - y1, 2. ) + pow( z2 - z1, 2. ) ) ); // this is theta angle in reality

  std::pair<double,double> retVal;
  retVal=std::make_pair (phi,eta);
  return retVal;
  //return std::make_pair<double,double>( phi, eta );
  
}



///
///
///
void QuadrupletSeedMerger::printHit( const TransientTrackingRecHit::ConstRecHitPointer& aRecHitPointer ) const {

  printHit( aRecHitPointer->hit() );

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
void QuadrupletSeedMerger::setLayerListName( std::string layerListName ) {
  layerListName_ = layerListName;
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
bool QuadrupletSeedMerger::isValidQuadruplet( std::vector<TransientTrackingRecHit::ConstRecHitPointer> &quadruplet, const std::vector<SeedMergerPixelLayer>& layers ) const {

  const unsigned int quadrupletSize = quadruplet.size();

  // basic size test..
  if( quadrupletSize != layers.size() ) {
    std::cout << " [QuadrupletSeedMerger::isValidQuadruplet] ** WARNING: size mismatch: "
	      << quadrupletSize << "/" << layers.size() << std::endl;
    return false;
  }

  // go along layers and check if all (parallel) quadruplet hits match
  for( unsigned int index = 0; index < quadrupletSize; ++index ) {
    if( ! layers[index].isContainsDetector( quadruplet[index]->geographicalId() ) ) {
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
    if( layer > 0 && layer < 4 ) {
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
bool SeedMergerPixelLayer::isContainsDetector( const DetId& detId ) const {

  PixelSubdetector::SubDetector subdet = getSubdet();

  // same subdet?
  if( detId.subdetId() == subdet ) {
    
    // same barrel layer?
    if( PixelSubdetector::PixelBarrel == subdet ) {
      const PXBDetId cmssw_numbering(detId);
      if (cmssw_numbering.layer() == getLayerNumber()) {
	return true;
      }
    }
    
    // same endcap disk?
    else if( PixelSubdetector::PixelEndcap == subdet ) {
      PXFDetId px_numbering(detId);
      if (px_numbering.disk() == getLayerNumber()) {
	if (px_numbering.side() == (unsigned)getSide()) {
	  return true;
	}
      }
    }
    
  }
  
  return false;
  
}


std::vector<TransientTrackingRecHit::ConstRecHitPointer> QuadrupletSeedMerger::mySort(TransientTrackingRecHit::ConstRecHitPointer &h1,
										      TransientTrackingRecHit::ConstRecHitPointer &h2,
										      TransientTrackingRecHit::ConstRecHitPointer &h3,
										      TransientTrackingRecHit::ConstRecHitPointer &h4) {
  // create an intermediate vector with all hits
  std::vector<TransientTrackingRecHit::ConstRecHitPointer> unsortedHits;
  unsortedHits.reserve(4);
  unsortedHits.push_back( h1);
  unsortedHits.push_back( h2);
  unsortedHits.push_back( h3);
  unsortedHits.push_back( h4);
  
  float radiiSq[4];
  for ( unsigned int iR=0; iR<4; iR++){
    const GeomDet* geom1=theTrackerGeometry_->idToDet( unsortedHits[iR]->hit()->geographicalId() );
    GlobalPoint p1=geom1->toGlobal(  unsortedHits[iR]->hit()->localPosition() );
    radiiSq[iR]=( p1.x()*p1.x()+p1.y()*p1.y()); // no need to take the sqrt
  }
  TransientTrackingRecHit::ConstRecHitPointer tempRHP;
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
  return unsortedHits;
}



