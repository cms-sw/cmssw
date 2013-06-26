
#include "RecoPixelVertexing/PixelTriplets/interface/QuadrupletSeedMerger.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include <time.h>

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

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  es.get<IdealGeometryRecord>().get(tTopoHand);
  const TrackerTopology *tTopo=tTopoHand.product();
  
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

  //std::vector<bool> phiEtaClose(nInputTriplets*nInputTriplets,true);

  //  for (unsigned int t1=0; t1<nInputTriplets-1; t1++) {
  //  for (unsigned int t2=t1+1; t2<nInputTriplets; t2++) {
  //    if( fabs( phiEtaCache[t1].second - phiEtaCache[t2].second ) > 0.05 ) {
  //	phiEtaClose[t1*nInputTriplets+t2]=false;
  //	phiEtaClose[t2*nInputTriplets+t1]=false;
  //	continue;
  //   }
  //   double temp = fabs( phiEtaCache[t1].first - phiEtaCache[t2].first );
  //   if( (temp > 0.15) && (temp <6.133185) ) {
  //phiEtaClose[t1*nInputTriplets+t2]=false;
  //phiEtaClose[t2*nInputTriplets+t1]=false;
  //  }
  //}
  //}
 
  std::vector<unsigned int> t1List;
  std::vector<unsigned int> t2List;
  for (unsigned int t1=0; t1<nInputTriplets-1; t1++) {
    for (unsigned int t2=t1+1; t2<nInputTriplets; t2++) {
      if( fabs( phiEtaCache[t1].second - phiEtaCache[t2].second ) > 0.05 ) 
  	continue;
      double temp = fabs( phiEtaCache[t1].first - phiEtaCache[t2].first );
      if( (temp > 0.15) && (temp <6.133185) ) {
	continue;
      }
      t1List.push_back(t1);
      t2List.push_back(t2);
    }
  }

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

	    //    if ( !phiEtaClose[t1*nInputTriplets+t2] ) continue;

	    // do both triplets have shared hits on these two layers?
	    if( isTripletsShareHitsOnLayers( (tripletCache[t1]), (tripletCache[t2]), 
					     currentLayers[s1],
					     currentLayers[s2], sharedHits, tTopo ) ) {

	      // are the remaining hits on different layers?
	      if( isMergeableHitsInTriplets( (tripletCache[t1]), (tripletCache[t2]), 
					     currentLayers[nonSharedLayerNums[0]],
					     currentLayers[nonSharedLayerNums[1]], nonSharedHits, tTopo ) ) {


		std::vector<TransientTrackingRecHit::ConstRecHitPointer> unsortedHits=mySort(sharedHits.first,
											     sharedHits.second,
											     nonSharedHits.first,
											     nonSharedHits.second);

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
  // leak????
  SeedCreator* seedCreator = SeedCreatorFactory::get()->create( creatorName, creatorPSet );
  seedCreator->init(region, es, 0);
  for ( unsigned int i=0; i< quadrupletHitSets.size(); i++) {
    // add trajectory seed to result collection
    seedCreator->makeSeed( theResult, quadrupletHitSets[i]);
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
  const double eta = acos( (z2 - z1) / sqrt( pow( x2 - x1, 2. ) + pow( y2 - y1, 2. ) + pow( z2 - z1, 2. ) ) );

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
bool QuadrupletSeedMerger::isValidQuadruplet( std::vector<TransientTrackingRecHit::ConstRecHitPointer> &quadruplet, const std::vector<SeedMergerPixelLayer>& layers,
					      const TrackerTopology *tTopo) const {

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
/// check if both triplets share a hit
/// on either of the sharedLayers
/// and return both hits (unsorted)
///
bool QuadrupletSeedMerger::isTripletsShareHitsOnLayers( const SeedingHitSet& firstTriplet, const SeedingHitSet& secondTriplet, 
							const SeedMergerPixelLayer &shared1, const SeedMergerPixelLayer &shared2,
							std::pair<TransientTrackingRecHit::ConstRecHitPointer,TransientTrackingRecHit::ConstRecHitPointer>& hits,
							const TrackerTopology *tTopo ) const {

  bool isSuccess1[2],isSuccess2[2];
  isSuccess1[0]=false;
  isSuccess1[1]=false;
  isSuccess2[0]=false;
  isSuccess2[1]=false;

  std::pair<TransientTrackingRecHit::ConstRecHitPointer,TransientTrackingRecHit::ConstRecHitPointer> hitsTriplet1, hitsTriplet2;

  // check if firstTriplet and secondTriplet have hits on sharedLayers
  for( unsigned int index = 0; index < 3; ++index )
    { // first triplet
      if( ! firstTriplet[index]->isValid() ) return false; // catch invalid TTRH pointers (tbd: erase triplet)
      bool firsthit(false); // Don't look in second layer if found in first
      DetId const& thisDetId = firstTriplet[index]->hit()->geographicalId();
      
      if( ! isSuccess1[0] ) { // first triplet on shared layer 1
	if( shared1.isContainsDetector( thisDetId, tTopo ) ) {
	  isSuccess1[0] = true;
	  firsthit = true;
	  hitsTriplet1.first = firstTriplet[index];
	}
      }
      
      if ( (! firsthit) && (! isSuccess1[1] ) && ((index !=3) || isSuccess1[0]) ) { // first triplet on shared layer 2
	if( shared2.isContainsDetector( thisDetId, tTopo ) ) {
	  isSuccess1[1] = true;
	  hitsTriplet1.second = firstTriplet[index];
	}
      } 
    }
  
  if ( isSuccess1[0] && isSuccess1[1]) { // Don't do second triplet if first unsuccessful
    for( unsigned int index = 0; index < 3; ++index )
      { // second triplet
	if( ! secondTriplet[index]->isValid() ) { return false; } // catch invalid TTRH pointers (tbd: erase triplet)
	bool firsthit(false); // Don't look in second layer if found in first
	DetId const& thisDetId = secondTriplet[index]->hit()->geographicalId();
	
	if( ! isSuccess2[0] ) { // second triplet on shared layer 1
	  if( shared1.isContainsDetector( thisDetId, tTopo ) ) {
	    isSuccess2[0] = true;
	    firsthit = true;
	    hitsTriplet2.first = secondTriplet[index];
	  }
	}
	
	if( (! firsthit) && (! isSuccess2[1]) && ((index !=3) || isSuccess2[0]) ) { // second triplet on shared layer 2
	  if( shared2.isContainsDetector( thisDetId, tTopo ) ) {
	    isSuccess2[1] = true;
	    hitsTriplet2.second = secondTriplet[index];
	  }
	}
      }
    
    // check if these hits are pairwise equal
    if( isSuccess2[0] && isSuccess2[1] ) {
      if( isEqual( hitsTriplet1.first->hit(),  hitsTriplet2.first->hit()  ) &&
	  isEqual( hitsTriplet1.second->hit(), hitsTriplet2.second->hit() )    ) {
	
	// copy to output, take triplet1 since they're equal anyway
	hits.first  = hitsTriplet1.first;
	hits.second = hitsTriplet1.second;
	return true;
      }
    }
  }
  
  // empty output, careful
  return false;
  
}



///
/// check if the triplets have hits on the nonSharedLayers
/// triplet1 on layer1 && triplet2 on layer2, or vice versa,
/// and return the hits on those layers (unsorted)
bool QuadrupletSeedMerger::isMergeableHitsInTriplets( const SeedingHitSet& firstTriplet, const SeedingHitSet& secondTriplet, 
						      const SeedMergerPixelLayer &nonShared1, const SeedMergerPixelLayer &nonShared2,
						      std::pair<TransientTrackingRecHit::ConstRecHitPointer,TransientTrackingRecHit::ConstRecHitPointer>& hits,
						      const TrackerTopology *tTopo ) const {

  // check if firstTriplet and secondTriplet have hits on sharedLayers
  for( unsigned int index1 = 0; index1 < 3; ++index1 ) {
    
    { // first triplet on non-shared layer 1
      DetId const& aDetId = firstTriplet[index1]->hit()->geographicalId();
      if( nonShared1.isContainsDetector( aDetId, tTopo ) ) {
	
	// look for hit in other (second) triplet on other layer
	for( unsigned int index2 = 0; index2 < 3; ++index2 ) {
	  
	  DetId const& anotherDetId = secondTriplet[index2]->hit()->geographicalId();
	  if( nonShared2.isContainsDetector( anotherDetId, tTopo ) ) {
	    
	    // ok!
	    hits.first  = firstTriplet[index1];
	    hits.second = secondTriplet[index2];
	    return true;
	    
	  }
	}
      }
    }

    // and vice versa..

    { // second triplet on non-shared layer 1
      DetId const& aDetId = secondTriplet[index1]->hit()->geographicalId();
      if( nonShared1.isContainsDetector( aDetId, tTopo ) ) {
	
	// look for hit in other (second) triplet on other layer
	for( unsigned int index2 = 0; index2 < 3; ++index2 ) {
	  
	  DetId const& anotherDetId = firstTriplet[index2]->hit()->geographicalId();
	  if( nonShared2.isContainsDetector( anotherDetId, tTopo ) ) {
	    
	    // ok!
	    hits.first  = firstTriplet[index1];
	    hits.second = secondTriplet[index2];
	    return true;
	    
	  }
	}
      }
    }

  } // for( index1

  return false;
    
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



