/*! \brief   Implementation of methods of TTTrackAlgorithm_trackletLB
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 *  \author Anders Ryd
 *  \author Emmanuele Salvati
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

#include "L1Trigger/TrackTrigger/interface/TTTrackAlgorithm_trackletLB.h"

template< >
void TTTrackAlgorithm_trackletLB< Ref_PixelDigi_ >::CreateSeeds( std::vector< TTTrack< Ref_PixelDigi_ > > &output,
                                                                 std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > > *outputSectorMap,
                                                                 edm::Handle< std::vector< TTStub< Ref_PixelDigi_ > > > &input ) const
{
  /// Prepare output
  output.clear();

  /// Map the Barrel Stubs per layer/rod
  std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > > stubBarrelMap;
  stubBarrelMap.clear();

  /// Map the Barrel Stubs per sector
  outputSectorMap->clear();

  typename std::vector< TTStub< Ref_PixelDigi_ > >::const_iterator inputIter;
  unsigned int j = 0; /// Counter needed to build the edm::Ptr to the TTStub

  for ( inputIter = input->begin();
        inputIter != input->end();
        ++inputIter )
  {
    /// Make the pointer to be put in the map and, later on, in the Track
    edm::Ptr< TTStub< Ref_PixelDigi_ > > tempStubPtr( input, j++ );

    /// Calculate Sector
    /// From 0 to nSectors-1
    /// Sector 0 centered on Phi = 0 and symmetric around it
    double stubPhi = TTTrackAlgorithm< Ref_PixelDigi_ >::theStackedTracker->findGlobalPosition( tempStubPtr.get() ).phi();
    stubPhi += M_PI/nSectors;
    if ( stubPhi < 0 )
    {
      stubPhi += 2*M_PI;
    }
    unsigned int thisSector = floor( 0.5*stubPhi*nSectors/M_PI );
    /// Calculate Wedge
    /// From 0 to nWedges-1
    /// 
    double stubEta = TTTrackAlgorithm< Ref_PixelDigi_ >::theStackedTracker->findGlobalPosition( tempStubPtr.get() ).eta();
    stubEta += 2.5; /// bring eta = -2.5 to 0
    //stubEta += 2.5/nWedges;

    /// Accept only stubs within -2.5, 2.5 range
    if ( stubEta < 0.0 || stubEta > 5.0 )
      continue;

    unsigned int thisWedge = floor( stubEta*nWedges/5.0 );

    /// Build the key to the map (by Sector / Wedge)
    std::pair< unsigned int, unsigned int > mapkey = std::make_pair( thisSector, thisWedge );

    StackedTrackerDetId detIdStub( inputIter->getDetId() );
    if ( detIdStub.isBarrel() )
    {
      /// If an entry already exists for this key, just add the stub
      /// to the vector, otherwise create the entry
      if ( stubBarrelMap.find( mapkey ) == stubBarrelMap.end() )
      {
        /// New entry
        std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > tempStubVec;
        tempStubVec.clear();
        tempStubVec.push_back( tempStubPtr );
        stubBarrelMap.insert( std::pair< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > > ( mapkey, tempStubVec ) );
      }
      else
      {
        /// Already existing entry
        stubBarrelMap[mapkey].push_back( tempStubPtr );
      }

      /// If an entry already exists for this Sector, just add the stub
      /// to the vector, otherwise create the entry
      if ( outputSectorMap->find( mapkey ) == outputSectorMap->end() )
      {
        /// New entry
        std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > tempStubVec;
        tempStubVec.clear();
        tempStubVec.push_back( tempStubPtr );
        outputSectorMap->insert( std::pair< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > > ( mapkey, tempStubVec ) );
      }
      else
      {
        /// Already existing entry
        outputSectorMap->find( mapkey )->second.push_back( tempStubPtr );
      }
    }
  }

  /// Loop over the map
  /// Create Seeds and map detectors
  typename std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > >::const_iterator mapIter;
  for ( mapIter = stubBarrelMap.begin();
        mapIter != stubBarrelMap.end();
        ++mapIter )
  {
    /// Here we have ALL the stubs in one single Hermetic ROD (if applicable) 
    std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > tempStubVec = mapIter->second;

    for ( unsigned int i = 0; i < tempStubVec.size(); i++ )
    {
      StackedTrackerDetId detId1( tempStubVec.at(i)->getDetId() );
      GlobalPoint pos1 = TTTrackAlgorithm< Ref_PixelDigi_ >::theStackedTracker->findAverageGlobalPosition( tempStubVec.at(i)->getClusterPtr(0).get() );

      for ( unsigned int k = i+1; k < tempStubVec.size(); k++ )
      {
        StackedTrackerDetId detId2( tempStubVec.at(k)->getDetId() );
        GlobalPoint pos2 = TTTrackAlgorithm< Ref_PixelDigi_ >::theStackedTracker->findAverageGlobalPosition( tempStubVec.at(k)->getClusterPtr(0).get() );
        /// Skip different rod pairs
        if ( detId1.iPhi() != detId2.iPhi() ) continue;
        /// Skip same layer pairs
        if ( detId1.iLayer()+1 != detId2.iLayer() ) continue;
        /// Skip off-hermetic-rod
        if ( detId1.iLayer()%2 == 0 ) continue;

        /// Perform standard trigonometric operations
        double deltaPhi = pos1.phi() - pos2.phi();
        if ( fabs(deltaPhi) >= M_PI )
        {
          if ( deltaPhi>0 )
            deltaPhi = deltaPhi - 2*M_PI;
          else
            deltaPhi = 2*M_PI + deltaPhi;
        }

        double distance = sqrt( pos2.perp2() + pos1.perp2() - 2*pos2.perp()*pos1.perp()*cos(deltaPhi) );
        double rInvOver2 = sin(deltaPhi)/distance; /// Sign is maintained to keep track of the charge

        /// Perform cut on Pt
        if ( fabs(rInvOver2) > mMagneticField*0.0015*0.5 ) continue;

        /// Calculate projected vertex
        /// NOTE: cotTheta0 = Pz/Pt
        double rhoPsi1 = asin( pos1.perp()*rInvOver2 )/rInvOver2;
        double rhoPsi2 = asin( pos2.perp()*rInvOver2 )/rInvOver2;
        double cotTheta0 = ( pos1.z() - pos2.z() ) / ( rhoPsi1 - rhoPsi2 );
        double z0 = pos2.z() - rhoPsi2 * cotTheta0;

        /// Perform projected vertex cut
        if ( fabs(z0) > 30.0 ) continue;

        /// Calculate direction at vertex
        double phi0 = pos2.phi() + asin( pos2.perp() * rInvOver2 );

        /// Calculate Pt
        double roughPt = fabs( mMagneticField*0.0015 / rInvOver2 );

        /// Create the Seed in the form of a Track and store it in the output
        std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > tempVec;
        tempVec.push_back( tempStubVec.at(i) );
        tempVec.push_back( tempStubVec.at(k) );
        TTTrack< Ref_PixelDigi_ > tempTrack( tempVec );
        tempTrack.setRInv( 2*rInvOver2 );
        tempTrack.setMomentum( GlobalVector( roughPt*cos(phi0),
                                             roughPt*sin(phi0),
                                             roughPt*cotTheta0 ) );
        tempTrack.setVertex( GlobalPoint( 0, 0, z0 ) );
        tempTrack.setSector( mapIter->first.first );
        tempTrack.setWedge( mapIter->first.second );
        output.push_back( tempTrack );
      }
    } /// End of double loop over pairs of stubs

  } /// End of loop over map elements
}

/// Match a Stub to a Seed/Track
template< >
void TTTrackAlgorithm_trackletLB< Ref_PixelDigi_ >::AttachStubToSeed( TTTrack< Ref_PixelDigi_ > &seed,
                                                                      edm::Ptr< TTStub< Ref_PixelDigi_ > > &candidate ) const
{
  /// Get the track Stubs
  std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > theStubs = seed.getStubPtrs();

  /// Compare SuperLayers
  unsigned int seedSuperLayer = (unsigned int)(( StackedTrackerDetId( theStubs.at(0)->getDetId() ).iLayer() + 1 )/2 );
  unsigned int targetSuperLayer = (unsigned int)(( StackedTrackerDetId( candidate->getDetId() ).iLayer() + 1 )/2 );

  if ( seedSuperLayer == targetSuperLayer )
    return;

  /// Skip if the seed and the stub are in the same
  /// SuperLayer in case of SL 3-4-5
  if ( seedSuperLayer > 2 && targetSuperLayer > 2 )
    return;

  unsigned int seedSL = ( seedSuperLayer > 2 ) ? 3 : seedSuperLayer;
  unsigned int targSL = ( targetSuperLayer > 2 ) ? 3 : targetSuperLayer;

  /// Check that the candidate is NOT the one under examination
  for ( unsigned int i = 0; i < theStubs.size(); i++ )
  {
    if ( theStubs.at(i) == candidate )
      return;
  }

  /// Get the track momentum and propagate it
  GlobalVector curMomentum = seed.getMomentum();
  GlobalPoint curVertex = seed.getVertex();
  double curRInv = seed.getRInv();

  /// Get the candidate Stub position
  GlobalPoint candPos = TTTrackAlgorithm< Ref_PixelDigi_ >::theStackedTracker->findAverageGlobalPosition( candidate->getClusterPtr(0).get() );

  /// Propagate
  double propPsi = asin( candPos.perp() * 0.5 * curRInv );
  double propRhoPsi = 2 * propPsi / curRInv;
  double propPhi = curMomentum.phi() - propPsi;
  double propZ = curVertex.z() + propRhoPsi * tan( M_PI_2 - curMomentum.theta() );

  /// Calculate displacement
  /// Perform standard trigonometric operations
  double deltaPhi = propPhi - candPos.phi();
  if ( fabs(deltaPhi) >= M_PI )
  {
    if ( deltaPhi>0 )
      deltaPhi = deltaPhi - 2*M_PI;
    else
      deltaPhi = 2*M_PI + deltaPhi;
  }
  double deltaRPhi = fabs( deltaPhi * candPos.perp() );
  double deltaZ = fabs( propZ - candPos.z() );

  /// First get the vector corresponding to the seed SL
  /// Then get in this vector the entry corresponding to the targer SL
  if ( deltaRPhi < (tableRPhi.at(seedSL)).at(targSL) && deltaZ < (tableZ.at(seedSL)).at(targSL) )
  //if ( deltaRPhi < 4 && deltaZ < 8 )
  {
    seed.addStubPtr( candidate );
  }
}

