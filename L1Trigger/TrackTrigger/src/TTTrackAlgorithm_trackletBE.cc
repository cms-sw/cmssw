/*! \brief   Implementation of methods of TTTrackAlgorithm_trackletBE
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 *  \author Anders Ryd
 *  \author Emmanuele Salvati
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

#include "L1Trigger/TrackTrigger/interface/TTTrackAlgorithm_trackletBE.h"

/// Create Seeds
template< >
void TTTrackAlgorithm_trackletBE< Ref_PixelDigi_ >::CreateSeeds( std::vector< TTTrack< Ref_PixelDigi_ > > &output,
                                                                 std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > > *outputSectorMap,
                                                                 edm::Handle< std::vector< TTStub< Ref_PixelDigi_ > > > &input ) const
{
  /// STEP 0
  /// Prepare output
  output.clear();
  outputSectorMap->clear();

  /// Map the Stubs per Sector/Wedge for seeding purposes
  std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > > *stubBarrelMap;
  stubBarrelMap = new std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > >();
  stubBarrelMap->clear();
  std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > > *stubEndcapMap;
  stubEndcapMap = new std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > >();
  stubEndcapMap->clear();

  /// STEP 1
  /// Create the maps
  /// Loop over input stubs
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
    double stubEta = TTTrackAlgorithm< Ref_PixelDigi_ >::theStackedTracker->findGlobalPosition( tempStubPtr.get() ).eta();
    stubEta += 2.5; /// Bring eta = -2.5 to 0
    if ( stubEta < 0.0 || stubEta > 5.0 )
    {
      /// Accept only stubs within -2.5, 2.5 range
      continue;
    }
    unsigned int thisWedge = floor( stubEta*nWedges/5.0 );

    /// Build the key to the map (by Sector / Wedge)
    std::pair< unsigned int, unsigned int > mapKey = std::make_pair( thisSector, thisWedge );

    /// Do the same but separating Barrel and Endcap Stubs
    /// NOTE this is internal to build seeds
    /// The previous one goes into the output
    StackedTrackerDetId detIdStub( inputIter->getDetId() );
    if ( detIdStub.isBarrel() )
    {
      /// If an entry already exists for this key, just add the stub
      /// to the vector, otherwise create the entry
      if ( stubBarrelMap->find( mapKey ) == stubBarrelMap->end() )
      {
        /// New entry
        std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > tempStubVec;
        tempStubVec.clear();
        tempStubVec.push_back( tempStubPtr );
        stubBarrelMap->insert( std::pair< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > > ( mapKey, tempStubVec ) );
      }
      else
      {
        /// Already existing entry
        stubBarrelMap->find( mapKey )->second.push_back( tempStubPtr );
      }
    }
    else if ( detIdStub.isEndcap() )
    {
      /// If an entry already exists for this key, just add the stub
      /// to the vector, otherwise create the entry
      if ( stubEndcapMap->find( mapKey ) == stubEndcapMap->end() )
      {
        /// New entry
        std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > tempStubVec;
        tempStubVec.clear();
        tempStubVec.push_back( tempStubPtr );
        stubEndcapMap->insert( std::pair< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > > ( mapKey, tempStubVec ) );
      }
      else
      {
        /// Already existing entry
        stubEndcapMap->find( mapKey )->second.push_back( tempStubPtr );
      }
    } /// End of Barrel-Endcap switch
  } /// End of loop over input stubs

  /// STEP 2
  /// Create the seeds

  /// At this point, all the maps are available
  /// there are stubBarrelMap and stubEndcapMap
  /// Then, tracklets must be found
  /// the idea is to loop over neigbor sectors

  /// BARREL-BARREL
  /// BARREL-ENDCAP
  /// Loop over map elements
  typename std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > >::const_iterator mapIter;
  typename std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > >::const_iterator anotherMapIter;
  for ( mapIter = stubBarrelMap->begin();
        mapIter != stubBarrelMap->end();
        ++mapIter )
  {
    /// Get the list of stubs in the present Sector/Wedge
    std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > tempBarrelStubVec0 = mapIter->second;

    /// Get the Sector/Wedge of the present list
    unsigned int curSector0 = mapIter->first.first + this->ReturnNumberOfSectors(); /// This is to use the %nSectors later
    unsigned int curWedge0 = mapIter->first.second;

    /// Loop over the sector and its two neighbors
    for ( unsigned int iSector = 0; iSector < 2; iSector++ )
    {
      for ( unsigned int iWedge = 0; iWedge < 2; iWedge++)
      {
        /// Find the correct sector index
        unsigned int curSector = ( curSector0 + iSector -1 )%(this->ReturnNumberOfSectors());
        int curWedge = curWedge0 + iWedge - 1;
        if ( curWedge < 0 || curWedge >= (int)(this->ReturnNumberOfWedges()) )
          continue;

        /// Now we are in the correct Sector/Wedge pair
        /// Get also the stubs of this Sector/Wedge
        anotherMapIter = stubBarrelMap->find( std::make_pair( curSector, curWedge ) );
        std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > tempBarrelStubVec1;
        if ( anotherMapIter != stubBarrelMap->end() )
        {
          tempBarrelStubVec1 = anotherMapIter->second;
        }

        /// Include also the Endcap stubs (mixed seeding)
        anotherMapIter = stubEndcapMap->find( std::make_pair( curSector, curWedge ) );
        std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > tempEndcapStubVec1; 
        if ( anotherMapIter != stubEndcapMap->end() )
        {
          tempEndcapStubVec1 = anotherMapIter->second;
        }

        /// Double loop over all pairs of stubs
        for ( unsigned int i = 0; i < tempBarrelStubVec0.size(); i++ )
        {
          StackedTrackerDetId detId1( tempBarrelStubVec0.at(i)->getDetId() );
          GlobalPoint pos1 = TTTrackAlgorithm< Ref_PixelDigi_ >::theStackedTracker->findAverageGlobalPosition( tempBarrelStubVec0.at(i)->getClusterPtr(0).get() );
          double rho1 = pos1.perp();
          double phi1 = pos1.phi();
          double z1 = pos1.z();

          /// Layer-Disk-Ring constraint
          if ( detId1.iLayer() > 4 )
            continue;

          bool barrelSeed2S = !( TTTrackAlgorithm< Ref_PixelDigi_ >::theStackedTracker->isPSModule( detId1 ) );

          /// Find the index of the first element of the nested loop
          unsigned int startIndex = 0;
          if ( curSector == curSector0%(this->ReturnNumberOfSectors()) &&
               curWedge == (int)curWedge0 ) /// If they are in the same Sector/Wedge the loop can be simplified
          {
            /// This means tempBarrelStubVec1 == tempBarrelStubVec0
            startIndex = i+1;
          }

          /// Loop over other barrel stubs
          for ( unsigned int k = startIndex; k < tempBarrelStubVec1.size(); k++ )
          {
            StackedTrackerDetId detId2( tempBarrelStubVec1.at(k)->getDetId() );
            GlobalPoint pos2 = TTTrackAlgorithm< Ref_PixelDigi_ >::theStackedTracker->findAverageGlobalPosition( tempBarrelStubVec1.at(k)->getClusterPtr(0).get() );
            double rho2 = pos2.perp();
            double phi2 = pos2.phi();
            double z2 = pos2.z();

            /// Skip same layer pairs
            /// Skip pairs with distance larger than 1 layer
            //if (( detId2.iLayer() != detId1.iLayer() + 1 ) && ( detId1.iLayer() != detId2.iLayer() + 1 ))
            if ( detId2.iLayer() != detId1.iLayer() + 1 )
              continue;

            /// Safety cross-check
            if ( rho1 > rho2 )
            {
              //std::cerr << "TTTrackAlgorithm_exactBarrelEndcap::CreateSeeds()" << std::endl;
              //std::cerr << "   A L E R T ! pos1.perp() > pos2.perp() in Barrel-Barrel tracklet" << std::endl;
              continue;
            }

            /// Apply cosine theorem to find 1/(2R) = rInvOver2
#include "L1Trigger/TrackTrigger/src/TTTrackAlgorithm_trackletBE_SeedTrigonometry.icc"

            /// Perform cut on Pt
            if ( fabs(rInvOver2) > mMagneticField*0.0015*0.5 )
              continue;

            /// Calculate tracklet parameters with helix model
            /// roughPt, z0, cotTheta0, phi0
#include "L1Trigger/TrackTrigger/src/TTTrackAlgorithm_trackletBE_SeedParameters.icc"

            /// Correct for seeds in 2S Barrel layers
            if ( barrelSeed2S )
            {
              if ( fabs( z1 - z2 ) < 10 )
              {
                z0 = 0;
                cotTheta0 = z1 / rhoPsi1;
              }
            }

            /// Perform projected vertex cut
            if ( fabs(z0) > 30.0 )
              continue;

            /// Create the Seed in the form of a Track and store it in the output
            std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > tempVec;
            tempVec.push_back( tempBarrelStubVec0.at(i) );
            tempVec.push_back( tempBarrelStubVec1.at(k) );
            TTTrack< Ref_PixelDigi_ > tempTrack( tempVec );
            tempTrack.setRInv( 2*rInvOver2 );
            tempTrack.setMomentum( GlobalVector( roughPt*cos(phi0),
                                                 roughPt*sin(phi0),
                                                 roughPt*cotTheta0 ) );
            tempTrack.setVertex( GlobalPoint( 0, 0, z0 ) );
            tempTrack.setSector( mapIter->first.first );
            tempTrack.setWedge( mapIter->first.second );
            output.push_back( tempTrack );
          } /// End of loop over other barrel stubs

          /// Loop over endcap stubs in the same sector (mixed seeding)
          for ( unsigned int k = 0; k < tempEndcapStubVec1.size(); k++ )
          {
            StackedTrackerDetId detId2( tempEndcapStubVec1.at(k)->getDetId() );
            GlobalPoint pos2 = TTTrackAlgorithm< Ref_PixelDigi_ >::theStackedTracker->findAverageGlobalPosition( tempEndcapStubVec1.at(k)->getClusterPtr(0).get() );
            double rho2 = pos2.perp();
            double phi2 = pos2.phi();
            double z2 = pos2.z();

            bool endcapSeedPS = TTTrackAlgorithm< Ref_PixelDigi_ >::theStackedTracker->isPSModule( detId2 );

            /// Skip non-PS in endcaps for the mixed seeding
            if ( !endcapSeedPS )
            {}//  continue;

            /// Layer-Disk-Ring constraint
            if ( detId2.iDisk() > 1 )
              continue;

            if ( detId2.iRing() > 11 )
              continue;

            /// Safety cross-check
            if ( rho1 > rho2 )
            {
              //std::cerr << "TTTrackAlgorithm_trackletBE::CreateSeeds()" << std::endl;
              //std::cerr << "   A L E R T ! pos1.perp() > pos2.perp() in Barrel-Endcap tracklet" << std::endl;
              continue;
            }

            /// Apply cosine theorem to find 1/(2R) = rInvOver2
#include "L1Trigger/TrackTrigger/src/TTTrackAlgorithm_trackletBE_SeedTrigonometry.icc"

            /// Perform cut on Pt
            if ( fabs(rInvOver2) > mMagneticField*0.0015*0.5 )
              continue;

            /// Calculate tracklet parameters with helix model
            /// roughPt, z0, cotTheta0, phi0
#include "L1Trigger/TrackTrigger/src/TTTrackAlgorithm_trackletBE_SeedParameters.icc"

            /// Correct for Endcap 2S in the seed
            if ( !endcapSeedPS )
            {
              if ( fabs( z1 - z2 ) < 10 )
              {
                z0 = 0;
                cotTheta0 = z1 / rhoPsi1;
              }
            }

            /// Perform projected vertex cut
            if ( fabs(z0) > 30.0 )
              continue;

            /// Create the Seed in the form of a Track and store it in the output
            std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > tempVec;
            tempVec.push_back( tempBarrelStubVec0.at(i) );
            tempVec.push_back( tempEndcapStubVec1.at(k) );
            TTTrack< Ref_PixelDigi_ > tempTrack( tempVec );
            tempTrack.setRInv( 2*rInvOver2 );
            tempTrack.setMomentum( GlobalVector( roughPt*cos(phi0),
                                                 roughPt*sin(phi0),
                                                 roughPt*cotTheta0 ) );
            tempTrack.setVertex( GlobalPoint( 0, 0, z0 ) );
            tempTrack.setSector( mapIter->first.first );
            tempTrack.setWedge( mapIter->first.second );
            output.push_back( tempTrack );
          } /// End of loop over endcap stubs in the same sector (mixed seeding)

        } /// End of double loop over pairs of stubs
      }
    } /// End of loop over neighbor Sectors/Wedges
  } /// End of loop over map elements

  /// ENDCAP-ENDCAP
  /// Loop over the map
  /// Create Seeds in Endcap
  for ( mapIter = stubEndcapMap->begin();
        mapIter != stubEndcapMap->end();
        ++mapIter )
  {
    /// Here we have ALL the stubs in one single Sector/Wedge in the Endcap
    std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > tempEndcapStubVec0 = mapIter->second;

    /// Get the Sector/Wedge of the present list
    unsigned int curSector0 = mapIter->first.first + this->ReturnNumberOfSectors(); /// This is to use the %nSectors later
    unsigned int curWedge0 = mapIter->first.second;

    /// Loop over the sector and its two neighbors
    for ( unsigned int iSector = 0; iSector < 2; iSector++ )
    {
      for ( unsigned int iWedge = 0; iWedge < 2; iWedge++)
      {
        /// Find the correct sector index
        unsigned int curSector = ( curSector0 + iSector -1 )%(this->ReturnNumberOfSectors());
        int curWedge = curWedge0 + iWedge - 1;
        if ( curWedge < 0 || curWedge >= (int)(this->ReturnNumberOfWedges()) )
          continue;

        /// Now we are in the correct Sector/Wedge pair
        /// Get also the stubs of this Sector/Wedge
        anotherMapIter = stubEndcapMap->find( std::make_pair( curSector, curWedge ) );
        std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > tempEndcapStubVec1;
        if ( anotherMapIter != stubEndcapMap->end() )
        {
          tempEndcapStubVec1 = anotherMapIter->second;
        }

        /// Double loop over all pairs of stubs
        for ( unsigned int i = 0; i < tempEndcapStubVec0.size(); i++ )
        {
          StackedTrackerDetId detId1( tempEndcapStubVec0.at(i)->getDetId() );
          GlobalPoint pos1 = TTTrackAlgorithm< Ref_PixelDigi_ >::theStackedTracker->findAverageGlobalPosition( tempEndcapStubVec0.at(i)->getClusterPtr(0).get() );
          double rho1 = pos1.perp();
          double phi1 = pos1.phi();
          double z1 = pos1.z();

          bool endcapSeedPS1 = TTTrackAlgorithm< Ref_PixelDigi_ >::theStackedTracker->isPSModule( detId1 );
          if ( !endcapSeedPS1 )
            continue;

          /// Find the index of the first element of the nested loop
          unsigned int startIndex = 0;
          if ( curSector == curSector0%(this->ReturnNumberOfSectors()) &&
               curWedge == (int)curWedge0 ) /// If they are in the same Sector/Wedge the loop can be simplified
          {
            /// This means tempEndcapStubVec1 == tempEndcapStubVec0
            startIndex = i+1;
          }

          /// Loop over other endcap stubs
          for ( unsigned int k = startIndex; k < tempEndcapStubVec1.size(); k++ )
          {
            StackedTrackerDetId detId2( tempEndcapStubVec1.at(k)->getDetId() );
            GlobalPoint pos2 = TTTrackAlgorithm< Ref_PixelDigi_ >::theStackedTracker->findAverageGlobalPosition( tempEndcapStubVec1.at(k)->getClusterPtr(0).get() );
            double rho2 = pos2.perp();
            double phi2 = pos2.phi();
            double z2 = pos2.z();

            bool endcapSeedPS2 = TTTrackAlgorithm< Ref_PixelDigi_ >::theStackedTracker->isPSModule( detId2 );
            if ( !endcapSeedPS2 )
              continue;

            /// Skip same disk pairs
            if ( detId2.iSide() != detId1.iSide() )
              continue;

            /// Skip pairs with distance larger than 1 disk
            //if (( detId2.iDisk() != detId1.iDisk() + 1 ) && ( detId1.iDisk() != detId2.iDisk() + 1 ))
            if ( detId2.iDisk() != detId1.iDisk() + 1 )
              continue;

            /// Safety cross check
            if ( fabs(pos1.z()) > fabs(pos2.z()) )
            {
              //std::cerr << "TTTrackAlgorithm_exactBarrelEndcap::CreateSeeds()" << std::endl;
              //std::cerr << "   A L E R T ! fabs(pos1.z()) > fabs(pos2.z()) in Endcap-Endcap tracklet" << std::endl;
              continue;
            }

            /// More robust additional cross check
            if ( fabs(rho1 - rho2) / fabs(z1 - z2) < 0.1 )
              continue;

            /// Apply cosine theorem to find 1/(2R) = rInvOver2
#include "L1Trigger/TrackTrigger/src/TTTrackAlgorithm_trackletBE_SeedTrigonometry.icc"

            /// Perform cut on Pt
            if ( fabs(rInvOver2) > mMagneticField*0.0015*0.5 )
              continue;

            /// Calculate tracklet parameters with helix model
            /// roughPt, z0, cotTheta0, phi0
#include "L1Trigger/TrackTrigger/src/TTTrackAlgorithm_trackletBE_SeedParameters.icc"

            /// Perform projected vertex cut
            if ( fabs(z0) > 30.0 )
              continue;

            /// Create the Seed in the form of a Track and store it in the output
            std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > tempVec;
            tempVec.push_back( tempEndcapStubVec0.at(i) );
            tempVec.push_back( tempEndcapStubVec1.at(k) );
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
      }
    } /// End of loop over neighbor Sectors/Wedges
  } /// End of loop over map elements

  /// Fill the output map at the very end of everything
  /// This should be faster than doing it filling it
  /// stub-by-stub, also because we already have everything
  /// This way the number of searches with map::find will be lower
  /// and also the memory usage is smaller than in the other case

  /// Just to keep it simple and to avoid any overload of the memory
  /// the two maps are merged into one which is eventually
  /// passed to the output
  for ( mapIter = stubEndcapMap->begin();
        mapIter != stubEndcapMap->end();
        ++mapIter )
  {
    /// Get the key
    std::pair< unsigned int, unsigned int > mapKey = mapIter->first;

    /// If an entry already exists for this Sector/Wedge, just add the vector
    /// to the existing vector, otherwise create the entry
    if ( stubBarrelMap->find( mapKey ) == stubBarrelMap->end() )
    {
      /// New entry
      stubBarrelMap->insert( std::make_pair( mapKey, mapIter->second ) );
    }
    else
    {
      /// Already existing entry
      std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > >::iterator itVec = stubBarrelMap->find( mapKey )->second.end();
      stubBarrelMap->find( mapKey )->second.insert( itVec, mapIter->second.begin(), mapIter->second.end() );
    }
  }

  /// Now copy the merged map in the output
  outputSectorMap->insert( stubBarrelMap->begin(), stubBarrelMap->end() );

}

/// Match a Stub to a Seed/Track
template< >
void TTTrackAlgorithm_trackletBE< Ref_PixelDigi_ >::AttachStubToSeed( TTTrack< Ref_PixelDigi_ > &seed,
                                                                      edm::Ptr< TTStub< Ref_PixelDigi_ > > &candidate ) const
{
  /// Get the track Stubs
  std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > theStubs = seed.getStubPtrs();

  /// Check that the candidate is NOT the one under examination
  for ( unsigned int i = 0; i < theStubs.size(); i++ )
  {
    if ( theStubs.at(i) == candidate )
      return;
  }

  /// Skip if the stub is in one of the seed layers/disks  
  StackedTrackerDetId stDetId0( theStubs.at(0)->getDetId() );
  StackedTrackerDetId stDetId1( theStubs.at(1)->getDetId() );
  StackedTrackerDetId stDetIdCand( candidate->getDetId() );

  bool endcapCandPS = false;
  bool endcapSeedPS = false;
  if ( endcapCandPS || endcapSeedPS )
  {} /// This is needed when I comment lines down there in order to perform tests and not to get compilation warnings etc...

  if ( stDetId0.isBarrel() && stDetIdCand.isBarrel() )
  {
    if ( stDetId0.iLayer() == stDetIdCand.iLayer() || stDetId1.iLayer() == stDetIdCand.iLayer() )
      return;
  }
  else
  {
    if ( stDetId0.isEndcap() && stDetIdCand.isEndcap() )
    {
      if ( stDetId0.iSide() == stDetIdCand.iSide() )
      {
        if ( stDetId0.iDisk() == stDetIdCand.iDisk() || stDetId1.iDisk() == stDetIdCand.iDisk() )
          return;
      }
    }
  }

  endcapCandPS = stDetIdCand.isEndcap() && TTTrackAlgorithm< Ref_PixelDigi_ >::theStackedTracker->isPSModule( stDetIdCand );
  endcapSeedPS = stDetId0.isEndcap() && TTTrackAlgorithm< Ref_PixelDigi_ >::theStackedTracker->isPSModule( stDetId0 );

  /// Here we have either Barrel-Barrel with different Layer,
  /// either Endcap-Endcap with different Side/Disk,
  /// either Barrel-Endcap or Endcap-Barrel

  /// Get the track momentum and propagate it
  GlobalVector curMomentum = seed.getMomentum();
  GlobalPoint curVertex = seed.getVertex();
  double curRInv = seed.getRInv();
  double curPhi = curMomentum.phi();
  double curTheta = curMomentum.theta();
  double curZVtx = curVertex.z();

  /// Get the candidate Stub position
  GlobalPoint candPos = TTTrackAlgorithm< Ref_PixelDigi_ >::theStackedTracker->findGlobalPosition( candidate.get() );
  double rhoCand = candPos.perp();
  double phiCand = candPos.phi();
  double zCand = candPos.z();

  /// Propagate seed to Barrel candidate
  if ( stDetIdCand.isBarrel() )
  {
    /// Calculate deltaRPhi and deltaZ
#include "L1Trigger/TrackTrigger/src/TTTrackAlgorithm_trackletBE_BarrelSeedPropagation.icc"

    /// First get the vector corresponding to the seed SL
    /// Then get in this vector the entry corresponding to the targer SL
    if ( stDetId0.isBarrel() )
    {
      if ( deltaRPhi < (tableRPhiBB.at(stDetId0.iLayer())).at(stDetIdCand.iLayer()) &&
           deltaZ < (tableZBB.at(stDetId0.iLayer())).at(stDetIdCand.iLayer()) )
      {
        seed.addStubPtr( candidate );
      }
    }
    else if ( stDetId0.isEndcap() )
    {
      if ( endcapSeedPS )
      {
        if ( deltaRPhi < (tableRPhiEB_PS.at(stDetId0.iDisk())).at(stDetIdCand.iLayer()) && 
             deltaZ < (tableZEB_PS.at(stDetId0.iDisk())).at(stDetIdCand.iLayer()) )
        {
          seed.addStubPtr( candidate );
        }
      }
      else
      {
        if ( deltaRPhi < (tableRPhiEB.at(stDetId0.iDisk())).at(stDetIdCand.iLayer()) && 
             deltaZ < (tableZEB.at(stDetId0.iDisk())).at(stDetIdCand.iLayer()) )
        {
          seed.addStubPtr( candidate );
        }
      }
    }
  }
  /// Propagate to Endcap candidate
  else if ( stDetIdCand.isEndcap() )
  {
    /// Calculate a correction for non-pointing-strips in square modules
    /// Relevant angle is the one between hit and module center, with
    /// vertex at (0, 0). Take snippet from HitMatchingAlgorithm_window201*
    /// POSITION IN TERMS OF PITCH MULTIPLES:
    ///       0 1 2 3 4 5 5 6 8 9 ...
    /// COORD: 0 1 2 3 4 5 6 7 8 9 ...
    /// OUT   | | | | | |x| | | | | | | | | |
    ///
    /// IN    | | | |x|x| | | | | | | | | | |
    ///             THIS is 3.5 (COORD) and 4.0 (POS)
    /// The center of the module is at NROWS/2 (position) and NROWS-0.5 (coordinates)
    StackedTrackerDetId stDetId( candidate->getClusterPtr(0)->getDetId() );
    const GeomDetUnit* det0 = TTTrackAlgorithm< Ref_PixelDigi_ >::theStackedTracker->idToDetUnit( stDetId, 0 );
    const PixelGeomDetUnit* pix0 = dynamic_cast< const PixelGeomDetUnit* >( det0 );
    const PixelTopology* top0 = dynamic_cast< const PixelTopology* >( &(pix0->specificTopology()) );
    std::pair< float, float > pitch0 = top0->pitch();
    MeasurementPoint stubCoord = candidate->getClusterPtr(0)->findAverageLocalCoordinates();
    double stubTransvDispl = pitch0.first * ( stubCoord.x() - (top0->nrows()/2 - 0.5) ); /// Difference in coordinates is the same as difference in position

    if ( zCand > 0 )
    {
      stubTransvDispl = - stubTransvDispl;
    }

    /// Calculate deltaRPhi and deltaRho
#include "L1Trigger/TrackTrigger/src/TTTrackAlgorithm_trackletBE_EndcapSeedPropagation.icc"

    if ( stDetId0.isBarrel() )
    {
      if ( endcapCandPS )
      {
        if ( deltaRPhi < (tableRPhiBE_PS.at(stDetId0.iLayer())).at(stDetIdCand.iDisk()) &&
             deltaR < (tableZBE_PS.at(stDetId0.iLayer())).at(stDetIdCand.iDisk()) )
        {
          seed.addStubPtr( candidate );
        }
      }
      else
      {
        if ( deltaRPhi < (tableRPhiBE.at(stDetId0.iLayer())).at(stDetIdCand.iDisk()) && 
             deltaR < (tableZBE.at(stDetId0.iLayer())).at(stDetIdCand.iDisk()) )
        {
          seed.addStubPtr( candidate );
        }
      }
    }
    else if ( stDetId0.isEndcap() )
    {
      if ( endcapCandPS )
      {
        if ( deltaRPhi < (tableRPhiEE_PS.at(stDetId0.iDisk())).at(stDetIdCand.iDisk()) &&
             deltaR < (tableZEE_PS.at(stDetId0.iDisk())).at(stDetIdCand.iDisk()) )
        {
          seed.addStubPtr( candidate );
        }
      }
      else
      {
        if ( deltaRPhi < (tableRPhiEE.at(stDetId0.iDisk())).at(stDetIdCand.iDisk()) && 
             deltaR < (tableZEE.at(stDetId0.iDisk())).at(stDetIdCand.iDisk()) )
        {
          seed.addStubPtr( candidate );
        }
      }
    }
  }
}

