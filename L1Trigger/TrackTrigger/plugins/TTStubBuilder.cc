/*!  \brief   Implementation of methods of TTClusterBuilder.h
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 * \author Andrew W. Rose
 * \author Nicola Pozzobon
 * \author Ivan Reid
 * \date 2013, Jul 18
 *
 */

#include "L1Trigger/TrackTrigger/plugins/TTStubBuilder.h"

/// Implement the producer
template< >
void TTStubBuilder< Ref_Phase2TrackerDigi_ >::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  //Retrieve tracker topology from geometry                                                                                                              
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
  edm::ESHandle< TrackerGeometry > tGeomHandle;
  iSetup.get< TrackerDigiGeometryRecord >().get( tGeomHandle );
  const TrackerGeometry* const theTrackerGeom = tGeomHandle.product();
	
  /// Prepare output
  auto ttClusterDSVForOutput      = std::make_unique<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>>();
  auto ttStubDSVForOutputTemp     = std::make_unique<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>();
  auto ttStubDSVForOutputAccepted = std::make_unique<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>();
  auto ttStubDSVForOutputRejected = std::make_unique<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>();

  /// Get the Clusters already stored away
  edm::Handle< edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > > > clusterHandle;
  iEvent.getByToken( clustersToken, clusterHandle );

  /// Get the maximum number of stubs per ROC
  /// (CBC3-style)
  //  unsigned maxStubs = theStackedTracker->getCBC3MaxStubs();
  unsigned maxStubs = 3;

  for (auto gd=theTrackerGeom->dets().begin(); gd != theTrackerGeom->dets().end(); gd++) {
      DetId detid = (*gd)->geographicalId();
      if(detid.subdetId()!=StripSubdetector::TOB && detid.subdetId()!=StripSubdetector::TID ) continue; // only run on OT
      if(!tTopo->isLower(detid) ) continue; // loop on the stacks: choose the lower arbitrarily
      DetId lowerDetid = detid;
      DetId upperDetid = tTopo->partnerDetId(detid);
      DetId stackDetid = tTopo->stack(detid);

    /// Go on only if both detectors have Clusters
    if ( clusterHandle->find( lowerDetid ) == clusterHandle->end() ||
         clusterHandle->find( upperDetid ) == clusterHandle->end() )
      continue;

    /// Get the DetSets of the Clusters
    edmNew::DetSet< TTCluster< Ref_Phase2TrackerDigi_ > > lowerClusters = (*clusterHandle)[ lowerDetid ];
    edmNew::DetSet< TTCluster< Ref_Phase2TrackerDigi_ > > upperClusters = (*clusterHandle)[ upperDetid ];

    /// If there are Clusters in both sensors
    /// you can try and make a Stub
    /// This is ~redundant
    if ( lowerClusters.size() == 0 || upperClusters.size() == 0 )
      continue;

    /// Create the vectors of objects to be passed to the FastFillers
    std::vector< TTCluster< Ref_Phase2TrackerDigi_ > > tempInner; 
    std::vector< TTCluster< Ref_Phase2TrackerDigi_ > > tempOuter; 
    std::vector< TTStub< Ref_Phase2TrackerDigi_ > >   tempAccepted; 
    tempInner.clear();
    tempOuter.clear();
    tempAccepted.clear();

    /// Get chip size information
    const GeomDetUnit* det0 = theTrackerGeom->idToDetUnit( lowerDetid );
    const PixelGeomDetUnit* pix0 = dynamic_cast< const PixelGeomDetUnit* >( det0 );
    const PixelTopology* top0 = dynamic_cast< const PixelTopology* >( &(pix0->specificTopology()) );
    const int chipSize = 2 * top0->rowsperroc(); /// Need to find ASIC size in half-strip units

    std::unordered_map< int, std::vector< TTStub< Ref_Phase2TrackerDigi_ > > > moduleStubs; /// Temporary storage for stubs before max check

    /// Loop over pairs of Clusters
    for ( auto lowerClusterIter = lowerClusters.begin();
               lowerClusterIter != lowerClusters.end();
               ++lowerClusterIter ) {

      /// Temporary storage to allow only one stub per inner cluster
      /// if requested in cfi
      std::vector< TTStub< Ref_Phase2TrackerDigi_ > > tempOutput;
      // tempOutput.clear();

      for ( auto upperClusterIter = upperClusters.begin();
                 upperClusterIter != upperClusters.end();
                 ++upperClusterIter ) {

        /// Build a temporary Stub
        TTStub< Ref_Phase2TrackerDigi_ > tempTTStub( stackDetid );
        tempTTStub.addClusterRef( edmNew::makeRefTo( clusterHandle, lowerClusterIter ) );
        tempTTStub.addClusterRef( edmNew::makeRefTo( clusterHandle, upperClusterIter ) );

        /// Check for compatibility
        bool thisConfirmation = false;
        int thisDisplacement = 999999;
        int thisOffset = 0;

        theStubFindingAlgoHandle->PatternHitCorrelation( thisConfirmation, thisDisplacement, thisOffset, tempTTStub );

        /// If the Stub is above threshold
        if ( thisConfirmation )
        {
          tempTTStub.setTriggerDisplacement( thisDisplacement );
          tempTTStub.setTriggerOffset( thisOffset );
	  tempOutput.push_back( tempTTStub );
        } /// Stub accepted
      } /// End of loop over outer clusters

      /// Here tempOutput stores all the stubs from this inner cluster
      /// Check if there is need to store only one (if only one already, skip this step)
      if ( ForbidMultipleStubs && tempOutput.size() > 1 )
      {
        /// If so, sort the stubs by bend and keep only the first one (smallest bend)
        std::sort( tempOutput.begin(), tempOutput.end(), TTStubBuilder< Ref_Phase2TrackerDigi_ >::SortStubsBend );

        /// Get to the second element (the switch above ensures there are min 2)
        typename std::vector< TTStub< Ref_Phase2TrackerDigi_ > >::iterator tempIter = tempOutput.begin();
        ++tempIter;

        /// tempIter points now to the second element

        /// Delete all-but-the first one from tempOutput
        tempOutput.erase( tempIter, tempOutput.end() );
      }

      /// Here, tempOutput is either of size 1 (if entering the switch)
      /// either of size N with all the valid combinations ...

      /// Now loop over the accepted stubs (1 or N) for this inner cluster
      for ( unsigned int iTempStub = 0; iTempStub < tempOutput.size(); ++iTempStub )
      {
        /// Get the stub
        const TTStub< Ref_Phase2TrackerDigi_ >& tempTTStub = tempOutput[iTempStub];

        /// Put in the output
        if ( maxStubs == 0 )
        {
          /// This means that ALL stubs go into the output
          tempInner.push_back( *(tempTTStub.getClusterRef(0)) );
          tempOuter.push_back( *(tempTTStub.getClusterRef(1)) );
          tempAccepted.push_back( tempTTStub );
        }
        else
        {
          /// This means that only some of them do
          /// Put in the temporary output
          int chip = tempTTStub.getTriggerPosition() / chipSize; /// Find out which ASIC
          if ( moduleStubs.find( chip ) == moduleStubs.end() ) /// Already a stub for this ASIC?
          {
            /// No, so new entry
            std::vector< TTStub< Ref_Phase2TrackerDigi_ > > tempStubs(1,tempTTStub);
            //tempStubs.clear();
            //tempStubs.push_back( tempTTStub );
            moduleStubs.insert( std::pair< int, std::vector< TTStub< Ref_Phase2TrackerDigi_ > > >( chip, tempStubs ) );
          }
          else
          {
            /// Already existing entry
            moduleStubs[chip].push_back( tempTTStub );
          }
        } /// End of check on max number of stubs per module
      } /// End of nested loop
    } /// End of loop over pairs of Clusters

    /// If we are working with max no. stub/ROC, then clean the temporary output
    /// and store only the selected stubs
    if ( moduleStubs.empty() == false )
    {
      /// Loop over ROC's
      /// the ROC ID is not important
      for ( auto const & is : moduleStubs )
      {
        /// Put the stubs into the output
        if ( is.second.size() <= maxStubs )
        {
          for ( auto const & ts: is.second )
          {
            tempInner.push_back( *(ts.getClusterRef(0)) );
            tempOuter.push_back( *(ts.getClusterRef(1)) );
            tempAccepted.push_back( ts );
          }
        }
        else
        {
          /// Sort them and pick up only the first N.
          std::vector< std::pair< unsigned int, double > > bendMap;
          bendMap.reserve(is.second.size());
          for ( unsigned int i = 0; i < is.second.size(); ++i )
          {
            bendMap.push_back( std::pair< unsigned int, double >( i, is.second[i].getTriggerBend() ) );
          }
          std::sort( bendMap.begin(), bendMap.end(), TTStubBuilder< Ref_Phase2TrackerDigi_ >::SortStubBendPairs );

          for ( unsigned int i = 0; i < maxStubs; ++i )
          {
            /// Put the highest momenta (lowest bend) stubs into the event
            tempInner.push_back( *(is.second[bendMap[i].first].getClusterRef(0)) );
            tempOuter.push_back( *(is.second[bendMap[i].first].getClusterRef(1)) );
            tempAccepted.push_back( is.second[bendMap[i].first] );
          }
        }
      } /// End of loop over temp output
    } /// End store only the selected stubs if max no. stub/ROC is set
    /// Create the FastFillers
    if ( tempInner.size() > 0 )
    {
      typename edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > >::FastFiller lowerOutputFiller( *ttClusterDSVForOutput, lowerDetid );
      for ( unsigned int m = 0; m < tempInner.size(); m++ )
      {
        lowerOutputFiller.push_back( tempInner.at(m) );
      }
      if ( lowerOutputFiller.empty() )
        lowerOutputFiller.abort();
    }

    if ( tempOuter.size() > 0 )
    {
      typename edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > >::FastFiller upperOutputFiller( *ttClusterDSVForOutput, upperDetid );
      for ( unsigned int m = 0; m < tempOuter.size(); m++ )
      {
        upperOutputFiller.push_back( tempOuter.at(m) );
      }
      if ( upperOutputFiller.empty() )
        upperOutputFiller.abort();
    }

    if ( tempAccepted.size() > 0 )
    {
      typename edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >::FastFiller tempAcceptedFiller( *ttStubDSVForOutputTemp, stackDetid);
      for ( unsigned int m = 0; m < tempAccepted.size(); m++ )
      {
        tempAcceptedFiller.push_back( tempAccepted.at(m) );
      }
      if ( tempAcceptedFiller.empty() )
        tempAcceptedFiller.abort();
    }

  } /// End of loop over detector elements

  /// Put output in the event (1)
  /// Get also the OrphanHandle of the accepted clusters
  edm::OrphanHandle< edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > > > ttClusterAcceptedHandle = iEvent.put( std::move(ttClusterDSVForOutput), "ClusterAccepted" );

  /// Now, correctly reset the output
  typename edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >::const_iterator stubDetIter;

  for ( stubDetIter = ttStubDSVForOutputTemp->begin();
        stubDetIter != ttStubDSVForOutputTemp->end();
        ++stubDetIter ) {
    /// Get the DetId and prepare the FastFiller
    DetId thisStackedDetId = stubDetIter->id();
    typename edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >::FastFiller acceptedOutputFiller( *ttStubDSVForOutputAccepted, thisStackedDetId );

    /// detid of the two components. 
    ///This should be done via a TrackerTopology method that is not yet available.
    DetId lowerDetid = thisStackedDetId+1;
    DetId upperDetid = thisStackedDetId+2;

    /// Get the DetSets of the clusters
    edmNew::DetSet< TTCluster< Ref_Phase2TrackerDigi_ > > lowerClusters = (*ttClusterAcceptedHandle)[ lowerDetid ];
    edmNew::DetSet< TTCluster< Ref_Phase2TrackerDigi_ > > upperClusters = (*ttClusterAcceptedHandle)[ upperDetid ];

    /// Get the DetSet of the stubs
    edmNew::DetSet< TTStub< Ref_Phase2TrackerDigi_ > > theseStubs = (*ttStubDSVForOutputTemp)[ thisStackedDetId ];

    /// Prepare the new DetSet to replace the current one
    /// Loop over the stubs
    typename edmNew::DetSet< TTCluster< Ref_Phase2TrackerDigi_ > >::iterator clusterIter;
    typename edmNew::DetSet< TTStub< Ref_Phase2TrackerDigi_ > >::iterator stubIter;
    for ( stubIter = theseStubs.begin();
          stubIter != theseStubs.end();
          ++stubIter ) {
      /// Create a temporary stub
      TTStub< Ref_Phase2TrackerDigi_ > tempTTStub( stubIter->getDetId() );

      /// Compare the clusters stored in the stub with the ones of this module
      edm::Ref< edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > >, TTCluster< Ref_Phase2TrackerDigi_ > > lowerClusterToBeReplaced = stubIter->getClusterRef(0);
      edm::Ref< edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > >, TTCluster< Ref_Phase2TrackerDigi_ > > upperClusterToBeReplaced = stubIter->getClusterRef(1);

      bool lowerOK = false;
      bool upperOK = false;

      for ( clusterIter = lowerClusters.begin();
            clusterIter != lowerClusters.end() && !lowerOK;
            ++clusterIter ) {
        if ( clusterIter->getHits() == lowerClusterToBeReplaced->getHits() ) {
          tempTTStub.addClusterRef( edmNew::makeRefTo( ttClusterAcceptedHandle, clusterIter ) );
          lowerOK = true;
        }
      }

      for ( clusterIter = upperClusters.begin();
            clusterIter != upperClusters.end() && !upperOK;
            ++clusterIter ) {
        if ( clusterIter->getHits() == upperClusterToBeReplaced->getHits() ) {
          tempTTStub.addClusterRef( edmNew::makeRefTo( ttClusterAcceptedHandle, clusterIter ) );
          upperOK = true;
        }
      }

      /// If no compatible clusters were found, skip to the next one
      if ( !lowerOK || !upperOK ) continue;

      tempTTStub.setTriggerDisplacement( 2.*stubIter->getTriggerDisplacement() ); /// getter is in FULL-strip units, setter is in HALF-strip units
      tempTTStub.setTriggerOffset( 2.*stubIter->getTriggerOffset() );             /// getter is in FULL-strip units, setter is in HALF-strip units

      acceptedOutputFiller.push_back( tempTTStub );

    } /// End of loop over stubs of this module

    if ( acceptedOutputFiller.empty() )
      acceptedOutputFiller.abort();
   
  } /// End of loop over stub DetSetVector
    
  /// Put output in the event (2)
  iEvent.put( std::move(ttStubDSVForOutputAccepted), "StubAccepted" );
  iEvent.put( std::move(ttStubDSVForOutputRejected), "StubRejected" );
}


