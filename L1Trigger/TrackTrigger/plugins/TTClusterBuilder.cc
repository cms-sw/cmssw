/*! \brief   Implementation of methods of TTClusterBuilder.h
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 *  \author Nicola
 *  \date   2013, Jul 12
 *
 */

#include "L1Trigger/TrackTrigger/plugins/TTClusterBuilder.h"

/// Implement the producer
template< >
void TTClusterBuilder< Ref_PixelDigi_ >::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  /// Prepare output
  std::auto_ptr< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > > > TTClusterDSVForOutput( new edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > > );

  std::map< DetId, std::vector< Ref_PixelDigi_ > > rawHits; /// This is a map containing hits:
                                                            /// a vector of type Ref_PixelDigi_ is mapped wrt
                                                            /// the DetId
  this->RetrieveRawHits( rawHits, iEvent );

  /// Loop over the detector elements
  StackedTrackerGeometry::StackContainerIterator StackedTrackerIterator;
  for ( StackedTrackerIterator = theStackedTrackers->stacks().begin();
        StackedTrackerIterator != theStackedTrackers->stacks().end();
        ++StackedTrackerIterator )
  {
    StackedTrackerDetUnit* Unit = *StackedTrackerIterator;
    StackedTrackerDetId Id = Unit->Id();
    assert(Unit == theStackedTrackers->idToStack(Id));

    const GeomDetUnit* det0 = theStackedTrackers->idToDetUnit( Id, 0 );
    const GeomDetUnit* det1 = theStackedTrackers->idToDetUnit( Id, 1 );

    /// Find pixel pitch and topology related information
    const PixelGeomDetUnit* pix0 = dynamic_cast< const PixelGeomDetUnit* >( det0 );
    const PixelGeomDetUnit* pix1 = dynamic_cast< const PixelGeomDetUnit* >( det1 );
    const PixelTopology* top0 = dynamic_cast< const PixelTopology* >( &(pix0->specificTopology()) );
    const PixelTopology* top1 = dynamic_cast< const PixelTopology* >( &(pix1->specificTopology()) );
    //std::pair< float, float > pitch0 = top0->pitch();
    //std::pair< float, float > pitch1 = top1->pitch();

    /// Stop if the clusters are not in the same z-segment
    int cols0 = top0->ncolumns();
    int cols1 = top1->ncolumns();
    int ratio = cols0/cols1; /// This assumes the ratio is integer!

    bool isPS = (ratio != 1);

    /// Temp vectors containing the vectors of the
    /// hits used to build each cluster
    std::vector< std::vector< Ref_PixelDigi_ > > innerHits, outerHits;

    /// Find the hits in each stack member
    typename std::map< DetId, std::vector< Ref_PixelDigi_ > >::const_iterator innerHitFind = rawHits.find( Unit->stackMember(0) );
    typename std::map< DetId, std::vector< Ref_PixelDigi_ > >::const_iterator outerHitFind = rawHits.find( Unit->stackMember(1) );

    /// If there are hits, cluster them
    /// It is the TTClusterAlgorithm::Cluster method which
    /// calls the constructor to the Cluster class!
    if ( innerHitFind != rawHits.end() ) theClusterFindingAlgoHandle->Cluster( innerHits, innerHitFind->second, isPS );
    if ( outerHitFind != rawHits.end() ) theClusterFindingAlgoHandle->Cluster( outerHits, outerHitFind->second, false );

    /// Create TTCluster objects and store them
    /// Use the FastFiller with edmNew::DetSetVector
    {
      edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >::FastFiller innerOutputFiller( *TTClusterDSVForOutput, Unit->stackMember(0) );
      for ( unsigned int i = 0; i < innerHits.size(); i++ )
      {
        TTCluster< Ref_PixelDigi_ > temp( innerHits.at(i), Id, 0, storeLocalCoord );
        innerOutputFiller.push_back( temp );
      }
      if ( innerOutputFiller.empty() )
        innerOutputFiller.abort();
    }
    {
      edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >::FastFiller outerOutputFiller( *TTClusterDSVForOutput, Unit->stackMember(1) );
      for ( unsigned int i = 0; i < outerHits.size(); i++ )
      {
        TTCluster< Ref_PixelDigi_ > temp( outerHits.at(i), Id, 1, storeLocalCoord );
        outerOutputFiller.push_back( temp );
      }
      if ( outerOutputFiller.empty() )
        outerOutputFiller.abort();
    }
  } /// End of loop over detector elements

  /// Put output in the event
  iEvent.put( TTClusterDSVForOutput, "ClusterInclusive" );
}

/// Retrieve hits from the event
/// Specialize template for PixelDigis
template< >
void TTClusterBuilder< Ref_PixelDigi_ >::RetrieveRawHits( std::map< DetId, std::vector< Ref_PixelDigi_ > > &mRawHits,
                                                          const edm::Event& iEvent )
{
  mRawHits.clear();

  /// Loop over the tags used to identify hits in the cfg file
  std::vector< edm::InputTag >::iterator it;
  for ( it = rawHitInputTags.begin();
        it != rawHitInputTags.end();
        ++it )
  {
    /// For each tag, get the corresponding handle
    edm::Handle< edm::DetSetVector< PixelDigi > > HitHandle;
    iEvent.getByLabel( it->label(), HitHandle );

    edm::DetSetVector<PixelDigi>::const_iterator detsIter;
    edm::DetSet<PixelDigi>::const_iterator       hitsIter;

    /// Loop over detector elements identifying PixelDigis
    for ( detsIter = HitHandle->begin();
          detsIter != HitHandle->end();
          detsIter++ )
    {
      DetId id = detsIter->id;

      /// Is it Pixel?
      if ( id.subdetId()==1 || id.subdetId()==2 )
      {
        /// Loop over Digis in this specific detector element
        for ( hitsIter = detsIter->data.begin();
              hitsIter != detsIter->data.end();
              hitsIter++ )
        {
          if ( hitsIter->adc() >= ADCThreshold )
          {
            /// If the Digi is over threshold,
            /// accept it as a raw hit and put into map
            mRawHits[id].push_back( edm::makeRefTo( HitHandle, id , hitsIter ) );
          } /// End of threshold selection
        } /// End of loop over digis
      } /// End of "is Pixel"
    } /// End of loop over detector elements
  } /// End of loop over tags
}

