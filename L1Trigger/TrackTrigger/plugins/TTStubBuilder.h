/*! \class   TTStubBuilder
 *  \brief   Plugin to load the Stub finding algorithm and produce the
 *           collection of Stubs that goes in the event content.
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Andrew W. Rose
 *  \author Nicola Pozzobon
 *  \author Ivan Reid
 *  \date   2013, Jul 18
 *
 */

#ifndef L1_TRACK_TRIGGER_STUB_BUILDER_H
#define L1_TRACK_TRIGGER_STUB_BUILDER_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithmRecord.h"
//#include "classNameFinder.h"

#include <memory>
#include <map>
#include <vector>

template< typename T >
class TTStubBuilder : public edm::EDProducer
{
  public:
    typedef std::pair< StackedTrackerDetId, unsigned int >                    ClusterKey;   /// This is the key
    typedef std::map< ClusterKey, std::vector< edm::Ptr< TTCluster< T > > > > TTClusterMap; /// This is the map

    /// Constructor
    explicit TTStubBuilder( const edm::ParameterSet& iConfig );

    /// Destructor;
    ~TTStubBuilder();

  private:
    /// Data members
    const StackedTrackerGeometry          *theStackedTracker;
    edm::ESHandle< TTStubAlgorithm< T > > theStubFindingAlgoHandle;
    edm::InputTag                         TTClustersInputTag;

    /// Mandatory methods
    virtual void beginRun( const edm::Run& run, const edm::EventSetup& iSetup );
    virtual void endRun( const edm::Run& run, const edm::EventSetup& iSetup );
    virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );

    /// Sorting method for stubs
    /// NOTE: this must be static!
    static bool SortStubBendPairs( const std::pair< unsigned int, double >& left, const std::pair< unsigned int, double >& right );


}; /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Constructors
template< typename T >
TTStubBuilder< T >::TTStubBuilder( const edm::ParameterSet& iConfig )
{
  produces< std::vector< TTCluster< T > > >( "ClusterAccepted" );
  produces< std::vector< TTStub< T > > >( "StubAccepted" );
  produces< std::vector< TTStub< T > > >( "StubRejected" );
  TTClustersInputTag = iConfig.getParameter< edm::InputTag >( "TTClusters" );
}

/// Destructor
template< typename T >
TTStubBuilder< T >::~TTStubBuilder(){}

/// Begin run
template< typename T >
void TTStubBuilder< T >::beginRun( const edm::Run& run, const edm::EventSetup& iSetup )
{
  /// Get the geometry references
  edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
  iSetup.get< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );
  theStackedTracker = StackedTrackerGeomHandle.product();

  /// Get the clustering algorithm 
  iSetup.get< TTStubAlgorithmRecord >().get( theStubFindingAlgoHandle );

  /// Print some information when loaded
  std::cout << std::endl;
  std::cout << "TTStubBuilder< " << templateNameFinder< T >() << " > loaded modules:"
            << "\n\tTTStubAlgorithm:\t" << theStubFindingAlgoHandle->AlgorithmName()
            << std::endl;
  std::cout << std::endl;
}

/// End run
template< typename T >
void TTStubBuilder< T >::endRun( const edm::Run& run, const edm::EventSetup& iSetup ){}

/// Implement the producer
template< typename T >
void TTStubBuilder< T >::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{  
  /// Prepare output
  std::auto_ptr< std::vector< TTCluster< T > > > TTClustersForOutput( new std::vector< TTCluster< T > > );
  std::auto_ptr< std::vector< TTStub< T > > > TTStubsForOutputAccepted( new std::vector< TTStub< T > > );
  std::auto_ptr< std::vector< TTStub< T > > > TTStubsForOutputRejected( new std::vector< TTStub< T > > );

  /// Get the Clusters already stored away
  edm::Handle< std::vector< TTCluster< T > > > TTClusterHandle;
  iEvent.getByLabel( TTClustersInputTag, TTClusterHandle);   

  /// Get the maximum number of stubs per ROC
  /// (CBC3-style)
  unsigned maxStubs = theStackedTracker->getCBC3MaxStubs();

  /// Map the Clusters according to detector elements
  TTClusterMap clusterMap;
  clusterMap.clear();

  unsigned int j = 0; /// Counter needed to build the edm::Ptr to the TTCluster
  typename std::vector< TTCluster< T > >::const_iterator inputIter;
  for ( inputIter = TTClusterHandle->begin();
        inputIter != TTClusterHandle->end();
        ++inputIter )
  {
    /// Make the pointer to be put in the map and, later on, in the Stub
    /// as reference to lower-class bricks composing the Stub itself
    edm::Ptr< TTCluster< T > > tempCluPtr( TTClusterHandle, j++ );

    /// Build the key to the map
    ClusterKey mapkey = std::make_pair( StackedTrackerDetId( inputIter->getDetId() ), inputIter->getStackMember() );

    /// If an entry already exists for this key, just add the cluster
    /// to the vector, otherwise create the entry
    if ( clusterMap.find( mapkey ) == clusterMap.end() )
    {
      /// New entry
      std::vector< edm::Ptr< TTCluster< T > > > tempCluVec;
      tempCluVec.clear();
      tempCluVec.push_back( tempCluPtr );
      clusterMap.insert( std::pair< ClusterKey, std::vector< edm::Ptr< TTCluster< T > > > > ( mapkey, tempCluVec ) );
    }
    else
    {
      /// Already existing entry
      clusterMap[mapkey].push_back( tempCluPtr );   
    }
  }

  /// Loop over the detector elements
  StackedTrackerGeometry::StackContainerIterator StackedTrackerIterator;
  for ( StackedTrackerIterator = theStackedTracker->stacks().begin();
        StackedTrackerIterator != theStackedTracker->stacks().end();
        ++StackedTrackerIterator )
  {
    StackedTrackerDetUnit* Unit = *StackedTrackerIterator;
    StackedTrackerDetId Id = Unit->Id();
    assert(Unit == theStackedTracker->idToStack(Id));
    
    /// Build the keys to get the Clusters
    ClusterKey inmapkey  = std::make_pair(Id, 0);
    ClusterKey outmapkey = std::make_pair(Id, 1);

    /// Get the vectors of Clusters for the current Pt module
    /// Go on only if the entry in the map is found
    typename TTClusterMap::const_iterator innerIter = clusterMap.find( inmapkey );
    typename TTClusterMap::const_iterator outerIter = clusterMap.find( outmapkey );

    if ( innerIter == clusterMap.end() || outerIter == clusterMap.end() )
      continue;

    std::vector< edm::Ptr< TTCluster< T > > > innerClusters = innerIter->second;
    std::vector< edm::Ptr< TTCluster< T > > > outerClusters = outerIter->second;

    typename std::vector< edm::Ptr< TTCluster< T > > >::iterator innerClusterIter, outerClusterIter;

    /// If there are Clusters in both sensors
    /// you can try and make a Stub
    if ( innerClusters.size() == 0 || outerClusters.size() == 0 )
      continue;

    /// Get chip size information
    const GeomDetUnit* det0 = theStackedTracker->idToDetUnit( Id, 0 );
    const PixelGeomDetUnit* pix0 = dynamic_cast< const PixelGeomDetUnit* >( det0 );
    const PixelTopology* top0 = dynamic_cast< const PixelTopology* >( &(pix0->specificTopology()) );
    const int chipSize = 2 * top0->rowsperroc();             /// Need to find ASIC size in half-strip units
    std::map< int, std::vector< TTStub< T > > > moduleStubs; /// Temporary storage for stubs before max check

    /// Loop over pairs of Clusters
    for ( innerClusterIter = innerClusters.begin();
          innerClusterIter != innerClusters.end();
          ++innerClusterIter )
    {
      for ( outerClusterIter = outerClusters.begin();
            outerClusterIter != outerClusters.end();
            ++outerClusterIter )
      {
        /// Build a temporary Stub
        TTStub< T > tempTTStub( Id );
        tempTTStub.addClusterPtr( *innerClusterIter ); /// innerClusterIter is an iterator pointing to the edm::Ptr
        tempTTStub.addClusterPtr( *outerClusterIter );

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

          /// Put in the output
          if ( maxStubs == 0 )
          {
            /// This means that ALL stubs go into the output
            TTStubsForOutputAccepted->push_back( tempTTStub );
            TTClustersForOutput->push_back( **innerClusterIter );
            TTClustersForOutput->push_back( **outerClusterIter );
          }
          else
          {
            /// This means that only some of them do
            /// Put in the temporary output
            int chip = tempTTStub.getTriggerPosition() / chipSize; /// Find out which ASIC
            if ( moduleStubs.find( chip ) == moduleStubs.end() )   /// Already a stub for this ASIC?
            {
              /// No, so new entry
              std::vector< TTStub<T> > tempStubs;
              tempStubs.clear();
              tempStubs.push_back( tempTTStub );
              moduleStubs.insert( std::pair< int, std::vector< TTStub< T > > >( chip, tempStubs ) );
            }
            else
            {
              /// Already existing entry
              moduleStubs[chip].push_back( tempTTStub );
            }
          }
        } /// Stub accepted
        else
          TTStubsForOutputRejected->push_back( tempTTStub );

      } /// End of nested loop
    } /// End of loop over pairs of Clusters

    /// If we are working with max no. stub/ROC, then clean the temporary output
    /// and store only the selected stubs
    if ( moduleStubs.empty() ) /// NOTE: this includes "if ( maxStubs == 0 )"
      continue;

    /// Loop over ROC's
    /// the ROC ID is not important
    for ( auto const & is : moduleStubs )
    {
      /// Put the stubs into the output
      if ( is.second.size() <= maxStubs )
      {
        for ( auto const & ts: is.second )
        {
          TTStubsForOutputAccepted->push_back( ts );
          TTClustersForOutput->push_back( *(ts.getClusterPtr(0)) );
          TTClustersForOutput->push_back( *(ts.getClusterPtr(1)) );
        }
      }
      else
      {
        /// Sort them and pick up only the first N.
        std::vector< std::pair< unsigned int, double > > bendMap;
        for ( unsigned int i = 0; i < is.second.size(); ++i )
        {
          bendMap.push_back( std::pair< unsigned int, double >( i, is.second[i].getTriggerBend() ) );
        }
        std::sort( bendMap.begin(), bendMap.end(), TTStubBuilder< T >::SortStubBendPairs );

        for ( unsigned int i = 0; i < maxStubs; ++i )
        {
          /// Put the highest momenta (lowest bend) stubs into the event
          TTStubsForOutputAccepted->push_back(is.second[bendMap[i].first]);
          TTClustersForOutput->push_back( *(is.second[bendMap[i].first].getClusterPtr(0)) );
          TTClustersForOutput->push_back( *(is.second[bendMap[i].first].getClusterPtr(1)) );
        }
        for ( unsigned int i = maxStubs; i < is.second.size(); ++i )
        {
          /// Reject the rest
          TTStubsForOutputRejected->push_back( is.second[bendMap[i].first] );
        }
      }
    } /// End of loop over temp output

  } /// End of loop over detector elements

  /// Put output in the event
  iEvent.put( TTStubsForOutputAccepted, "StubAccepted" );
  iEvent.put( TTStubsForOutputRejected, "StubRejected" );
  iEvent.put( TTClustersForOutput, "ClusterAccepted" );

}

/// Sort routine for stub ordering
template< typename T >
bool TTStubBuilder< T >::SortStubBendPairs( const std::pair< unsigned int, double >& left, const std::pair< unsigned int, double >& right )
{
  return left.second < right.second;
}


#endif

