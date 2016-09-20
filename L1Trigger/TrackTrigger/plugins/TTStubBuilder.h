/*! \class TTStubBuilder
* \brief Plugin to load the Stub finding algorithm and produce the
* collection of Stubs that goes in the event content.
* \details After moving from SimDataFormats to DataFormats,
* the template structure of the class was maintained
* in order to accomodate any types other than PixelDigis
* in case there is such a need in the future.
*
* \author Andrew W. Rose
* \author Nicola Pozzobon
* \author Ivan Reid
* \date 2013, Jul 18
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

#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include <memory>
#include <map>
#include <vector>

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

template< typename T >
class TTStubBuilder : public edm::EDProducer
{
  public:
    /// Constructor
    explicit TTStubBuilder( const edm::ParameterSet& iConfig );

    /// Destructor;
    ~TTStubBuilder();

  private:
    /// Data members
    edm::ESHandle< TTStubAlgorithm< T > > theStubFindingAlgoHandle;
    edm::EDGetTokenT< TTCluster< T > > clustersToken;
    
    /// Mandatory methods
    virtual void beginRun( const edm::Run& run, const edm::EventSetup& iSetup );
    virtual void endRun( const edm::Run& run, const edm::EventSetup& iSetup );
    virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );

    /// Sorting method for stubs
    /// NOTE: this must be static!
    static bool SortStubBendPairs( const std::pair< unsigned int, double >& left, const std::pair< unsigned int, double >& right );

}; /// Close class

/*! \brief Implementation of methods
* \details Here, in the header file, the methods which do not depend
* on the specific type <T> that can fit the template.
* Other methods, with type-specific features, are implemented
* in the source file.
*/

/// Constructors
template< typename T >
TTStubBuilder< T >::TTStubBuilder( const edm::ParameterSet& iConfig )
{
  clustersToken = consumes< TTCluster< T > >(iConfig.getParameter< edm::InputTag >( "TTClusters" ));
  produces< edmNew::DetSetVector< TTCluster< T > > >( "ClusterAccepted" );
  produces< edmNew::DetSetVector< TTStub< T > > >( "StubAccepted" );
  produces< edmNew::DetSetVector< TTStub< T > > >( "StubRejected" );
}

/// Destructor
template< typename T >
TTStubBuilder< T >::~TTStubBuilder(){}

/// Begin run
template< typename T >
void TTStubBuilder< T >::beginRun( const edm::Run& run, const edm::EventSetup& iSetup )
{
  /// Get the stub finding algorithm
  iSetup.get< TTStubAlgorithmRecord >().get( theStubFindingAlgoHandle );
  /// Print some information when loaded
  std::cout << std::endl;
  std::cout << "TTStubBuilder< " << templateNameFinder< T >() << " > loaded modules:"
            << "\n\tTTStubAlgorithm:\t" << theStubFindingAlgoHandle->AlgorithmName()
            << std::endl;
}

/// End run
template< typename T >
void TTStubBuilder< T >::endRun( const edm::Run& run, const edm::EventSetup& iSetup ){}

/// Implement the producer
template< typename T >
void TTStubBuilder< T >::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  //Retrieve tracker topology from geometry                                                                                                              
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
  edm::ESHandle< TrackerGeometry > tGeomHandle;
  iSetup.get< TrackerDigiGeometryRecord >().get( tGeomHandle );
  const TrackerGeometry* const theTrackerGeom = tGeomHandle.product();

  /// Prepare output
  std::unique_ptr< edmNew::DetSetVector< TTCluster< T > > > TTClusterDSVForOutput( new edmNew::DetSetVector< TTCluster< T > > );
  std::unique_ptr< edmNew::DetSetVector< TTStub< T > > > TTStubDSVForOutputTemp( new edmNew::DetSetVector< TTStub< T > > );
  std::unique_ptr< edmNew::DetSetVector< TTStub< T > > > TTStubDSVForOutputAccepted( new edmNew::DetSetVector< TTStub< T > > );
  std::unique_ptr< edmNew::DetSetVector< TTStub< T > > > TTStubDSVForOutputRejected( new edmNew::DetSetVector< TTStub< T > > );

  /// Get the Clusters already stored away
  edm::Handle< edmNew::DetSetVector< TTCluster< T > > > clusterHandle;
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
    edmNew::DetSet< TTCluster< T > > lowerClusters = (*clusterHandle)[ lowerDetid ];
    edmNew::DetSet< TTCluster< T > > upperClusters = (*clusterHandle)[ upperDetid ];

    /// If there are Clusters in both sensors
    /// you can try and make a Stub
    /// This is ~redundant
    if ( lowerClusters.size() == 0 || upperClusters.size() == 0 )
      continue;

    /// Create the vectors of objects to be passed to the FastFillers
    std::vector< TTCluster< T > > tempInner; 
    std::vector< TTCluster< T > > tempOuter; 
    std::vector< TTStub< T > >   tempOutput; 
    tempInner.clear();
    tempOuter.clear();
    tempOutput.clear();

    /// Get chip size information
    const GeomDetUnit* det0 = theTrackerGeom->idToDetUnit( lowerDetid );
    const PixelGeomDetUnit* pix0 = dynamic_cast< const PixelGeomDetUnit* >( det0 );
    const PixelTopology* top0 = dynamic_cast< const PixelTopology* >( &(pix0->specificTopology()) );
    const int chipSize = 2 * top0->rowsperroc(); /// Need to find ASIC size in half-strip units

    std::map< int, std::vector< TTStub< T > > > moduleStubs; /// Temporary storage for stubs before max check

    /// Loop over pairs of Clusters
    for ( auto lowerClusterIter = lowerClusters.begin();
               lowerClusterIter != lowerClusters.end();
               ++lowerClusterIter ) {
      for ( auto upperClusterIter = upperClusters.begin();
                 upperClusterIter != upperClusters.end();
                 ++upperClusterIter ) {

        /// Build a temporary Stub
        TTStub< T > tempTTStub( stackDetid );
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

          /// Put in the output
          if ( maxStubs == 0 )
          {
            /// This means that ALL stubs go into the output
            tempInner.push_back( *lowerClusterIter );
            tempOuter.push_back( *upperClusterIter );
            tempOutput.push_back( tempTTStub );
          }
          else
          {
            /// This means that only some of them do
            /// Put in the temporary output
            int chip = tempTTStub.getTriggerPosition() / chipSize; /// Find out which ASIC
            if ( moduleStubs.find( chip ) == moduleStubs.end() ) /// Already a stub for this ASIC?
            {
              /// No, so new entry
              std::vector< TTStub< T > > tempStubs;
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
            tempOutput.push_back( ts );
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
          std::sort( bendMap.begin(), bendMap.end(), TTStubBuilder< T >::SortStubBendPairs );

          for ( unsigned int i = 0; i < maxStubs; ++i )
          {
            /// Put the highest momenta (lowest bend) stubs into the event
            tempInner.push_back( *(is.second[bendMap[i].first].getClusterRef(0)) );
            tempOuter.push_back( *(is.second[bendMap[i].first].getClusterRef(1)) );
            tempOutput.push_back( is.second[bendMap[i].first] );
          }
        }
      } /// End of loop over temp output
    } /// End store only the selected stubs if max no. stub/ROC is set
    /// Create the FastFillers
    if ( tempInner.size() > 0 )
    {
      typename edmNew::DetSetVector< TTCluster< T > >::FastFiller lowerOutputFiller( *TTClusterDSVForOutput, lowerDetid );
      for ( unsigned int m = 0; m < tempInner.size(); m++ )
      {
        lowerOutputFiller.push_back( tempInner.at(m) );
      }
      if ( lowerOutputFiller.empty() )
        lowerOutputFiller.abort();
    }

    if ( tempOuter.size() > 0 )
    {
      typename edmNew::DetSetVector< TTCluster< T > >::FastFiller upperOutputFiller( *TTClusterDSVForOutput, upperDetid );
      for ( unsigned int m = 0; m < tempOuter.size(); m++ )
      {
        upperOutputFiller.push_back( tempOuter.at(m) );
      }
      if ( upperOutputFiller.empty() )
        upperOutputFiller.abort();
    }

    if ( tempOutput.size() > 0 )
    {
      typename edmNew::DetSetVector< TTStub< T > >::FastFiller tempOutputFiller( *TTStubDSVForOutputTemp, stackDetid);
      for ( unsigned int m = 0; m < tempOutput.size(); m++ )
      {
        tempOutputFiller.push_back( tempOutput.at(m) );
      }
      if ( tempOutputFiller.empty() )
        tempOutputFiller.abort();
    }

  } /// End of loop over detector elements

  /// Put output in the event (1)
  /// Get also the OrphanHandle of the accepted clusters
  edm::OrphanHandle< edmNew::DetSetVector< TTCluster< T > > > TTClusterAcceptedHandle = iEvent.put( std::move(TTClusterDSVForOutput), "ClusterAccepted" );

  /// Now, correctly reset the output
  typename edmNew::DetSetVector< TTStub< T > >::const_iterator stubDetIter;

  for ( stubDetIter = TTStubDSVForOutputTemp->begin();
        stubDetIter != TTStubDSVForOutputTemp->end();
        ++stubDetIter ) {
    /// Get the DetId and prepare the FastFiller
    DetId thisStackedDetId = stubDetIter->id();
    typename edmNew::DetSetVector< TTStub< T > >::FastFiller acceptedOutputFiller( *TTStubDSVForOutputAccepted, thisStackedDetId );

    /// detid of the two components. 
    ///This should be done via a TrackerTopology method that is not yet available.
    DetId lowerDetid = thisStackedDetId+1;
    DetId upperDetid = thisStackedDetId+2;

    /// Get the DetSets of the clusters
    edmNew::DetSet< TTCluster< T > > lowerClusters = (*TTClusterAcceptedHandle)[ lowerDetid ];
    edmNew::DetSet< TTCluster< T > > upperClusters = (*TTClusterAcceptedHandle)[ upperDetid ];

    /// Get the DetSet of the stubs
    edmNew::DetSet< TTStub< T > > theseStubs = (*TTStubDSVForOutputTemp)[ thisStackedDetId ];

    /// Prepare the new DetSet to replace the current one
    /// Loop over the stubs
    typename edmNew::DetSet< TTCluster< T > >::iterator clusterIter;
    typename edmNew::DetSet< TTStub< T > >::iterator stubIter;
    for ( stubIter = theseStubs.begin();
          stubIter != theseStubs.end();
          ++stubIter ) {
      /// Create a temporary stub
      TTStub< T > tempTTStub( stubIter->getDetId() );

      /// Compare the clusters stored in the stub with the ones of this module
      edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > lowerClusterToBeReplaced = stubIter->getClusterRef(0);
      edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > upperClusterToBeReplaced = stubIter->getClusterRef(1);

      bool lowerOK = false;
      bool upperOK = false;

      for ( clusterIter = lowerClusters.begin();
            clusterIter != lowerClusters.end() && !lowerOK;
            ++clusterIter ) {
        if ( clusterIter->getHits() == lowerClusterToBeReplaced->getHits() ) {
          tempTTStub.addClusterRef( edmNew::makeRefTo( TTClusterAcceptedHandle, clusterIter ) );
          lowerOK = true;
        }
      }

      for ( clusterIter = upperClusters.begin();
            clusterIter != upperClusters.end() && !upperOK;
            ++clusterIter ) {
        if ( clusterIter->getHits() == upperClusterToBeReplaced->getHits() ) {
          tempTTStub.addClusterRef( edmNew::makeRefTo( TTClusterAcceptedHandle, clusterIter ) );
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
  iEvent.put( std::move(TTStubDSVForOutputAccepted), "StubAccepted" );
  iEvent.put( std::move(TTStubDSVForOutputRejected), "StubRejected" );
}

/// Sort routine for stub ordering
template< typename T >
bool TTStubBuilder< T >::SortStubBendPairs( const std::pair< unsigned int, double >& left, const std::pair< unsigned int, double >& right )
{
  return left.second < right.second;
}

#endif
