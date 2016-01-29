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
void TTClusterBuilder< Ref_Phase2TrackerDigi_ >::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  std::cout<<"debug0"<<std::endl;
  /// Prepare output
  std::auto_ptr< edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > > > TTClusterDSVForOutput( new edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > > );
  std::cout<<"debug01"<<std::endl;
  std::map< DetId, std::vector< Ref_Phase2TrackerDigi_ > > rawHits; /// This is a map containing hits:
                                                            /// a vector of type Ref_PixelDigi_ is mapped wrt
                                                            /// the DetId
  std::cout<<"debug02"<<std::endl;
  this->RetrieveRawHits( rawHits, iEvent );
  std::cout<<"debug1"<<std::endl;
  //added from here
  //Retrieve tracker topology from geometry                                                                                                               
  
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
 
  edm::ESHandle< TrackerGeometry > tGeomHandle;
  iSetup.get< TrackerDigiGeometryRecord >().get( tGeomHandle );
  const TrackerGeometry* const theTrackerGeom = tGeomHandle.product();
  
  //added by me                                                                                                                                           
  for (TrackerGeometry::DetContainer::const_iterator gd=theTrackerGeom->dets().begin(); gd != theTrackerGeom->dets().end(); gd++)
    {     
      std::cout<<"debug3"<<std::endl;
      DetId detid = (*gd)->geographicalId();
      std::cout << " TTClusterBuilder taking into account the DetId: " << detid.rawId();
      std::cout << " is lower? " << tTopo->isLower(detid.rawId()) << std::endl;

      const GeomDetUnit* det0 = theTrackerGeom->idToDetUnit( tTopo->Lower(detid) );
      const GeomDetUnit* det1 = theTrackerGeom->idToDetUnit( tTopo->Upper(detid) );
      
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
      std::vector< std::vector< Ref_Phase2TrackerDigi_ > > innerHits, outerHits;
      
      /// Find the hits in each stack member                                                                                                                
      typename std::map< DetId, std::vector< Ref_Phase2TrackerDigi_ > >::const_iterator innerHitFind = rawHits.find(tTopo->isLower(detid) );
      typename std::map< DetId, std::vector< Ref_Phase2TrackerDigi_ > >::const_iterator outerHitFind = rawHits.find(tTopo->PartnerDetId(detid) );
      
      /// If there are hits, cluster them                                                                                                                    
      /// It is the TTClusterAlgorithm::Cluster method which                                                                                                
      /// calls the constructor to the Cluster class!                                                                                                        
      if ( innerHitFind != rawHits.end() ) theClusterFindingAlgoHandle->Cluster( innerHits, innerHitFind->second, isPS );  
      if ( outerHitFind != rawHits.end() ) theClusterFindingAlgoHandle->Cluster( outerHits, outerHitFind->second, false ); 
      
      /// Create TTCluster objects and store them                                                                                                          
      /// Use the FastFiller with edmNew::DetSetVector                                                                                                       
      { 
	edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > >::FastFiller innerOutputFiller( *TTClusterDSVForOutput, tTopo->isLower(detid) ); 
      for ( unsigned int i = 0; i < innerHits.size(); i++ )
	{
	  TTCluster< Ref_Phase2TrackerDigi_ > temp( innerHits.at(i), detid, 0, storeLocalCoord );
	  innerOutputFiller.push_back( temp );
	}
      if ( innerOutputFiller.empty() )
        innerOutputFiller.abort();
     }
     {  
       edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > >::FastFiller outerOutputFiller( *TTClusterDSVForOutput, tTopo->PartnerDetId(detid) );
     for ( unsigned int i = 0; i < outerHits.size(); i++ )
       {
	 TTCluster< Ref_Phase2TrackerDigi_ > temp( outerHits.at(i), detid, 1, storeLocalCoord );
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
void TTClusterBuilder< Ref_Phase2TrackerDigi_ >::RetrieveRawHits( std::map< DetId, std::vector< Ref_Phase2TrackerDigi_ > > &mRawHits,
                                                          const edm::Event& iEvent )
{ std::cout<<"debug02i"<<std::endl;
  mRawHits.clear();
  std::cout<<"debug02ii"<<std::endl;
  /// Loop over the tags used to identify hits in the cfg file
  std::vector< edm::InputTag >::iterator it;
  for ( it = rawHitInputTags.begin();
        it != rawHitInputTags.end();
        ++it )
  { std::cout<<"debug02iii"<<std::endl;
    /// For each tag, get the corresponding handle
    //    edm::Handle< edm::DetSetVector< PixelDigi > > HitHandle;
    edm::Handle< edm::DetSetVector< Phase2TrackerDigi > > HitHandle;
    iEvent.getByLabel( it->label(), HitHandle );
    std::cout<<"debug02iv"<<std::endl;
    //    edm::DetSetVector<PixelDigi>::const_iterator detsIter; std::cout<<"debug02v"<<std::endl;
    //    edm::DetSet<PixelDigi>::const_iterator       hitsIter; std::cout<<"debug02vi"<<std::endl;
    edm::DetSetVector<Phase2TrackerDigi>::const_iterator detsIter; std::cout<<"debug02v"<<std::endl;
    edm::DetSet<Phase2TrackerDigi>::const_iterator       hitsIter; std::cout<<"debug02vi"<<std::endl;

    /// Loop over detector elements identifying PixelDigis
    for ( detsIter = HitHandle->begin();
          detsIter != HitHandle->end();
          detsIter++ )
    { std::cout<<"debug02vii"<<std::endl;
      DetId id = detsIter->id;

      /// Is it Pixel?
      //      if ( id.subdetId()==1 || id.subdetId()==2 )
      //  {
        /// Loop over Digis in this specific detector element
        for ( hitsIter = detsIter->data.begin();
              hitsIter != detsIter->data.end();
              hitsIter++ )
        {
	  //  if ( hitsIter->adc() >= ADCThreshold )
	  // {
            /// If the Digi is over threshold,
            /// accept it as a raw hit and put into map
            mRawHits[id].push_back( edm::makeRefTo( HitHandle, id , hitsIter ) );
	    // } /// End of threshold selection
        } /// End of loop over digis
	//} /// End of "is Pixel"
    } /// End of loop over detector elements
  } /// End of loop over tags
}

