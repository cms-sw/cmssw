#include "MeasurementTrackerEventProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"

MeasurementTrackerEventProducer::MeasurementTrackerEventProducer(const edm::ParameterSet &iConfig) :
    measurementTrackerLabel_(iConfig.getParameter<std::string>("measurementTracker")),
    pset_(iConfig),
    theInactivePixelDetectorLabels(iConfig.getParameter<std::vector<edm::InputTag> >("inactivePixelDetectorLabels")),
    theInactiveStripDetectorLabels(iConfig.getParameter<std::vector<edm::InputTag> >("inactiveStripDetectorLabels"))
{
    //the measurement tracking is set to skip clusters, the other option is set from outside
    selfUpdateSkipClusters_=iConfig.exists("skipClusters");
    if (selfUpdateSkipClusters_)
    {
        edm::InputTag skip=iConfig.getParameter<edm::InputTag>("skipClusters");
        if (skip==edm::InputTag("")) selfUpdateSkipClusters_=false;
    }
    LogDebug("MeasurementTracker")<<"skipping clusters: "<<selfUpdateSkipClusters_;

    produces<MeasurementTrackerEvent>();
}

void
MeasurementTrackerEventProducer::produce(edm::Event &iEvent, const edm::EventSetup& iSetup)
{
    edm::ESHandle<MeasurementTracker> measurementTracker;
    iSetup.get<CkfComponentsRecord>().get(measurementTrackerLabel_, measurementTracker);

    // create new data structures from templates
    std::auto_ptr<StMeasurementDetSet> stripData(new StMeasurementDetSet(measurementTracker->stripDetConditions()));
    std::auto_ptr<PxMeasurementDetSet> pixelData(new PxMeasurementDetSet(measurementTracker->pixelDetConditions()));
    //std::cout << "Created new strip data @" << &* stripData << std::endl;
    std::vector<bool> stripClustersToSkip;
    std::vector<bool> pixelClustersToSkip;

    // fill them
    updateStrips(iEvent, *stripData, stripClustersToSkip);
    updatePixels(iEvent, *pixelData, pixelClustersToSkip);

    // put into MTE
    std::auto_ptr<MeasurementTrackerEvent> out(new MeasurementTrackerEvent(*measurementTracker, stripData.release(), pixelData.release(), stripClustersToSkip, pixelClustersToSkip));

    // put into event
    iEvent.put(out);
}

void 
MeasurementTrackerEventProducer::updatePixels( const edm::Event& event, PxMeasurementDetSet & thePxDets, std::vector<bool> & pixelClustersToSkip ) const
{
  // start by clearinng everything
  thePxDets.setEmpty();

  bool switchOffPixelsIfEmpty = (!pset_.existsAs<bool>("switchOffPixelsIfEmpty")) ||
                                (pset_.getParameter<bool>("switchOffPixelsIfEmpty"));
  std::vector<uint32_t> rawInactiveDetIds; 
  if (!theInactivePixelDetectorLabels.empty()) {
    edm::Handle<DetIdCollection> detIds;
    for (std::vector<edm::InputTag>::const_iterator itt = theInactivePixelDetectorLabels.begin(), edt = theInactivePixelDetectorLabels.end(); 
            itt != edt; ++itt) {
      if (event.getByLabel(*itt, detIds)){
        rawInactiveDetIds.insert(rawInactiveDetIds.end(), detIds->begin(), detIds->end());
      }else{
        static std::atomic<bool> iFailedAlready{false};
        bool expected = false;
        if (iFailedAlready.compare_exchange_strong(expected,true,std::memory_order_acq_rel)){
          edm::LogError("MissingProduct")<<"I fail to get the list of inactive pixel modules, because of 4.2/4.4 event content change.";
        }
      }
    }
    if (!rawInactiveDetIds.empty()) std::sort(rawInactiveDetIds.begin(), rawInactiveDetIds.end());
    // mark as inactive if in rawInactiveDetIds
    int i=0, endDet = thePxDets.size();
    unsigned int idp=0;
    for ( auto id : rawInactiveDetIds) {
        if (id==idp) continue; // skip multiple id
        idp=id;
        i=thePxDets.find(id,i);
        assert(i!=endDet && id == thePxDets.id(i));
        thePxDets.setActiveThisEvent(i,false);
    }
  }

  // Pixel Clusters
  std::string pixelClusterProducer = pset_.getParameter<std::string>("pixelClusterProducer");
  if( pixelClusterProducer.empty() ) { //clusters have not been produced
    if (switchOffPixelsIfEmpty) {
      thePxDets.setActiveThisEvent(false);
    }
  }else{  

    edm::Handle<edmNew::DetSetVector<SiPixelCluster> > & pixelClusters = thePxDets.handle();
    event.getByLabel(pixelClusterProducer, pixelClusters);
    
    const  edmNew::DetSetVector<SiPixelCluster>* pixelCollection = pixelClusters.product();
   
    if (switchOffPixelsIfEmpty && pixelCollection->empty()) {
       thePxDets.setActiveThisEvent(false);
    } else { 

       //std::cout <<"updatePixels "<<pixelCollection->dataSize()<<std::endl;
       pixelClustersToSkip.resize(pixelCollection->dataSize());
       std::fill(pixelClustersToSkip.begin(),pixelClustersToSkip.end(),false);
       
       if(selfUpdateSkipClusters_) {
          edm::Handle<edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > > pixelClusterMask;
         //and get the collection of pixel ref to skip
         event.getByLabel(pset_.getParameter<edm::InputTag>("skipClusters"),pixelClusterMask);
         LogDebug("MeasurementTracker")<<"getting pxl refs to skip";
         if (pixelClusterMask.failedToGet())edm::LogError("MeasurementTracker")<<"not getting the pixel clusters to skip";
	 if (pixelClusterMask->refProd().id()!=pixelClusters.id()){
	   edm::LogError("ProductIdMismatch")<<"The pixel masking does not point to the proper collection of clusters: "<<pixelClusterMask->refProd().id()<<"!="<<pixelClusters.id();
	 }
	 pixelClusterMask->copyMaskTo(pixelClustersToSkip);
       }
          

       // FIXME: should check if lower_bound is better
       int i = 0, endDet = thePxDets.size();
       for (edmNew::DetSetVector<SiPixelCluster>::const_iterator it = pixelCollection->begin(), ed = pixelCollection->end(); it != ed; ++it) {
         edmNew::DetSet<SiPixelCluster> set(*it);
         unsigned int id = set.id();
         while ( id != thePxDets.id(i)) { 
             ++i;
             if (endDet==i) throw "we have a problem!!!!";
         }
         // push cluster range in det
         if ( thePxDets.isActive(i) ) {
             thePxDets.update(i,set);         
	 }
       }
    }
  }

}

void 
MeasurementTrackerEventProducer::updateStrips( const edm::Event& event, StMeasurementDetSet & theStDets, std::vector<bool> & stripClustersToSkip ) const
{
  typedef edmNew::DetSet<SiStripCluster>   StripDetSet;

  std::vector<uint32_t> rawInactiveDetIds;
  getInactiveStrips(event,rawInactiveDetIds);

  // Strip Clusters
  std::string stripClusterProducer = pset_.getParameter<std::string>("stripClusterProducer");
  //first clear all of them
  theStDets.setEmpty();


  if( !stripClusterProducer.compare("") )  return;  //clusters have not been produced

  const int endDet = theStDets.size();
 

  // mark as inactive if in rawInactiveDetIds
  int i=0;
  unsigned int idp=0;
  for ( auto id : rawInactiveDetIds) {
    if (id==idp) continue; // skip multiple id
    idp=id;
    i=theStDets.find(id,i);
    assert(i!=endDet && id == theStDets.id(i));
    theStDets.setActiveThisEvent(i,false);
  }

  //=========  actually load cluster =============
  if(!theStDets.isRegional()){
    edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusterHandle;
    event.getByLabel(stripClusterProducer, clusterHandle);
    const edmNew::DetSetVector<SiStripCluster>* clusterCollection = clusterHandle.product();
    
    
    if (selfUpdateSkipClusters_){
      edm::Handle<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > > stripClusterMask;
      //and get the collection of pixel ref to skip
      LogDebug("MeasurementTracker")<<"getting strp refs to skip";
      event.getByLabel(pset_.getParameter<edm::InputTag>("skipClusters"),stripClusterMask);
      if (stripClusterMask.failedToGet())  edm::LogError("MeasurementTracker")<<"not getting the strip clusters to skip";
      if (stripClusterMask->refProd().id()!=clusterHandle.id()){
	edm::LogError("ProductIdMismatch")<<"The strip masking does not point to the proper collection of clusters: "<<stripClusterMask->refProd().id()<<"!="<<clusterHandle.id();
      }
      stripClusterMask->copyMaskTo(stripClustersToSkip);
    }
    
    theStDets.handle() = clusterHandle;
    int i=0;
    // cluster and det and in order (both) and unique so let's use set intersection
    for ( auto j = 0U; j< (*clusterCollection).size(); ++j) {
      unsigned int id = (*clusterCollection).id(j);
      while ( id != theStDets.id(i)) { // eventually change to lower_bound
	++i;
	if (endDet==i) throw "we have a problem in strips!!!!";
      }
      
      // push cluster range in det
      if ( theStDets.isActive(i) )
	theStDets.update(i,j);
    }

  }else{   // regional
    throw cms::Exception("NotSupported") << "Regional non-on-demand unpacking is not supported." << std::endl;
#if 0 
    //then set the not-empty ones only
    edm::Handle<edm::RefGetter<SiStripCluster> > refClusterHandle;
    event.getByLabel(stripClusterProducer, refClusterHandle);
    
    std::string lazyGetter = pset_.getParameter<std::string>("stripLazyGetterProducer");
    edm::Handle<edm::LazyGetter<SiStripCluster> > lazyClusterHandle;
    event.getByLabel(lazyGetter,lazyClusterHandle);
    
    if(selfUpdateSkipClusters_){
      edm::Handle<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > > stripClusterMask;
      LogDebug("MeasurementTracker")<<"getting reg strp refs to skip";
      event.getByLabel(pset_.getParameter<edm::InputTag>("skipClusters"),stripClusterMask);
      if (stripClusterMask.failedToGet())edm::LogError("MeasurementTracker")<<"not getting the strip clusters to skip";
      if (stripClusterMask->refProd().id()!=lazyClusterHandle.id()){
	edm::LogError("ProductIdMismatch")<<"The strip masking does not point to the proper collection of clusters: "<<stripClusterMask->refProd().id()<<"!="<<lazyClusterHandle.id();
      }       
      stripClusterMask->copyMaskTo(stripClustersToSkip);
    }
    
    theStDets.regionalHandle() =  lazyClusterHandle;
    
    uint32_t tmpId=0;
    std::vector<SiStripCluster>::const_iterator beginIterator;
    edm::RefGetter<SiStripCluster>::const_iterator iregion = refClusterHandle->begin();
    for(;iregion!=refClusterHandle->end();++iregion) {
      const edm::RegionIndex<SiStripCluster>& region = *iregion;
      std::vector<SiStripCluster>::const_iterator icluster = region.begin();
      const std::vector<SiStripCluster>::const_iterator endIterator = region.end();
      tmpId = icluster->geographicalId();
      beginIterator = icluster;
      
      //std::cout << "== tmpId ad inizio loop dentro region: " << tmpId << std::endl;
      
      for (;icluster!=endIterator;icluster++) {
	//std::cout << "===== cluster id,pos " 
	//  << icluster->geographicalId() << " , " << icluster->barycenter()
	//  << std::endl;
	//std::cout << "=====making ref in recHits() " << std::endl;
	if( icluster->geographicalId() != tmpId){ 
	  //std::cout << "geo!=tmpId" << std::endl;
	  
	  //cannot we avoid to update the det with detId of itself??  (sure we can!, done!)
	  theStDets.update(concreteDetUpdatable(tmpId)->index(),beginIterator,icluster);
	  
	  tmpId = icluster->geographicalId();
	  beginIterator = icluster;
	  if( icluster == (endIterator-1)){
	    theStDets.update(concreteDetUpdatable(tmpId)->index(),icluster,endIterator);
	  }   
	}else if( icluster == (endIterator-1)){	   
	  theStDets.update(concreteDetUpdatable(tmpId)->index(),beginIterator,endIterator);	 
	}
      }//end loop cluster in one ragion
    }
#endif
  }//end of block for updating with regional clusters 

}

void 
MeasurementTrackerEventProducer::getInactiveStrips(const edm::Event& event,std::vector<uint32_t> & rawInactiveDetIds) const
{
  if (!theInactiveStripDetectorLabels.empty()) {
    edm::Handle<DetIdCollection> detIds;
    for (std::vector<edm::InputTag>::const_iterator itt = theInactiveStripDetectorLabels.begin(), edt = theInactiveStripDetectorLabels.end(); 
	 itt != edt; ++itt) {
      event.getByLabel(*itt, detIds);
      rawInactiveDetIds.insert(rawInactiveDetIds.end(), detIds->begin(), detIds->end());
    }
    if (!rawInactiveDetIds.empty()) std::sort(rawInactiveDetIds.begin(), rawInactiveDetIds.end());
  }

}



DEFINE_FWK_MODULE(MeasurementTrackerEventProducer);
