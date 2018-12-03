#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"

#include "FWCore/Framework/interface/ESWatcher.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"

#include "RecoLocalFastTime/Records/interface/MTDCPERecord.h"
#include "RecoLocalFastTime/FTLClusterizer/interface/MTDClusterParameterEstimator.h"
#include "RecoLocalFastTime/FTLClusterizer/interface/MTDCPEBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class MTDTrackingRecHitProducer : public edm::stream::EDProducer<> {
  
 public:
  explicit MTDTrackingRecHitProducer(const edm::ParameterSet& ps);
  ~MTDTrackingRecHitProducer() override;

  void produce(edm::Event& evt, const edm::EventSetup& es) override;
  void run(edm::Handle<edmNew::DetSetVector<FTLCluster> >  inputHandle,
	   MTDTrackingDetSetVector &output);
  
 private:
  const edm::EDGetTokenT<FTLClusterCollection> ftlbClusters_; // collection of barrel digis
  const edm::EDGetTokenT<FTLClusterCollection> ftleClusters_; // collection of endcap digis
  
  const std::string ftlbInstance_; // instance name of barrel hits
  const std::string ftleInstance_; // instance name of endcap hits
  
  edm::ESWatcher<MTDDigiGeometryRecord> geomwatcher_;
  const MTDGeometry* geom_;

  edm::ESWatcher<MTDCPERecord> cpewatcher_;
  const MTDCPEBase* cpe_;
};

MTDTrackingRecHitProducer::MTDTrackingRecHitProducer(const edm::ParameterSet& ps) :
  ftlbClusters_( consumes<FTLClusterCollection>( ps.getParameter<edm::InputTag>("barrelClusters") ) ),
  ftleClusters_( consumes<FTLClusterCollection>( ps.getParameter<edm::InputTag>("endcapClusters") ) )
{  
  produces< MTDTrackingDetSetVector >();
}

MTDTrackingRecHitProducer::~MTDTrackingRecHitProducer() {
}

void
MTDTrackingRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  
  edm::ESHandle<MTDGeometry> geom;
  if( geomwatcher_.check(es) || geom_ == nullptr ) {
    es.get<MTDDigiGeometryRecord>().get(geom);
    geom_ = geom.product();
  }

  edm::ESHandle<MTDClusterParameterEstimator> cpe;
  if( cpewatcher_.check(es) || cpe_ == nullptr ) {
    es.get<MTDCPERecord>().get("MTDCPEBase",cpe);
    cpe_ = dynamic_cast< const MTDCPEBase* >(cpe.product());
  }

  if ( ! geom_ ) 
    {
      throw cms::Exception("MTDTrackingRecHitProducer") << "Geometry is not available -- can't run!";
      return;   // clusterizer is invalid, bail out
    }
  
  if ( ! cpe_ ) 
    {
      throw cms::Exception("MTDTrackingRecHitProducer") << "CPE is not ready -- can't run!";
      return;   // clusterizer is invalid, bail out
    }
  
  edm::Handle< FTLClusterCollection > inputBarrel;
  evt.getByToken( ftlbClusters_, inputBarrel);
  
  edm::Handle< FTLClusterCollection > inputEndcap;
  evt.getByToken( ftleClusters_, inputEndcap);
  
  auto outputhits = std::make_unique<MTDTrackingDetSetVector>();
  auto& theoutputhits = *outputhits;
  
  run(inputBarrel,theoutputhits);
  run(inputEndcap,theoutputhits);
  
  evt.put(std::move(outputhits));
}

//---------------------------------------------------------------------------
//!  Iterate over DetUnits, then over Clusters and invoke the CPE on each,
//!  and make a RecHit to store the result.
//---------------------------------------------------------------------------
void MTDTrackingRecHitProducer::run(edm::Handle<FTLClusterCollection>  inputHandle,
				    MTDTrackingDetSetVector &output) 
{
  int numberOfDetUnits = 0;
  int numberOfClusters = 0;
  
  const edmNew::DetSetVector<FTLCluster>& input = *inputHandle;
  edmNew::DetSetVector<FTLCluster>::const_iterator DSViter=input.begin();
  
  for ( ; DSViter != input.end() ; DSViter++) {
    numberOfDetUnits++;
    unsigned int detid = DSViter->detId();
    DetId detIdObject( detid );  
    const auto& genericDet = geom_->idToDetUnit(detIdObject);
    if( genericDet == nullptr ) {
      throw cms::Exception("MTDTrackingRecHitProducer") << "GeographicalID: " << std::hex
						<< detid
						<< " is invalid!" << std::dec
						<< std::endl;
    }

    MTDTrackingDetSetVector::FastFiller recHitsOnDet(output,detid);
      
    edmNew::DetSet<FTLCluster>::const_iterator clustIt = DSViter->begin(), clustEnd = DSViter->end();
    
    for ( ; clustIt != clustEnd; clustIt++) {
	numberOfClusters++;
	MTDClusterParameterEstimator::ReturnType tuple = cpe_->getParameters( *clustIt, *genericDet );
	LocalPoint lp( std::get<0>(tuple) );
	LocalError le( std::get<1>(tuple) );
	// Create a persistent edm::Ref to the cluster
	edm::Ref< edmNew::DetSetVector<FTLCluster>, FTLCluster > cluster = edmNew::makeRefTo( inputHandle, clustIt);
	// Make a RecHit and add it to the DetSet
	MTDTrackingRecHit hit( lp, le, *genericDet, cluster);
	// Now save it =================
	recHitsOnDet.push_back(hit);
    } //  <-- End loop on Clusters      
  } //    <-- End loop on DetUnits
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( MTDTrackingRecHitProducer );
