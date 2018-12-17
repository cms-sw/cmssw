//---------------------------------------------------------------------------
//! \class MTDClusterProducer
//!
//! \brief EDProducer to cluster FTLRecHits into FTLClusters.
//!
//---------------------------------------------------------------------------
// Our own stuff
#include "MTDThresholdClusterizer.h"
#include "MTDClusterizerBase.h"

// Data Formats
#include "DataFormats/FTLRecHit/interface/FTLRecHit.h"

// STL
#include <vector>
#include <memory>
#include <string>
#include <iostream>

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

class MTDClusterProducer : public edm::stream::EDProducer<> {
  public:
    //--- Constructor, virtual destructor (just in case)
    explicit MTDClusterProducer(const edm::ParameterSet& conf);
    ~MTDClusterProducer() override;

    //--- The top-level event method.
    void produce(edm::Event& e, const edm::EventSetup& c) override;

    //--- Execute the algorithm(s).
    template<typename T>
    void run(const T& input,
             FTLClusterCollection & output);

    void setupClusterizer(const edm::ParameterSet& conf);

  private:
    edm::EDGetTokenT< FTLRecHitCollection >  btlHits_;
    edm::EDGetTokenT< FTLRecHitCollection >  etlHits_;

    std::string ftlbInstance_; // instance name of barrel clusters
    std::string ftleInstance_; // instance name of endcap clusters

    const std::string clusterMode_;         // user's choice of the clusterizer
    std::unique_ptr<MTDClusterizerBase> clusterizer_;    // what we got (for now, one ptr to base class)
    bool readyToCluster_;                   // needed clusterizers valid => good to go!

    edm::ESWatcher<MTDDigiGeometryRecord> geomwatcher_;
    const MTDGeometry* geom_;
    edm::ESWatcher<MTDTopologyRcd> topowatcher_;
    const MTDTopology* topo_;
};


//---------------------------------------------------------------------------
//!  Constructor: set the ParameterSet and defer all thinking to setupClusterizer().
//---------------------------------------------------------------------------
MTDClusterProducer::MTDClusterProducer(edm::ParameterSet const& conf) 
  : 
  clusterMode_( conf.getUntrackedParameter<std::string>("ClusterMode","MTDThresholdClusterizer") ),
  clusterizer_(nullptr),          // the default, in case we fail to make one
  readyToCluster_(false)   // since we obviously aren't
{
  btlHits_ = consumes< FTLRecHitCollection >( conf.getParameter<edm::InputTag>("srcBarrel") );
  etlHits_ = consumes< FTLRecHitCollection >( conf.getParameter<edm::InputTag>("srcEndcap") );
  ftlbInstance_ = conf.getParameter<std::string>("BarrelClusterName");
  ftleInstance_ = conf.getParameter<std::string>("EndcapClusterName");

  //--- Declare to the EDM what kind of collections we will be making.
  produces<FTLClusterCollection>(ftlbInstance_); 
  produces<FTLClusterCollection>(ftleInstance_); 
  
  //--- Make the algorithm(s) according to what the user specified
  //--- in the ParameterSet.
  setupClusterizer(conf);
}

// Destructor
MTDClusterProducer::~MTDClusterProducer() { 
}  

  
//---------------------------------------------------------------------------
//! The "Event" entrypoint: gets called by framework for every event
//---------------------------------------------------------------------------
void MTDClusterProducer::produce(edm::Event& e, const edm::EventSetup& es)
{
  // Step A.1: get input data
  edm::Handle< FTLRecHitCollection >  inputBarrel;
  edm::Handle< FTLRecHitCollection >  inputEndcap;
  e.getByToken(btlHits_, inputBarrel);
  e.getByToken(etlHits_, inputEndcap);
      
  // Step A.2: get event setup
  edm::ESHandle<MTDGeometry> geom;
  if( geomwatcher_.check(es) || geom_ == nullptr ) {
    es.get<MTDDigiGeometryRecord>().get(geom);
    geom_ = geom.product();
  }
  edm::ESHandle<MTDTopology> topo;
  if( topowatcher_.check(es) || topo_ == nullptr ) {
    es.get<MTDTopologyRcd>().get(topo);
    topo_ = topo.product();
  }
  
  // Step B: create the final output collection
  auto outputBarrel = std::make_unique< FTLClusterCollection>();
  auto outputEndcap = std::make_unique< FTLClusterCollection>();
  
  run(*inputBarrel, *outputBarrel );
  run(*inputEndcap, *outputEndcap );

  e.put(std::move(outputBarrel),ftlbInstance_);
  e.put(std::move(outputEndcap),ftleInstance_);
}

//---------------------------------------------------------------------------
void MTDClusterProducer::setupClusterizer(const edm::ParameterSet& conf)  {

    if ( clusterMode_ == "MTDThresholdClusterizer" ) {
      clusterizer_ = std::unique_ptr<MTDClusterizerBase>(new MTDThresholdClusterizer(conf));
      readyToCluster_ = true;
    } 
    else {
      readyToCluster_ = false;
      throw cms::Exception("MTDClusterProducer") << "[MTDClusterProducer]:"
						 <<" choice " << clusterMode_ << " is invalid.\n"
						 << "Possible choices:\n" 
						 << "    MTDThresholdClusterizer";
    }
}


//---------------------------------------------------------------------------
//!  Iterate over DetUnits, and invoke the PixelClusterizer on each.
//---------------------------------------------------------------------------
template<typename T>
void MTDClusterProducer::run(const T                              & input, 
			     FTLClusterCollection & output) 
{
  if ( ! readyToCluster_ ) {
    edm::LogError("MTDClusterProducer")
      <<" at least one clusterizer is not ready -- can't run!" ;
    // TO DO: throw an exception here?  The user may want to know...
    return;   // clusterizer is invalid, bail out
  }
  
  clusterizer_->clusterize( input , geom_, topo_, output);
  
  LogDebug ("MTDClusterProducer") << " Executing " 
				  <<  clusterMode_ << " resulted in " << output.size()
				  << " MTDClusters for " << input.size() << " Hits."; 
}




#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MTDClusterProducer);

