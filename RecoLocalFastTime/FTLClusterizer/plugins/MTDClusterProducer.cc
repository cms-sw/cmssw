/** MTDClusterProducer.cc
 */

// Our own stuff
#include "MTDClusterProducer.h"
#include "MTDThresholdClusterizer.h"

// Data Formats
#include "DataFormats/FTLRecHit/interface/FTLRecHit.h"

// Framework
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

// STL
#include <vector>
#include <memory>
#include <string>
#include <iostream>

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"


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
  //--- Declare to the EDM what kind of collections we will be making.
  produces<FTLClusterCollection>("FTLBarrel"); 
  produces<FTLClusterCollection>("FTLEndcap"); 
  
  //--- Make the algorithm(s) according to what the user specified
  //--- in the ParameterSet.
  setupClusterizer(conf);
}

// Destructor
MTDClusterProducer::~MTDClusterProducer() { 
  delete clusterizer_;
  
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
  
  // Step B: create the final output collection
  auto outputBarrel = std::make_unique< FTLClusterCollection>();
  auto outputEndcap = std::make_unique< FTLClusterCollection>();
  
  run(*inputBarrel, *outputBarrel );
  run(*inputEndcap, *outputEndcap );

  e.put(std::move(outputBarrel),"FTLBarrel");
  e.put(std::move(outputEndcap),"FTLEndcap");
}

//---------------------------------------------------------------------------
void MTDClusterProducer::setupClusterizer(const edm::ParameterSet& conf)  {

    if ( clusterMode_ == "MTDThresholdClusterizer" ) {
      clusterizer_ = new MTDThresholdClusterizer(conf);
      readyToCluster_ = true;
    } 
    else {
      edm::LogError("MTDClusterProducer") << "[MTDClusterProducer]:"
		<<" choice " << clusterMode_ << " is invalid.\n"
		<< "Possible choices:\n" 
		<< "    MTDThresholdClusterizer";
      readyToCluster_ = false;
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
  
  int numberOfHits = input.size();
  //  int numberOfClusters = 0;

  clusterizer_->clusterize( input , geom_, output);
  
  // if ((maxTotalClusters_ >= 0) && (numberOfClusters > maxTotalClusters_)) {
  //     edm::LogError("TooManyClusters") <<  "Limit on the number of clusters exceeded. An empty cluster collection will be produced instead.\n";
  //     edmNew::DetSetVector<MTDCluster> empty;
  //     empty.swap(output);
  //     break;
  // }
  
  LogDebug ("MTDClusterProducer") << " Executing " 
				  <<  clusterMode_ << " resulted in " << output.size()
				  << " MTDClusters for " << input.size() << " Hits."; 
}




#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MTDClusterProducer);

