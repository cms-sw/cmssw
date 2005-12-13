/** SiPixelClusterProducer.cc
 * -------------------------------------------- 
 * Description:  see SiPixelClusterProducer.h
 * Author:  P. Maksimovic, massaging the strip version from Oliver Gutsche.
 * History: Oct 14, 2005, initial version
 * -------------------------------------------- 
 */


// Our own stuff
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelClusterProducer.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/PixelThresholdClusterizer.h"

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
class GeometricDet;   // hack in 0.2.0pre5, should be OK for pre6
#include "Geometry/TrackerSimAlgo/interface/PixelGeomDetUnit.h"

// Data Formats
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelClusterCollection.h"
#include "DataFormats/DetId/interface/DetId.h"

// Framework
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

// STL
#include <vector>
#include <memory>
#include <string>
#include <iostream>

namespace cms
{

  //---------------------------------------------------------------------------
  //!  Constructor: set the ParameterSet and defer all thinking to setupClusterizer().
  //---------------------------------------------------------------------------
  SiPixelClusterProducer::SiPixelClusterProducer(edm::ParameterSet const& conf) 
    : 
    conf_(conf),
    clusterMode_("None"),     // bogus
    clusterizer_(0),          // the default, in case we fail to make one
    readyToCluster_(false)    // since we obviously aren't
  {
    //--- Declare to the EDM what kind of collections we will be making.
    produces<SiPixelClusterCollection>();

    //--- Make the algorithm(s) according to what the user specified
    //--- in the ParameterSet.
    setupClusterizer();
  }


  // Virtual destructor needed, just in case.
  SiPixelClusterProducer::~SiPixelClusterProducer() { }  



  //---------------------------------------------------------------------------
  //! The "Event" entrypoint: gets called by framework for every event
  //---------------------------------------------------------------------------
  void SiPixelClusterProducer::produce(edm::Event& e, const edm::EventSetup& es)
  {
    // retrieve producer name of input SiPixelDigiCollection
    std::string digiProducer = conf_.getParameter<std::string>("DigiProducer");

    // Step A.1: get input data
    edm::Handle<PixelDigiCollection> stripDigis;
    e.getByLabel(digiProducer, stripDigis);

    // Step A.2: get event setup
    edm::ESHandle<TrackerGeom> geom;
    es.get<TrackerDigiGeometryRecord>().get( geom );


    // Step B: create empty output collection
    std::auto_ptr<SiPixelClusterCollection> output(new SiPixelClusterCollection);

    // Step C: Iterate over DetIds and invoke the strip clusterizer algorithm
    // on each DetUnit
    run( stripDigis.product(), *output, geom );

    // Step D: write output to file
    e.put(output);

  }



  //---------------------------------------------------------------------------
  //!  Set up the specific algorithm we are going to use.  
  //!  TO DO: in the future, we should allow for a different algorithm for 
  //!  each detector subset (e.g. barrel vs forward, per layer, etc).
  //---------------------------------------------------------------------------
  void SiPixelClusterProducer::setupClusterizer() 
  {
    clusterMode_ = conf_.getParameter<std::string>("ClusterMode");

    if ( clusterMode_ == "PixelThresholdClusterizer" ) {
      clusterizer_ = new PixelThresholdClusterizer(conf_);
      readyToCluster_ = true;
    } 
    else {
      std::cout << "[SiPixelClusterProducer]:"
		<<" choice " << clusterMode_ << " is invalid.\n"
		<< "Possible choices:\n" 
		<< "    PixelThresholdClusterizer" << std::endl;
      readyToCluster_ = false;
    }
  }


  //---------------------------------------------------------------------------
  //!  Iterate over DetUnits, and invoke the PixelClusterizer on each.
  //---------------------------------------------------------------------------
  void SiPixelClusterProducer::run(const PixelDigiCollection* input, 
				   SiPixelClusterCollection &output,
				   edm::ESHandle<TrackerGeom> & geom)
  {
    if ( ! readyToCluster_ ) {
      std::cout << "[SiPixelClusterProducer]:"
		<<" at least one clusterizer is not ready -- can't run!" 
		<< std::endl;
      // TO DO: throw an exception here?  The user may want to know...
      return;   // clusterizer is invalid, bail out
    }

    int numberOfDetUnits = 0;
    int numberOfClusters = 0;
    
    // get vector of detunit ids
    const std::vector<unsigned int> detIDs = input->detIDs();
    

    //--- Loop over detunits.
    std::vector<unsigned int>::const_iterator 
      detunit_it  = detIDs.begin(),
      detunit_end = detIDs.end();
    for ( ; detunit_it != detunit_end; ++detunit_it ) {
      //
      unsigned int detid = *detunit_it;
      ++numberOfDetUnits;
      const PixelDigiCollection::Range digiRange = input->get(detid);
      PixelDigiCollection::ContainerIterator 
	digiRangeIteratorBegin = digiRange.first;
      PixelDigiCollection::ContainerIterator 
	digiRangeIteratorEnd   = digiRange.second;

      // TO DO: if we were to allow concurrent clusterizers, we would
      // be picking one from the map and running it.

      // geometry information for this DetUnit. TrackerGeom:DetContainer is
      // a vector<GeomDetUnit*>.  We need to call 
      //     const GeomDet&  TrackerGeom::idToDet(DetId) const;
      // to get a GeomDet (a GeomDetUnit*) from a DetId.

      // convert detid (unsigned int) to DetId
      DetId detIdObject( detid );      

      GeomDetUnit * geoUnit = geom->idToDet( detIdObject );
      PixelGeomDetUnit * pixDet = dynamic_cast<PixelGeomDetUnit*>(geoUnit);
      if (! pixDet) {
	// Fatal error!  TO DO: throw an exception!
	assert(0);
      }

      // dummies, right now, all empty
      std::vector<float> noiseVec(768,2);
      std::vector<short> badChannels;

      std::vector<SiPixelCluster> clustersOnDetUnit = 
	clusterizer_->clusterizeDetUnit(digiRangeIteratorBegin, 
					digiRangeIteratorEnd,
					detid,
					pixDet,
					noiseVec,
					badChannels);
      
      SiPixelClusterCollection::Range inputRange;
      inputRange.first = clustersOnDetUnit.begin();
      inputRange.second = clustersOnDetUnit.end();
      output.put(inputRange, detid);
      numberOfClusters += clustersOnDetUnit.size();

    }
    // end of the loop over detunits
    
    std::cout << "[SiPixelClusterProducer]: executing " 
	      << clusterMode_ << " resulted in " << numberOfClusters 
	      << " SiPixelClusters in " << numberOfDetUnits << " DetUnits." 
	      << std::endl; 
  }


  


}  // end of namespace cms
