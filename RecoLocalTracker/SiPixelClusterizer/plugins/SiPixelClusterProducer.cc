/** SiPixelClusterProducer.cc
 * ---------------------------------------------------------------
 * Description:  see SiPixelClusterProducer.h
 * Author:  P. Maksimovic (porting from original ORCA version)
 * History: Oct 14, 2005, initial version
 * Get rid of the noiseVector. d.k. 28/3/06
 * Implementation of the DetSetVector container.    V.Chiochia, May 06
 * SiPixelClusterCollection typedef of DetSetVector V.Chiochia, June 06
 * Introduce the DetSet local container (cache) for speed. d.k. 05/07
 * 
 * ---------------------------------------------------------------
 */

// Our own stuff
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelClusterProducer.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/PixelThresholdClusterizer.h"

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

// Data Formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/DetId/interface/DetId.h"

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

namespace cms
{

  //---------------------------------------------------------------------------
  //!  Constructor: set the ParameterSet and defer all thinking to setupClusterizer().
  //---------------------------------------------------------------------------
  SiPixelClusterProducer::SiPixelClusterProducer(edm::ParameterSet const& conf) 
    : 
    conf_(conf),
    theSiPixelGainCalibration_(conf),
    clusterMode_("None"),     // bogus
    clusterizer_(0),          // the default, in case we fail to make one
    readyToCluster_(false),   // since we obviously aren't
    src_( conf.getParameter<edm::InputTag>( "src" ) )
  {
    //--- Declare to the EDM what kind of collections we will be making.
    produces<SiPixelClusterCollection>(); 

    //--- Make the algorithm(s) according to what the user specified
    //--- in the ParameterSet.
    setupClusterizer();
  }

  // Destructor
  SiPixelClusterProducer::~SiPixelClusterProducer() { 
    delete clusterizer_;
  }  

  void SiPixelClusterProducer::beginJob( const edm::EventSetup& es ) {
    edm::LogInfo("SiPixelClusterizer") << "[SiPixelClusterizer::beginJob]";
    clusterizer_->setSiPixelGainCalibrationService(& theSiPixelGainCalibration_);
  }

  //---------------------------------------------------------------------------
  //! The "Event" entrypoint: gets called by framework for every event
  //---------------------------------------------------------------------------
  void SiPixelClusterProducer::produce(edm::Event& e, const edm::EventSetup& es)
  {

    //Setup gain calibration service
    theSiPixelGainCalibration_.setESObjects( es );

   // Step A.1: get input data
    //edm::Handle<PixelDigiCollection> pixDigis;
    edm::Handle< edm::DetSetVector<PixelDigi> >  input;
    e.getByLabel( src_, input);

    // Step A.2: get event setup
    edm::ESHandle<TrackerGeometry> geom;
    es.get<TrackerDigiGeometryRecord>().get( geom );

    // Step B: Iterate over DetIds and invoke the pixel clusterizer algorithm
    // on each DetUnit
    run(*input, geom );

    // Step C: create the final output collection
    std::auto_ptr< SiPixelClusterCollection > 
      output( new SiPixelClusterCollection (theClusterVector));


    // Step D: write output to file
    e.put( output );

  }

  //---------------------------------------------------------------------------
  //!  Set up the specific algorithm we are going to use.  
  //!  TO DO: in the future, we should allow for a different algorithm for 
  //!  each detector subset (e.g. barrel vs forward, per layer, etc).
  //---------------------------------------------------------------------------
  void SiPixelClusterProducer::setupClusterizer()  {
    clusterMode_ = 
      conf_.getUntrackedParameter<std::string>("ClusterMode","PixelThresholdClusterizer");

    if ( clusterMode_ == "PixelThresholdClusterizer" ) {
      clusterizer_ = new PixelThresholdClusterizer(conf_);
      readyToCluster_ = true;
    } 
    else {
      edm::LogError("SiPixelClusterProducer") << "[SiPixelClusterProducer]:"
		<<" choice " << clusterMode_ << " is invalid.\n"
		<< "Possible choices:\n" 
		<< "    PixelThresholdClusterizer";
      readyToCluster_ = false;
    }
  }

  //---------------------------------------------------------------------------
  //!  Iterate over DetUnits, and invoke the PixelClusterizer on each.
  //---------------------------------------------------------------------------
  void SiPixelClusterProducer::run(const edm::DetSetVector<PixelDigi>& input, 
				   edm::ESHandle<TrackerGeometry> & geom) {
    if ( ! readyToCluster_ ) {
      edm::LogError("SiPixelClusterProducer")
		<<" at least one clusterizer is not ready -- can't run!" ;
      // TO DO: throw an exception here?  The user may want to know...
      return;   // clusterizer is invalid, bail out
    }

    int numberOfDetUnits = 0;
    int numberOfClusters = 0;
 
    // Iterate on detector units
    edm::DetSetVector<PixelDigi>::const_iterator DSViter = input.begin();
    for( ; DSViter != input.end(); DSViter++) {
      ++numberOfDetUnits;
      LogDebug("SiStripClusterizer") << "[SiPixelClusterProducer::run] DetID" << DSViter->id;

      std::vector<short> badChannels; 
      DetId detIdObject(DSViter->id);
      
      // Comment: At the moment the clusterizer depends on geometry
      // to access information as the pixel topology (number of columns
      // and rows in a detector module). 
      // In the future the geometry service will be replaced with
      // a ES service.
      const GeomDetUnit      * geoUnit = geom->idToDetUnit( detIdObject );
      const PixelGeomDetUnit * pixDet  = dynamic_cast<const PixelGeomDetUnit*>(geoUnit);
      if (! pixDet) {
	// Fatal error!  TO DO: throw an exception!
	assert(0);
      }
      // Produce clusters for this DetUnit and store them in 
      // a DetSet
      edm::DetSet<SiPixelCluster> spc(DSViter->id);
      clusterizer_->clusterizeDetUnit(*DSViter, pixDet, badChannels, spc);
      if( spc.data.size() > 0) {
	//output.insert( spc );  // very slow
	theClusterVector.push_back( spc );  // fill the cache
	numberOfClusters += spc.data.size();
      }

    } // end of DetUnit loop
    
    LogDebug ("SiPixelClusterProducer") << " Executing " 
	      << clusterMode_ << " resulted in " << numberOfClusters
					    << " SiPixelClusters in " << numberOfDetUnits << " DetUnits."; 
  }

}  // end of namespace cms
