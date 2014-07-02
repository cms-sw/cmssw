/** SiPixelRecHitConverter.cc
 * ------------------------------------------------------
 * Description:  see SiPixelRecHitConverter.h
 * Authors:  P. Maksimovic (JHU), V.Chiochia (Uni Zurich)
 * History: Feb 27, 2006 -  initial version
 *          May 30, 2006 -  edm::DetSetVector and edm::Ref
 *          Aug 30, 2007 -  edmNew::DetSetVector
*			Jan 31, 2008 -  change to use Lorentz angle from DB (Lotte Wilke)
 * ------------------------------------------------------
 */

// Our own stuff
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelRecHitConverter.h"
// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

// Data Formats
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSet2RangeMap.h"


// STL
#include <vector>
#include <memory>
#include <string>
#include <iostream>

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"

using namespace std;

namespace cms
{
  //---------------------------------------------------------------------------
  //!  Constructor: set the ParameterSet and defer all thinking to setupCPE().
  //---------------------------------------------------------------------------
  SiPixelRecHitConverter::SiPixelRecHitConverter(edm::ParameterSet const& conf) 
    : 
    conf_(conf),
    src_( conf.getParameter<edm::InputTag>( "src" ) ),
    tPixelCluster(consumes< edmNew::DetSetVector<SiPixelCluster> >( src_)) {
    //--- Declare to the EDM what kind of collections we will be making.
    produces<SiPixelRecHitCollection>();
    
  }
  
  // Destructor
  SiPixelRecHitConverter::~SiPixelRecHitConverter() 
  { 
  }  
  
  //---------------------------------------------------------------------------
  //! The "Event" entrypoint: gets called by framework for every event
  //---------------------------------------------------------------------------
  void SiPixelRecHitConverter::produce(edm::Event& e, const edm::EventSetup& es)
  {

    // Step A.1: get input data
    edm::Handle< edmNew::DetSetVector<SiPixelCluster> > input;
    e.getByToken( tPixelCluster, input);
    
    // Step A.2: get event setup
    edm::ESHandle<TrackerGeometry> geom;
    es.get<TrackerDigiGeometryRecord>().get( geom );

    // Step B: create empty output collection
    std::auto_ptr<SiPixelRecHitCollectionNew> output(new SiPixelRecHitCollectionNew);
    
    // Step B*: create CPE
    edm::ESHandle<PixelClusterParameterEstimator> hCPE;
    std::string cpeName_ = conf_.getParameter<std::string>("CPE");
    es.get<TkPixelCPERecord>().get(cpeName_,hCPE);
    cpe_ = dynamic_cast< const PixelCPEBase* >(&(*hCPE));
    
    // Step C: Iterate over DetIds and invoke the strip CPE algorithm
    // on each DetUnit

    run( input, *output, geom );


    e.put(output);

  }

  //---------------------------------------------------------------------------
  //!  Iterate over DetUnits, then over Clusters and invoke the CPE on each,
  //!  and make a RecHit to store the result.
  //!  New interface reading DetSetVector by V.Chiochia (May 30th, 2006)
  //---------------------------------------------------------------------------
  void SiPixelRecHitConverter::run(edm::Handle<edmNew::DetSetVector<SiPixelCluster> >  inputhandle,
				   SiPixelRecHitCollectionNew &output,
				   edm::ESHandle<TrackerGeometry> & geom) {
    if ( ! cpe_ ) 
      {
	edm::LogError("SiPixelRecHitConverter") << " at least one CPE is not ready -- can't run!";
	// TO DO: throw an exception here?  The user may want to know...
	assert(0);
	return;   // clusterizer is invalid, bail out
      }
    
    int numberOfDetUnits = 0;
    int numberOfClusters = 0;
    
    const edmNew::DetSetVector<SiPixelCluster>& input = *inputhandle;
    
    edmNew::DetSetVector<SiPixelCluster>::const_iterator DSViter=input.begin();
    
    for ( ; DSViter != input.end() ; DSViter++) {
      numberOfDetUnits++;
      unsigned int detid = DSViter->detId();
      DetId detIdObject( detid );  
      const GeomDetUnit * genericDet = geom->idToDetUnit( detIdObject );
      const PixelGeomDetUnit * pixDet = dynamic_cast<const PixelGeomDetUnit*>(genericDet);
      assert(pixDet); 
      SiPixelRecHitCollectionNew::FastFiller recHitsOnDetUnit(output,detid);
      
      edmNew::DetSet<SiPixelCluster>::const_iterator clustIt = DSViter->begin(), clustEnd = DSViter->end();
      
      for ( ; clustIt != clustEnd; clustIt++) {
	numberOfClusters++;
	std::tuple<LocalPoint, LocalError,SiPixelRecHitQuality::QualWordType> tuple = cpe_->getParameters( *clustIt, *genericDet );
	LocalPoint lp( std::get<0>(tuple) );
	LocalError le( std::get<1>(tuple) );
        SiPixelRecHitQuality::QualWordType rqw( std::get<2>(tuple) );
	// Create a persistent edm::Ref to the cluster
	edm::Ref< edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster > cluster = edmNew::makeRefTo( inputhandle, clustIt);
	// Make a RecHit and add it to the DetSet
	// old : recHitsOnDetUnit.push_back( new SiPixelRecHit( lp, le, detIdObject, &*clustIt) );
	SiPixelRecHit hit( lp, le, rqw, *genericDet, cluster);
	// 
	// Now save it =================
	recHitsOnDetUnit.push_back(hit);
	// =============================

	// std::cout << "SiPixelRecHitConverterVI " << numberOfClusters << ' '<< lp << " " << le << std::endl;
      } //  <-- End loop on Clusters
	

      //  LogDebug("SiPixelRecHitConverter")
      //std::cout << "SiPixelRecHitConverterVI "
	//	<< " Found " << recHitsOnDetUnit.size() << " RecHits on " << detid //;
	//	<< std::endl;
      
      
    } //    <-- End loop on DetUnits
    
    //    LogDebug ("SiPixelRecHitConverter") 
    //  std::cout << "SiPixelRecHitConverterVI "
    //  << cpeName_ << " converted " << numberOfClusters 
    //  << " SiPixelClusters into SiPixelRecHits, in " 
    //  << numberOfDetUnits << " DetUnits." //; 
    //  << std::endl;
	
  }
}  // end of namespace cms
