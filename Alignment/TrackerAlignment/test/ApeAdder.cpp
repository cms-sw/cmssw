// -*- C++ -*-
//
//
// Description: Module to add a custom APE to the alignment object
//
//
// Original Author:  Frederic Ronga
//         Created:  December 5, 2006
//


// system include files

// user include files
//#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"


// Class declaration
class ApeAdder : public edm::EDAnalyzer {
public:
  explicit ApeAdder( const edm::ParameterSet& );
  ~ApeAdder() {};
  
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );

private:
    // methods
    void addApe( std::vector<Alignable*> alignables );
    
private:
    // members
  std::string theErrorRecordName;
    std::vector<double> theApe; // Amount of APE to add (from config.)
  
};

ApeAdder::ApeAdder( const edm::ParameterSet& iConfig )  :
  theErrorRecordName( "TrackerAlignmentErrorExtendedRcd" )
{ 

  // The APE to set to all GeomDets
  theApe = iConfig.getUntrackedParameter< std::vector<double> >("apeVector");
  
}

//__________________________________________________________________________________________________
void ApeAdder::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )

{

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  // Get geometry from ES
  edm::ESHandle<TrackerGeometry> trackerGeometry;
  iSetup.get<TrackerDigiGeometryRecord>().get( trackerGeometry );
    
  // Create the alignable hierarchy
  AlignableTracker* theAlignableTracker = new AlignableTracker( &(*trackerGeometry), tTopo );
  
  // Now loop on alignable dets and add alignment error
  if ( theAlignableTracker->barrelGeomDets().size() ) 
    this->addApe(theAlignableTracker->barrelGeomDets());
  if ( theAlignableTracker->pixelHalfBarrelGeomDets().size() ) 
    this->addApe(theAlignableTracker->pixelHalfBarrelGeomDets());
  if ( theAlignableTracker->endcapGeomDets().size() ) 
    this->addApe(theAlignableTracker->endcapGeomDets());
  if ( theAlignableTracker->TIDGeomDets().size() ) 
    this->addApe(theAlignableTracker->TIDGeomDets());
  if ( theAlignableTracker->pixelEndcapGeomDets().size() ) 
    this->addApe(theAlignableTracker->pixelEndcapGeomDets());
  
  // Store to DB
  AlignmentErrorsExtended* alignmentErrors = theAlignableTracker->alignmentErrors();

  // Call service
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if( !poolDbService.isAvailable() ) // Die if not available
    throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";

  // Save to DB
  poolDbService->writeOne<AlignmentErrorsExtended>(alignmentErrors, poolDbService->beginOfTime(),
                                           theErrorRecordName);


  delete theAlignableTracker;

}

void ApeAdder::addApe( std::vector<Alignable*> alignables )
{
  
  AlignmentPositionError ape( theApe[0], theApe[1], theApe[2] );
  for ( std::vector<Alignable*>::iterator iDet = alignables.begin();
		iDet != alignables.end(); ++iDet )
    (*iDet)->setAlignmentPositionError( ape, true ); // true: propagate to components
    
}

//define this as a plug-in
DEFINE_FWK_MODULE(ApeAdder);
