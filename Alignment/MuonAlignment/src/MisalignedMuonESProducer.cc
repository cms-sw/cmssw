#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
//#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"

#include "CondFormats/Alignment/interface/Alignments.h"

#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/MuonAlignment/interface/MisalignmentScenarioBuilder.h"
#include "Alignment/MuonAlignment/interface/MisalignedMuonESProducer.h"

///
/// An ESProducer that fills the MuonDigiGeometryRcd with a misaligned Muon
/// 
/// FIXME: configuration file, output POOL-ORA object?

#include <memory>


//__________________________________________________________________________________________________
MisalignedMuonESProducer::MisalignedMuonESProducer(const edm::ParameterSet& p) :
  theParameterSet( p )
{
  
  setWhatProduced(this);

}


//__________________________________________________________________________________________________
MisalignedMuonESProducer::~MisalignedMuonESProducer() {}


//__________________________________________________________________________________________________
boost::shared_ptr<DTGeometry> 
MisalignedMuonESProducer::produce( const MuonGeometryRecord& iRecord )
{ 

  edm::LogInfo("MisalignedMuon") << "Producer called";
  

  // Create the Muon geometry from ideal geometry
  edm::ESHandle<DDCompactView> cpv;
  iRecord.getRecord<IdealGeometryRecord>().get( cpv );

  DTGeometryBuilderFromDDD  DTGeometryBuilder;
  CSCGeometryBuilderFromDDD CSCGeometryBuilder;

  theDTGeometry   = boost::shared_ptr<DTGeometry>( DTGeometryBuilder.build( &(*cpv) ) );
  theCSCGeometry  = boost::shared_ptr<CSCGeometry>( CSCGeometryBuilder.build( &(*cpv) ) );


  // Dump BEFORE
  if ( theParameterSet.getUntrackedParameter<bool>("dumpBefore", false) ){
    for ( std::vector<DTChamber*>::const_iterator iGeomDet = theDTGeometry->chambers().begin();
 		iGeomDet != theDTGeometry->chambers().end(); iGeomDet++ )
 	std::cout << (*iGeomDet)->geographicalId().rawId()
 			  << " " << (*iGeomDet)->position() << std::endl;

    for ( std::vector<CSCChamber*>::const_iterator iGeomDet = theCSCGeometry->chambers().begin();
 		iGeomDet != theCSCGeometry->chambers().end(); iGeomDet++ )
 	std::cout << (*iGeomDet)->geographicalId().rawId()
 			  << " " << (*iGeomDet)->position() << std::endl;
  }

  // Create the alignable hierarchy
  AlignableMuon* theAlignableMuon = new AlignableMuon( *theDTGeometry , *theCSCGeometry );

  // Create misalignment scenario
  MisalignmentScenarioBuilder scenarioBuilder( theAlignableMuon );
  scenarioBuilder.applyScenario( theParameterSet );

  
  // Apply to geometry
  // 
  // theMuon->applyAlignments( theAlignableMuon->alignments() );
  //


  // Dump AFTER
  if ( theParameterSet.getUntrackedParameter<bool>("dumpAfter", false) ){

    for ( std::vector<DTChamber*>::const_iterator iGeomDet = theDTGeometry->chambers().begin();
 		iGeomDet != theDTGeometry->chambers().end(); iGeomDet++ )
 	std::cout << (*iGeomDet)->geographicalId().rawId()
 			  << " " << (*iGeomDet)->position() << std::endl;

    for ( std::vector<CSCChamber*>::const_iterator iGeomDet = theCSCGeometry->chambers().begin();
 		iGeomDet != theCSCGeometry->chambers().end(); iGeomDet++ )
 	std::cout << (*iGeomDet)->geographicalId().rawId()
 			  << " " << (*iGeomDet)->position() << std::endl;

  }
  edm::LogInfo("MisalignedMuon") << "Producer done";

  return theDTGeometry; 
  
}


DEFINE_FWK_EVENTSETUP_MODULE(MisalignedMuonESProducer)
