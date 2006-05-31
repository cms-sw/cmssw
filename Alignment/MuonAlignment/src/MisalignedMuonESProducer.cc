#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"

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
boost::shared_ptr<MuonGeometry> 
MisalignedMuonESProducer::produce( const MuonGeometryRecord& iRecord )
{ 

  edm::LogInfo("MisalignedMuon") << "Producer called";
  
//  edm::ESHandle<GeometricDet> gD;
//  iRecord.getRecord<IdealGeometryRecord>().get( gD );

  // Create the Muon geometry from ideal geometry
  edm::ESHandle<DDCompactView> cpv;
  iRecord.getRecord<IdealGeometryRecord>().get( cpv );

  DTGeometryBuilderFromDDD  DTMuonBuilder;
  CSCGeometryBuilderFromDDD CSCMuonBuilder;

  theDTMuon   = boost::shared_ptr<MuonGeometry>( DTMuonBuilder.build( &cpv ) );
  theCSCMuon  = boost::shared_ptr<MuonGeometry>( CSCMuonBuilder.build( &cpv ) );

  // Dump BEFORE
  for ( std::vector<GeomDet*>::const_iterator iGeomDet = theDTMuon->chambers().begin();
 		iGeomDet != theDTMuon->chambers().end(); iGeomDet++ )
 	std::cout << (*iGeomDet)->geographicalId().rawId()
 			  << " " << (*iGeomDet)->position() << std::endl;

  for ( std::vector<GeomDet*>::const_iterator iGeomDet = theCSCMuon->chambers().begin();
 		iGeomDet != theCSCMuon->chambers().end(); iGeomDet++ )
 	std::cout << (*iGeomDet)->geographicalId().rawId()
 			  << " " << (*iGeomDet)->position() << std::endl;

  // Create the alignable hierarchy
  AlignableMuon* theAlignableMuon = new AlignableMuon(  const edm::EventSetup&  iSetup );

  // Create misalignment scenario
  MisalignmentScenarioBuilder scenarioBuilder( theAlignableMuon );
  scenarioBuilder.applyScenario( theParameterSet );

  
  // Apply to geometry
  // 
  // theMuon->applyAlignments( theAlignableMuon->alignments() );
  //


  // Dump AFTER
  for ( std::vector<GeomDet*>::const_iterator iGeomDet = theDTMuon->chambers().begin();
 		iGeomDet != theDTMuon->chambers().end(); iGeomDet++ )
 	std::cout << (*iGeomDet)->geographicalId().rawId()
 			  << " " << (*iGeomDet)->position() << std::endl;

  for ( std::vector<GeomDet*>::const_iterator iGeomDet = theCSCMuon->chambers().begin();
 		iGeomDet != theCSCMuon->chambers().end(); iGeomDet++ )
 	std::cout << (*iGeomDet)->geographicalId().rawId()
 			  << " " << (*iGeomDet)->position() << std::endl;

 
  edm::LogInfo("MisalignedMuon") << "Producer done";

  // Store result to EventSetup
  return theMuon;
  
}


DEFINE_FWK_EVENTSETUP_MODULE(MisalignedMuonESProducer)
