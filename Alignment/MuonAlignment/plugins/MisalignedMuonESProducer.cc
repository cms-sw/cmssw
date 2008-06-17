/** \file
 *
 *  $Date: 2008/03/05 20:43:59 $
 *  $Revision: 1.6 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */
 

// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// Alignment
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h" 
#include "Alignment/MuonAlignment/interface/MuonScenarioBuilder.h"
#include "Alignment/CommonAlignment/interface/Alignable.h" 
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"
#include "Alignment/MuonAlignment/plugins/MisalignedMuonESProducer.h"

///
/// An ESProducer that fills the MuonDigiGeometryRcd with a misaligned Muon
/// 
#include <memory>


//__________________________________________________________________________________________________
MisalignedMuonESProducer::MisalignedMuonESProducer(const edm::ParameterSet& p) :
  theParameterSet( p ),
  theDTAlignRecordName( "DTAlignmentRcd" ),
  theDTErrorRecordName( "DTAlignmentErrorRcd" ),
  theCSCAlignRecordName( "CSCAlignmentRcd" ),
  theCSCErrorRecordName( "CSCAlignmentErrorRcd" )
{
  
  setWhatProduced(this);

}


//__________________________________________________________________________________________________
MisalignedMuonESProducer::~MisalignedMuonESProducer() {}


//__________________________________________________________________________________________________
edm::ESProducts< boost::shared_ptr<DTGeometry>, boost::shared_ptr<CSCGeometry> >
MisalignedMuonESProducer::produce( const MuonGeometryRecord& iRecord )
{ 

  edm::LogInfo("MisalignedMuon") << "Producer called";
  

  // Create the Muon geometry from ideal geometry
  edm::ESHandle<DDCompactView> cpv;
  iRecord.getRecord<IdealGeometryRecord>().get( cpv );

  edm::ESHandle<MuonDDDConstants> mdc;
  iRecord.getRecord<MuonNumberingRecord>().get(mdc);

  DTGeometryBuilderFromDDD  DTGeometryBuilder;
  CSCGeometryBuilderFromDDD CSCGeometryBuilder;

  theDTGeometry   = boost::shared_ptr<DTGeometry>(  DTGeometryBuilder.build( &(*cpv), *mdc ) );
  //theCSCGeometry  = boost::shared_ptr<CSCGeometry>( CSCGeometryBuilder.build( &(*cpv), *mdc ) );
  theCSCGeometry  = boost::shared_ptr<CSCGeometry>( new CSCGeometry );
  CSCGeometryBuilder.build( theCSCGeometry,  &(*cpv), *mdc );


  // Create the alignable hierarchy
  AlignableMuon* theAlignableMuon = new AlignableMuon( &(*theDTGeometry) , &(*theCSCGeometry) );

  // Create misalignment scenario
  MuonScenarioBuilder scenarioBuilder( theAlignableMuon );
  scenarioBuilder.applyScenario( theParameterSet );
  
  // Get alignments and errors
  dt_Alignments = theAlignableMuon->dtAlignments() ;
  dt_AlignmentErrors = theAlignableMuon->dtAlignmentErrors();
  csc_Alignments = theAlignableMuon->cscAlignments();
  csc_AlignmentErrors = theAlignableMuon->cscAlignmentErrors();

 
  // Misalign the EventSetup geometry
  GeometryAligner aligner;

  aligner.applyAlignments<DTGeometry>( &(*theDTGeometry),
                                       dt_Alignments, 
				       dt_AlignmentErrors,
				       AlignTransform() );
  aligner.applyAlignments<CSCGeometry>( &(*theCSCGeometry ),
                                        csc_Alignments,
					csc_AlignmentErrors,
					AlignTransform() );  

  // Write alignments to DB
  if ( theParameterSet.getUntrackedParameter<bool>("saveToDbase", false) ) saveToDB();

  edm::LogInfo("MisalignedMuon") << "Producer done";

  return edm::es::products( theDTGeometry, theCSCGeometry ); 
  
}


//__________________________________________________________________________________________________
void MisalignedMuonESProducer::saveToDB( void )
{

  // Call service
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if( !poolDbService.isAvailable() ) // Die if not available
	throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";

  // Store DT alignments and errors
  poolDbService->writeOne<Alignments>( &(*dt_Alignments), poolDbService->beginOfTime(), theDTAlignRecordName);
  poolDbService->writeOne<AlignmentErrors>( &(*dt_AlignmentErrors), poolDbService->beginOfTime(), theDTErrorRecordName);

  // Store CSC alignments and errors
  poolDbService->writeOne<Alignments>( &(*csc_Alignments), poolDbService->beginOfTime(), theCSCAlignRecordName);
  poolDbService->writeOne<AlignmentErrors>( &(*csc_AlignmentErrors), poolDbService->beginOfTime(), theCSCErrorRecordName);

}
//__________________________________________________________________________________________________

DEFINE_FWK_EVENTSETUP_MODULE(MisalignedMuonESProducer);
