/** \file
 *
 *  $Date: 2007/04/18 14:09:59 $
 *  $Revision: 1.2 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */
 

// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// Alignment
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignmentSorter.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/MuonAlignment/interface/MuonScenarioBuilder.h"
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
				       dt_AlignmentErrors );
  aligner.applyAlignments<CSCGeometry>( &(*theCSCGeometry ),
                                        csc_Alignments,
					csc_AlignmentErrors);  

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
  if ( poolDbService->isNewTagRequest(theDTAlignRecordName) ){
   poolDbService->createNewIOV<Alignments>( &(*dt_Alignments), 
	                                    poolDbService->endOfTime(), 
                                            theDTAlignRecordName );
  } else {
    poolDbService->appendSinceTime<Alignments>( &(*dt_Alignments),
                                                poolDbService->currentTime(), 
                                                theDTAlignRecordName );
  }
      
  if ( poolDbService->isNewTagRequest(theDTErrorRecordName) ){
   poolDbService->createNewIOV<AlignmentErrors>( &(*dt_AlignmentErrors),
                                                 poolDbService->endOfTime(), 
                                                 theDTErrorRecordName );
  } else {
   poolDbService->appendSinceTime<AlignmentErrors>( &(*dt_AlignmentErrors),
                                                    poolDbService->currentTime(),
                                                    theDTErrorRecordName );
  }							  


  // Store CSC alignments and errors
  if ( poolDbService->isNewTagRequest(theCSCAlignRecordName) ){
   poolDbService->createNewIOV<Alignments>( &(*csc_Alignments), 
	                                    poolDbService->endOfTime(), 
                                            theCSCAlignRecordName );
  } else {
    poolDbService->appendSinceTime<Alignments>( &(*csc_Alignments),
                                                poolDbService->currentTime(), 
                                                theCSCAlignRecordName );
  }
      
  if ( poolDbService->isNewTagRequest(theCSCErrorRecordName) ){
   poolDbService->createNewIOV<AlignmentErrors>( &(*csc_AlignmentErrors),
                                                 poolDbService->endOfTime(), 
                                                 theCSCErrorRecordName );
  } else {
   poolDbService->appendSinceTime<AlignmentErrors>( &(*csc_AlignmentErrors),
                                                    poolDbService->currentTime(),
                                                    theCSCErrorRecordName );
  }							  



  

}
//__________________________________________________________________________________________________

DEFINE_FWK_EVENTSETUP_MODULE(MisalignedMuonESProducer);
