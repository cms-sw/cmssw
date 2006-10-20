/** \file
 *
 *  $Date: 2006/10/16 14:39:31 $
 *  $Revision: 1.6 $
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
#include "Alignment/MuonAlignment/interface/MisalignmentScenarioBuilder.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"
#include "Alignment/MuonAlignment/interface/MisalignedMuonESProducer.h"

///
/// An ESProducer that fills the MuonDigiGeometryRcd with a misaligned Muon
/// 
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
  theCSCGeometry  = boost::shared_ptr<CSCGeometry>( CSCGeometryBuilder.build( &(*cpv), *mdc ) );


  // Create the alignable hierarchy
  AlignableMuon* theAlignableMuon = new AlignableMuon( &(*theDTGeometry) , &(*theCSCGeometry) );

  // Create misalignment scenario
  MisalignmentScenarioBuilder scenarioBuilder( theAlignableMuon );
  scenarioBuilder.applyScenario( theParameterSet );

  // Retrieve muon barrel alignments and errors
  dtAlignments      = theAlignableMuon->DTBarrel().front()->alignments();
  dtAlignmentErrors = theAlignableMuon->DTBarrel().front()->alignmentErrors();

  // Retrieve muon endcaps alignments and errors
  Alignments* cscEndCap1    = theAlignableMuon->CSCEndcaps().front()->alignments();
  Alignments* cscEndCap2    = theAlignableMuon->CSCEndcaps().back()->alignments();
  cscAlignments = new Alignments();
  std::copy( cscEndCap1->m_align.begin(), cscEndCap1->m_align.end(), back_inserter( cscAlignments->m_align ) );
  std::copy( cscEndCap2->m_align.begin(), cscEndCap2->m_align.end(), back_inserter( cscAlignments->m_align ) );

  AlignmentErrors* cscEndCap1Errors = theAlignableMuon->CSCEndcaps().front()->alignmentErrors();
  AlignmentErrors* cscEndCap2Errors = theAlignableMuon->CSCEndcaps().back()->alignmentErrors();
  cscAlignmentErrors    = new AlignmentErrors();
  std::copy(cscEndCap1Errors->m_alignError.begin(), cscEndCap1Errors->m_alignError.end(), back_inserter(cscAlignmentErrors->m_alignError) );
  std::copy(cscEndCap2Errors->m_alignError.begin(), cscEndCap2Errors->m_alignError.end(), back_inserter(cscAlignmentErrors->m_alignError) );
  
  // Misalign the EventSetup geometry
  GeometryAligner aligner;
  aligner.applyAlignments<DTGeometry>( &(*theDTGeometry),
                                         &(*dtAlignments), &(*dtAlignmentErrors) );
  aligner.applyAlignments<CSCGeometry>( &(*theCSCGeometry),
                                         &(*cscAlignments), &(*cscAlignmentErrors) );
  
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

  // Define callback tokens for the records 
  size_t dtAlignmentsToken = poolDbService->callbackToken("dtAlignments");
  size_t dtAlignmentErrorsToken  = poolDbService->callbackToken("dtAlignmentErrors");

  size_t cscAlignmentsToken = poolDbService->callbackToken("cscAlignments");
  size_t cscAlignmentErrorsToken = poolDbService->callbackToken("cscAlignmentErrors");

  // Store in the database
  poolDbService->newValidityForNewPayload<Alignments>( dtAlignments, poolDbService->endOfTime(), dtAlignmentsToken );
  poolDbService->newValidityForNewPayload<AlignmentErrors>( dtAlignmentErrors, poolDbService->endOfTime(), dtAlignmentErrorsToken );

  poolDbService->newValidityForNewPayload<Alignments>( cscAlignments, poolDbService->endOfTime(), cscAlignmentsToken );
  poolDbService->newValidityForNewPayload<AlignmentErrors>( cscAlignmentErrors, poolDbService->endOfTime(), cscAlignmentErrorsToken );
  

}
//__________________________________________________________________________________________________

DEFINE_FWK_EVENTSETUP_MODULE(MisalignedMuonESProducer)
