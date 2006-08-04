/** \file
 *
 *  $Date: 2006/8/4 10:10:07 $
 *  $Revision: 1.0 $
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

  DTGeometryBuilderFromDDD  DTGeometryBuilder;
  CSCGeometryBuilderFromDDD CSCGeometryBuilder;

  theDTGeometry   = boost::shared_ptr<DTGeometry>( DTGeometryBuilder.build( &(*cpv) ) );
  theCSCGeometry  = boost::shared_ptr<CSCGeometry>( CSCGeometryBuilder.build( &(*cpv) ) );


  // Create the alignable hierarchy
  AlignableMuon* theAlignableMuon = new AlignableMuon( &(*theDTGeometry) , &(*theCSCGeometry) );

  // Dump BEFORE
  Alignments* alignments;
  if ( theParameterSet.getUntrackedParameter<bool>("dumpBefore", false) )
 	{
 	  alignments = theAlignableMuon->alignments();
 	  std::vector<AlignTransform> alignTransforms = alignments->m_align;
 	  edm::LogInfo("DumpPositions") << alignTransforms.size() << " alignTransforms found";
 	  for ( std::vector<AlignTransform>::iterator it = alignTransforms.begin();
 			it != alignTransforms.end(); it++ )
		edm::LogInfo("DumpPositions") << (*it).rawId() << " " << (*it).translation();
 	}


  // Create misalignment scenario
  MisalignmentScenarioBuilder scenarioBuilder( theAlignableMuon );
  scenarioBuilder.applyScenario( theParameterSet );

  // Dump AFTER
  if ( theParameterSet.getUntrackedParameter<bool>("dumpAfter", false) )
	{
 	  alignments = theAlignableMuon->alignments();
 	  std::vector<AlignTransform> alignTransforms = alignments->m_align;
 	  edm::LogInfo("DumpPositions") << alignTransforms.size() << " alignTransforms found";
 	  for ( std::vector<AlignTransform>::iterator it = alignTransforms.begin();
 			it != alignTransforms.end(); it++ )
		edm::LogInfo("DumpPositions") << (*it).rawId() << " " << (*it).translation();
	}

  // Write alignments to DB
  if ( theParameterSet.getUntrackedParameter<bool>("saveToDbase", false) )
	{

	  // Call service
	  edm::Service<cond::service::PoolDBOutputService> poolDbService;
	  if( !poolDbService.isAvailable() ) // Die if not available
		throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";

          // Retrieve muon barrel alignments and errors
          Alignments*      dtAlignments       = theAlignableMuon->DTBarrel().front()->alignments();
          AlignmentErrors* dtAlignmentErrors = theAlignableMuon->DTBarrel().front()->alignmentErrors();

          // Retrieve muon endcaps alignments and errors
          Alignments* cscEndCap1    = theAlignableMuon->CSCEndcaps().front()->alignments();
          Alignments* cscEndCap2    = theAlignableMuon->CSCEndcaps().back()->alignments();
          Alignments* cscAlignments = new Alignments();
          std::copy( cscEndCap1->m_align.begin(), cscEndCap1->m_align.end(), back_inserter( cscAlignments->m_align ) );
          std::copy( cscEndCap2->m_align.begin(), cscEndCap2->m_align.end(), back_inserter( cscAlignments->m_align ) );

          AlignmentErrors* cscEndCap1Errors = theAlignableMuon->CSCEndcaps().front()->alignmentErrors();
          AlignmentErrors* cscEndCap2Errors = theAlignableMuon->CSCEndcaps().back()->alignmentErrors();
          AlignmentErrors* cscAlignmentErrors    = new AlignmentErrors();
          std::copy(cscEndCap1Errors->m_alignError.begin(), cscEndCap1Errors->m_alignError.end(), 
                     back_inserter(cscAlignmentErrors->m_alignError) );
          std::copy(cscEndCap2Errors->m_alignError.begin(), cscEndCap2Errors->m_alignError.end(),
                     back_inserter(cscAlignmentErrors->m_alignError) );

          // Define callback tokens for the records 
          size_t dtAlignmentsToken = poolDbService->callbackToken("dtAlignments");
          size_t dtAlignmentErrorsToken  = poolDbService->callbackToken("dtAlignmentErrors");

          size_t cscAlignmentsToken = poolDbService->callbackToken("cscAlignments");
          size_t cscAlignmentErrorsToken = poolDbService->callbackToken("cscAlignmentErrors");

          // Sort by DetID
          std::sort( dtAlignments->m_align.begin(),  dtAlignments->m_align.end(),  lessAlignmentDetId<AlignTransform>() );
          std::sort( dtAlignmentErrors->m_alignError.begin(),  dtAlignmentErrors->m_alignError.end(),  lessAlignmentDetId<AlignTransformError>() );

          std::sort( cscAlignments->m_align.begin(), cscAlignments->m_align.end(), lessAlignmentDetId<AlignTransform>() );
          std::sort( cscAlignmentErrors->m_alignError.begin(), cscAlignmentErrors->m_alignError.end(), lessAlignmentDetId<AlignTransformError>() );

          // Store in the database
          poolDbService->newValidityForNewPayload<Alignments>( dtAlignments, poolDbService->endOfTime(), dtAlignmentsToken );
          poolDbService->newValidityForNewPayload<AlignmentErrors>( dtAlignmentErrors, poolDbService->endOfTime(), dtAlignmentErrorsToken );

          poolDbService->newValidityForNewPayload<Alignments>( cscAlignments, poolDbService->endOfTime(), cscAlignmentsToken );
          poolDbService->newValidityForNewPayload<AlignmentErrors>( cscAlignmentErrors, poolDbService->endOfTime(), cscAlignmentErrorsToken );


/*	  
	  // Define callback tokens for the two records
	  size_t alignmentsToken = poolDbService->callbackToken("Alignments");
	  size_t alignmentErrorsToken = poolDbService->callbackToken("AlignmentErrors");

	  // Retrieve and sort
	  Alignments = theAlignableMuon->alignments();
	  AlignmentErrors* alignmentErrors = theAlignableMuon->alignmentErrors();
	  std::sort( alignments->m_align.begin(), alignments->m_align.end(), 
				 lessAlignmentDetId<AlignTransform>() );
	  std::sort( alignmentErrors->m_alignError.begin(), alignmentErrors->m_alignError.end(), 
				 lessAlignmentDetId<AlignTransformError>() );
	  // Store
	  poolDbService->newValidityForNewPayload<Alignments>( alignments, poolDbService->endOfTime(), alignmentsToken );
	  poolDbService->newValidityForNewPayload<AlignmentErrors>( alignmentErrors, poolDbService->endOfTime(),alignmentErrorsToken );
*/


	}

  edm::LogInfo("MisalignedMuon") << "Producer done";

  // Store result to EventSetup
  return edm::es::products( theDTGeometry, theCSCGeometry ); 
  
}


DEFINE_FWK_EVENTSETUP_MODULE(MisalignedMuonESProducer)
