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

	  // Define callback tokens for the two records
	  size_t alignmentsToken = poolDbService->callbackToken("Alignments");
	  size_t alignmentErrorsToken = poolDbService->callbackToken("AlignmentErrors");
	  
	  // Retrieve and sort
	  alignments = theAlignableMuon->alignments();
	  AlignmentErrors* alignmentErrors = theAlignableMuon->alignmentErrors();
	  std::sort( alignments->m_align.begin(), alignments->m_align.end(), 
				 lessAlignmentDetId<AlignTransform>() );
	  std::sort( alignmentErrors->m_alignError.begin(), alignmentErrors->m_alignError.end(), 
				 lessAlignmentDetId<AlignTransformError>() );

	  // Store
	  poolDbService->newValidityForNewPayload<Alignments>( alignments, 
														   poolDbService->endOfTime(),
														   alignmentsToken );
	  poolDbService->newValidityForNewPayload<AlignmentErrors>( alignmentErrors, 
																poolDbService->endOfTime(),
																alignmentErrorsToken );
	}

  edm::LogInfo("MisalignedMuon") << "Producer done";

  // Store result to EventSetup
  return edm::es::products( theDTGeometry, theCSCGeometry ); 
  
}


DEFINE_FWK_EVENTSETUP_MODULE(MisalignedMuonESProducer)
