///
/// An ESProducer that fills the MuonDigiGeometryRcd with a misaligned Muon
/// 
/// This should replace the standard DTGeometry and CSCGeometry producers 
/// when producing Misalignment scenarios.
///
/// \file
/// $Date: 2010/05/14 21:07:59 $
/// $Revision: 1.12 $
/// \author Andre Sznajder - UERJ(Brazil)
///
 

// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"

#include <boost/shared_ptr.hpp>
#include <memory>


class MisalignedMuonESProducer: public edm::ESProducer
{
public:

  /// Constructor
  MisalignedMuonESProducer( const edm::ParameterSet & p );
  
  /// Destructor
  virtual ~MisalignedMuonESProducer(); 
  
  /// Produce the misaligned Muon geometry and store it
  edm::ESProducts< boost::shared_ptr<DTGeometry>,
 				   boost::shared_ptr<CSCGeometry> > produce( const MuonGeometryRecord&  );

  /// Save alignemnts and error to database
  void saveToDB();
  
private:
  const bool theSaveToDB; /// whether or not writing to DB
  const edm::ParameterSet theScenario;  /// misalignment scenario

  std::string theDTAlignRecordName, theDTErrorRecordName;
  std::string theCSCAlignRecordName, theCSCErrorRecordName;
  
  boost::shared_ptr<DTGeometry> theDTGeometry;
  boost::shared_ptr<CSCGeometry> theCSCGeometry;

  Alignments*      dt_Alignments;
  AlignmentErrors* dt_AlignmentErrors;
  Alignments*      csc_Alignments;
  AlignmentErrors* csc_AlignmentErrors;

};

//__________________________________________________________________________________________________
//__________________________________________________________________________________________________
//__________________________________________________________________________________________________


//__________________________________________________________________________________________________
MisalignedMuonESProducer::MisalignedMuonESProducer(const edm::ParameterSet& p) :
  theSaveToDB(p.getUntrackedParameter<bool>("saveToDbase")),
  theScenario(p.getParameter<edm::ParameterSet>("scenario")),
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
  edm::ESTransientHandle<DDCompactView> cpv;
  iRecord.getRecord<IdealGeometryRecord>().get( cpv );

  edm::ESHandle<MuonDDDConstants> mdc;
  iRecord.getRecord<MuonNumberingRecord>().get(mdc);

  DTGeometryBuilderFromDDD  DTGeometryBuilder;
  CSCGeometryBuilderFromDDD CSCGeometryBuilder;

  theDTGeometry = boost::shared_ptr<DTGeometry>(new DTGeometry );
  DTGeometryBuilder.build(theDTGeometry,  &(*cpv), *mdc );
  theCSCGeometry  = boost::shared_ptr<CSCGeometry>( new CSCGeometry );
  CSCGeometryBuilder.build( theCSCGeometry,  &(*cpv), *mdc );


  // Create the alignable hierarchy
  AlignableMuon* theAlignableMuon = new AlignableMuon( &(*theDTGeometry) , &(*theCSCGeometry) );

  // Create misalignment scenario
  MuonScenarioBuilder scenarioBuilder( theAlignableMuon );
  scenarioBuilder.applyScenario( theScenario );
  
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
  if (theSaveToDB) this->saveToDB();

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
