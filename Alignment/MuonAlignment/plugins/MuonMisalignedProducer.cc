///
/// An ESProducer that fills the MuonDigiGeometryRcd with a misaligned Muon
///
/// This should replace the standard DTGeometry and CSCGeometry producers
/// when producing Misalignment scenarios.
///
/// \file
/// $Date: 2009/03/26 09:56:51 $
/// $Revision: 1.11 $
/// \author Andre Sznajder - UERJ(Brazil)
///

// Framework
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// Alignment
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Alignment/MuonAlignment/interface/MuonScenarioBuilder.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include <memory>

class MuonMisalignedProducer : public edm::one::EDAnalyzer<> {
public:
  /// Constructor
  MuonMisalignedProducer(const edm::ParameterSet&);

  /// Destructor
  ~MuonMisalignedProducer() override;

  /// Produce the misaligned Muon geometry and store iti
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  /// Save alignemnts and error to database
  void saveToDB();

private:
  const bool theSaveToDB;               /// whether or not writing to DB
  const edm::ParameterSet theScenario;  /// misalignment scenario

  std::string theDTAlignRecordName, theDTErrorRecordName;
  std::string theCSCAlignRecordName, theCSCErrorRecordName;
  std::string theIdealGeometryLabel;  

  Alignments* dt_Alignments;
  AlignmentErrorsExtended* dt_AlignmentErrorsExtended;
  Alignments* csc_Alignments;
  AlignmentErrorsExtended* csc_AlignmentErrorsExtended;
};

//__________________________________________________________________________________________________
MuonMisalignedProducer::MuonMisalignedProducer(const edm::ParameterSet& p)
    : theSaveToDB(p.getUntrackedParameter<bool>("saveToDbase")),
      theScenario(p.getParameter<edm::ParameterSet>("scenario")),
      theDTAlignRecordName("DTAlignmentRcd"),
      theDTErrorRecordName("DTAlignmentErrorExtendedRcd"),
      theCSCAlignRecordName("CSCAlignmentRcd"),
      theCSCErrorRecordName("CSCAlignmentErrorExtendedRcd"),
      theIdealGeometryLabel("idealForMuonMisalignedProducer") {
}

//__________________________________________________________________________________________________
MuonMisalignedProducer::~MuonMisalignedProducer() {}

//__________________________________________________________________________________________________
void MuonMisalignedProducer::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  edm::LogInfo("MisalignedMuon") << "Producer called";
  // Create the Muon geometry from ideal geometry
  edm::ESHandle<DTGeometry> theDTGeometry;
  edm::ESHandle<CSCGeometry> theCSCGeometry;
  eventSetup.get<MuonGeometryRecord>().get(theIdealGeometryLabel, theDTGeometry);
  eventSetup.get<MuonGeometryRecord>().get(theIdealGeometryLabel, theCSCGeometry);

  // Create the alignable hierarchy
  AlignableMuon* theAlignableMuon = new AlignableMuon(&(*theDTGeometry), &(*theCSCGeometry));

  // Create misalignment scenario
  MuonScenarioBuilder scenarioBuilder(theAlignableMuon);
  scenarioBuilder.applyScenario(theScenario);

  // Get alignments and errors
  dt_Alignments = theAlignableMuon->dtAlignments();
  dt_AlignmentErrorsExtended = theAlignableMuon->dtAlignmentErrorsExtended();
  csc_Alignments = theAlignableMuon->cscAlignments();
  csc_AlignmentErrorsExtended = theAlignableMuon->cscAlignmentErrorsExtended();

  // Misalign the EventSetup geometry
  GeometryAligner aligner;

  aligner.applyAlignments<DTGeometry>(&(*theDTGeometry), dt_Alignments, dt_AlignmentErrorsExtended, AlignTransform());
  aligner.applyAlignments<CSCGeometry>(
      &(*theCSCGeometry), csc_Alignments, csc_AlignmentErrorsExtended, AlignTransform());

  // Write alignments to DB
  if (theSaveToDB)
    this->saveToDB();

  edm::LogInfo("MisalignedMuon") << "Producer done";
}

//__________________________________________________________________________________________________
void MuonMisalignedProducer::saveToDB(void) {
  // Call service
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (!poolDbService.isAvailable())  // Die if not available
    throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";

  // Store DT alignments and errors
  poolDbService->writeOne<Alignments>(&(*dt_Alignments), poolDbService->beginOfTime(), theDTAlignRecordName);
  poolDbService->writeOne<AlignmentErrorsExtended>(
      &(*dt_AlignmentErrorsExtended), poolDbService->beginOfTime(), theDTErrorRecordName);

  // Store CSC alignments and errors
  poolDbService->writeOne<Alignments>(&(*csc_Alignments), poolDbService->beginOfTime(), theCSCAlignRecordName);
  poolDbService->writeOne<AlignmentErrorsExtended>(
      &(*csc_AlignmentErrorsExtended), poolDbService->beginOfTime(), theCSCErrorRecordName);
}
//____________________________________________________________________________________________
DEFINE_FWK_MODULE(MuonMisalignedProducer);
