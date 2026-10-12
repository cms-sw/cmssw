///
/// An ESProducer that fills the MTDGeometryRcd with a misaligned MTD
///
/// This should replace the standard MTD geometry producers
/// when producing Misalignment scenarios.
///
/// \file
/// $Date: 2024/12/26 09:56:51 $
/// $Revision: 1.0 $
/// \author Pablo Martinez Ruiz del Arbol - IFCA
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
#include "Alignment/MTDAlignment/interface/AlignableMTD.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Alignment/MTDAlignment/interface/MTDScenarioBuilder.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Geometry/GeometryAligner/interface/GeometryAligner.h"
#include "Geometry/Records/interface/MTDGeometryRecord.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"

#include <memory>

#include <iostream>

class MTDMisalignedProducer : public edm::one::EDAnalyzer<> {
public:
  /// Constructor
  MTDMisalignedProducer(const edm::ParameterSet&);

  /// Destructor
  ~MTDMisalignedProducer() override;

  /// Produce the misaligned MTD geometry and store it
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  /// Save alignemnts and error to database
  void saveToDB();

private:
  const bool theSaveToDB;               /// whether or not writing to DB
  const edm::ParameterSet theScenario;  /// misalignment scenario

  std::string theMTDAlignRecordName, theMTDErrorRecordName;

  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> esTokenMTD_;

  Alignments mtd_Alignments;
  AlignmentErrorsExtended mtd_AlignmentErrorsExtended;
};

//__________________________________________________________________________________________________
MTDMisalignedProducer::MTDMisalignedProducer(const edm::ParameterSet& p)
    : theSaveToDB(p.getUntrackedParameter<bool>("saveToDbase")),
      theScenario(p.getParameter<edm::ParameterSet>("scenario")),
      theMTDAlignRecordName("MTDAlignmentRcd"),
      theMTDErrorRecordName("MTDAlignmentErrorExtendedRcd"),
      esTokenMTD_(esConsumes(edm::ESInputTag("", "idealForMTDMisalignedProducer"))) {}

//__________________________________________________________________________________________________
MTDMisalignedProducer::~MTDMisalignedProducer() = default;

//__________________________________________________________________________________________________
void MTDMisalignedProducer::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  edm::LogInfo("MisalignedMTD") << "Producer called";
  // Create the MTD geometry from ideal geometry
  edm::ESHandle<MTDGeometry> theMTDGeometry = eventSetup.getHandle(esTokenMTD_);

  // Create the alignable hierarchy
  AlignableMTD* theAlignableMTD = new AlignableMTD(&(*theMTDGeometry));

  // Create misalignment scenario
  MTDScenarioBuilder scenarioBuilder(theAlignableMTD);
  scenarioBuilder.applyScenario(theScenario);

  // Get alignments and errors
  mtd_Alignments = *(theAlignableMTD->mtdAlignments());
  mtd_AlignmentErrorsExtended = *(theAlignableMTD->mtdAlignmentErrorsExtended());

  if (theSaveToDB)
    this->saveToDB();

  edm::LogInfo("MisalignedMTD") << "Producer done";
}

//__________________________________________________________________________________________________
void MTDMisalignedProducer::saveToDB(void) {
  // Call service
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (!poolDbService.isAvailable())  // Die if not available
    throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";

  // Store MTD alignments and errors
  poolDbService->writeOneIOV<Alignments>(mtd_Alignments, poolDbService->beginOfTime(), theMTDAlignRecordName);
  poolDbService->writeOneIOV<AlignmentErrorsExtended>(
      mtd_AlignmentErrorsExtended, poolDbService->beginOfTime(), theMTDErrorRecordName);
}
//____________________________________________________________________________________________
DEFINE_FWK_MODULE(MTDMisalignedProducer);
