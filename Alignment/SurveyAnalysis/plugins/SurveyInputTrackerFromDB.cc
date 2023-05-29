#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "Geometry/Records/interface/PTrackerAdditionalParametersPerDetRcd.h"

// Database
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"

#include "Alignment/SurveyAnalysis/plugins/SurveyInputTrackerFromDB.h"

SurveyInputTrackerFromDB::SurveyInputTrackerFromDB(const edm::ParameterSet& cfg)
    : tTopoToken_(esConsumes()),
      geomDetToken_(esConsumes()),
      ptpToken_(esConsumes()),
      textFileName(cfg.getParameter<std::string>("textFileName")) {}

void SurveyInputTrackerFromDB::analyze(const edm::Event&, const edm::EventSetup& setup) {
  if (theFirstEvent) {
    //  std::cout << "***************ENTERING INITIALIZATION******************" << std::endl;

    //Retrieve tracker topology from geometry
    const TrackerTopology* const tTopo = &setup.getData(tTopoToken_);

    //Get map from textreader
    SurveyInputTextReader dataReader;
    dataReader.readFile(textFileName);
    uIdMap = dataReader.UniqueIdMap();

    const GeometricDet* geom = &setup.getData(geomDetToken_);
    const PTrackerParameters& ptp = setup.getData(ptpToken_);
    TrackerGeometry* tracker = TrackerGeomBuilderFromGeometricDet().build(geom, ptp, tTopo);

    addComponent(new AlignableTracker(tracker, tTopo));
    addSurveyInfo(detector());

    //write out to a DB ...
    Alignments* myAlignments = detector()->alignments();
    AlignmentErrorsExtended* myAlignmentErrorsExtended = detector()->alignmentErrors();

    // 2. Store alignment[Error]s to DB
    edm::Service<cond::service::PoolDBOutputService> poolDbService;
    // Call service

    if (!poolDbService.isAvailable())  // Die if not available
      throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";

    poolDbService->writeOneIOV<Alignments>(*myAlignments, poolDbService->beginOfTime(), "TrackerAlignmentRcd");
    poolDbService->writeOneIOV<AlignmentErrorsExtended>(
        *myAlignmentErrorsExtended, poolDbService->beginOfTime(), "TrackerAlignmentErrorExtendedRcd");

    theFirstEvent = false;
  }
}

void SurveyInputTrackerFromDB::addSurveyInfo(Alignable* ali) {
  const align::Alignables& comp = ali->components();
  unsigned int nComp = comp.size();

  for (unsigned int i = 0; i < nComp; ++i)
    addSurveyInfo(comp[i]);

  align::ErrorMatrix error;

  SurveyInputTextReader::MapType::const_iterator it = uIdMap.find(std::make_pair(ali->id(), ali->alignableObjectId()));

  if (it != uIdMap.end()) {
    const align::Scalars& parameters = (it)->second;

    //move the surface
    //displacement
    align::LocalVector lvector(parameters[0], parameters[1], parameters[2]);
    align::GlobalVector gvector = ali->surface().toGlobal(lvector);
    ali->move(gvector);
    //rotation
    Basic3DVector<align::Scalar> rot_aa(parameters[3], parameters[4], parameters[5]);
    align::RotationType rotation(rot_aa, rot_aa.mag());
    ali->rotateInLocalFrame(rotation);

    //sets the errors for the hierarchy level
    double* errorData = error.Array();
    for (unsigned int i = 0; i < 21; ++i) {
      errorData[i] = parameters[i + 6];
    }

    ali->setSurvey(new SurveyDet(ali->surface(), error * (1e-6)));
  } else {
    error = ROOT::Math::SMatrixIdentity();
    ali->setSurvey(new SurveyDet(ali->surface(), error * 1e-6));
  }
}
// Plug in to framework

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SurveyInputTrackerFromDB);
