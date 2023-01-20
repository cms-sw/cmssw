#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/PTrackerAdditionalParametersPerDetRcd.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Alignment/SurveyAnalysis/plugins/CreateSurveyRcds.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"
#include "CLHEP/Random/RandGauss.h"

CreateSurveyRcds::CreateSurveyRcds(const edm::ParameterSet& cfg)
    : tTopoToken_(esConsumes()),
      geomDetToken_(esConsumes()),
      ptpToken_(esConsumes()),
      aliToken_(esConsumes()),
      aliErrToken_(esConsumes()) {
  m_inputGeom = cfg.getUntrackedParameter<std::string>("inputGeom");
  m_inputSimpleMis = cfg.getUntrackedParameter<double>("simpleMis");
  m_generatedRandom = cfg.getUntrackedParameter<bool>("generatedRandom");
  m_generatedSimple = cfg.getUntrackedParameter<bool>("generatedSimple");
}

void CreateSurveyRcds::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  //Retrieve tracker topology from geometry
  const TrackerTopology* const tTopo = &setup.getData(tTopoToken_);
  const GeometricDet* geom = &setup.getData(geomDetToken_);
  const PTrackerParameters& ptp = setup.getData(ptpToken_);
  TrackerGeometry* tracker = TrackerGeomBuilderFromGeometricDet().build(geom, ptp, tTopo);

  //take geometry from DB or randomly generate geometry
  if (m_inputGeom == "sqlite") {
    //build the tracker
    const Alignments* alignments = &setup.getData(aliToken_);
    const AlignmentErrorsExtended* alignmentErrors = &setup.getData(aliErrToken_);

    //apply the latest alignments
    GeometryAligner aligner;
    aligner.applyAlignments<TrackerGeometry>(&(*tracker), alignments, alignmentErrors, AlignTransform());
  }

  addComponent(new AlignableTracker(tracker, tTopo));

  Alignable* ali = detector();
  if (m_inputGeom == "generated") {
    setGeometry(ali);
  }

  setSurveyErrors(ali);
}

void CreateSurveyRcds::setGeometry(Alignable* ali) {
  const align::Alignables& comp = ali->components();
  unsigned int nComp = comp.size();
  //move then do for lower level object
  //for issue of det vs detunit
  bool usecomps = true;
  if ((ali->alignableObjectId() == 2) && (nComp >= 1))
    usecomps = false;
  for (unsigned int i = 0; i < nComp; ++i) {
    if (usecomps)
      setGeometry(comp[i]);
  }
  DetId id(ali->id());
  int subdetlevel = id.subdetId();
  int level = ali->alignableObjectId();

  //for random misalignment
  if (m_generatedRandom) {
    if (subdetlevel > 0) {
      AlgebraicVector value = getStructureErrors(level, subdetlevel);

      double value0 = CLHEP::RandGauss::shoot(0, value[0]);
      double value1 = CLHEP::RandGauss::shoot(0, value[1]);
      double value2 = CLHEP::RandGauss::shoot(0, value[2]);
      double value3 = CLHEP::RandGauss::shoot(0, value[3]);
      double value4 = CLHEP::RandGauss::shoot(0, value[4]);
      double value5 = CLHEP::RandGauss::shoot(0, value[5]);

      //move/rotate the surface
      align::LocalVector diffR(value0, value1, value2);
      align::Scalar diffWx = value3;
      align::Scalar diffWy = value4;
      align::Scalar diffWz = value5;
      ali->move(ali->surface().toGlobal(diffR));
      ali->rotateAroundLocalX(diffWx);
      ali->rotateAroundLocalY(diffWy);
      ali->rotateAroundLocalZ(diffWz);
    }
  }

  // for simple misalignments
  if (m_generatedSimple) {
    if ((level == 2) || ((level == 1) && (ali->mother()->alignableObjectId() != 2))) {
      const double constMis = m_inputSimpleMis;
      const double dAngle = constMis / ali->surface().length();
      //std::cout << "Shift: " << constMis << ", Rot: " << dAngle << std::endl;
      double value0 = CLHEP::RandGauss::shoot(0, constMis);
      double value1 = CLHEP::RandGauss::shoot(0, constMis);
      double value2 = CLHEP::RandGauss::shoot(0, constMis);
      double value3 = CLHEP::RandGauss::shoot(0, dAngle);
      double value4 = CLHEP::RandGauss::shoot(0, dAngle);
      double value5 = CLHEP::RandGauss::shoot(0, dAngle);

      align::LocalVector diffR(value0, value1, value2);
      ali->move(ali->surface().toGlobal(diffR));
      align::Scalar diffWx = value3;
      align::Scalar diffWy = value4;
      align::Scalar diffWz = value5;
      ali->rotateAroundLocalX(diffWx);
      ali->rotateAroundLocalY(diffWy);
      ali->rotateAroundLocalZ(diffWz);
    }
  }
}

void CreateSurveyRcds::setSurveyErrors(Alignable* ali) {
  const align::Alignables& comp = ali->components();
  unsigned int nComp = comp.size();
  //move then do for lower level object
  //for issue of det vs detunit
  for (unsigned int i = 0; i < nComp; ++i) {
    setSurveyErrors(comp[i]);
  }

  DetId id(ali->id());
  int subdetlevel = id.subdetId();
  int level = ali->alignableObjectId();

  AlgebraicVector error = getStructureErrors(level, subdetlevel);

  double error0 = error[0];
  double error1 = error[1];
  double error2 = error[2];
  double error3 = error[3];
  double error4 = error[4];
  double error5 = error[5];

  // ----------------INFLATING ERRORS----------------------
  // inflating sensitive coordinates in each subdetector
  // tib
  if ((level <= 2) && (subdetlevel == 3)) {
    error0 = 0.01;
    error5 = 0.001;
    if ((level == 2) && (nComp == 2)) {
      error1 = 0.01;
    }
  }
  // tid
  if ((level <= 2) && (subdetlevel == 4)) {
    error0 = 0.01;
    error1 = 0.01;
    error2 = 0.01;
    error3 = 0.001;
    error4 = 0.001;
    error5 = 0.001;
    //error0=0.01; error1=0.002; error2=0.002; error3=0.0002; error4=0.0002; error5=0.001;
    //if ((level == 2)&&(nComp == 2)){
    //	error1 = 0.01;
    //}
  }
  if ((level == 23) && (subdetlevel == 4)) {  //Ring is a Disk
    error0 = 0.02;
    error1 = 0.02;
    error2 = 0.03;
    error3 = 0.0002;
    error4 = 0.0002;
    error5 = 0.0002;
  }
  if ((level == 22) && (subdetlevel == 4)) {  //Side of a Ring
    error0 = 0.01;
    error1 = 0.01;
    error2 = 0.01;
    error3 = 0.0002;
    error4 = 0.0002;
    error5 = 0.0002;
  }
  // tob
  if ((level <= 2) && (subdetlevel == 5)) {
    //error0 = 0.015; error1 = 0.015; error2 = 0.05; error3 = 0.001; error4 = 0.001; error5 = 0.001;
    error0 = 0.015;
    error1 = 0.003;
    error2 = 0.003;
    error3 = 0.0002;
    error4 = 0.0002;
    error5 = 0.001;
    if ((level == 2) && (nComp == 2)) {
      error1 = 0.015;
    }
  }
  if ((level == 27) && (subdetlevel == 5)) {  //Rod in a Layer
    error0 = 0.02;
    error1 = 0.02;
    error2 = 0.03;
    error3 = 0.001;
    error4 = 0.001;
    error5 = 0.001;
  }
  // tec
  if ((level <= 2) && (subdetlevel == 6)) {
    error0 = 0.02;
    error5 = 0.0005;
    if ((level == 2) && (nComp == 2)) {
      error1 = 0.02;
    }
  }
  if ((level == 34) && (subdetlevel == 6)) {  //Side on a Disk
    error0 = 0.01;
    error1 = 0.01;
    error2 = 0.02;
    error3 = 0.00005;
    error4 = 0.00005;
    error5 = 0.00005;
  }
  if ((level == 33) && (subdetlevel == 6)) {  //Petal on a Side of a Disk
    error0 = 0.01;
    error1 = 0.01;
    error2 = 0.02;
    error3 = 0.0001;
    error4 = 0.0001;
    error5 = 0.0001;
  }
  if ((level == 32) && (subdetlevel == 6)) {  //Ring on a Petal
    error0 = 0.007;
    error1 = 0.007;
    error2 = 0.015;
    error3 = 0.00015;
    error4 = 0.00015;
    error5 = 0.00015;
  }
  // ----------------INFLATING ERRORS----------------------

  //create the error matrix
  align::ErrorMatrix error_Matrix;
  double* errorData = error_Matrix.Array();
  errorData[0] = error0 * error0;
  errorData[2] = error1 * error1;
  errorData[5] = error2 * error2;
  errorData[9] = error3 * error3;
  errorData[14] = error4 * error4;
  errorData[20] = error5 * error5;
  errorData[1] = 0.0;
  errorData[3] = 0.0;
  errorData[4] = 0.0;
  errorData[6] = 0.0;
  errorData[7] = 0.0;
  errorData[8] = 0.0;
  errorData[10] = 0.0;
  errorData[11] = 0.0;
  errorData[12] = 0.0;
  errorData[13] = 0.0;
  errorData[15] = 0.0;
  errorData[16] = 0.0;
  errorData[17] = 0.0;
  errorData[18] = 0.0;
  errorData[19] = 0.0;

  ali->setSurvey(new SurveyDet(ali->surface(), error_Matrix));
}

//-------------------------------------------------------
// DEFAULT VALUES FOR THE ASSEMBLY PRECISION
//-------------------------------------------------------
AlgebraicVector CreateSurveyRcds::getStructurePlacements(int level, int subdetlevel) {
  AlgebraicVector deltaRW(6);
  deltaRW(1) = 0.0;
  deltaRW(2) = 0.0;
  deltaRW(3) = 0.0;
  deltaRW(4) = 0.0;
  deltaRW(5) = 0.0;
  deltaRW(6) = 0.0;
  //PIXEL
  if ((level == 37) && (subdetlevel == 1)) {
    deltaRW(1) = 0.3;
    deltaRW(2) = 0.3;
    deltaRW(3) = 0.3;
    deltaRW(4) = 0.0017;
    deltaRW(5) = 0.0017;
    deltaRW(6) = 0.0017;
  }
  //STRIP
  if ((level == 38) && (subdetlevel == 3)) {
    deltaRW(1) = 0.3;
    deltaRW(2) = 0.3;
    deltaRW(3) = 0.3;
    deltaRW(4) = 0.0004;
    deltaRW(5) = 0.0004;
    deltaRW(6) = 0.0004;
  }
  //TRACKER
  if ((level == 39) && (subdetlevel == 1)) {
    deltaRW(1) = 0.0;
    deltaRW(2) = 0.0;
    deltaRW(3) = 0.0;
    deltaRW(4) = 0.0;
    deltaRW(5) = 0.0;
    deltaRW(6) = 0.0;
  }
  //TPB
  if ((level == 7) && (subdetlevel == 1)) {
    deltaRW(1) = 0.2;
    deltaRW(2) = 0.2;
    deltaRW(3) = 0.2;
    deltaRW(4) = 0.003;
    deltaRW(5) = 0.003;
    deltaRW(6) = 0.003;
  }
  if ((level == 6) && (subdetlevel == 1)) {
    deltaRW(1) = 0.05;
    deltaRW(2) = 0.05;
    deltaRW(3) = 0.05;
    deltaRW(4) = 0.0008;
    deltaRW(5) = 0.0008;
    deltaRW(6) = 0.0008;
  }
  if ((level == 5) && (subdetlevel == 1)) {
    deltaRW(1) = 0.02;
    deltaRW(2) = 0.02;
    deltaRW(3) = 0.02;
    deltaRW(4) = 0.0004;
    deltaRW(5) = 0.0004;
    deltaRW(6) = 0.0004;
  }
  if ((level == 4) && (subdetlevel == 1)) {
    deltaRW(1) = 0.01;
    deltaRW(2) = 0.01;
    deltaRW(3) = 0.005;
    deltaRW(4) = 0.0002;
    deltaRW(5) = 0.0002;
    deltaRW(6) = 0.0002;
  }
  if ((level == 2) && (subdetlevel == 1)) {
    deltaRW(1) = 0.005;
    deltaRW(2) = 0.005;
    deltaRW(3) = 0.003;
    deltaRW(4) = 0.001;
    deltaRW(5) = 0.001;
    deltaRW(6) = 0.001;
  }
  if ((level == 1) && (subdetlevel == 1)) {
    deltaRW(1) = 0.005;
    deltaRW(2) = 0.005;
    deltaRW(3) = 0.003;
    deltaRW(4) = 0.001;
    deltaRW(5) = 0.001;
    deltaRW(6) = 0.001;
  }
  //TPE
  if ((level == 13) && (subdetlevel == 2)) {
    deltaRW(1) = 0.2;
    deltaRW(2) = 0.2;
    deltaRW(3) = 0.2;
    deltaRW(4) = 0.0017;
    deltaRW(5) = 0.0017;
    deltaRW(6) = 0.0017;
  }
  if ((level == 12) && (subdetlevel == 2)) {
    deltaRW(1) = 0.05;
    deltaRW(2) = 0.05;
    deltaRW(3) = 0.05;
    deltaRW(4) = 0.0004;
    deltaRW(5) = 0.0004;
    deltaRW(6) = 0.0004;
  }
  if ((level == 11) && (subdetlevel == 2)) {
    deltaRW(1) = 0.02;
    deltaRW(2) = 0.02;
    deltaRW(3) = 0.02;
    deltaRW(4) = 0.001;
    deltaRW(5) = 0.001;
    deltaRW(6) = 0.001;
  }
  if ((level == 10) && (subdetlevel == 2)) {
    deltaRW(1) = 0.01;
    deltaRW(2) = 0.01;
    deltaRW(3) = 0.01;
    deltaRW(4) = 0.001;
    deltaRW(5) = 0.001;
    deltaRW(6) = 0.001;
  }
  if ((level == 9) && (subdetlevel == 2)) {
    deltaRW(1) = 0.01;
    deltaRW(2) = 0.01;
    deltaRW(3) = 0.005;
    deltaRW(4) = 0.002;
    deltaRW(5) = 0.002;
    deltaRW(6) = 0.002;
  }
  if ((level == 2) && (subdetlevel == 2)) {
    deltaRW(1) = 0.005;
    deltaRW(2) = 0.005;
    deltaRW(3) = 0.003;
    deltaRW(4) = 0.001;
    deltaRW(5) = 0.001;
    deltaRW(6) = 0.001;
  }
  if ((level == 1) && (subdetlevel == 2)) {
    deltaRW(1) = 0.005;
    deltaRW(2) = 0.005;
    deltaRW(3) = 0.003;
    deltaRW(4) = 0.001;
    deltaRW(5) = 0.001;
    deltaRW(6) = 0.001;
  }
  //TIB
  if ((level == 20) && (subdetlevel == 3)) {
    deltaRW(1) = 0.2;
    deltaRW(2) = 0.2;
    deltaRW(3) = 0.2;
    deltaRW(4) = 0.0017;
    deltaRW(5) = 0.0017;
    deltaRW(6) = 0.0017;
  }
  if ((level == 19) && (subdetlevel == 3)) {
    deltaRW(1) = 0.1;
    deltaRW(2) = 0.1;
    deltaRW(3) = 0.1;
    deltaRW(4) = 0.0008;
    deltaRW(5) = 0.0008;
    deltaRW(6) = 0.0008;
  }
  if ((level == 18) && (subdetlevel == 3)) {
    deltaRW(1) = 0.04;
    deltaRW(2) = 0.04;
    deltaRW(3) = 0.02;
    deltaRW(4) = 0.0006;
    deltaRW(5) = 0.0006;
    deltaRW(6) = 0.0006;
  }
  if ((level == 17) && (subdetlevel == 3)) {
    deltaRW(1) = 0.03;
    deltaRW(2) = 0.03;
    deltaRW(3) = 0.015;
    deltaRW(4) = 0.0004;
    deltaRW(5) = 0.0004;
    deltaRW(6) = 0.0004;
  }
  if ((level == 16) && (subdetlevel == 3)) {
    deltaRW(1) = 0.01;
    deltaRW(2) = 0.01;
    deltaRW(3) = 0.01;
    deltaRW(4) = 0.0004;
    deltaRW(5) = 0.0002;
    deltaRW(6) = 0.0002;
  }
  if ((level == 15) && (subdetlevel == 3)) {
    deltaRW(1) = 0.01;
    deltaRW(2) = 0.01;
    deltaRW(3) = 0.01;
    deltaRW(4) = 0.0004;
    deltaRW(5) = 0.0002;
    deltaRW(6) = 0.0002;
  }
  if ((level == 2) && (subdetlevel == 3)) {
    deltaRW(1) = 0.005;
    deltaRW(2) = 0.005;
    deltaRW(3) = 0.005;
    deltaRW(4) = 0.001;
    deltaRW(5) = 0.0005;
    deltaRW(6) = 0.0005;
  }
  if ((level == 1) && (subdetlevel == 3)) {
    deltaRW(1) = 0.005;
    deltaRW(2) = 0.005;
    deltaRW(3) = 0.005;
    deltaRW(4) = 0.001;
    deltaRW(5) = 0.0005;
    deltaRW(6) = 0.0005;
  }
  //TID
  if ((level == 25) && (subdetlevel == 4)) {
    deltaRW(1) = 0.2;
    deltaRW(2) = 0.2;
    deltaRW(3) = 0.2;
    deltaRW(4) = 0.0013;
    deltaRW(5) = 0.0013;
    deltaRW(6) = 0.0013;
  }
  if ((level == 24) && (subdetlevel == 4)) {
    deltaRW(1) = 0.05;
    deltaRW(2) = 0.05;
    deltaRW(3) = 0.05;
    deltaRW(4) = 0.0004;
    deltaRW(5) = 0.0004;
    deltaRW(6) = 0.0004;
  }
  if ((level == 23) && (subdetlevel == 4)) {
    deltaRW(1) = 0.01;
    deltaRW(2) = 0.01;
    deltaRW(3) = 0.01;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  if ((level == 22) && (subdetlevel == 4)) {
    deltaRW(1) = 0.005;
    deltaRW(2) = 0.005;
    deltaRW(3) = 0.005;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  if ((level == 2) && (subdetlevel == 4)) {
    deltaRW(1) = 0.005;
    deltaRW(2) = 0.005;
    deltaRW(3) = 0.005;
    deltaRW(4) = 0.0005;
    deltaRW(5) = 0.0005;
    deltaRW(6) = 0.0005;
  }
  if ((level == 1) && (subdetlevel == 4)) {
    deltaRW(1) = 0.005;
    deltaRW(2) = 0.005;
    deltaRW(3) = 0.005;
    deltaRW(4) = 0.0005;
    deltaRW(5) = 0.0005;
    deltaRW(6) = 0.0005;
  }
  //TOB
  if ((level == 30) && (subdetlevel == 5)) {
    deltaRW(1) = 0.2;
    deltaRW(2) = 0.2;
    deltaRW(3) = 0.2;
    deltaRW(4) = 0.0008;
    deltaRW(5) = 0.0008;
    deltaRW(6) = 0.0008;
  }
  if ((level == 29) && (subdetlevel == 5)) {
    deltaRW(1) = 0.014;
    deltaRW(2) = 0.014;
    deltaRW(3) = 0.05;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  if ((level == 28) && (subdetlevel == 5)) {
    deltaRW(1) = 0.02;
    deltaRW(2) = 0.02;
    deltaRW(3) = 0.02;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  if ((level == 27) && (subdetlevel == 5)) {
    deltaRW(1) = 0.01;
    deltaRW(2) = 0.01;
    deltaRW(3) = 0.02;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  if ((level == 2) && (subdetlevel == 5)) {
    deltaRW(1) = 0.003;
    deltaRW(2) = 0.003;
    deltaRW(3) = 0.01;
    deltaRW(4) = 0.0002;
    deltaRW(5) = 0.0002;
    deltaRW(6) = 0.0002;
  }
  if ((level == 1) && (subdetlevel == 5)) {
    deltaRW(1) = 0.003;
    deltaRW(2) = 0.003;
    deltaRW(3) = 0.01;
    deltaRW(4) = 0.0002;
    deltaRW(5) = 0.0002;
    deltaRW(6) = 0.0002;
  }
  //TEC
  if ((level == 36) && (subdetlevel == 6)) {
    deltaRW(1) = 0.2;
    deltaRW(2) = 0.2;
    deltaRW(3) = 0.2;
    deltaRW(4) = 0.0008;
    deltaRW(5) = 0.0008;
    deltaRW(6) = 0.0008;
  }
  if ((level == 35) && (subdetlevel == 6)) {
    deltaRW(1) = 0.05;
    deltaRW(2) = 0.05;
    deltaRW(3) = 0.05;
    deltaRW(4) = 0.0003;
    deltaRW(5) = 0.0003;
    deltaRW(6) = 0.0003;
  }
  if ((level == 34) && (subdetlevel == 6)) {
    deltaRW(1) = 0.01;
    deltaRW(2) = 0.01;
    deltaRW(3) = 0.02;
    deltaRW(4) = 0.00005;
    deltaRW(5) = 0.00005;
    deltaRW(6) = 0.00005;
  }
  if ((level == 33) && (subdetlevel == 6)) {
    deltaRW(1) = 0.01;
    deltaRW(2) = 0.01;
    deltaRW(3) = 0.02;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  if ((level == 32) && (subdetlevel == 6)) {
    deltaRW(1) = 0.007;
    deltaRW(2) = 0.007;
    deltaRW(3) = 0.015;
    deltaRW(4) = 0.00015;
    deltaRW(5) = 0.00015;
    deltaRW(6) = 0.00015;
  }
  if ((level == 2) && (subdetlevel == 6)) {
    deltaRW(1) = 0.002;
    deltaRW(2) = 0.002;
    deltaRW(3) = 0.005;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  if ((level == 1) && (subdetlevel == 6)) {
    deltaRW(1) = 0.002;
    deltaRW(2) = 0.002;
    deltaRW(3) = 0.005;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }

  return deltaRW;
}
//-------------------------------------------------------
// DEFAULT VALUES FOR THE PRECISION OF THE SURVEY
//-------------------------------------------------------
AlgebraicVector CreateSurveyRcds::getStructureErrors(int level, int subdetlevel) {
  AlgebraicVector deltaRW(6);
  deltaRW(1) = 0.0;
  deltaRW(2) = 0.0;
  deltaRW(3) = 0.0;
  deltaRW(4) = 0.0;
  deltaRW(5) = 0.0;
  deltaRW(6) = 0.0;
  //PIXEL
  if ((level == 37) && (subdetlevel == 1)) {  //Pixel Detector in Tracker
    deltaRW(1) = 0.2;
    deltaRW(2) = 0.2;
    deltaRW(3) = 0.2;
    deltaRW(4) = 0.0017;
    deltaRW(5) = 0.0017;
    deltaRW(6) = 0.0017;
  }
  //STRIP
  if ((level == 38) && (subdetlevel == 3)) {  //Strip Tracker in Tracker
    deltaRW(1) = 0.2;
    deltaRW(2) = 0.2;
    deltaRW(3) = 0.2;
    deltaRW(4) = 0.0004;
    deltaRW(5) = 0.0004;
    deltaRW(6) = 0.0004;
  }
  //TRACKER
  if ((level == 39) && (subdetlevel == 1)) {  //Tracker
    deltaRW(1) = 0.0;
    deltaRW(2) = 0.0;
    deltaRW(3) = 0.0;
    deltaRW(4) = 0.0;
    deltaRW(5) = 0.0;
    deltaRW(6) = 0.0;
  }
  //TPB
  if ((level == 7) && (subdetlevel == 1)) {  //Barrel Pixel in Pixel
    deltaRW(1) = 0.05;
    deltaRW(2) = 0.05;
    deltaRW(3) = 0.1;
    deltaRW(4) = 0.0008;
    deltaRW(5) = 0.0008;
    deltaRW(6) = 0.0008;
  }
  if ((level == 6) && (subdetlevel == 1)) {  //HalfBarrel in Barrel Pixel
    deltaRW(1) = 0.015;
    deltaRW(2) = 0.015;
    deltaRW(3) = 0.03;
    deltaRW(4) = 0.0003;
    deltaRW(5) = 0.0003;
    deltaRW(6) = 0.0003;
  }
  if ((level == 5) && (subdetlevel == 1)) {  //HalfShell in HalfBarrel
    deltaRW(1) = 0.005;
    deltaRW(2) = 0.005;
    deltaRW(3) = 0.01;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  if ((level == 4) && (subdetlevel == 1)) {  //Ladder in HalfShell
    deltaRW(1) = 0.001;
    deltaRW(2) = 0.001;
    deltaRW(3) = 0.002;
    deltaRW(4) = 0.00005;
    deltaRW(5) = 0.00005;
    deltaRW(6) = 0.00005;
  }
  if ((level == 2) && (subdetlevel == 1)) {  //Det in Ladder
    deltaRW(1) = 0.0005;
    deltaRW(2) = 0.001;
    deltaRW(3) = 0.001;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  if ((level == 1) && (subdetlevel == 1)) {  //DetUnit in Ladder
    deltaRW(1) = 0.0005;
    deltaRW(2) = 0.001;
    deltaRW(3) = 0.001;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  //TPE
  if ((level == 13) && (subdetlevel == 2)) {  //Forward Pixel in Pixel
    deltaRW(1) = 0.05;
    deltaRW(2) = 0.05;
    deltaRW(3) = 0.1;
    deltaRW(4) = 0.0004;
    deltaRW(5) = 0.0004;
    deltaRW(6) = 0.0004;
  }
  if ((level == 12) && (subdetlevel == 2)) {  //HalfCylinder in Forward Pixel
    deltaRW(1) = 0.015;
    deltaRW(2) = 0.015;
    deltaRW(3) = 0.03;
    deltaRW(4) = 0.00015;
    deltaRW(5) = 0.00015;
    deltaRW(6) = 0.00015;
  }
  if ((level == 11) && (subdetlevel == 2)) {  //HalfDisk in HalfCylinder
    deltaRW(1) = 0.005;
    deltaRW(2) = 0.005;
    deltaRW(3) = 0.01;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  if ((level == 10) && (subdetlevel == 2)) {  //Blade in HalfDisk
    deltaRW(1) = 0.001;
    deltaRW(2) = 0.001;
    deltaRW(3) = 0.002;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  if ((level == 9) && (subdetlevel == 2)) {  //Panel in Blade
    deltaRW(1) = 0.001;
    deltaRW(2) = 0.0008;
    deltaRW(3) = 0.0006;
    deltaRW(4) = 0.0002;
    deltaRW(5) = 0.0002;
    deltaRW(6) = 0.0002;
  }
  if ((level == 2) && (subdetlevel == 2)) {  //Det in Panel
    deltaRW(1) = 0.0005;
    deltaRW(2) = 0.0004;
    deltaRW(3) = 0.0006;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0003;
    deltaRW(6) = 0.0001;
  }
  if ((level == 1) && (subdetlevel == 2)) {  //DetUnit in Panel
    deltaRW(1) = 0.0005;
    deltaRW(2) = 0.0004;
    deltaRW(3) = 0.0006;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0003;
    deltaRW(6) = 0.0001;
  }
  //TIB
  if ((level == 20) && (subdetlevel == 3)) {  //TIB in Strip Tracker
    deltaRW(1) = 0.08;
    deltaRW(2) = 0.08;
    deltaRW(3) = 0.04;
    deltaRW(4) = 0.0017;
    deltaRW(5) = 0.0017;
    deltaRW(6) = 0.0017;
  }
  if ((level == 19) && (subdetlevel == 3)) {  //HalfBarrel in TIB
    deltaRW(1) = 0.04;
    deltaRW(2) = 0.04;
    deltaRW(3) = 0.02;
    deltaRW(4) = 0.0003;
    deltaRW(5) = 0.0003;
    deltaRW(6) = 0.0003;
  }
  if ((level == 18) && (subdetlevel == 3)) {  //Layer in HalfBarrel
    deltaRW(1) = 0.02;
    deltaRW(2) = 0.02;
    deltaRW(3) = 0.01;
    deltaRW(4) = 0.0006;
    deltaRW(5) = 0.0006;
    deltaRW(6) = 0.0006;
  }
  if ((level == 17) && (subdetlevel == 3)) {  //HalfShell in Layer
    deltaRW(1) = 0.01;
    deltaRW(2) = 0.01;
    deltaRW(3) = 0.005;
    deltaRW(4) = 0.0002;
    deltaRW(5) = 0.0002;
    deltaRW(6) = 0.0002;
  }
  if ((level == 16) && (subdetlevel == 3)) {  //Surface in a HalfShell
    deltaRW(1) = 0.004;
    deltaRW(2) = 0.004;
    deltaRW(3) = 0.008;
    deltaRW(4) = 0.0002;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  if ((level == 15) && (subdetlevel == 3)) {  //String in a Surface
    deltaRW(1) = 0.004;
    deltaRW(2) = 0.004;
    deltaRW(3) = 0.008;
    deltaRW(4) = 0.0002;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  if ((level == 2) && (subdetlevel == 3)) {  //Det in a String
    deltaRW(1) = 0.002;
    deltaRW(2) = 0.002;
    deltaRW(3) = 0.004;
    deltaRW(4) = 0.0004;
    deltaRW(5) = 0.0002;
    deltaRW(6) = 0.0002;
  }
  if ((level == 1) && (subdetlevel == 3)) {  //DetUnit in a String
    deltaRW(1) = 0.002;
    deltaRW(2) = 0.002;
    deltaRW(3) = 0.004;
    deltaRW(4) = 0.0004;
    deltaRW(5) = 0.0002;
    deltaRW(6) = 0.0002;
  }
  //TID
  if ((level == 25) && (subdetlevel == 4)) {  //TID in Strip Tracker
    deltaRW(1) = 0.05;
    deltaRW(2) = 0.05;
    deltaRW(3) = 0.1;
    deltaRW(4) = 0.0003;
    deltaRW(5) = 0.0003;
    deltaRW(6) = 0.0003;
  }
  if ((level == 24) && (subdetlevel == 4)) {  //Disk in a TID
    deltaRW(1) = 0.01;
    deltaRW(2) = 0.01;
    deltaRW(3) = 0.02;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  if ((level == 23) && (subdetlevel == 4)) {  //Ring is a Disk
    deltaRW(1) = 0.004;
    deltaRW(2) = 0.004;
    deltaRW(3) = 0.005;
    deltaRW(4) = 0.00004;
    deltaRW(5) = 0.00004;
    deltaRW(6) = 0.00004;
  }
  if ((level == 22) && (subdetlevel == 4)) {  //Side of a Ring
    deltaRW(1) = 0.002;
    deltaRW(2) = 0.002;
    deltaRW(3) = 0.002;
    deltaRW(4) = 0.00004;
    deltaRW(5) = 0.00004;
    deltaRW(6) = 0.00004;
  }
  if ((level == 2) && (subdetlevel == 4)) {  //Det in a Side
    deltaRW(1) = 0.002;
    deltaRW(2) = 0.002;
    deltaRW(3) = 0.002;
    deltaRW(4) = 0.0002;
    deltaRW(5) = 0.0002;
    deltaRW(6) = 0.0002;
  }
  if ((level == 1) && (subdetlevel == 4)) {  //DetUnit is a Side
    deltaRW(1) = 0.002;
    deltaRW(2) = 0.002;
    deltaRW(3) = 0.002;
    deltaRW(4) = 0.0002;
    deltaRW(5) = 0.0002;
    deltaRW(6) = 0.0002;
  }
  //TOB
  if ((level == 30) && (subdetlevel == 5)) {  // TOB in Strip Tracker
    deltaRW(1) = 0.06;
    deltaRW(2) = 0.06;
    deltaRW(3) = 0.06;
    deltaRW(4) = 0.00025;
    deltaRW(5) = 0.00025;
    deltaRW(6) = 0.00025;
  }
  if ((level == 29) && (subdetlevel == 5)) {  //HalfBarrel in the TOB
    deltaRW(1) = 0.014;
    deltaRW(2) = 0.014;
    deltaRW(3) = 0.05;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  if ((level == 28) && (subdetlevel == 5)) {  //Layer in a HalfBarrel
    deltaRW(1) = 0.02;
    deltaRW(2) = 0.02;
    deltaRW(3) = 0.02;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  if ((level == 27) && (subdetlevel == 5)) {  //Rod in a Layer
    deltaRW(1) = 0.01;
    deltaRW(2) = 0.01;
    deltaRW(3) = 0.02;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  if ((level == 2) && (subdetlevel == 5)) {  //Det in a Rod
    deltaRW(1) = 0.003;
    deltaRW(2) = 0.003;
    deltaRW(3) = 0.01;
    deltaRW(4) = 0.0002;
    deltaRW(5) = 0.0002;
    deltaRW(6) = 0.0002;
  }
  if ((level == 1) && (subdetlevel == 5)) {  //DetUnit in a Rod
    deltaRW(1) = 0.003;
    deltaRW(2) = 0.003;
    deltaRW(3) = 0.01;
    deltaRW(4) = 0.0002;
    deltaRW(5) = 0.0002;
    deltaRW(6) = 0.0002;
  }
  //TEC
  if ((level == 36) && (subdetlevel == 6)) {  //TEC in the Strip Tracker
    deltaRW(1) = 0.06;
    deltaRW(2) = 0.06;
    deltaRW(3) = 0.1;
    deltaRW(4) = 0.0003;
    deltaRW(5) = 0.0003;
    deltaRW(6) = 0.0003;
  }
  if ((level == 35) && (subdetlevel == 6)) {  //Disk in the TEC
    deltaRW(1) = 0.015;
    deltaRW(2) = 0.015;
    deltaRW(3) = 0.03;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  if ((level == 34) && (subdetlevel == 6)) {  //Side on a Disk
    deltaRW(1) = 0.01;
    deltaRW(2) = 0.01;
    deltaRW(3) = 0.02;
    deltaRW(4) = 0.00005;
    deltaRW(5) = 0.00005;
    deltaRW(6) = 0.00005;
  }
  if ((level == 33) && (subdetlevel == 6)) {  //Petal on a Side of a Disk
    deltaRW(1) = 0.01;
    deltaRW(2) = 0.01;
    deltaRW(3) = 0.02;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  if ((level == 32) && (subdetlevel == 6)) {  //Ring on a Petal
    deltaRW(1) = 0.007;
    deltaRW(2) = 0.007;
    deltaRW(3) = 0.015;
    deltaRW(4) = 0.00015;
    deltaRW(5) = 0.00015;
    deltaRW(6) = 0.00015;
  }
  if ((level == 2) && (subdetlevel == 6)) {  //Det on a Ring
    deltaRW(1) = 0.002;
    deltaRW(2) = 0.002;
    deltaRW(3) = 0.005;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }
  if ((level == 1) && (subdetlevel == 6)) {  // DetUnit on a Ring
    deltaRW(1) = 0.002;
    deltaRW(2) = 0.002;
    deltaRW(3) = 0.005;
    deltaRW(4) = 0.0001;
    deltaRW(5) = 0.0001;
    deltaRW(6) = 0.0001;
  }

  return deltaRW;
}

// Plug in to framework

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CreateSurveyRcds);
