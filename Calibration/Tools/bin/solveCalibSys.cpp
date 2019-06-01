#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "Calibration/Tools/interface/calibXMLwriter.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibTools.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapEcal.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalBarrel.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalEndcap.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Calibration/Tools/interface/matrixSaver.h"
#include "Calibration/Tools/interface/BlockSolver.h"

//#include "Calibration/EcalAlCaRecoProducers/interface/trivialParser.h"
//#include "Calibration/EcalAlCaRecoProducers/bin/trivialParser.h"

#include "TH2.h"
#include "TProfile.h"
#include "TH1.h"
#include "TFile.h"

#include "CLHEP/Matrix/GenMatrix.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/Vector.h"

#define PI_GRECO 3.14159265

inline int etaShifter(const int etaOld) {
  if (etaOld < 0)
    return etaOld + 85;
  else if (etaOld > 0)
    return etaOld + 84;
  assert(false);
}

// ------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  if (argc < 3)
    return 1;
  std::string chi2MtrFile = argv[0];
  std::string chi2VtrFile = argv[1];
  std::string cfgFile = argv[2];

  matrixSaver leggo;

  CLHEP::HepMatrix *chi2Mtr = dynamic_cast<CLHEP::HepMatrix *>(leggo.getMatrix(chi2MtrFile));
  CLHEP::HepVector *chi2Vtr = dynamic_cast<CLHEP::HepVector *>(leggo.getMatrix(chi2VtrFile));

  double min = 0.5;               //FIXME
  double max = 1.5;               //FIXME
  bool usingBlockSolver = false;  //FIXME
  int region = 0;                 //FIXME

  CLHEP::HepVector result = CLHEP::solve(*chi2Mtr, *chi2Vtr);
  if (result.normsq() < min * chi2Mtr->num_row() || result.normsq() > max * chi2Mtr->num_row()) {
    if (usingBlockSolver) {
      edm::LogWarning("IML") << "using  blocSlover " << std::endl;
      BlockSolver()(*chi2Mtr, *chi2Vtr, result);
    } else {
      edm::LogWarning("IML") << "coeff out of range " << std::endl;
      for (int i = 0; i < chi2Vtr->num_row(); ++i)
        result[i] = 1.;
    }
  }

  unsigned int numberOfElements = chi2Mtr->num_row();
  std::map<unsigned int, float> coefficients;
  for (unsigned int i = 0; i < numberOfElements; ++i)
    coefficients[i] = result[i];

  if (region == 0)  //PG EB
  {
    //      int index =

  } else if (region == -1)  //PG EB-
  {
  } else  //PG EB+
  {
  }

  //FIXME salva la mappa
  return 0;
}
