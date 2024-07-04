#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/FWLite/interface/Record.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "MagneticField/ParametrizedEngine/interface/alpaka/ParabolicParametrizedMagneticField.h"

#include <iostream>
#include <fstream>
#include <Eigen/Core>

using namespace edm;
using namespace std;
using namespace ALPAKA_ACCELERATOR_NAMESPACE::magneticFieldParabolicPortable;
using Vector3f = Eigen::Matrix<float, 3, 1>;

int main() {
  ifstream file;
  edm::FileInPath mydata("MagneticField/Engine/data/Regression/referenceField_160812_RII_3_8T.bin");
  file.open(mydata.fullPath().c_str(), ios::binary);

  int count = 0;

  float px, py, pz;
  float bx, by, bz;
  vector<Vector3f> points;
  vector<GlobalVector> referenceB_vec;

  int numberOfPoints = 100;
  do {
    if (!(file.read((char*)&px, sizeof(float)) && file.read((char*)&py, sizeof(float)) &&
          file.read((char*)&pz, sizeof(float)) && file.read((char*)&bx, sizeof(float)) &&
          file.read((char*)&by, sizeof(float)) && file.read((char*)&bz, sizeof(float))))
      break;

    const auto gp = GlobalPoint(px, py, pz);
    if (gp.perp2() > Parameters::max_radius2 || fabs(gp.z()) > Parameters::max_z)
      continue;

    points.push_back(Vector3f(px, py, pz));
    referenceB_vec.push_back(GlobalVector(bx, by, bz));
    count++;
  } while (count < numberOfPoints);

  float resolution = 0.2;
  float maxdelta = 0.;
  int fail = 0;

  for (uint i=0; i<points.size(); i++) {
    const auto point = points[i];
    const auto referenceB = referenceB_vec[i];
    GlobalVector deviceB(0, 0, magneticFieldAtPoint(point));
    if ((referenceB - deviceB).mag() > resolution) {
      ++fail;
      float delta = (referenceB - deviceB).mag();
      if (delta > maxdelta)
        maxdelta = delta;
      if (fail < 10) {
        const GlobalPoint gp(point(0), point(1), point(2));
        cout << " Discrepancy at point  # " << count + 1 << ": " << gp << ", R " << gp.perp() << ", Phi " << gp.phi()
             << ", delta: " << referenceB - deviceB << " " << delta << endl;
        cout << " Old: " << referenceB << ", New: " << deviceB << endl;
      } else if (fail == 10) {
        cout << "..." << endl;
      }
    }
  }

  if (fail != 0)
    throw cms::Exception("RegressionFailure") << "MF regression found: at least " << fail << " failures";
}
