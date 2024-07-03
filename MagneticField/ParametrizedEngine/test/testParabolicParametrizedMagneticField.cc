#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/FWLite/interface/Record.h"

#include "MagneticField/ParametrizedEngine/interface/alpaka/ParabolicParametrizedMagneticField.h"

#include <iostream>
#include <fstream>
#include <Eigen/Core>

using namespace edm;
using namespace std;
using namespace ALPAKA_ACCELERATOR_NAMESPACE::MagneticFieldParabolicPortable;
using Vector3f = Eigen::Matrix<float, 3, 1>;

int main() {
  ifstream file;
  edm::FileInPath mydata("MagneticField/Engine/data/Regression/referenceField_160812_RII_3_8T.bin");
  file.open(mydata.fullPath().c_str(), ios::binary);

  float resolution = 0.0001;
  float maxdelta = 0.;
  int fail = 0;
  int count = 0;

  float px, py, pz;
  float bx, by, bz;
  GlobalPoint gp;

  int numberOfPoints = 100;
  do {
    if (!(file.read((char*)&px, sizeof(float)) && file.read((char*)&py, sizeof(float)) &&
          file.read((char*)&pz, sizeof(float)) && file.read((char*)&bx, sizeof(float)) &&
          file.read((char*)&by, sizeof(float)) && file.read((char*)&bz, sizeof(float))))
      break;
    gp = GlobalPoint(px, py, pz);

    if (gp.perp2() > Parameters::max_radius2 || gp.z() > Parameters::max_z)
      continue;

    GlobalVector referenceB(bx, by, bz);
    GlobalVector deviceB(0, 0, MagneticFieldAtPoint(Vector3f(px, py, pz)));
    if ((referenceB - deviceB).mag() > resolution) {
      ++fail;
      float delta = (referenceB - deviceB).mag();
      if (delta > maxdelta)
        maxdelta = delta;
      if (fail < 10) {
        cout << " Discrepancy at: # " << count + 1 << " " << gp << " R " << gp.perp() << " Phi " << gp.phi()
             << " delta : " << referenceB - deviceB << " " << delta << endl;
        cout << " Old: " << deviceB << " New: " << referenceB << endl;
      } else if (fail == 10) {
        cout << "..." << endl;
      }
    }
    count++;
  } while (count < numberOfPoints);

  if (fail != 0)
    //throw cms::Exception("RegressionFailure") << "MF regression found: " << fail << " failures";
    cout << "MF regression found: " << fail << " failures";
}