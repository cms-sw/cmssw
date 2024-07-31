#include <alpaka/alpaka.hpp>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/FWLite/interface/Record.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Geometry/Vector3D.h"

#include "MagneticField/ParametrizedEngine/interface/alpaka/ParabolicParametrizedMagneticField.h"

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include <iostream>
#include <fstream>
#include <Eigen/Core>

using namespace edm;
using namespace std;
using namespace alpaka;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace ALPAKA_ACCELERATOR_NAMESPACE::magneticFieldParabolicPortable;
using Vector3f = Eigen::Matrix<float, 3, 1>;
using Vec3 = CLHEP::Hep3Vector;

struct MagneticFieldKernel {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* __restrict__ in, T* __restrict__ out, size_t size) const {
    for (auto index : cms::alpakatools::uniform_elements(acc, size)) {
      out[index][0] = 0;
      out[index][1] = 0;
      out[index][2] = magneticFieldAtPoint(in[index]);
    }
  }
};

int main() {

  // get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

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

    const auto point = Vec3(px, py, pz);
    if (isValid(point))
      continue;

    points.push_back(Vector3f(px, py, pz));
    referenceB_vec.push_back(GlobalVector(bx, by, bz));
    count++;
  } while (count < numberOfPoints);

  const size_t size = points.size();
  // allocate the input and output host buffer in pinned memory accessible by the Platform devices
  auto in_host = cms::alpakatools::make_host_buffer<float[], Platform>(size);
  auto out_host = cms::alpakatools::make_host_buffer<float[], Platform>(size);
  // fill the input buffers, and the output buffer with zeros
  for (size_t i = 0; i < size; ++i) {
    in_host[i] = points[i];
    out_host[i] = {0.};
  }

  float resolution = 0.2;
  float maxdelta = 0.;
  int fail = 0;

  // run the test on each device
  for (auto const& device : devices) {
    auto queue = Queue(device);
    // allocate input and output buffers on the device
    auto points_dev = cms::alpakatools::make_device_buffer<float[]>(queue, size);
    auto deviceB = cms::alpakatools::make_device_buffer<float[]>(queue, size);

    // copy the input data to the device; the size is known from the buffer objects
    alpaka::memcpy(queue, points_dev, in_host);
    // fill the output buffer with zeros; the size is known from the buffer objects
    alpaka::memset(queue, deviceB, 0.);

    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(1, size);
    alpaka::exec<Acc1D>(queue, workDiv, MagneticFieldKernel, points_dev.data(), deviceB.data(), size);

    // copy the results from the device to the host
    alpaka::memcpy(queue, out_host, deviceB);

    // wait for the kernel and the potential copy to complete
    alpaka::wait(queue);

    // check the results
    for (uint i = 0; i < points.size(); i++) {
      const auto point = points[i];
      const auto referenceB = referenceB_vec[i];
      const auto out_host_element = out_host[i];
      GlobalVector parametricB(out_host_element[0], out_host_element[1], out_host_element[2]);
      float delta = (referenceB - parametricB).mag();
      if (delta > resolution) {
        ++fail;
        if (delta > maxdelta)
          maxdelta = delta;
        if (fail < 10) {
          const GlobalPoint gp(point(0), point(1), point(2));
          cout << " Discrepancy at point  # " << count + 1 << ": " << gp << ", R " << gp.perp() << ", Phi " << gp.phi()
               << ", delta: " << referenceB - parametricB << " " << delta << endl;
          cout << " Reference: " << referenceB << ", Approximation: " << parametricB << endl;
        } else if (fail == 10) {
          cout << "..." << endl;
        }
      }
    }

    if (fail != 0)
      throw cms::Exception("RegressionFailure") << "MF regression found: " << fail << " failures; max delta = "
        << maxdelta << endl;
  }
}
