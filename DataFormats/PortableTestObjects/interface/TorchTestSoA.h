#ifndef DataFormats_PortableTestObjects_interface_TorchTestSoA_h
#define DataFormats_PortableTestObjects_interface_TorchTestSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace torchportabletest {

  GENERATE_SOA_LAYOUT(ParticleLayout, SOA_COLUMN(float, pt), SOA_COLUMN(float, eta), SOA_COLUMN(float, phi))
  using ParticleSoA = ParticleLayout<>;

  GENERATE_SOA_LAYOUT(SimpleNetLayout, SOA_COLUMN(float, reco_pt))
  using SimpleNetSoA = SimpleNetLayout<>;

  using ClassificationHead = Eigen::Vector<float, 3>;
  GENERATE_SOA_LAYOUT(MultiHeadNetLayout,
                      SOA_COLUMN(float, regression_head),
                      SOA_EIGEN_COLUMN(ClassificationHead, classification_head))
  using MultiHeadNetSoA = MultiHeadNetLayout<>;

  using ColorChannel = Eigen::Matrix<float, 9, 9>;
  GENERATE_SOA_LAYOUT(ImageLayout,
                      SOA_EIGEN_COLUMN(ColorChannel, r),
                      SOA_EIGEN_COLUMN(ColorChannel, g),
                      SOA_EIGEN_COLUMN(ColorChannel, b))
  using Image = ImageLayout<>;

  using LogitsType = Eigen::Vector<float, 10>;
  GENERATE_SOA_LAYOUT(LogitsLayout, SOA_EIGEN_COLUMN(LogitsType, logits))
  using Logits = LogitsLayout<>;

}  // namespace torchportabletest

#endif  // DataFormats_PortableTestObjects_interface_TorchTestSoA_h
