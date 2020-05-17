#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
using namespace reco;

GsfTrackExtra::GsfTrackExtra(const std::vector<GsfComponent5D>& outerStates,
                             const double& outerLocalPzSign,
                             const std::vector<GsfComponent5D>& innerStates,
                             const double& innerLocalPzSign,
                             const std::vector<GsfTangent>& tangents)
    : outerStates_(outerStates),
      positiveOuterStatePz_(outerLocalPzSign > 0.),
      innerStates_(innerStates),
      positiveInnerStatePz_(innerLocalPzSign > 0.),
      tangents_(tangents) {}

std::vector<double> GsfTrackExtra::weights(const std::vector<GsfComponent5D>& states) const {
  std::vector<double> result(states.size());
  auto ir(result.begin());
  for (const auto& state : states) {
    *(ir++) = state.weight();
  }
  return result;
}

std::vector<GsfTrackExtra::LocalParameterVector> GsfTrackExtra::parameters(
    const std::vector<GsfComponent5D>& states) const {
  std::vector<LocalParameterVector> result(states.size());
  auto ir(result.begin());
  for (const auto& state : states) {
    *(ir++) = state.parameters();
  }
  return result;
}

std::vector<GsfTrackExtra::LocalCovarianceMatrix> GsfTrackExtra::covariances(
    const std::vector<GsfComponent5D>& states) const {
  std::vector<LocalCovarianceMatrix> result(states.size());
  auto ir(result.begin());
  for (const auto& state : states) {
    state.covariance(*(ir++));
  }
  return result;
}
