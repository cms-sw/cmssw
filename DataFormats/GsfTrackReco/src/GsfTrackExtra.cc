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
  std::vector<double>::iterator ir(result.begin());
  for (std::vector<GsfComponent5D>::const_iterator i = states.begin(); i != states.end(); ++i) {
    *(ir++) = (*i).weight();
  }
  return result;
}

std::vector<GsfTrackExtra::LocalParameterVector> GsfTrackExtra::parameters(
    const std::vector<GsfComponent5D>& states) const {
  std::vector<LocalParameterVector> result(states.size());
  std::vector<LocalParameterVector>::iterator ir(result.begin());
  for (std::vector<GsfComponent5D>::const_iterator i = states.begin(); i != states.end(); ++i) {
    *(ir++) = (*i).parameters();
  }
  return result;
}

std::vector<GsfTrackExtra::LocalCovarianceMatrix> GsfTrackExtra::covariances(
    const std::vector<GsfComponent5D>& states) const {
  std::vector<LocalCovarianceMatrix> result(states.size());
  std::vector<LocalCovarianceMatrix>::iterator ir(result.begin());
  for (std::vector<GsfComponent5D>::const_iterator i = states.begin(); i != states.end(); ++i) {
    (*i).covariance(*(ir++));
  }
  return result;
}
