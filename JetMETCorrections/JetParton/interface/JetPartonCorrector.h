#ifndef JetPartonCorrector_h
#define JetPartonCorrector_h

///
/// jet parton energy corrections
///

#include <map>
#include <string>
#include <vector>
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
namespace edm {
  class ParameterSet;
}

namespace JetPartonNamespace {
  class ParametrizationJetParton;
  class UserPartonMixture;
}  // namespace JetPartonNamespace

class JetPartonCorrector : public JetCorrector {
public:
  JetPartonCorrector(const edm::ParameterSet& fConfig);
  ~JetPartonCorrector() override;

  double correction(const LorentzVector& fJet) const override;

  void setParameters(std::string aCalibrationType, double aJetFinderRadius, int aPartonMixture);

  /// if correction needs event information
  bool eventRequired() const override { return false; }

private:
  typedef std::map<double, JetPartonNamespace::ParametrizationJetParton*> ParametersMap;
  ParametersMap parametrization;
  int thePartonMixture;
  double theJetFinderRadius;
};
#endif
