#ifndef RecoBTau_JetTagComputer_TagInfoMVACategorySelector_h
#define RecoBTau_JetTagComputer_TagInfoMVACategorySelector_h

#include <string>
#include <vector>

#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"

class TagInfoMVACategorySelector {
public:
  TagInfoMVACategorySelector(const edm::ParameterSet &params);
  ~TagInfoMVACategorySelector();

  inline const std::vector<std::string> &getCategoryLabels() const { return categoryLabels; }

  int findCategory(const reco::TaggingVariableList &taggingVariables) const;

  static void fillPSetDescription(edm::ParameterSetDescription &desc);

private:
  reco::TaggingVariableName categoryVariable;
  std::vector<std::string> categoryLabels;
};

#endif  // RecoBTau_JetTagComputer_TagInfoMVACategorySelector_h
