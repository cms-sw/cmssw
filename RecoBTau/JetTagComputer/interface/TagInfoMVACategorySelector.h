#ifndef RecoBTau_JetTagComputer_TagInfoMVACategorySelector_h
#define RecoBTau_JetTagComputer_TagInfoMVACategorySelector_h

#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"

class TagInfoMVACategorySelector {
    public:
	TagInfoMVACategorySelector(const edm::ParameterSet &params);
	~TagInfoMVACategorySelector();

	inline const std::vector<std::string> &getCategoryLabels() const
	{ return categoryLabels; }

	int
	findCategory(const reco::TaggingVariableList &taggingVariables) const;

    private:
	reco::TaggingVariableName	categoryVariable;
	std::vector<std::string>	categoryLabels;
};

#endif // RecoBTau_JetTagComputer_TagInfoMVACategorySelector_h
