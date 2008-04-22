#ifndef RecoBTau_JetTagComputer_CombinedMVAJetTagComputer_h
#define RecoBTau_JetTagComputer_CombinedMVAJetTagComputer_h

#include <string>
#include <memory>
#include <vector>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAJetTagComputer.h"

class CombinedMVAJetTagComputer : public GenericMVAJetTagComputer {
    public:
	CombinedMVAJetTagComputer(const edm::ParameterSet &parameters);
	virtual ~CombinedMVAJetTagComputer();

	virtual void setEventSetup(const edm::EventSetup &es) const;

	virtual reco::TaggingVariableList
	taggingVariables(const TagInfoHelper &info) const;

    private:
	struct Computer {
		std::string		name;
		bool			discriminator;
		bool			variables;

		const JetTagComputer	*computer;
		std::vector<int>	indices;
	};

	mutable std::vector<Computer>	computers;
};

#endif // RecoBTau_JetTagComputer_CombinedMVAJetTagComputer_h
