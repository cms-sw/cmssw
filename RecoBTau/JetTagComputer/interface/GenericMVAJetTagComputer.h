#ifndef RecoBTau_JetTagComputer_GenericMVAJetTagComputer_h
#define RecoBTau_JetTagComputer_GenericMVAJetTagComputer_h

#include <memory>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAComputerCache.h"
#include "RecoBTau/JetTagComputer/interface/TagInfoMVACategorySelector.h"

class JetTagComputerRecord;

class GenericMVAJetTagComputer : public JetTagComputer {
    public:
	GenericMVAJetTagComputer(const edm::ParameterSet &parameters);
	virtual ~GenericMVAJetTagComputer();

	virtual void initialize(const JetTagComputerRecord &);

	virtual float discriminator(const TagInfoHelper &info) const;

	virtual reco::TaggingVariableList
	taggingVariables(const reco::BaseTagInfo &tagInfo) const;
	virtual reco::TaggingVariableList
	taggingVariables(const TagInfoHelper &info) const;

    private:
	std::auto_ptr<TagInfoMVACategorySelector> categorySelector_;
	GenericMVAComputerCache computerCache_;
        std::string recordLabel_;
};

#endif // RecoBTau_JetTagComputer_GenericMVAJetTagComputer_h
