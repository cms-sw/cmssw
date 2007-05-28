#ifndef RecoBTau_JetTagMVALearning_JetTagMVATrainer_h
#define RecoBTau_JetTagMVALearning_JetTagMVATrainer_h

#include <string>
#include <vector>
#include <memory>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

#include "RecoBTau/JetTagComputer/interface/GenericMVAComputer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAComputerCache.h"
#include "RecoBTau/JetTagComputer/interface/TagInfoMVACategorySelector.h"

class JetTagMVATrainer : public edm::EDAnalyzer {
    public:
	explicit JetTagMVATrainer(const edm::ParameterSet &params);
	~JetTagMVATrainer();

	virtual void analyze(const edm::Event &event,
	                     const edm::EventSetup &es);

    protected:
	bool isSignalFlavour(int flavour) const;
	bool isIgnoreFlavour(int flavour) const;

	edm::InputTag					jetFlavour;
	edm::InputTag					tagInfo;
	std::auto_ptr<TagInfoMVACategorySelector>	categorySelector;
	std::auto_ptr<GenericMVAComputerCache>		computerCache;

	double						minPt;
	double						minEta;
	double						maxEta;

    private:
	std::vector<int>				signalFlavours;
	std::vector<int>				ignoreFlavours;
};

#endif // RecoBTau_JetTagMVALearning_JetTagMVATrainer_h
