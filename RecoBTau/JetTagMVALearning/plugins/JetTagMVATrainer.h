#ifndef RecoBTau_JetTagMVALearning_JetTagMVATrainer_h
#define RecoBTau_JetTagMVALearning_JetTagMVATrainer_h

#include <string>
#include <memory>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

#include "DataFormats/BTauReco/interface/TaggingVariable.h"

#include "RecoBTau/JetTagComputer/interface/GenericMVAComputer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAComputerCache.h"
#include "RecoBTau/JetTagComputer/interface/TagInfoMVACategorySelector.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAJetTagComputer.h"

class JetTagMVATrainer : public edm::EDAnalyzer {
    public:
	explicit JetTagMVATrainer(const edm::ParameterSet &params);
	~JetTagMVATrainer();

	virtual void analyze(const edm::Event &event,
	                     const edm::EventSetup &es);

    protected:
	struct JetInfo;

	static bool isFlavour(const JetInfo &info,
	                      const std::vector<int> &list);
	bool isSignalFlavour(const JetInfo &jetInfo) const;
	bool isIgnoreFlavour(const JetInfo &jetInfo) const;

	edm::InputTag					jetFlavour;
	std::auto_ptr<TagInfoMVACategorySelector>	categorySelector;
	std::auto_ptr<GenericMVAComputerCache>		computerCache;

	double						minPt;
	double						minEta;
	double						maxEta;

    private:
	void setup(const JetTagComputer &computer);

	bool						setupDone;
	std::string					jetTagComputer;

	std::vector<edm::InputTag>			tagInfos;

	std::vector<int>				signalFlavours;
	std::vector<int>				ignoreFlavours;
};

#endif // RecoBTau_JetTagMVALearning_JetTagMVATrainer_h
