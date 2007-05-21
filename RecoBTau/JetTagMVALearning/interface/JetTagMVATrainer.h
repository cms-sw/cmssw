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

#include "RecoBTau/JetTagComputer/interface/GenericMVAComputer.h"
#include "RecoBTag/MCTools/interface/JetFlavourIdentifier.h"

class JetTagMVATrainer : public edm::EDAnalyzer {
    public:
	explicit JetTagMVATrainer(const edm::ParameterSet &params);
	~JetTagMVATrainer();

	virtual void analyze(const edm::Event &event,
	                     const edm::EventSetup &es);

    protected:
	bool updateComputer(const edm::EventSetup &es);

	bool isSignalFlavour(int flavour) const;
	bool isIgnoreFlavour(int flavour) const;

	JetFlavourIdentifier			jetId;
	edm::InputTag				tagInfo;
	std::string				calibrationLabel;
	std::auto_ptr<GenericMVAComputer>	mvaComputer;

	double					minPt;
	double					minEta;
	double					maxEta;

    private:
	std::vector<int>			signalFlavours;
	std::vector<int>			ignoreFlavours;

	PhysicsTools::Calibration::MVAComputer::CacheId			computerCacheId;
	PhysicsTools::Calibration::MVAComputerContainer::CacheId	containerCacheId;
};

#endif // RecoBTau_JetTagMVALearning_JetTagMVATrainer_h
