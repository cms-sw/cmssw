#ifndef RecoBTau_JetTagComputer_GenericMVAComputerCache_h
#define RecoBTau_JetTagComputer_GenericMVAComputerCache_h

#include <string>
#include <vector>
#include <memory>

#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAComputer.h"

class GenericMVAComputerCache {
    public:
	GenericMVAComputerCache(const std::vector<std::string> &labels);
	~GenericMVAComputerCache();

	bool
	update(const PhysicsTools::Calibration::MVAComputerContainer *calib);

	GenericMVAComputer const* getComputer(int index) const;

	bool isEmpty() const;

    private:
	struct IndividualComputer {
		IndividualComputer();
		IndividualComputer(const IndividualComputer &orig);
		~IndividualComputer();

		std::string						label;
		std::unique_ptr<GenericMVAComputer>			computer;
		PhysicsTools::Calibration::MVAComputer::CacheId		cacheId;
	};

	std::vector<IndividualComputer>					computers;
	PhysicsTools::Calibration::MVAComputerContainer::CacheId	cacheId;
	bool								initialized;
	bool								empty;
        std::string errorUpdatingLabel;
};

#endif // RecoBTau_JetTagComputer_GenericMVAComputerCache_h
