#ifndef PhysicsTools_MVATrainer_MVATrainerContainerSave_h
#define PhysicsTools_MVATrainer_MVATrainerContainerSave_h

#include <memory>
#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

namespace PhysicsTools {

class MVATrainerContainerSave : public edm::EDAnalyzer {
    public:
	explicit MVATrainerContainerSave(const edm::ParameterSet &params);

	virtual void analyze(const edm::Event& iEvent,
	                     const edm::EventSetup& iSetup);

	virtual void endJob();

    protected:
	virtual const Calibration::MVAComputerContainer *
	getToPut(const edm::EventSetup& es) const = 0;

	virtual const Calibration::MVAComputerContainer *
	getToCopy(const edm::EventSetup& es) const = 0;

	virtual std::string getRecordName() const = 0;

    private:
	std::vector<std::string>				toPut;
	std::vector<std::string>				toCopy;
	std::auto_ptr<Calibration::MVAComputerContainer>	calib;
	bool							saved;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVATrainer_MVATrainerContainerSave_h
