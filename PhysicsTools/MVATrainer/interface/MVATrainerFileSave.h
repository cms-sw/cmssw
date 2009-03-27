#ifndef PhysicsTools_MVATrainer_MVATrainerFileSave_h
#define PhysicsTools_MVATrainer_MVATrainerFileSave_h

#include <memory>
#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

namespace PhysicsTools {

class MVATrainerFileSave : public edm::EDAnalyzer {
    public:
	explicit MVATrainerFileSave(const edm::ParameterSet &params);

	virtual void analyze(const edm::Event& iEvent,
	                     const edm::EventSetup& iSetup);

	virtual void endJob();

    protected:
	virtual const Calibration::MVAComputerContainer *
	getToPut(const edm::EventSetup& es) const = 0;

	bool							trained;

    private:
	typedef std::map<std::string, std::string> LabelFileMap;

	LabelFileMap						toPut;
	std::auto_ptr<Calibration::MVAComputerContainer>	calib;
	bool							saved;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVATrainer_MVATrainerFileSave_h
