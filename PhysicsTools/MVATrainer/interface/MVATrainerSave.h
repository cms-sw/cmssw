#ifndef PhysicsTools_MVATrainer_MVATrainerSave_h
#define PhysicsTools_MVATrainer_MVATrainerSave_h

#include <memory>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

namespace PhysicsTools {

class MVATrainerSave : public edm::EDAnalyzer {
    public:
	explicit MVATrainerSave(const edm::ParameterSet &params);

	virtual void analyze(const edm::Event& iEvent,
	                     const edm::EventSetup& iSetup);

	virtual void endJob();

    protected:
	virtual const Calibration::MVAComputer *
	getToPut(const edm::EventSetup& es) const = 0;

	virtual std::string getRecordName() const = 0;

    private:
	std::auto_ptr<Calibration::MVAComputer>	calib;
	bool					saved;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVATrainer_MVATrainerSave_h
