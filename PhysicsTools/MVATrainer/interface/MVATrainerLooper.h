#ifndef PhysicsTools_MVATrainer_MVATrainerLooper_h
#define PhysicsTools_MVATrainer_MVATrainerLooper_h

#include <memory>

#include <boost/shared_ptr.hpp>

#include "FWCore/Framework/interface/ESProducerLooper.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"

namespace PhysicsTools {

class MVATrainerLooper : public edm::ESProducerLooper {
    public:
	MVATrainerLooper(const edm::ParameterSet &params);
	virtual ~MVATrainerLooper() {}

	virtual void startingNewLoop(unsigned int iteration);
	virtual Status duringLoop(const edm::Event &ev,
	                          const edm::EventSetup &es);
	virtual Status endOfLoop(const edm::EventSetup &es,
	                         unsigned int iteration);

	typedef boost::shared_ptr<Calibration::MVAComputer> TrainObject;

    protected:
	virtual void
	storeCalibration(std::auto_ptr<Calibration::MVAComputer> calib) const = 0;

	inline const MVATrainer *getTrainer() const { return trainer.get(); }
	inline TrainObject getCalibration() const { return trainCalib; }

    private:
	void updateTrainer();

	std::auto_ptr<MVATrainer>	trainer;
	TrainObject			trainCalib;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVATrainer_MVATrainerLooper_h
