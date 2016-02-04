#ifndef PhysicsTools_MVATrainer_MVATrainerContainer_h
#define PhysicsTools_MVATrainer_MVATrainerContainer_h

#include <vector>
#include <string>
#include <memory>

#include <boost/shared_ptr.hpp>

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerLooper.h"

namespace PhysicsTools {

class MVATrainerContainer : public Calibration::MVAComputerContainer {
    public:
	typedef MVATrainerLooper::TrainObject	Value_t;

	virtual const Calibration::MVAComputer &
	find(const std::string &label) const
	{
		Map_t::const_iterator pos = trainCalibs.find(label);
		if (pos != trainCalibs.end())
			return *pos->second.get();

		return Calibration::MVAComputerContainer::find(label);
	}

	void addTrainer(const std::string &label, const Value_t &calibration)
	{ trainCalibs[label] = calibration; }

    private:
	typedef std::map<std::string, Value_t>	Map_t;

	Map_t	trainCalibs;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVATrainer_MVATrainerContainer_h
