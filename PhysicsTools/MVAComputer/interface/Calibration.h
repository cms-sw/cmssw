#ifndef PhysicsTools_MVAComputer_Calibration_h
#define PhysicsTools_MVAComputer_Calibration_h

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

#include "PhysicsTools/MVAComputer/interface/Variable.h"

namespace PhysicsTools {

/// for internal use by MVATrainer
class TrainMVAComputerCalibration : public Calibration::MVAComputer {
    public:
	virtual void initFlags(std::vector<Variable::Flags> &flags) const = 0;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVAComputer_Calibration_h
