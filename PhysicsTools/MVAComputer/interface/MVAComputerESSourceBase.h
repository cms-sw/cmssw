#ifndef PhysicsTools_MVAComputer_MVAComputerESSourceBase_h
#define PhysicsTools_MVAComputer_MVAComputerESSourceBase_h

#include <string>
#include <map>

#include <memory>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"

namespace PhysicsTools {

class MVAComputerESSourceBase : public edm::ESProducer {
    public:
	using ReturnType = std::unique_ptr<Calibration::MVAComputerContainer>;

	MVAComputerESSourceBase(const edm::ParameterSet &params);
	~MVAComputerESSourceBase() override;

    protected:
	ReturnType produce() const;

	typedef std::map<std::string, std::string> LabelFileMap;

	LabelFileMap	mvaCalibrations;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVAComputer_MVAComputerESSourceBase_h
