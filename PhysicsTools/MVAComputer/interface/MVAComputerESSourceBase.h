#ifndef PhysicsTools_MVAComputer_MVAComputerESSourceBase_h
#define PhysicsTools_MVAComputer_MVAComputerESSourceBase_h

#include <string>
#include <map>

#include <boost/shared_ptr.hpp>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"

namespace PhysicsTools {

class MVAComputerESSourceBase : public edm::ESProducer {
    public:
	typedef boost::shared_ptr<Calibration::MVAComputerContainer> ReturnType;

	MVAComputerESSourceBase(const edm::ParameterSet &params);
	virtual ~MVAComputerESSourceBase();

    protected:
	ReturnType produce() const;

	typedef std::map<std::string, std::string> LabelFileMap;

	LabelFileMap	mvaCalibrations;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVAComputer_MVAComputerESSourceBase_h
