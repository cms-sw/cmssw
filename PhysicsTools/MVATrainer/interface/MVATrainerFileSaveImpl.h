#ifndef PhysicsTools_MVATrainer_MVATrainerFileSaveImpl_h
#define PhysicsTools_MVATrainer_MVATrainerFileSaveImpl_h

#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

#include "PhysicsTools/MVATrainer/interface/MVATrainerFileSave.h"

namespace PhysicsTools {

template<typename Record_t>
class MVATrainerFileSaveImpl : public MVATrainerFileSave {
    public:
	explicit MVATrainerFileSaveImpl(const edm::ParameterSet &params) :
		MVATrainerFileSave(params) {}

    protected:
	virtual const Calibration::MVAComputerContainer *
	getToPut(const edm::EventSetup& es) const
	{
		edm::ESHandle<Calibration::MVAComputerContainer> handle;
		if (trained)
			es.get<Record_t>().get("trained", handle);
		else
			es.get<Record_t>().get(handle);
		return handle.product();
	}
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVATrainer_MVATrainerFileSaveImpl_h
