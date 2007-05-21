#ifndef PhysicsTools_MVATrainer_MVATrainerSaveImpl_h
#define PhysicsTools_MVATrainer_MVATrainerSaveImpl_h

#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

#include "PhysicsTools/MVATrainer/interface/MVATrainerSave.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerContainerSave.h"

namespace PhysicsTools {

template<typename Record_t>
class MVATrainerSaveImpl : public MVATrainerSave {
    public:
	explicit MVATrainerSaveImpl(const edm::ParameterSet &params) :
		MVATrainerSave(params) {}

    protected:
	virtual const Calibration::MVAComputer *
	getToPut(const edm::EventSetup& es) const
	{
		edm::ESHandle<Calibration::MVAComputer> handle;
		es.get<Record_t>().get("trained", handle);
		return handle.product();
	}

	virtual std::string getRecordName() const
	{ return Record_t::keyForClass().type().name(); }
};

template<typename Record_t>
class MVATrainerContainerSaveImpl : public MVATrainerContainerSave {
    public:
	explicit MVATrainerContainerSaveImpl(const edm::ParameterSet &params) :
		MVATrainerContainerSave(params) {}

    protected:
	virtual const Calibration::MVAComputerContainer *
	getToPut(const edm::EventSetup& es) const
	{
		edm::ESHandle<Calibration::MVAComputerContainer> handle;
		es.get<Record_t>().get("trained", handle);
		return handle.product();
	}

	virtual const Calibration::MVAComputerContainer *
	getToCopy(const edm::EventSetup& es) const
	{
		edm::ESHandle<Calibration::MVAComputerContainer> handle;
		es.get<Record_t>().get(handle);
		return handle.product();
	}

	virtual std::string getRecordName() const
	{ return Record_t::keyForClass().type().name(); }
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVATrainer_MVATrainerSaveImpl_h
