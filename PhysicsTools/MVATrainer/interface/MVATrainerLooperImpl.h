#ifndef PhysicsTools_MVATrainer_MVATrainerLooperImpl_h
#define PhysicsTools_MVATrainer_MVATrainerLooperImpl_h

#include <string>

#include <boost/shared_ptr.hpp>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/MVATrainer/interface/MVATrainerLooper.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerContainer.h"

namespace PhysicsTools {

template<class Record_t>
class MVATrainerLooperImpl : public MVATrainerLooper {
    public:
	MVATrainerLooperImpl(const edm::ParameterSet &params) :
		MVATrainerLooperImpl(params)
	{ setWhatProduced(this); }

	virtual ~MVATrainerLooperImpl() {}

	TrainObject produce(const Record_t &record)
	{ return getCalibration(); }
};

template<class Record_t>
class MVATrainerContainerLooperImpl : public MVATrainerLooper {
    public:
	MVATrainerContainerLooperImpl(const edm::ParameterSet &params) :
		MVATrainerLooper(params),
		calibrationRecord(params.getParameter<std::string>(
							"calibrationRecord"))
	{ setWhatProduced(this); }

	virtual ~MVATrainerContainerLooperImpl() {}

	boost::shared_ptr<Calibration::MVAComputerContainer>
	produce(const Record_t &record)
	{
		boost::shared_ptr<MVATrainerContainer> container(
						new MVATrainerContainer());
		container->addTrainer(calibrationRecord, getCalibration());
		return container;
	}

    private:
	std::string	calibrationRecord;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVATrainer_MVATrainerLooperImpl_h
