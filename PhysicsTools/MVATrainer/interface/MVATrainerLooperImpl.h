#ifndef PhysicsTools_MVATrainer_MVATrainerLooperImpl_h
#define PhysicsTools_MVATrainer_MVATrainerLooperImpl_h

#include <string>
#include <memory>

#include <boost/shared_ptr.hpp>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

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

    protected:
	virtual void
	storeCalibration(std::auto_ptr<Calibration::MVAComputer> calib) const
	{
		edm::Service<cond::service::PoolDBOutputService> dbService;
		if (!dbService.isAvailable())
			throw cms::Exception("MVATrainerLooper")
				<< "No PoolDBOutputService available!"
				<< std::endl;

		dbService->createNewIOV<Calibration::MVAComputerContainer>(
			calib.release(), dbService->endOfTime(),
			"BTagCombinedSVDiscriminatorComputerRcd");
	}
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

	virtual Status duringLoop(const edm::Event &event,
	                         const edm::EventSetup &es)
	{
#if 0
		if (!dbService.get()) {
			dbService = std::auto_ptr<DBService>(new DBService);

			if (!dbService->isAvailable())
				throw cms::Exception("MVATrainerLooper")
					<< "No PoolDBOutputService available!"
					<< std::endl;
		}
#endif

		return MVATrainerLooper::duringLoop(event, es);
	}

    protected:
	virtual void
	storeCalibration(std::auto_ptr<Calibration::MVAComputer> calib) const
	{
#if 0
		Calibration::MVAComputerContainer *container =
					new Calibration::MVAComputerContainer;
		container->add(calibrationRecord) = *calib;

		(*dbService)->createNewIOV<Calibration::MVAComputerContainer>(
			container, (*dbService)->endOfTime(),
			Record_t::keyForClass().type().name());
#endif	
	}

    private:
	typedef edm::Service<cond::service::PoolDBOutputService> DBService;

	std::string			calibrationRecord;
	std::auto_ptr<DBService>	dbService;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVATrainer_MVATrainerLooperImpl_h
