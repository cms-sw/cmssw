#ifndef PhysicsTools_MVATrainer_MVATrainerLooperImpl_h
#define PhysicsTools_MVATrainer_MVATrainerLooperImpl_h

#include <string>
#include <memory>

#include <boost/shared_ptr.hpp>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "PhysicsTools/MVATrainer/interface/MVATrainerLooper.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerContainer.h"

namespace PhysicsTools {

template<class Record_t>
class MVATrainerLooperImpl : public MVATrainerLooper {
    public:
	MVATrainerLooperImpl(const edm::ParameterSet &params) :
		MVATrainerLooper(params)
	{
		setWhatProduced(this, "trainer");
		addTrainer(new Trainer(params));
	}

	virtual ~MVATrainerLooperImpl() {}

	boost::shared_ptr<Calibration::MVAComputer>
	produce(const Record_t &record)
	{ return (*getTrainers().begin())->getCalibration(); }
};

template<class Record_t>
class MVATrainerContainerLooperImpl : public MVATrainerLooper {
    public:
	enum { kTrainer, kTrained };

	MVATrainerContainerLooperImpl(const edm::ParameterSet &params) :
		MVATrainerLooper(params)
	{
		setWhatProduced(this, edm::es::label("trainer", kTrainer)
		                                    ("trained", kTrained));

		std::vector<edm::ParameterSet> trainers =
			params.getParameter<std::vector<edm::ParameterSet> >(
								"trainers");

		for(std::vector<edm::ParameterSet>::const_iterator iter =
			trainers.begin(); iter != trainers.end(); iter++)

			addTrainer(new Trainer(*iter));
	}

	virtual ~MVATrainerContainerLooperImpl() {}

	edm::ESProducts<
		edm::es::L<Calibration::MVAComputerContainer, kTrainer>,
		edm::es::L<Calibration::MVAComputerContainer, kTrained> >
	produce(const Record_t &record)
	{
		boost::shared_ptr<MVATrainerContainer> trainerCalib(
						new MVATrainerContainer());
		TrainContainer trainedCalib;

		bool untrained = false;
		for(TrainerContainer::const_iterator iter =
							getTrainers().begin();
		    iter != getTrainers().end(); iter++) {
			Trainer *trainer = dynamic_cast<Trainer*>(*iter);
			TrainObject calib = trainer->getCalibration();

			trainerCalib->addTrainer(trainer->calibrationRecord,
			                         calib);
			if (calib) {
				untrained = true;
				continue;
			}

			if (!trainedCalib)
				trainedCalib = TrainContainer(
					new Calibration::MVAComputerContainer);

			trainedCalib->add(trainer->calibrationRecord) =
				*trainer->getTrainer()->getCalibration();
		}

		if (untrained)
			trainedCalib = TrainContainer(
					new UntrainedMVAComputerContainer);

		edm::es::L<Calibration::MVAComputerContainer, kTrainer>
						trainedESLabel(trainerCalib);

		return edm::es::products(trainedESLabel,
		                         edm::es::l<kTrained>(trainedCalib));
	}

    protected:
	class Trainer : public MVATrainerLooper::Trainer {
	    public:
		Trainer(const edm::ParameterSet &params) :
			MVATrainerLooper::Trainer(params),
			calibrationRecord(params.getParameter<std::string>(
						"calibrationRecord")) {}

		std::string calibrationRecord;
	};
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVATrainer_MVATrainerLooperImpl_h
