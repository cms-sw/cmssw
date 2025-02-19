#ifndef PhysicsTools_MVATrainer_MVATrainerLooper_h
#define PhysicsTools_MVATrainer_MVATrainerLooper_h

#include <string>
#include <vector>
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
	virtual ~MVATrainerLooper();

	virtual void startingNewLoop(unsigned int iteration);
	virtual Status duringLoop(const edm::Event &ev,
	                          const edm::EventSetup &es);
	virtual Status endOfLoop(const edm::EventSetup &es,
	                         unsigned int iteration);

	typedef boost::shared_ptr<Calibration::MVAComputer> TrainObject;
	typedef boost::shared_ptr<Calibration::MVAComputerContainer>
							TrainContainer;

	template<class T>
	static inline bool isUntrained(const T *ptr);

    protected:
	class Trainer {
	    public:
		Trainer(const edm::ParameterSet &params);
		virtual ~Trainer() {}

		inline const MVATrainer *getTrainer() const
		{ return trainer.get(); }
		inline const TrainObject getCalibration() const
		{ return trainCalib; }

	    private:
		friend class MVATrainerLooper;

		std::auto_ptr<MVATrainer>	trainer;
		TrainObject			trainCalib;
	};

	class TrainerContainer {
	    public:
		~TrainerContainer();
		void clear();

		typedef std::vector<Trainer*>::const_iterator const_iterator;

		inline const_iterator begin() const { return content.begin(); }
		inline const_iterator end() const { return content.end(); }
		inline bool empty() const { return content.empty(); }

		inline void add(Trainer *trainer)
		{ content.push_back(trainer); }

	    private:
		std::vector<Trainer*> content;
	};

	class UntrainedMVAComputer : public Calibration::MVAComputer {};
	class UntrainedMVAComputerContainer :
				public Calibration::MVAComputerContainer {};

	void addTrainer(Trainer *trainer) { trainers.add(trainer); }

	inline const TrainerContainer &getTrainers() const { return trainers; }

    private:
	TrainerContainer	trainers;
        bool dataProcessedInLoop;
};

template<> inline bool
MVATrainerLooper::isUntrained(const Calibration::MVAComputer *ptr)
{ return dynamic_cast<const UntrainedMVAComputer*>(ptr) != 0; }

template<> inline bool
MVATrainerLooper::isUntrained(const Calibration::MVAComputerContainer *ptr)
{ return dynamic_cast<const UntrainedMVAComputerContainer*>(ptr) != 0; }

} // namespace PhysicsTools

#endif // PhysicsTools_MVATrainer_MVATrainerLooper_h
