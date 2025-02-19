#ifndef PhysicsTools_MVATrainer_TrainProcessor_h
#define PhysicsTools_MVATrainer_TrainProcessor_h

#include <vector>
#include <string>

#include <boost/version.hpp>
#include <boost/filesystem.hpp>

#include <xercesc/dom/DOM.hpp>

#include "FWCore/PluginManager/interface/PluginFactory.h"

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"
#include "PhysicsTools/MVAComputer/interface/Variable.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/ProcessRegistry.h"

#include "PhysicsTools/MVATrainer/interface/Source.h"
#include "PhysicsTools/MVATrainer/interface/TrainerMonitoring.h"

class TH1F;

namespace PhysicsTools {

class MVATrainer;

class TrainProcessor : public Source,
	public ProcessRegistry<TrainProcessor, AtomicId, MVATrainer>::Factory {
    public:
	template<typename Instance_t>
	struct Registry {
		typedef typename ProcessRegistry<
			TrainProcessor,
			AtomicId,
			MVATrainer
		>::Registry<Instance_t, AtomicId> Type;
	};

	typedef TrainerMonitoring::Module Monitoring;

	TrainProcessor(const char *name,
	               const AtomicId *id,
	               MVATrainer *trainer);
	virtual ~TrainProcessor();

	virtual Variable::Flags getDefaultFlags() const
	{ return Variable::FLAG_ALL; }

	virtual void
	configure(XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *config) {}

	virtual void
	passFlags(const std::vector<Variable::Flags> &flags) {}

	virtual Calibration::VarProcessor *getCalibration() const { return 0; }

	void doTrainBegin();
	void doTrainData(const std::vector<double> *values,
	                 bool target, double weight, bool train, bool test);
	void doTrainEnd();

	virtual bool load() { return true; }
	virtual void save() {}
	virtual void cleanup() {}

	inline const char *getId() const { return name.c_str(); }

	struct Dummy {};
	typedef edmplugin::PluginFactory<Dummy*()> PluginFactory;

    protected:
	virtual void trainBegin() {}
	virtual void trainData(const std::vector<double> *values,
	                       bool target, double weight) {}
	virtual void testData(const std::vector<double> *values,
	                      bool target, double weight, bool trainedOn) {}
	virtual void trainEnd() { trained = true; }

	virtual void *requestObject(const std::string &name) const
	{ return 0; }

	inline bool exists(const std::string &name)
	{ return boost::filesystem::exists(name.c_str()); }

	std::string		name;
	MVATrainer		*trainer;
	Monitoring		*monitoring;

    private:
	struct SigBkg {
		bool		sameBinning;
		double		min;
		double		max;
		unsigned long	entries[2];
		double		underflow[2];
		double		overflow[2];
		TH1F		*histo[2];
	};
		
	std::vector<SigBkg>	monHistos;
	Monitoring		*monModule;
};

template<>
TrainProcessor *ProcessRegistry<TrainProcessor, AtomicId,
                                MVATrainer>::Factory::create(
			const char*, const AtomicId*, MVATrainer*);

} // namespace PhysicsTools

#define MVA_TRAINER_DEFINE_PLUGIN(T) \
	DEFINE_EDM_PLUGIN(::PhysicsTools::TrainProcessor::PluginFactory, \
	                  ::PhysicsTools::TrainProcessor::Dummy, \
	                  "TrainProcessor/" #T)

#endif // PhysicsTools_MVATrainer_TrainProcessor_h
