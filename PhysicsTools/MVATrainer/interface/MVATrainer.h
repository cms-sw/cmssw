#ifndef PhysicsTools_MVATrainer_MVATrainer_h
#define PhysicsTools_MVATrainer_MVATrainer_h

#include <memory>
#include <string>
#include <map>

#include <xercesc/dom/DOM.hpp>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/Variable.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"

#include "PhysicsTools/MVATrainer/interface/XMLDocument.h"
#include "PhysicsTools/MVATrainer/interface/SourceVariable.h"
#include "PhysicsTools/MVATrainer/interface/SourceVariableSet.h"
#include "PhysicsTools/MVATrainer/interface/TrainerMonitoring.h"

namespace PhysicsTools {

class Source;
class TrainProcessor;

class MVATrainer {
    public:
	MVATrainer(const std::string &fileName, bool useXSLT = false,
	           const char *styleSheet = 0);
	~MVATrainer();

	inline void setAutoSave(bool autoSave) { doAutoSave = autoSave; }
	inline void setCleanup(bool cleanup) { doCleanup = cleanup; }
	inline void setMonitoring(bool monitoring) { doMonitoring = monitoring; }
	inline void setRandomSeed(UInt_t seed) { randomSeed = seed; }
	inline void setCrossValidation(double split) { crossValidation = split; }

	void loadState();
	void saveState();

	Calibration::MVAComputer *getTrainCalibration() const;
	void doneTraining(Calibration::MVAComputer *trainCalibration) const;

	Calibration::MVAComputer *getCalibration() const;

	// used by TrainProcessors

	std::string trainFileName(const TrainProcessor *proc,
	                          const std::string &ext,
	                          const std::string &arg = "") const;

	inline const std::string &getName() const { return name; }

	TrainerMonitoring::Module *bookMonitor(const std::string &name);

	// constants

	static const AtomicId kTargetId;
	static const AtomicId kWeightId;

    private:
	SourceVariable *getVariable(AtomicId source, AtomicId name) const;

	SourceVariable *createVariable(Source *source, AtomicId name,
	                               Variable::Flags flags);

	struct CalibratedProcessor {
		CalibratedProcessor(TrainProcessor *processor,
		                    Calibration::VarProcessor *calib) :
			processor(processor), calib(calib) {}

		TrainProcessor			*processor;
		Calibration::VarProcessor	*calib;
	};

	void fillInputVars(SourceVariableSet &vars,
	                   XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *xml);

	void fillOutputVars(SourceVariableSet &vars, Source *source,
	                    XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *xml);

	void makeProcessor(XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *elem,
	                   AtomicId id, const char *name);

	void connectProcessors(Calibration::MVAComputer *calib,
	                       const std::vector<CalibratedProcessor> &procs,
	                       bool withTarget) const;

	Calibration::MVAComputer *
	makeTrainCalibration(const AtomicId *compute,
	                     const AtomicId *train) const;

	void
	findUntrainedComputers(std::vector<AtomicId> &compute,
	                       std::vector<AtomicId> &train) const;

	std::vector<AtomicId> findFinalProcessors() const;

	std::map<AtomicId, Source*>		sources;
	std::vector<SourceVariable*>		variables;
	std::vector<AtomicId>			processors;
	Source					*input;
	TrainProcessor				*output;

	std::auto_ptr<TrainerMonitoring>	monitoring;
	std::auto_ptr<XMLDocument>		xml;
	std::string				trainFileMask;
	std::string				name;
	bool					doAutoSave;
	bool					doCleanup;
	bool					doMonitoring;

	UInt_t					randomSeed;
	double					crossValidation;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVATrainer_MVATrainer_h
