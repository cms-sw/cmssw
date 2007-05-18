#ifndef PhysicsTools_MVATrainer_MVATrainer_h
#define PhysicsTools_MVATrainer_MVATrainer_h

#include <sstream>
#include <memory>
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

namespace PhysicsTools {

class Source;
class Processor;

class MVATrainer {
    public:
	MVATrainer(const std::string &fileName);
	~MVATrainer();

	Calibration::MVAComputer *getTrainCalibration() const;
	void doneTraining(Calibration::MVAComputer *trainCalibration) const;

	Calibration::MVAComputer *getCalibration() const;

	std::string trainFileName(const Processor *proc,
	                          const std::string &ext,
	                          const std::string &arg = "") const;

	static const AtomicId kTargetId;

    private:
	SourceVariable *getVariable(AtomicId source, AtomicId name) const;

	SourceVariable *createVariable(Source *source, AtomicId name,
	                               Variable::Flags flags);

	typedef std::pair<Processor*,
	                  Calibration::VarProcessor*> CalibratedProcessor;

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

	std::map<AtomicId, Source*>	sources;
	std::vector<SourceVariable*>	variables;
	std::vector<AtomicId>		processors;
	Source				*input;
	Source				*output;

	std::auto_ptr<XMLDocument>	xml;
	std::string			trainFileMask;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVATrainer_MVATrainer_h
