#ifndef MVATrainer_h
#define MVATrainer_h

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

class Source;
class Processor;

class MVATrainer {
    public:
	MVATrainer(const std::string &fileName);
	~MVATrainer();

	PhysicsTools::Calibration::MVAComputer *getTrainCalibration() const;
	PhysicsTools::Calibration::MVAComputer *getCalibration() const;

	static const PhysicsTools::AtomicId kTargetId;

	std::string trainFileName(const Processor *proc,
	                          const std::string &ext,
	                          const std::string &arg = "") const;

    private:
	SourceVariable *getVariable(PhysicsTools::AtomicId source,
	                            PhysicsTools::AtomicId name) const;

	SourceVariable *createVariable(Source *source,
	                               PhysicsTools::AtomicId name,
	                               PhysicsTools::Variable::Flags flags);

	typedef std::pair<Processor*,
		PhysicsTools::Calibration::VarProcessor*> CalibratedProcessor;

	void fillInputVars(SourceVariableSet &vars,
	                   XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *xml);

	void fillOutputVars(SourceVariableSet &vars, Source *source,
	                    XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *xml);

	void makeProcessor(XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *elem,
	                   PhysicsTools::AtomicId id, const char *name);

	void connectProcessors(PhysicsTools::Calibration::MVAComputer *calib,
	                       const std::vector<CalibratedProcessor> &procs,
	                       bool withTarget) const;

	PhysicsTools::Calibration::MVAComputer *
	makeTrainCalibration(const PhysicsTools::AtomicId *compute,
	                     const PhysicsTools::AtomicId *train) const;

	void
	findUntrainedComputers(std::vector<PhysicsTools::AtomicId> &compute,
	                       std::vector<PhysicsTools::AtomicId> &train) const;

	std::map<PhysicsTools::AtomicId, Source*>	sources;
	std::vector<SourceVariable*>			variables;
	std::vector<PhysicsTools::AtomicId>		processors;
	Source						*input;
	Source						*output;

	std::auto_ptr<XMLDocument>			xml;
	std::string					trainFileMask;
};

#endif // MVATrainer_h
