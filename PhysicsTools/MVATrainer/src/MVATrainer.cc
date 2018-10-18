#include <cassert>
#include <functional>
#include <ext/functional>
#include <algorithm>
#include <iostream>
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>

#include <xercesc/dom/DOM.hpp>

#include <TRandom.h>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"
#include "PhysicsTools/MVAComputer/interface/BitSet.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/Variable.h"

#include "PhysicsTools/MVATrainer/interface/Interceptor.h"
#include "PhysicsTools/MVATrainer/interface/XMLDocument.h"
#include "PhysicsTools/MVATrainer/interface/XMLSimpleStr.h"
#include "PhysicsTools/MVATrainer/interface/XMLUniStr.h"
#include "PhysicsTools/MVATrainer/interface/Source.h"
#include "PhysicsTools/MVATrainer/interface/SourceVariable.h"
#include "PhysicsTools/MVATrainer/interface/TrainProcessor.h"
#include "PhysicsTools/MVATrainer/interface/TrainerMonitoring.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"

XERCES_CPP_NAMESPACE_USE

namespace PhysicsTools {

namespace { // anonymous
	class MVATrainerComputer;

	class BaseInterceptor : public Calibration::Interceptor {
	    public:
		BaseInterceptor() : calib(nullptr) {}
		~BaseInterceptor() override {}

		inline void setCalibration(MVATrainerComputer *calib)
		{ this->calib = calib; }

		std::vector<Variable::Flags>
		configure(const MVAComputer *computer, unsigned int n,
		          const std::vector<Variable::Flags> &flags) override = 0;

		double
		intercept(const std::vector<double> *values) const override = 0;

		virtual void init() {}
		virtual void finish(bool save) {}

	    protected:
		MVATrainerComputer	*calib;
	};

	class InitInterceptor : public BaseInterceptor {
	    public:
		InitInterceptor() {}
		~InitInterceptor() override {}

		std::vector<Variable::Flags>
		configure(const MVAComputer *computer, unsigned int n,
		          const std::vector<Variable::Flags> &flags) override;

		double
		intercept(const std::vector<double> *values) const override;
	};

	class TrainInterceptor : public BaseInterceptor {
	    public:
		TrainInterceptor(TrainProcessor *proc) : proc(proc) {}
		~TrainInterceptor() override {}

		inline TrainProcessor *getProcessor() const { return proc; }

		std::vector<Variable::Flags>
		configure(const MVAComputer *computer, unsigned int n,
		          const std::vector<Variable::Flags> &flags) override;

		double
		intercept(const std::vector<double> *values) const override;

		void init() override;
		void finish(bool save) override;

	    private:
		unsigned int					targetIdx;
		unsigned int					weightIdx;
		mutable	std::vector<std::vector<double>	>	tmp;
		TrainProcessor					*const proc;
	};

	class MVATrainerComputer : public TrainMVAComputerCalibration {
	    public:
		typedef std::pair<unsigned int, BaseInterceptor*> Interceptor;

		MVATrainerComputer(const std::vector<Interceptor>
							&interceptors,
		                   bool autoSave, UInt_t seed, double split);

		~MVATrainerComputer() override;

		std::vector<Calibration::VarProcessor*>
							getProcessors() const override;
		void initFlags(std::vector<Variable::Flags>
							&flags) const override;

		void configured(BaseInterceptor *interceptor) const;
		void next();
		void done();

		inline void addFlag(Variable::Flags flag)
		{ flags.push_back(flag); }

		inline bool useForTraining() const { return splitResult; }
		inline bool useForTesting() const
		{ return split <= 0.0 || !splitResult; }

		inline bool isConfigured() const
		{ return nConfigured == interceptors.size(); }

	    private:
		std::vector<Interceptor>	interceptors;
		std::vector<Variable::Flags>	flags;
		mutable unsigned int		nConfigured;
		bool				doAutoSave;
		TRandom				random;
		double				split;
		bool				splitResult;
	};

	// useful litte helpers

 	template<typename T>
	static inline void deleter(T *ptr) { delete ptr; }

	template<typename T>
	struct auto_cleaner {
		inline ~auto_cleaner()
		{ std::for_each(clean.begin(), clean.end(), deleter<T>); }

		inline void add(T *ptr) { clean.push_back(ptr); }
		std::vector<T*>	clean;
	};
} // anonymous namespace

static std::string stdStringVPrintf(const char *format, std::va_list va)
{
	unsigned int size = std::min<unsigned int>(128, std::strlen(format));
	char *buffer = new char[size];
	for(;;) {
		int n = std::vsnprintf(buffer, size, format, va);
		if (n >= 0 && (unsigned int)n < size)
			break;

		if (n >= 0)
			size = n + 1;
		else
			size *= 2;

		delete[] buffer;
		buffer = new char[size];
	}

	std::string result(buffer);
	delete[] buffer;
	return result;
}

static std::string stdStringPrintf(const char *format, ...)
{
	std::va_list va;
	va_start(va, format);
	std::string result = stdStringVPrintf(format, va);
	va_end(va);
	return result;
}

// implementation for InitInterceptor

std::vector<Variable::Flags>
InitInterceptor::configure(const MVAComputer *computer, unsigned int n,
                           const std::vector<Variable::Flags> &flags)
{
	calib->configured(this);
	return std::vector<Variable::Flags>(n, Variable::FLAG_ALL);
}

double
InitInterceptor::intercept(const std::vector<double> *values) const
{
	calib->next();
	return 0.0;
}

// implementation for TrainInterceptor

std::vector<Variable::Flags>
TrainInterceptor::configure(const MVAComputer *computer, unsigned int n,
                            const std::vector<Variable::Flags> &flags)
{
	const SourceVariableSet &inputSet = 
		const_cast<const TrainProcessor*>(proc)->getInputs();
	SourceVariable *target = inputSet.find(SourceVariableSet::kTarget);
	SourceVariable *weight = inputSet.find(SourceVariableSet::kWeight);

	std::vector<SourceVariable*> inputs = inputSet.get(true);

	std::vector<SourceVariable*>::const_iterator pos;
	pos = std::find(inputs.begin(), inputs.end(), target);
	assert(pos != inputs.end());
	targetIdx = pos - inputs.begin();
	pos = std::find(inputs.begin(), inputs.end(), weight);
	assert(pos != inputs.end());
	weightIdx = pos - inputs.begin();

	calib->configured(this);

	std::vector<Variable::Flags> result = flags;
	if (targetIdx < weightIdx) {
		result.erase(result.begin() + weightIdx);
		result.erase(result.begin() + targetIdx);
	} else {
		result.erase(result.begin() + targetIdx);
		result.erase(result.begin() + weightIdx);
	}

	proc->passFlags(result);

	result.clear();
	result.resize(n, proc->getDefaultFlags());
	result[targetIdx] = Variable::FLAG_NONE;
	result[weightIdx] = Variable::FLAG_OPTIONAL;

	if (targetIdx >= 2 || weightIdx >= 2)
		tmp.resize(n - 2);

	return result;
}

void TrainInterceptor::init()
{
	edm::LogInfo("MVATrainer")
		<< "TrainProcessor \"" << (const char*)proc->getName()
		<< "\" training iteration starting...";

	proc->doTrainBegin();
}

double
TrainInterceptor::intercept(const std::vector<double> *values) const
{
	if (values[targetIdx].size() != 1) {
		if (values[targetIdx].empty())
			throw cms::Exception("MVATrainer")
				<< "Trainer input lacks target variable."
				<< std::endl;
		else
			throw cms::Exception("MVATrainer")
				<< "Multiple targets supplied in input."
				<< std::endl;
	}
	double target = values[targetIdx].front();

	double weight = 1.0;
	if (values[weightIdx].size() > 1)
		throw cms::Exception("MVATrainer")
			<< "Multiple weights supplied in input."
			<< std::endl;
	else if (values[weightIdx].size() == 1)
		weight = values[weightIdx].front();

	if (tmp.empty())
		proc->doTrainData(values + 2, target > 0.5, weight,
		                  calib->useForTraining(),
		                  calib->useForTesting());
	else {
		std::vector<std::vector<double> >::iterator pos = tmp.begin();
		for(unsigned int i = 0; pos != tmp.end(); i++)
			if (i != targetIdx && i != weightIdx)
				*pos++ = values[i];

		proc->doTrainData(&tmp.front(), target > 0.5, weight,
		                  calib->useForTraining(),
		                  calib->useForTesting());
	}

	return target;
}

void TrainInterceptor::finish(bool save)
{
	proc->doTrainEnd();

	edm::LogInfo("MVATrainer")
		<< "... processor \"" << (const char*)proc->getName()
		<< "\" training iteration done.";

	if (proc->isTrained()) {
		edm::LogInfo("MVATrainer")
			<< "* Completed training of \""
			<< (const char*)proc->getName() << "\".";

		if (save)
			proc->save();
	}
}

// implementation for MVATrainerComputer

MVATrainerComputer::MVATrainerComputer(const std::vector<Interceptor>
						&interceptors, bool autoSave,
                                       UInt_t seed, double split) :
	interceptors(interceptors), nConfigured(0), doAutoSave(autoSave),
	random(seed), split(split)
{
	for(std::vector<Interceptor>::const_iterator iter =
		interceptors.begin(); iter != interceptors.end(); ++iter)
		iter->second->setCalibration(this);
}

MVATrainerComputer::~MVATrainerComputer()
{
	done();
	
	for(std::vector<Interceptor>::const_iterator iter =
		interceptors.begin(); iter != interceptors.end(); ++iter)
		delete iter->second;
}

std::vector<Calibration::VarProcessor*>
MVATrainerComputer::getProcessors() const
{
	std::vector<Calibration::VarProcessor*> processors =
			Calibration::MVAComputer::getProcessors();

	for(std::vector<Interceptor>::const_iterator iter =
		interceptors.begin(); iter != interceptors.end(); ++iter)

		processors.insert(processors.begin() + iter->first,
		                  1, iter->second);

	return processors;
}

void MVATrainerComputer::initFlags(std::vector<Variable::Flags> &flags) const
{
	assert(flags.size() == this->flags.size());
	flags = this->flags;
}

void MVATrainerComputer::configured(BaseInterceptor *interceptor) const
{
	nConfigured++;
	if (isConfigured())
		for(std::vector<Interceptor>::const_iterator iter =
						interceptors.begin();
		    iter != interceptors.end(); ++iter)
			iter->second->init();
}

void MVATrainerComputer::next()
{
	splitResult = random.Uniform(1.0) >= split;
}

void MVATrainerComputer::done()
{
	if (isConfigured()) {
		for(std::vector<Interceptor>::const_iterator iter =
						interceptors.begin();
		    iter != interceptors.end(); ++iter)
			iter->second->finish(doAutoSave);
		nConfigured = 0;
	}
}

// implementation for MVATrainer

const AtomicId MVATrainer::kTargetId("__TARGET__");
const AtomicId MVATrainer::kWeightId("__WEIGHT__");

static const AtomicId kOutputId("__OUTPUT__");

static bool isMagic(AtomicId id)
{
	return id == MVATrainer::kTargetId ||
	       id == MVATrainer::kWeightId ||
	       id == kOutputId;
}

static std::string escape(const std::string &in)
{
	std::string result("'");
	for(std::string::const_iterator iter = in.begin();
	    iter != in.end(); ++iter) {
		switch(*iter) {
		    case '\'':
			result += "'\\''";
			break;
		    default:
			result += *iter;
		}
	}
	result += '\'';
	return result;
}

MVATrainer::MVATrainer(const std::string &fileName, bool useXSLT,
	const char *styleSheet) :
	input(nullptr), output(nullptr), name("MVATrainer"),
	doAutoSave(true), doCleanup(false),
	doMonitoring(false), randomSeed(65539), crossValidation(0.0)
{
	if (useXSLT) {
		std::string sheet;
		if (!styleSheet)
			sheet = edm::FileInPath(
				"PhysicsTools/MVATrainer/data/MVATrainer.xsl")
				.fullPath();
		else
			sheet = styleSheet;

		std::string preproc = "xsltproc --xinclude " + escape(sheet) +
		                      " " + escape(fileName);
		xml.reset(new XMLDocument(fileName, preproc));
	} else
		xml.reset(new XMLDocument(fileName));

	DOMNode *node = xml->getRootNode();

	if (std::strcmp(XMLSimpleStr(node->getNodeName()), "MVATrainer") != 0)
		throw cms::Exception("MVATrainer")
			<< "Invalid XML root node." << std::endl;

	enum State {
		STATE_GENERAL,
		STATE_FIRST,
		STATE_MIDDLE,
		STATE_LAST
	} state = STATE_GENERAL;

	for(node = node->getFirstChild();
	    node; node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		std::string name = XMLSimpleStr(node->getNodeName());
		DOMElement *elem = static_cast<DOMElement*>(node);

		switch(state) {
		    case STATE_GENERAL: {
			if (name != "general")
				throw cms::Exception("MVATrainer")
					<< "Expected general config as first "
					   "tag." << std::endl;

			for(DOMNode *subNode = elem->getFirstChild();
			    subNode; subNode = subNode->getNextSibling()) {
				if (subNode->getNodeType() !=
				    DOMNode::ELEMENT_NODE)
					continue;

				if (std::strcmp(XMLSimpleStr(
					subNode->getNodeName()), "option") != 0)
					throw cms::Exception("MVATrainer")
						<< "Expected option tag."
						<< std::endl;

				elem = static_cast<DOMElement*>(subNode);
				name = XMLDocument::readAttribute<std::string>(
								elem, "name");
				std::string content = XMLSimpleStr(
						elem->getTextContent());

				if (name == "id")
					this->name = content;
				else if (name == "trainfiles")
					trainFileMask = content;
				else
					throw cms::Exception("MVATrainer")
						<< "Unknown option \""
						<< name << "\"." << std::endl;
			}

			state = STATE_FIRST;
		    }	break;
		    case STATE_FIRST: {
			if (name != "input")
				throw cms::Exception("MVATrainer")
					<< "Expected input config as second "
					   "tag." << std::endl;

			AtomicId id = XMLDocument::readAttribute<std::string>(
								elem, "id");
			input = new Source(id, true);
			input->getOutputs().append(
				createVariable(input, kTargetId,
				               Variable::FLAG_NONE),
				SourceVariableSet::kTarget);
			input->getOutputs().append(
				createVariable(input, kWeightId,
				               Variable::FLAG_OPTIONAL),
				SourceVariableSet::kWeight);
			sources.insert(std::make_pair(id, input));
			fillOutputVars(input->getOutputs(), input, elem);

			state = STATE_MIDDLE;
		    }	break;
		    case STATE_MIDDLE: {
			if (name == "output") {
				AtomicId zero;
				output = new TrainProcessor("output",
				                            &zero, this);
				fillInputVars(output->getInputs(), elem);
				state = STATE_LAST;
				continue;
			} else if (name != "processor")
				throw cms::Exception("MVATrainer")
					<< "Unexpected tag after input "
					   "config." << std::endl;

			AtomicId id = XMLDocument::readAttribute<std::string>(
								elem, "id");
			std::string name =
				XMLDocument::readAttribute<std::string>(
					elem, "name");

			makeProcessor(elem, id, name.c_str());
		    }	break;
		    case STATE_LAST:
			throw cms::Exception("MVATrainer")
				<< "Unexpected tag found after output."
				<< std::endl;
			break;
		}
	}

	if (state == STATE_FIRST)
		throw cms::Exception("MVATrainer")
			<< "Expected input variable config." << std::endl;
	else if (state == STATE_MIDDLE)
		throw cms::Exception("MVATrainer")
			<< "Expected output variable config." << std::endl;

	if (trainFileMask.empty())
		trainFileMask = this->name + "_%s%s.%s";
}

MVATrainer::~MVATrainer()
{
	if (monitoring.get())
		monitoring->write();

	for(std::map<AtomicId, Source*>::const_iterator iter = sources.begin();
	    iter != sources.end(); iter++) {
		TrainProcessor *proc =
				dynamic_cast<TrainProcessor*>(iter->second);

		if (proc && doCleanup)
			proc->cleanup();

		delete iter->second;
	}
	delete output;
	std::for_each(variables.begin(), variables.end(),
	              deleter<SourceVariable>);
}

void MVATrainer::loadState()
{
	for(std::vector<AtomicId>::const_iterator iter =
						this->processors.begin();
	    iter != this->processors.end(); iter++) {
		std::map<AtomicId, Source*>::const_iterator pos =
							sources.find(*iter);
		assert(pos != sources.end());
		TrainProcessor *source =
				dynamic_cast<TrainProcessor*>(pos->second);
		assert(source);

		if (source->load())
			edm::LogInfo("MVATrainer")
				<< source->getId() << " configuration for \""
			 	<< (const char*)source->getName()
				<< "\" loaded from file.";
	}
}

void MVATrainer::saveState()
{
	doCleanup = false;

	for(std::vector<AtomicId>::const_iterator iter =
						this->processors.begin();
	    iter != this->processors.end(); iter++) {
		std::map<AtomicId, Source*>::const_iterator pos =
							sources.find(*iter);
		assert(pos != sources.end());
		TrainProcessor *source =
				dynamic_cast<TrainProcessor*>(pos->second);
		assert(source);

		if (source->isTrained())
			source->save();
	}
}

void MVATrainer::makeProcessor(DOMElement *elem, AtomicId id, const char *name)
{
	DOMElement *xmlInput = nullptr;
	DOMElement *xmlConfig = nullptr;
	DOMElement *xmlOutput = nullptr;
	DOMElement *xmlData = nullptr;

	static struct NameExpect {
		const char	*tag;
		bool		mandatory;
		DOMElement	**elem;
	} const expect[] = {
		{ "input",	true,	&xmlInput },
		{ "config",	true,	&xmlConfig },
		{ "output",	true,	&xmlOutput },
		{ "data",	false,	&xmlData },
		{ nullptr, }
	};

	const NameExpect *cur = expect;
	for(DOMNode *node = elem->getFirstChild();
	    node; node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		std::string tag = XMLSimpleStr(node->getNodeName());
		DOMElement *elem = static_cast<DOMElement*>(node);

		if (!cur->tag)
			throw cms::Exception("MVATrainer")
				<< "Superfluous tag " << tag
				<< "encountered in processor." << std::endl;
		else if (tag != cur->tag && cur->mandatory)
			throw cms::Exception("MVATrainer")
				<< "Expected tag " << cur->tag << ", got "
				<< tag << " instead in processor."
				<< std::endl;
		else if (tag != cur->tag) {
			cur++;
			continue;
		}
		*(cur++)->elem = elem;
	}

	while(cur->tag && !cur->mandatory)
		cur++;
	if (cur->tag)
		throw cms::Exception("MVATrainer")
			<< "Unexpected end of processor configuration, "
			<< "expected tag " << cur->tag << "." << std::endl;

	std::unique_ptr<TrainProcessor> proc(
				TrainProcessor::create(name, &id, this));
	if (!proc.get())
		throw cms::Exception("MVATrainer")
			<< "Variable processor trainer " << name
			<< " could not be instantiated. Most likely because"
			   " the trainer plugin for \"" << name << "\""
			   " does not exist." << std::endl;

	if (sources.find(id) != sources.end())
		throw cms::Exception("MVATrainer")
			<< "Duplicate variable processor id "
			<< (const char*)id << "."
			<< std::endl;

	fillInputVars(proc->getInputs(), xmlInput);
	fillOutputVars(proc->getOutputs(), proc.get(), xmlOutput);

	edm::LogInfo("MVATrainer")
		<< "Configuring " << (const char*)proc->getId()
		<< " \"" << (const char*)proc->getName() << "\".";
	proc->configure(xmlConfig);

	sources.insert(std::make_pair(id, proc.release()));
	processors.push_back(id);
}

std::string MVATrainer::trainFileName(const TrainProcessor *proc,
                                      const std::string &ext,
                                      const std::string &arg) const
{
	std::string arg_ = !arg.empty() ? ("_" + arg) : "";
	return stdStringPrintf(trainFileMask.c_str(),
	                       (const char*)proc->getName(),
	                       arg_.c_str(), ext.c_str());
}

TrainerMonitoring::Module *MVATrainer::bookMonitor(const std::string &name)
{
	if (!doMonitoring)
		return nullptr;

	if (!monitoring.get()) {
		std::string fileName = 
			stdStringPrintf(trainFileMask.c_str(),
			                "monitoring", "", "root");
		monitoring.reset(new TrainerMonitoring(fileName));
	}

	return monitoring->book(name);
}

SourceVariable *MVATrainer::getVariable(AtomicId source, AtomicId name) const
{
	std::map<AtomicId, Source*>::const_iterator pos = sources.find(source);
	if (pos == sources.end())
		return nullptr;

	return pos->second->getOutput(name);
}

SourceVariable *MVATrainer::createVariable(Source *source, AtomicId name,
                                           Variable::Flags flags)
{
	SourceVariable *var = getVariable(source->getName(), name);
	if (var)
		return nullptr;

	var = new SourceVariable(source, name, flags);
	variables.push_back(var);
	return var;
}

void MVATrainer::fillInputVars(SourceVariableSet &vars,
                               XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *xml)
{
	std::vector<SourceVariable*> tmp;
	SourceVariable *target = nullptr;
	SourceVariable *weight = nullptr;

	for(DOMNode *node = xml->getFirstChild(); node;
	    node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		if (std::strcmp(XMLSimpleStr(node->getNodeName()), "var") != 0)
			throw cms::Exception("MVATrainer")
				<< "Invalid input variable node." << std::endl;

		DOMElement *elem = static_cast<DOMElement*>(node);

		AtomicId source = XMLDocument::readAttribute<std::string>(
							elem, "source");
		AtomicId name = XMLDocument::readAttribute<std::string>(
							elem, "name");

		SourceVariable *var = getVariable(source, name);
		if (!var)
			throw cms::Exception("MVATrainer")
				<< "Input variable " << (const char*)source
				<< ":" << (const char*)name
				<< " not found." << std::endl;

		if (XMLDocument::readAttribute<bool>(elem, "target", false)) {
			if (target)
				throw cms::Exception("MVATrainer")
					<< "Target variable defined twice"
					<< std::endl;
			target = var;
		}
		if (XMLDocument::readAttribute<bool>(elem, "weight", false)) {
			if (weight)
				throw cms::Exception("MVATrainer")
					<< "Weight variable defined twice"
					<< std::endl;
			weight = var;
		}

		tmp.push_back(var);
	}

	if (!weight) {
		weight = input->getOutput(kWeightId);
		assert(weight);
		tmp.insert(tmp.begin() +
		           	(target == input->getOutput(kTargetId)),
		           1, weight);
	}
	if (!target) {
		target = input->getOutput(kTargetId);
		assert(target);
		tmp.insert(tmp.begin(), 1, target);
	}

	unsigned int n = 0;
	for(std::vector<SourceVariable*>::const_iterator iter = variables.begin();
	    iter != variables.end(); iter++) {
		std::vector<SourceVariable*>::const_iterator pos =
			std::find(tmp.begin(), tmp.end(), *iter);
		if (pos == tmp.end())
			continue;

		SourceVariableSet::Magic magic;
		if (*iter == target)
			magic = SourceVariableSet::kTarget;
		else if (*iter == weight)
			magic = SourceVariableSet::kWeight;
		else
			magic = SourceVariableSet::kRegular;

		if (vars.append(*iter, magic, pos - tmp.begin())) {
			AtomicId source = (*iter)->getSource()->getName();
			AtomicId name = (*iter)->getName();
			throw cms::Exception("MVATrainer")
				<< "Input variable " << (const char*)source
				<< ":" << (const char*)name
				<< " defined twice." << std::endl;
		}

		n++;
	}

	assert(tmp.size() == n);
}

void MVATrainer::fillOutputVars(SourceVariableSet &vars, Source *source,
                                XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *xml)
{
	for(DOMNode *node = xml->getFirstChild(); node;
	    node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		if (std::strcmp(XMLSimpleStr(node->getNodeName()), "var") != 0)
			throw cms::Exception("MVATrainer")
				<< "Invalid output variable node."
				<< std::endl;

		DOMElement *elem = static_cast<DOMElement*>(node);

		AtomicId name = XMLDocument::readAttribute<std::string>(
							elem, "name");
		if (!name)
			throw cms::Exception("MVATrainer")
				<< "Output variable tag missing name."
				<< std::endl;
		if (isMagic(name))
			throw cms::Exception("MVATrainer")
				<< "Cannot use magic variable names in output."
				<< std::endl;

		Variable::Flags flags = Variable::FLAG_NONE;

		if (XMLDocument::readAttribute<bool>(elem, "optional", true))
			flags = (PhysicsTools::Variable::Flags)
				(flags | Variable::FLAG_OPTIONAL);

		if (XMLDocument::readAttribute<bool>(elem, "multiple", true))
			flags = (PhysicsTools::Variable::Flags)
				(flags | Variable::FLAG_MULTIPLE);

		SourceVariable *var = createVariable(source, name, flags);
		if (!var || vars.append(var))
			throw cms::Exception("MVATrainer")
				<< "Output variable "
				<< (const char*)source->getName()
				<< ":" << (const char*)name
				<< " defined twice." << std::endl;
	}
}

void
MVATrainer::connectProcessors(Calibration::MVAComputer *calib,
                              const std::vector<CalibratedProcessor> &procs,
                              bool withTarget) const
{
	std::map<SourceVariable*, unsigned int> vars;
	unsigned int size = 0;

	MVATrainerComputer *trainCalib =
			dynamic_cast<MVATrainerComputer*>(calib);

	for(unsigned int i = 0;
	    i < input->getOutputs().size(true); i++) {
		if (i < 2 && !withTarget)
			continue;

		SourceVariable *var = variables[i];
		vars[var] = size++;

		Calibration::Variable calibVar;
		calibVar.name = (const char*)var->getName();
		calib->inputSet.push_back(calibVar);
		if (trainCalib)
			trainCalib->addFlag(var->getFlags());
	}

	for(std::vector<CalibratedProcessor>::const_iterator iter =
				procs.begin(); iter != procs.end(); iter++) {
		bool isInterceptor = dynamic_cast<BaseInterceptor*>(
							iter->calib) != nullptr;

		BitSet inputSet(size);

		unsigned int last = 0;
		std::vector<SourceVariable*> inoutVars;
		if (iter->processor)
			inoutVars = iter->processor->getInputs().get(
								isInterceptor);
		for(std::vector<SourceVariable*>::const_iterator iter2 =
			inoutVars.begin(); iter2 != inoutVars.end(); iter2++) {
			std::map<SourceVariable*,
			         unsigned int>::const_iterator pos =
							vars.find(*iter2);

			assert(pos != vars.end());

			if (pos->second < last)
				throw cms::Exception("MVATrainer")
					<< "Input variables not declared "
					   "in order of appearance in \""
					<< (const char*)iter->processor->getName()
					<< "\"." << std::endl;

			inputSet[last = pos->second] = true;
		}

		assert(!isInterceptor || withTarget);

		iter->calib->inputVars = Calibration::convert(inputSet);

		calib->output = size;

		if (isInterceptor) {
			size++;
			continue;
		}

		calib->addProcessor(iter->calib);

		inoutVars = iter->processor->getOutputs().get();
		for(std::vector<SourceVariable*>::const_iterator iter =
			inoutVars.begin(); iter != inoutVars.end(); iter++) {

			vars[*iter] = size++;
		}
	}

	if (output->getInputs().size() != 1)
		throw cms::Exception("MVATrainer")
			<< "Exactly one output variable has to be specified."
			<< std::endl;

	SourceVariable *outVar = output->getInputs().get()[0];
	std::map<SourceVariable*, unsigned int>::const_iterator pos =
							vars.find(outVar);
	if (pos != vars.end())
		calib->output = pos->second;
}

Calibration::MVAComputer *
MVATrainer::makeTrainCalibration(const AtomicId *compute,
                                 const AtomicId *train) const
{
	std::map<AtomicId, TrainInterceptor*> interceptors;
	std::vector<MVATrainerComputer::Interceptor> baseInterceptors;
	std::vector<CalibratedProcessor> processors;

	BaseInterceptor *interceptor = new InitInterceptor;
	baseInterceptors.push_back(std::make_pair(0, interceptor));
	processors.push_back(CalibratedProcessor(nullptr, interceptor));

	for(const AtomicId *iter = train; *iter; iter++) {
		TrainProcessor *source;
		if (*iter == kOutputId)
			source = output;
		else {
			std::map<AtomicId, Source*>::const_iterator pos =
							sources.find(*iter);
			assert(pos != sources.end());
			source = dynamic_cast<TrainProcessor*>(pos->second);
		}
		assert(source);

		interceptors[*iter] = new TrainInterceptor(source);
	}

	auto_cleaner<Calibration::VarProcessor> autoClean;

	std::set<AtomicId> done;
	for(const AtomicId *iter = compute; *iter; iter++) {
		if (done.erase(*iter))
			continue;

		std::map<AtomicId, Source*>::const_iterator pos =
							sources.find(*iter);
		assert(pos != sources.end());
		TrainProcessor *source =
				dynamic_cast<TrainProcessor*>(pos->second);
		assert(source);
		assert(source->isTrained());

		Calibration::VarProcessor *proc = source->getCalibration();
		if (!proc)
			continue;

		autoClean.add(proc);
		processors.push_back(CalibratedProcessor(source, proc));

		Calibration::ProcForeach *looper =
				dynamic_cast<Calibration::ProcForeach*>(proc);
		if (looper) {
			std::vector<AtomicId>::const_iterator pos2 =
				std::find(this->processors.begin(),
				          this->processors.end(), *iter);
			assert(pos2 != this->processors.end());
			++pos2;
			unsigned int n = 0;
			for(int i = 0; i < (int)looper->nProcs; ++i, ++pos2) {
				assert(pos2 != this->processors.end());

				const AtomicId *iter2 = compute;
				while(*iter2) {
					if (*iter2 == *pos2)
						break;
					iter2++;
				}

				if (*iter2) {
					n++;
					done.insert(*iter2);
					pos = sources.find(*iter2);
					assert(pos != sources.end());
					TrainProcessor *source =
						dynamic_cast<TrainProcessor*>(
								pos->second);
					assert(source);
					assert(source->isTrained());

					proc = source->getCalibration();
					if (proc) {
						autoClean.add(proc);
						processors.push_back(
							CalibratedProcessor(
								source, proc));
					}
				}

				std::map<AtomicId, TrainInterceptor*>::iterator
						pos3 = interceptors.find(*pos2);
				if (pos3 != interceptors.end()) {
					n++;
					baseInterceptors.push_back(
						std::make_pair(processors.size(),
							       pos3->second));
					processors.push_back(
						CalibratedProcessor(
							pos3->second->getProcessor(),
							pos3->second));
					interceptors.erase(pos3);
				}
			}

			looper->nProcs = n;
			if (!n) {
				baseInterceptors.pop_back();
				processors.pop_back();
			}
		}
	}

	for(std::map<AtomicId, TrainInterceptor*>::const_iterator iter =
		interceptors.begin(); iter != interceptors.end(); ++iter) {

		TrainProcessor *proc = iter->second->getProcessor();
		baseInterceptors.push_back(std::make_pair(processors.size(),
		                                          iter->second));
		processors.push_back(CalibratedProcessor(proc, iter->second));
	}

	std::unique_ptr<Calibration::MVAComputer> calib(
		new MVATrainerComputer(baseInterceptors, doAutoSave,
		                       randomSeed, crossValidation));

	connectProcessors(calib.get(), processors, true);

	return calib.release();
}

void MVATrainer::doneTraining(Calibration::MVAComputer *trainCalibration) const
{
	MVATrainerComputer *calib =
			dynamic_cast<MVATrainerComputer*>(trainCalibration);

	if (!calib)
		throw cms::Exception("MVATrainer")
			<< "Invalid training calibration passed to "
			   "doneTraining()" << std::endl;

	calib->done();
}

std::vector<AtomicId> MVATrainer::findFinalProcessors() const
{
	std::set<Source*> toCheck;
	toCheck.insert(output);

	std::set<Source*> done;
	while(!toCheck.empty()) {
		Source *source = *toCheck.begin();
		toCheck.erase(toCheck.begin());

		std::vector<SourceVariable*> inputs = source->inputs.get();
		for(std::vector<SourceVariable*>::const_iterator iter =
				inputs.begin(); iter != inputs.end(); ++iter) {
			source = (*iter)->getSource();
			if (done.insert(source).second)
				toCheck.insert(source);
		}
	}

	std::vector<AtomicId> result;
	for(std::vector<AtomicId>::const_iterator iter = processors.begin();
	    iter != processors.end(); ++iter) {
		std::map<AtomicId, Source*>::const_iterator pos =
							sources.find(*iter);
		if (pos != sources.end() && done.count(pos->second))
			result.push_back(*iter);
	}

	return result;
}

Calibration::MVAComputer *MVATrainer::getCalibration() const
{
	std::vector<CalibratedProcessor> processors;

	std::unique_ptr<Calibration::MVAComputer> calib(
						new Calibration::MVAComputer);

	std::vector<AtomicId> used = findFinalProcessors();
	for(std::vector<AtomicId>::const_iterator iter = used.begin();
	    iter != used.end(); iter++) {
		std::map<AtomicId, Source*>::const_iterator pos =
							sources.find(*iter);
		assert(pos != sources.end());
		TrainProcessor *source =
				dynamic_cast<TrainProcessor*>(pos->second);
		assert(source);
		if (!source->isTrained())
			return nullptr;

		Calibration::VarProcessor *proc = source->getCalibration();
		if (!proc)
			continue;

		Calibration::ProcForeach *foreach =
				dynamic_cast<Calibration::ProcForeach*>(proc);
		if (foreach) {
			std::vector<AtomicId>::const_iterator begin =
				std::find(this->processors.begin(),
				          this->processors.end(), *iter);
			assert(this->processors.end() - begin >
			       (int)(foreach->nProcs + 1));
			++begin;
			std::vector<AtomicId>::const_iterator end =
						begin + foreach->nProcs;
			foreach->nProcs = 0;
			for(std::vector<AtomicId>::const_iterator iter2 =
					iter; iter2 != used.end(); ++iter2)
				if (std::find(begin, end, *iter2) != end)
					foreach->nProcs++;
		}

		processors.push_back(CalibratedProcessor(source, proc));
	}

	connectProcessors(calib.get(), processors, false);

	return calib.release();
}

void MVATrainer::findUntrainedComputers(std::vector<AtomicId> &compute,
                                        std::vector<AtomicId> &train) const
{
	compute.clear();
	train.clear();

	std::set<Source*> trainedSources;
	trainedSources.insert(input);

	for(std::vector<AtomicId>::const_iterator iter =
		processors.begin(); iter != processors.end(); iter++) {
		std::map<AtomicId, Source*>::const_iterator pos =
							sources.find(*iter);
		assert(pos != sources.end());
		TrainProcessor *proc =
				dynamic_cast<TrainProcessor*>(pos->second);
		assert(proc);

		bool trainedDeps = true;
		std::vector<SourceVariable*> inputVars =
					proc->getInputs().get();
		for(std::vector<SourceVariable*>::const_iterator iter2 =
			inputVars.begin(); iter2 != inputVars.end(); iter2++) {
			if (trainedSources.find((*iter2)->getSource())
			    == trainedSources.end()) {
				trainedDeps = false;
				break;
			}
		}

		if (!trainedDeps)
			continue;

		if (proc->isTrained()) {
			trainedSources.insert(proc);
			compute.push_back(proc->getName());
		} else
			train.push_back(proc->getName());
	}

	if (doMonitoring && !output->isTrained() &&
	    trainedSources.find(output->getInputs().get()[0]->getSource())
						!= trainedSources.end())
		train.push_back(kOutputId);
}

Calibration::MVAComputer *MVATrainer::getTrainCalibration() const
{
	std::vector<AtomicId> compute, train;
	findUntrainedComputers(compute, train);

	if (train.empty())
		return nullptr;

	compute.push_back(nullptr);
	train.push_back(nullptr);

	return makeTrainCalibration(&compute.front(), &train.front());
}

} // namespace PhysicsTools
