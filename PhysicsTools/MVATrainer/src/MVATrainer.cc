#include <assert.h>
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

#include "FWCore/Utilities/interface/Exception.h"
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
#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"

XERCES_CPP_NAMESPACE_USE

namespace PhysicsTools {

namespace { // anonymous
	class MVATrainerComputer;

	class TrainInterceptor : public Calibration::Interceptor {
	    public:
		TrainInterceptor(TrainProcessor *proc) :
			proc(proc), calib(0) {}
		virtual ~TrainInterceptor() {}

		inline TrainProcessor *getProcessor() const { return proc; }

		inline void setCalibration(MVATrainerComputer *calib)
		{ this->calib = calib; }

		virtual std::vector<Variable::Flags>
		configure(const MVAComputer *computer, unsigned int n);

		virtual double
		intercept(const std::vector<double> *values) const;

		void init();
		void finish(bool save);

	    private:
		TrainProcessor		*const proc;
		MVATrainerComputer	*calib;
	};

	class MVATrainerComputer : public Calibration::MVAComputer {
	    public:
		MVATrainerComputer(const std::vector<TrainInterceptor*>
						&interceptors, bool autoSave);

		virtual ~MVATrainerComputer();

		virtual std::vector<Calibration::VarProcessor*>
		getProcessors() const;

		void configured(TrainInterceptor *interceptor) const;
		void done();

		inline bool isConfigured() const
		{ return nConfigured == interceptors.size(); }

	    private:
		std::vector<TrainInterceptor*>	interceptors;
		mutable unsigned int		nConfigured;
		bool				doAutoSave;
	};

	// useful litte helpers

	template<typename T>
	struct deleter : public std::unary_function<T*, void> {
		inline void operator() (T *ptr) const { delete ptr; }
	};

	template<typename T>
	struct auto_cleaner {
		inline ~auto_cleaner()
		{ std::for_each(clean.begin(), clean.end(), deleter<T>()); }
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

// implementation for TrainInterceptor

std::vector<Variable::Flags>
TrainInterceptor::configure(const MVAComputer *computer, unsigned int n)
{
	std::vector<Variable::Flags> flags(n, proc->getDefaultFlags());
	flags[0] = Variable::FLAG_NONE;
	flags[1] = Variable::FLAG_OPTIONAL;

	calib->configured(this);

	return flags;
}

void TrainInterceptor::init()
{
	edm::LogInfo("MVATrainer")
		<< "TrainProcessor \"" << (const char*)proc->getName()
		<< "\" training iteration starting...";

	proc->trainBegin();
}

double
TrainInterceptor::intercept(const std::vector<double> *values) const
{
	if (values[0].size() != 1) {
		if (values[0].size() == 0)
			throw cms::Exception("MVATrainer")
				<< "Trainer input lacks target variable."
				<< std::endl;
		else
			throw cms::Exception("MVATrainer")
				<< "Multiple targets supplied in input."
				<< std::endl;
	}
	double target = values[0].front();

	double weight = 1.0;
	if (values[1].size() > 1)
		throw cms::Exception("MVATrainer")
			<< "Multiple weights supplied in input."
			<< std::endl;
	else if (values[1].size() == 1)
		weight = values[1].front();

	proc->trainData(values + 2, target > 0.5, weight);

	return target;
}

void TrainInterceptor::finish(bool save)
{
	proc->trainEnd();

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

MVATrainerComputer::MVATrainerComputer(const std::vector<TrainInterceptor*>
						&interceptors, bool autoSave) :
	interceptors(interceptors), nConfigured(0), doAutoSave(autoSave)
{
	std::for_each(interceptors.begin(), interceptors.end(),
	              std::bind2nd(
	              	std::mem_fun(&TrainInterceptor::setCalibration),
	              	this));
}

MVATrainerComputer::~MVATrainerComputer()
{
	done();
	std::for_each(interceptors.begin(), interceptors.end(),
	              deleter<TrainInterceptor>());
}

std::vector<Calibration::VarProcessor*>
MVATrainerComputer::getProcessors() const
{
	std::vector<Calibration::VarProcessor*> processors =
			Calibration::MVAComputer::getProcessors();

	std::copy(interceptors.begin(), interceptors.end(),
	          std::back_inserter(processors));

	return processors;
}

void MVATrainerComputer::configured(TrainInterceptor *interceptor) const
{
	nConfigured++;
	if (isConfigured())
		std::for_each(interceptors.begin(), interceptors.end(),
		              std::mem_fun(&TrainInterceptor::init));
}

void MVATrainerComputer::done()
{
	if (isConfigured()) {
		std::for_each(interceptors.begin(), interceptors.end(),
		              std::bind2nd(
		              	std::mem_fun(&TrainInterceptor::finish),
		              	doAutoSave));
		nConfigured = 0;
	}
}

// implementation for MVATrainer

const AtomicId MVATrainer::kTargetId("__TARGET__");
const AtomicId MVATrainer::kWeightId("__WEIGHT__");

MVATrainer::MVATrainer(const std::string &fileName) :
	input(0), output(0), name("MVATrainer"),
	doAutoSave(true), doCleanup(false)
{
	xml = std::auto_ptr<XMLDocument>(new XMLDocument(fileName));

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
			sources.insert(std::make_pair(id, input));
			fillOutputVars(input->getOutputs(), input, elem);

			state = STATE_MIDDLE;
		    }	break;
		    case STATE_MIDDLE: {
			if (name == "output") {
				output = new Source(0);
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
	              deleter<SourceVariable>());
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
	DOMElement *xmlInput = 0;
	DOMElement *xmlConfig = 0;
	DOMElement *xmlOutput = 0;
	DOMElement *xmlData = 0;

	static struct NameExpect {
		const char	*tag;
		bool		mandatory;
		DOMElement	**elem;
	} const expect[] = {
		{ "input",	true,	&xmlInput },
		{ "config",	true,	&xmlConfig },
		{ "output",	true,	&xmlOutput },
		{ "data",	false,	&xmlData },
		{ 0, }
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

	std::auto_ptr<TrainProcessor> proc(
				TrainProcessor::create(name, &id, this));
	if (!proc.get())
		throw cms::Exception("MVATrainer")
			<< "Variable processor trainer " << name
			<< " could not be instantiated." << std::endl;

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
	std::string arg_ = arg.size() > 0 ? ("_" + arg) : "";
	return stdStringPrintf(trainFileMask.c_str(),
	                       (const char*)proc->getName(),
	                       arg_.c_str(), ext.c_str());
}

SourceVariable *MVATrainer::getVariable(AtomicId source, AtomicId name) const
{
	std::map<AtomicId, Source*>::const_iterator pos = sources.find(source);
	if (pos == sources.end())
		return 0;

	return pos->second->getOutput(name);
}

SourceVariable *MVATrainer::createVariable(Source *source, AtomicId name,
                                           Variable::Flags flags)
{
	SourceVariable *var = getVariable(source->getName(), name);
	if (var)
		return 0;

	var = new SourceVariable(source, name, flags);
	variables.push_back(var);
	return var;
}

void MVATrainer::fillInputVars(SourceVariableSet &vars,
                               XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *xml)
{
	std::vector<SourceVariable*> tmp;

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

		tmp.push_back(var);
	}

	unsigned int n = 0;
	for(std::vector<SourceVariable*>::const_iterator iter = variables.begin();
	    iter != variables.end(); iter++) {
		std::vector<SourceVariable*>::const_iterator pos =
			std::find(tmp.begin(), tmp.end(), *iter);
		if (pos == tmp.end())
			continue;

		int delta = pos - tmp.begin();
		vars.append(*iter, delta - vars.size());
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

		Variable::Flags flags = Variable::FLAG_NONE;

		if (XMLDocument::readAttribute<bool>(elem, "optional", true))
			(int&)flags |= Variable::FLAG_OPTIONAL;

		if (XMLDocument::readAttribute<bool>(elem, "multiple", true))
			(int&)flags |= Variable::FLAG_MULTIPLE;

		SourceVariable *var = createVariable(source, name, flags);
		if (!var)
			throw cms::Exception("MVATrainer")
				<< "Output variable "
				<< (const char*)source->getName()
				<< ":" << (const char*)name
				<< " defined twice." << std::endl;

		vars.append(var);
	}
}

void
MVATrainer::connectProcessors(Calibration::MVAComputer *calib,
                              const std::vector<CalibratedProcessor> &procs,
                              bool withTarget) const
{
	std::map<SourceVariable*, unsigned int> vars;
	unsigned int size = 0;

	if (withTarget) {
		Calibration::Variable calibVar;

		calibVar.name = (const char*)kTargetId;
		calib->inputSet.push_back(calibVar);
		size++;

		calibVar.name = (const char*)kWeightId;
		calib->inputSet.push_back(calibVar);
		size++;
	}

	for(unsigned int i = 0; i < input->getOutputs().size(); i++) {
		SourceVariable *var = variables[i];
		vars[var] = size++;

		Calibration::Variable calibVar;
		calibVar.name = (const char*)var->getName();
		calib->inputSet.push_back(calibVar);
	}

	for(std::vector<CalibratedProcessor>::const_iterator iter =
				procs.begin(); iter != procs.end(); iter++) {
		bool isInterceptor = dynamic_cast<TrainInterceptor*>(
							iter->calib) != 0;

		BitSet inputSet(size);

		unsigned int last = 0;
		std::vector<SourceVariable*> inoutVars =
					iter->processor->getInputs().get();
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

		if (isInterceptor) {
			assert(withTarget);
			inputSet[0] = true;
			inputSet[1] = true;
		}

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
	std::map<SourceVariable*,
	         unsigned int>::const_iterator pos = vars.find(outVar);
	if (pos != vars.end())
		calib->output = pos->second;
}

Calibration::MVAComputer *
MVATrainer::makeTrainCalibration(const AtomicId *compute,
                                 const AtomicId *train) const
{
	std::vector<TrainInterceptor*> interceptors;
	std::vector<CalibratedProcessor> processors;

	for(const AtomicId *iter = train; *iter; iter++) {
		std::map<AtomicId, Source*>::const_iterator pos =
							sources.find(*iter);
		assert(pos != sources.end());
		TrainProcessor *source =
				dynamic_cast<TrainProcessor*>(pos->second);
		assert(source);

		TrainInterceptor *interceptor =
					new TrainInterceptor(source);

		interceptors.push_back(interceptor);
	}

	std::auto_ptr<Calibration::MVAComputer> calib(
			new MVATrainerComputer(interceptors, doAutoSave));

	auto_cleaner<Calibration::VarProcessor> autoClean;

	for(const AtomicId *iter = compute; *iter; iter++) {
		std::map<AtomicId, Source*>::const_iterator pos =
							sources.find(*iter);
		assert(pos != sources.end());
		TrainProcessor *source =
				dynamic_cast<TrainProcessor*>(pos->second);
		assert(source);
		assert(source->isTrained());

		Calibration::VarProcessor *proc = source->getCalibration();

		autoClean.add(proc);
		processors.push_back(CalibratedProcessor(source, proc));
	}

	for(std::vector<TrainInterceptor*>::const_iterator iter =
		interceptors.begin(); iter != interceptors.end(); iter++)
		processors.push_back(
			CalibratedProcessor((*iter)->getProcessor(), *iter));

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

Calibration::MVAComputer *MVATrainer::getCalibration() const
{
	std::vector<CalibratedProcessor> processors;

	std::auto_ptr<Calibration::MVAComputer> calib(
						new Calibration::MVAComputer);

	for(std::vector<AtomicId>::const_iterator iter =
						this->processors.begin();
	    iter != this->processors.end(); iter++) {
		std::map<AtomicId, Source*>::const_iterator pos =
							sources.find(*iter);
		assert(pos != sources.end());
		TrainProcessor *source =
				dynamic_cast<TrainProcessor*>(pos->second);
		assert(source);
		if (!source->isTrained())
			return 0;

		Calibration::VarProcessor *proc = source->getCalibration();

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
}

Calibration::MVAComputer *MVATrainer::getTrainCalibration() const
{
	std::vector<AtomicId> compute, train;
	findUntrainedComputers(compute, train);

	if (train.empty())
		return 0;

	compute.push_back(0);
	train.push_back(0);

	return makeTrainCalibration(&compute.front(), &train.front());
}

} // namespace PhysicsTools
