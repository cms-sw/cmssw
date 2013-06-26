#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <cstddef>
#include <cstring>
#include <vector>
#include <set>

// ROOT version magic to support TMVA interface changes in newer ROOT   
#include <RVersion.h>

#include <TBufferFile.h>
#include <TClass.h>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/zstream.h"

#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"
#include "PhysicsTools/MVAComputer/interface/VarProcessor.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/Variable.h"
#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

// #define DEBUG_EVAL

#ifdef DEBUG_EVAL
#include "FWCore/Utilities/interface/TypeDemangler.h"
#endif

#define STANDALONE_HEADER "MVAComputer calibration\n"

namespace PhysicsTools {

MVAComputer::MVAComputer(const Calibration::MVAComputer *calib) :
	nVars(0), output(0)
{
	setup(calib);
}

MVAComputer::MVAComputer(Calibration::MVAComputer *calib, bool owned) :
	nVars(0), output(0)
{
	if (owned)
		this->owned.reset(calib);
	setup(calib);
}

MVAComputer::MVAComputer(const char *filename) :
	nVars(0), output(0), owned(readCalibration(filename))
{
	setup(owned.get());
}

MVAComputer::MVAComputer(std::istream &is) :
	nVars(0), output(0), owned(readCalibration(is))
{
	setup(owned.get());
}

void MVAComputer::setup(const Calibration::MVAComputer *calib)
{
	nVars = calib->inputSet.size();
	output = calib->output;

	std::vector<Variable::Flags> flags(nVars, Variable::FLAG_ALL);
	const TrainMVAComputerCalibration *trainCalib =
		dynamic_cast<const TrainMVAComputerCalibration*>(calib);
	if (trainCalib)
		trainCalib->initFlags(flags);

	VarProcessor::ConfigCtx config(flags);
	std::vector<Calibration::VarProcessor*> processors =
							calib->getProcessors();

	for(std::vector<Calibration::VarProcessor*>::const_iterator iter =
							processors.begin();
	    iter != processors.end(); ++iter) {
		std::string name = (*iter)->getInstanceName();
		VarProcessor *processor =
			VarProcessor::create(name.c_str(), *iter, this);
		if (!processor)
			throw cms::Exception("UnknownProcessor")
				<< name << " could not be instantiated."
				<< std::endl;

		VarProcessor::ConfigCtx::iterator::difference_type pos =
						config.end() - config.begin();
		processor->configure(config);
		unsigned int nOutput = (config.end() - config.begin()) - pos;
		if (!nOutput)
			throw cms::Exception("InvalidProcessor")
				<< name << " rejected input variable "
				"configuration" << std::endl;

		varProcessors.push_back(Processor(processor, nOutput));
	}

	for(VarProcessor::ConfigCtx::iterator iter = config.begin() + nVars;
	    iter != config.end(); iter++) {
		VarProcessor::Config *origin = &config[iter->origin];
		if (iter->origin >= nVars)
			iter->origin = origin->origin;

		if (iter->mask & Variable::FLAG_MULTIPLE) {
			iter->mask = (Variable::Flags)(iter->mask &
			                               origin->mask);
			config[iter->origin].origin++;
		}
	}

	nVars = config.size();

	if (output >= nVars)
			// FIXME || config[output].mask != Variable::FLAG_NONE)
		throw cms::Exception("InvalidOutput")
			<< "Output variable at index " << output
			<< " invalid." << std::endl;

	std::set<InputVar> variables;
	unsigned int i = 0;
	for(std::vector<Calibration::Variable>::const_iterator iter =
			calib->inputSet.begin(); iter != calib->inputSet.end();
	    ++iter, i++) {
		InputVar var;
		var.var = Variable(iter->name, config[i].mask);
		var.index = i;
                var.multiplicity = 0;
		variables.insert(var);
	}

	inputVariables.resize(i);
	std::copy(variables.begin(), variables.end(),
	          inputVariables.begin());

	for(unsigned int j = 0; j < i; j++)
		inputVariables[j].multiplicity = config[j].origin;
}

MVAComputer::~MVAComputer()
{
}

unsigned int MVAComputer::getVariableId(AtomicId name) const
{
	std::vector<InputVar>::const_iterator pos =
		std::lower_bound(inputVariables.begin(), inputVariables.end(),
		                 name);

	if (pos == inputVariables.end() || pos->var.getName() != name)
		throw cms::Exception("InvalidVariable")
			<< "Input variable " << (const char*)name
			<< " not found."  << std::endl;

	return pos->index;
}

template<class T>
void MVAComputer::evalInternal(T &ctx) const
{
	double *output = ctx.values() + ctx.n();
	int *outConf = ctx.conf() + inputVariables.size();

#ifdef DEBUG_EVAL
	std::cout << "Input" << std::endl;
	double *v = ctx.values();
	for(int *o = ctx.conf(); o < outConf; o++) {
		std::cout << "\tVar " << (o - ctx.conf()) << std::endl;
		for(int i = o[0]; i < o[1]; i++)
			std::cout << "\t\t" << *v++ << std::endl;
	}
#endif
	std::vector<Processor>::const_iterator iter = varProcessors.begin();
	while(iter != varProcessors.end()) {
		std::vector<Processor>::const_iterator loop = iter;
		int *loopOutConf = outConf;
		int *loopStart = 0;
		double *loopOutput = output;

		VarProcessor::LoopStatus status = VarProcessor::kNext;
		unsigned int offset = 0;
		while(status != VarProcessor::kStop) {
			std::vector<Processor>::const_iterator next = iter + 1;
			unsigned int nextOutput = (next != varProcessors.end())
			                          ? next->nOutput : 0;

#ifdef DEBUG_EVAL
                        std::string demangledName;
                        edm::typeDemangle(typeid(*iter->processor).name(), demangledName);
			std::cout << demangledName << std::endl;
#endif
			if (status != VarProcessor::kSkip)
				ctx.eval(&*iter->processor, outConf, output,
				         loopStart ? loopStart : loopOutConf,
				         offset, iter->nOutput);

#ifdef DEBUG_EVAL
			for(unsigned int i = 0; i < iter->nOutput;
			    i++, outConf++) {
				std::cout << "\tVar " << (outConf - ctx.conf())
				          << std::endl;
				for(int j = outConf[0]; j < outConf[1]; j++)
					std::cout << "\t\t" << *output++
					          << std::endl;
			}
#else
			int orig = *outConf;
			outConf += iter->nOutput;
			output += *outConf - orig;
#endif

			status = loop->processor->loop(output, outConf,
			                               nextOutput, offset);

			if (status == VarProcessor::kReset) {
				outConf = loopOutConf;
				output = loopOutput;
				loopStart = 0;
				offset = 0;
				iter = loop;
			} else {
				if (loop == iter)
					loopStart = outConf;
				iter = next;
			}
		}
	}
}

Calibration::MVAComputer *MVAComputer::readCalibration(const char *filename)
{
	std::ifstream file(filename);
	return readCalibration(file);
}

Calibration::MVAComputer *MVAComputer::readCalibration(std::istream &is)
{
	if (!is.good())
		throw cms::Exception("InvalidFileState")
			<< "Stream passed to MVAComputer::readCalibration "
			   "has an invalid state." << std::endl;

	char header[sizeof STANDALONE_HEADER - 1] = { 0, };
	if (is.readsome(header, sizeof header) != sizeof header ||
	    std::memcmp(header, STANDALONE_HEADER, sizeof header) != 0)
		throw cms::Exception("InvalidFileFormat")
			<< "Stream passed to MVAComputer::readCalibration "
			   "is not a valid calibration file." << std::endl;

	TClass *rootClass =
		TClass::GetClass("PhysicsTools::Calibration::MVAComputer");
	if (!rootClass)
		throw cms::Exception("DictionaryMissing")
			<< "CondFormats dictionary for "
			   "PhysicsTools::Calibration::MVAComputer missing"
			<< std::endl;

	ext::izstream izs(&is);
	std::ostringstream ss;
	ss << izs.rdbuf();
	std::string buf = ss.str();

	TBufferFile buffer(TBuffer::kRead, buf.size(), const_cast<void*>(
			static_cast<const void*>(buf.c_str())), kFALSE);
	buffer.InitMap();

	std::auto_ptr<Calibration::MVAComputer> calib(
					new Calibration::MVAComputer());
	buffer.StreamObject(static_cast<void*>(calib.get()), rootClass);

	return calib.release();
}

void MVAComputer::writeCalibration(const char *filename,
                                   const Calibration::MVAComputer *calib)
{
	std::ofstream file(filename);
	writeCalibration(file, calib);
}

void MVAComputer::writeCalibration(std::ostream &os,
                                   const Calibration::MVAComputer *calib)
{
	if (!os.good())
		throw cms::Exception("InvalidFileState")
			<< "Stream passed to MVAComputer::writeCalibration "
			   "has an invalid state." << std::endl;

	os << STANDALONE_HEADER;

	TClass *rootClass =
		TClass::GetClass("PhysicsTools::Calibration::MVAComputer");
	if (!rootClass)
		throw cms::Exception("DictionaryMissing")
			<< "CondFormats dictionary for "
			   "PhysicsTools::Calibration::MVAComputer missing"
			<< std::endl;

	TBufferFile buffer(TBuffer::kWrite);
	buffer.StreamObject(const_cast<void*>(static_cast<const void*>(calib)),
	                    rootClass);

	ext::ozstream ozs(&os);
	ozs.write(buffer.Buffer(), buffer.Length());
	ozs.flush();
}

// instantiate use cases fo MVAComputer::evalInternal

void MVAComputer::DerivContext::eval(
		const VarProcessor *proc, int *outConf, double *output,
		int *loop, unsigned int offset, unsigned int out) const
{
	proc->deriv(values(), conf(), output, outConf,
	            loop, offset, n(), out, deriv_);
}

double MVAComputer::DerivContext::output(unsigned int output,
                                         std::vector<double> &derivs) const
{
	derivs.clear();
	derivs.reserve(n_);

	int pos = conf_[output];
	if (pos >= (int)n_)
		std::copy(&deriv_.front() + n_ * (pos - n_),
		          &deriv_.front() + n_ * (pos - n_ + 1),
		          std::back_inserter(derivs));
	else {
		derivs.resize(n_);
		derivs[pos] = 1.;
	}

	return values_[pos];
}

template void MVAComputer::evalInternal(EvalContext &ctx) const;
template void MVAComputer::evalInternal(DerivContext &ctx) const;

} // namespace PhysicsTools
