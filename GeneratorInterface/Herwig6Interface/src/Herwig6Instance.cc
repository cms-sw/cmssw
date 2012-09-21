#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <string>
#include <cassert>

#ifdef _POSIX_C_SOURCE
#	include <sys/time.h>
#	include <signal.h>
#	include <setjmp.h>
#endif

#include <CLHEP/Random/RandomEngine.h>

#include <HepMC/HerwigWrapper.h>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "GeneratorInterface/Core/interface/RNDMEngineAccess.h"

#include "GeneratorInterface/Herwig6Interface/interface/herwig.h"
#include "GeneratorInterface/Herwig6Interface/interface/Herwig6Instance.h"

#include "params.inc"

extern "C" void jminit_();
__attribute__((visibility("hidden"))) void dummy()
{
  int dummyInt = 0;
  jminit_();
  jimmin_();
  jminit_();
  hwmsct_(&dummyInt);
  jmefin_();
}

// Added for reading in the paremeter files (orig. L. Sonnenschein)
extern "C" void lunread_(const char filename[], const int length);

using namespace gen;

// implementation for the Fortran callbacks from Herwig

// this gets a random number from the currently running instance
// so, if the have separate instances for let's say decayers based on
// Herwig than the Herwig pp collision instance, we get independent
// random numbers.  Also true for FastSim, since FastSim uses its own
// random numbers.  This means FastSim needs take a Herwig6Instance
// instance of its own instead of calling into Herwig directly.
double gen::hwrgen_(int *idummy)
{ 
  Herwig6Instance *instance = FortranInstance::getInstance<Herwig6Instance>();
  assert(instance != 0);
  assert(instance->randomEngine != 0);
  return instance->randomEngine->flat();
}

void gen::cms_hwwarn_(char fn[6], int *code, int *exit)
{
	std::string function(fn, fn + sizeof fn);
	*exit = FortranInstance::getInstance<Herwig6Instance>()->hwwarn(function, *code);
}

extern "C" {
	void hwaend_()
	{}

	void cmsending_(int *ecode)
	{
		throw cms::Exception("Herwig6Error")
			<< "Herwig6 stopped run with error code " << *ecode
			<< "." << std::endl;
	}
}

// Herwig6Instance methods

Herwig6Instance::Herwig6Instance(CLHEP::HepRandomEngine *randomEngine) :
	randomEngine(randomEngine ? randomEngine : &getEngineReference()),
	timeoutPrivate(0)
{
}

Herwig6Instance::Herwig6Instance(int dummy) :
	randomEngine(0),
	timeoutPrivate(0)
{
}

Herwig6Instance::~Herwig6Instance()
{
}

// timeout tool

#ifdef _POSIX_C_SOURCE
// some deep POSIX hackery to catch HERWIG sometimes (O(10k events) with
// complicated topologies) getting caught in and endless loop :-(

void Herwig6Instance::_timeout_sighandler(int signr)
{
  Herwig6Instance *instance = FortranInstance::getInstance<Herwig6Instance>();
  assert(instance != 0);
  assert(instance->timeoutPrivate != 0);
  siglongjmp(*(sigjmp_buf*)instance->timeoutPrivate, 1);
}

bool Herwig6Instance::timeout(unsigned int secs, void (*fn)())
{
	if (timeoutPrivate)
		throw cms::Exception("ReentrancyProblem")
			<< "Herwig6Instance::timeout() called recursively."
			<< std::endl;
	struct sigaction saOld;
	std::memset(&saOld, 0, sizeof saOld);

	struct itimerval itv;
	timerclear(&itv.it_value);
	timerclear(&itv.it_interval);
	itv.it_value.tv_sec = 0;
	itv.it_interval.tv_sec = 0;
	setitimer(ITIMER_VIRTUAL, &itv, NULL);

	sigset_t ss;
	sigemptyset(&ss);
	sigaddset(&ss, SIGVTALRM);

	sigprocmask(SIG_UNBLOCK, &ss, NULL);
	sigprocmask(SIG_BLOCK, &ss, NULL);

	timeoutPrivate = new sigjmp_buf;
	if (sigsetjmp(*(sigjmp_buf*)timeoutPrivate, 1)) {
		delete (sigjmp_buf*)timeoutPrivate;
		timeoutPrivate = 0;

		itv.it_value.tv_sec = 0;
		itv.it_interval.tv_sec = 0;
		setitimer(ITIMER_VIRTUAL, &itv, NULL);
		sigprocmask(SIG_UNBLOCK, &ss, NULL);
		return true;
	}

	itv.it_value.tv_sec = secs;
	itv.it_interval.tv_sec = secs;
	setitimer(ITIMER_VIRTUAL, &itv, NULL);

	struct sigaction sa;
	std::memset(&sa, 0, sizeof sa);
	sa.sa_handler = &Herwig6Instance::_timeout_sighandler;
	sa.sa_flags = SA_ONESHOT;
	sigemptyset(&sa.sa_mask);

	sigaction(SIGVTALRM, &sa, &saOld);
	sigprocmask(SIG_UNBLOCK, &ss, NULL);

	try {
		fn();
	} catch(...) {
		delete (sigjmp_buf*)timeoutPrivate;
		timeoutPrivate = 0;

		itv.it_value.tv_sec = 0;
		itv.it_interval.tv_sec = 0;
		setitimer(ITIMER_VIRTUAL, &itv, NULL);

		sigaction(SIGVTALRM, &saOld, NULL);

		throw;
	}

	delete (sigjmp_buf*)timeoutPrivate;
	timeoutPrivate = 0;

	itv.it_value.tv_sec = 0;
	itv.it_interval.tv_sec = 0;
	setitimer(ITIMER_VIRTUAL, &itv, NULL);

	sigaction(SIGVTALRM, &saOld, NULL);

	return false;
}
#else
bool Herwig6Instance::timeout(unsigned int secs, void (*fn)())
{
	fn();
	return false;
}
#endif

bool Herwig6Instance::hwwarn(const std::string &fn, int code)
{
	return false;
}

// regular Herwig6Instance methods

bool Herwig6Instance::give(const std::string &line)
{	
	typedef std::istringstream::traits_type traits;

	const char *p = line.c_str(), *q;
	p += std::strspn(p, " \t\r\n");

	for(q = p; std::isalnum(*q); q++);
	std::string name(p, q - p);

	const ConfigParam *param;
	for(param = configParams; param->name; param++)
		if (name == param->name)
			break;
	if (!param->name)
		return false;

	p = q + std::strspn(q, " \t\r\n");  

	std::size_t pos = 0;
	std::size_t mult = 1;
	for(unsigned int i = 0; i < 3; i++) {
		if (!param->dim[i].size)
			break;

		if (*p++ != (i ? ',' : '('))
			return false;

		p += std::strspn(p, " \t\r\n");

		for(q = p; std::isdigit(*q); q++);
		std::istringstream ss(std::string(p, q - p));
		std::size_t index;
		ss >> index;
		if (ss.bad() || ss.peek() != traits::eof())
			return false;

		if (index < param->dim[i].offset)
			return false;
		index -= param->dim[i].offset;
		if (index >= param->dim[i].size)
			return false;

		p = q + std::strspn(q, " \t\r\n");

		pos += mult * index;
		mult *= param->dim[i].size;
	}

	if (param->dim[0].size) {
		if (*p++ != ')')
			return false;
		p += std::strspn(p, " \t\r\n");
	}

	if (*p++ != '=')
		return false;
	p += std::strspn(p, " \t\r\n");

	for(q = p; *q && (std::isalnum(*q) || std::strchr(".-+", *q)); q++);
	std::istringstream ss(std::string(p, q - p));

	p = q + std::strspn(q, " \t\r\n");
	if (*p && *p != '!')
		return false;

	switch(param->type) {
	    case kInt: {
		int value;
		ss >> value;
		if (ss.bad() || ss.peek() != traits::eof())
			return false;

		((int*)param->ptr)[pos] = value;
		break;
	    }
	    case kDouble: {
		double value;
		ss >> value;
		if (ss.bad() || ss.peek() != traits::eof())
			return false;

		((double*)param->ptr)[pos] = value;
		break;
	    }
	    case kLogical: {
		std::string value_;
		ss >> value_;
		if (ss.bad() || ss.peek() != traits::eof())
			return false;

		for(std::string::iterator iter = value_.begin();
		    iter != value_.end(); ++iter)
			*iter = std::tolower(*iter);
		bool value;
		if (value_ == "yes" || value_ == "true" || value_ == "1")
			value = true;
		else if (value_ == "no" || value_ == "false" || value_ == "0")
			value = false;
		else
			return false;

		((int*)param->ptr)[pos] = value;
		break;
	    }
	}

	return true;
}

void Herwig6Instance::openParticleSpecFile(const std::string fileName) {

  edm::FileInPath fileAndPath( fileName );
  // WARING : This will call HWWARN if file does not exist.
  lunread_( fileAndPath.fullPath().c_str(),strlen(fileAndPath.fullPath().c_str()) );
  
  return;
}


