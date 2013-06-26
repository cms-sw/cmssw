#include <functional>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>
#include <memory>
#include <string>
#include <cctype>
#include <map>
#include <set>

#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string/trim.hpp>

#include <HepMC/GenEvent.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/JetMatching.h"

namespace lhef {

extern "C" {
	#define PARAMLEN 20
	namespace {
		struct Param {
			Param(const std::string &str)
			{
				int len = std::min(PARAMLEN,
				                   (int)str.length());
				std::memcpy(value, str.c_str(), len);
				std::memset(value + len, ' ', PARAMLEN - len);
			}

			char value[PARAMLEN];
		};
	}
	extern void mginit_(int *npara, Param *params, Param *values);
	extern void mgevnt_(void);
	extern void mgveto_(int *veto);

	extern struct UPPRIV {
		int	lnhin, lnhout;
		int	mscal, ievnt;
		int	ickkw, iscale;
	} uppriv_;

	extern struct MEMAIN {
		double 	etcjet, rclmax, etaclmax, qcut, clfact;
		int	maxjets, minjets, iexcfile, ktsche;
		int	nexcres, excres[30];
	} memain_;

	extern struct MEMAEV {
		double	ptclus[20];
		int	nljets, iexc, ifile;
	} memaev_;

	extern struct PYPART {
		int	npart, npartd, ipart[1000];
		double	ptpart[1000];
	} pypart_;
} // extern "C"

class JetMatchingMadgraph : public JetMatching {
    public:
	JetMatchingMadgraph(const edm::ParameterSet &params);
	~JetMatchingMadgraph();

    private:
	void init(const boost::shared_ptr<LHERunInfo> &runInfo);
	void beforeHadronisation(const boost::shared_ptr<LHEEvent> &event);

	double match(const HepMC::GenEvent *partonLevel,
	             const HepMC::GenEvent *finalState,
	             bool showeredFinalState);

	std::set<std::string> capabilities() const;

	template<typename T>
	static T parseParameter(const std::string &value);
	template<typename T>
	T getParameter(const std::string &var, const T &defValue = T()) const;

	std::map<std::string, std::string>	mgParams;

	bool					runInitialized;
	bool					eventInitialized;
	bool					soup;
	bool					exclusive;
};

template<typename T>
T JetMatchingMadgraph::parseParameter(const std::string &value)
{
	std::istringstream ss(value);
	T result;
	ss >> result;
	return result;
}

template<>
std::string JetMatchingMadgraph::parseParameter(const std::string &value)
{
	std::string result;
	if (!result.empty() && result[0] == '\'')
		result = result.substr(1);
	if (!result.empty() && result[result.length() - 1] == '\'')
		result.resize(result.length() - 1);
	return result;
}

template<>
bool JetMatchingMadgraph::parseParameter(const std::string &value_)
{
	std::string value(value_);
	std::transform(value.begin(), value.end(),
	               value.begin(), (int(*)(int))std::toupper);
	return value == "T" || value == "Y" ||
	       value == "1" || value == ".TRUE.";
}

template<typename T>
T JetMatchingMadgraph::getParameter(const std::string &var,
                                    const T &defValue) const
{
	std::map<std::string, std::string>::const_iterator pos =
							mgParams.find(var);
	if (pos == mgParams.end())
		return defValue;
	return parseParameter<T>(pos->second);
}

} // namespace lhef

using namespace lhef;

JetMatchingMadgraph::JetMatchingMadgraph(const edm::ParameterSet &params) :
	JetMatching(params),
	runInitialized(false)
{
	std::string mode = params.getParameter<std::string>("mode");
	if (mode == "inclusive") {
		soup = false;
		exclusive = false;
	} else if (mode == "exclusive") {
		soup = false;
		exclusive = true;
	} else if (mode == "auto")
		soup = true;
	else
		throw cms::Exception("Generator|LHEInterface")
			<< "Madgraph jet matching scheme requires \"mode\" "
			   "parameter to be set to either \"inclusive\", "
			   "\"exclusive\" or \"auto\"." << std::endl;

	memain_.etcjet = 0.;
	memain_.rclmax = 0.0;
	memain_.clfact = 0.0;
	memain_.iexcfile = 0;
	memain_.ktsche = 0;
	memain_.etaclmax = params.getParameter<double>("etaclmax");
	memain_.qcut = params.getParameter<double>("qcut");
	memain_.minjets = params.getParameter<int>("minjets");
	memain_.maxjets = params.getParameter<int>("maxjets");
}

JetMatchingMadgraph::~JetMatchingMadgraph()
{
}

std::set<std::string> JetMatchingMadgraph::capabilities() const
{
	std::set<std::string> result;
	result.insert("psFinalState");
	result.insert("hepevt");
	result.insert("pythia6");
	return result;
}

// implements the Madgraph method - use ME2pythia.f

void JetMatchingMadgraph::init(const boost::shared_ptr<LHERunInfo> &runInfo)
{
	// read MadGraph run card

	std::map<std::string, std::string> parameters;

	std::vector<std::string> header = runInfo->findHeader("MGRunCard");
	if (header.empty())
		throw cms::Exception("Generator|LHEInterface")
			<< "In order to use MadGraph jet matching, "
			   "the input file has to contain the corresponding "
			   "MadGraph headers." << std::endl;

	mgParams.clear();
	for(std::vector<std::string>::const_iterator iter = header.begin();
	    iter != header.end(); ++iter) {
		std::string line = *iter;
		if (line.empty() || line[0] == '#')
			continue;

		std::string::size_type pos = line.find('!');
		if (pos != std::string::npos)
			line.resize(pos);

		pos = line.find('=');
		if (pos == std::string::npos)
			continue;

		std::string var =
			boost::algorithm::trim_copy(line.substr(pos + 1));
		std::string value = 
			boost::algorithm::trim_copy(line.substr(0, pos));

		mgParams[var] = value;
	}

	// set variables in common block

	std::vector<Param> params;
	std::vector<Param> values;
	for(std::map<std::string, std::string>::const_iterator iter =
			mgParams.begin(); iter != mgParams.end(); ++iter) {
		params.push_back(" " + iter->first);
		values.push_back(iter->second);

	}

	// set MG matching parameters

	uppriv_.ickkw = getParameter<int>("ickkw", 0);

	// run Fortran initialization code

	int nparam = params.size();
	mginit_(&nparam, &params.front(), &values.front());
	runInitialized = true;
}

void JetMatchingMadgraph::beforeHadronisation(
				const boost::shared_ptr<LHEEvent> &event)
{
	if (!runInitialized)
		throw cms::Exception("Generator|LHEInterface")
			<< "Run not initialized in JetMatchingMadgraph"
			<< std::endl;

	if (uppriv_.ickkw) {
		std::vector<std::string> comments = event->getComments();
		if (comments.size() == 1) {
			std::istringstream ss(comments[0].substr(1));
			for(int i = 0; i < 1000; i++) {
				double pt;
				ss >> pt;
				if (!ss.good())
					break;
				pypart_.ptpart[i] = pt;
			}
		} else {
			edm::LogWarning("Generator|LHEInterface")
				<< "Expected exactly one comment line per "
				   "event containing MadGraph parton scale "
				   "information."
				<< std::endl;

			const HEPEUP *hepeup = event->getHEPEUP();
			for(int i = 2; i < hepeup->NUP; i++) {
				double mt2 =
					hepeup->PUP[i][0] * hepeup->PUP[i][0] +
					hepeup->PUP[i][1] * hepeup->PUP[i][1] +
					hepeup->PUP[i][4] * hepeup->PUP[i][4];
				pypart_.ptpart[i - 2] = std::sqrt(mt2);
			}
		}
	}

	mgevnt_();
	eventInitialized = true;
}

double JetMatchingMadgraph::match(const HepMC::GenEvent *partonLevel,
                                  const HepMC::GenEvent *finalState,
                                  bool showeredFinalState)
{
	if (!showeredFinalState)
		throw cms::Exception("Generator|LHEInterface")
			<< "MadGraph matching expected parton shower "
			   "final state." << std::endl;

	if (!runInitialized)
		throw cms::Exception("Generator|LHEInterface")
			<< "Run not initialized in JetMatchingMadgraph"
			<< std::endl;

	if (!eventInitialized)
		throw cms::Exception("Generator|LHEInterface")
			<< "Event not initialized in JetMatchingMadgraph"
			<< std::endl;

	if (soup)
		memaev_.iexc = (memaev_.nljets < memain_.maxjets);
	else
		memaev_.iexc = exclusive;

	int veto = 0;
	mgveto_(&veto);
	eventInitialized = false;

	return veto ? 0.0 : 1.0;
}

DEFINE_LHE_JETMATCHING_PLUGIN(JetMatchingMadgraph);
