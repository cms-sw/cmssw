#include <iostream>
#include <cstring>
#include <vector>
#include <memory>
#include <string>

#include <HepMC/GenEvent.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/JetMatching.h"

#include "GeneratorInterface/AlpgenInterface/interface/AlpgenHeader.h"
#include "GeneratorInterface/AlpgenInterface/interface/AlpgenCommonBlocks.h"

extern "C" {
  // Put here all the functions for interfacing Fortran code.
  
//	extern void alinit_();
	extern void alsetp_();
//	extern void alevnt_();
	extern void alveto_(int *ipveto);
//      extern void dbpart_();
//      extern void dbinrd_();
//      extern void pyupre_();

	void alshcd_(char csho[3]);
	void alshen_();
} // extern "C"

using namespace lhef;

class JetMatchingAlpgen : public JetMatching {
    public:
	JetMatchingAlpgen(const edm::ParameterSet &params);
	~JetMatchingAlpgen();

    private:
	void init(const boost::shared_ptr<LHERunInfo> &runInfo);
	void beforeHadronisation(const boost::shared_ptr<LHEEvent> &event);

	double match(const HepMC::GenEvent *partonLevel,
	             const HepMC::GenEvent *finalState,
	             bool showeredFinalState);

	std::set<std::string> capabilities() const;

        bool            applyMatching;
	bool		runInitialized;
	bool		eventInitialized;

	AlpgenHeader	header;
};

JetMatchingAlpgen::JetMatchingAlpgen(const edm::ParameterSet &params) :
	JetMatching(params),
	applyMatching(params.getParameter<bool>("applyMatching")),
	runInitialized(false)
{
	ahopts_.etclus = params.getParameter<double>("etMin");
	ahopts_.rclus = params.getParameter<double>("drMin");
	ahopts_.iexc = params.getParameter<bool>("exclusive");
}

JetMatchingAlpgen::~JetMatchingAlpgen()
{
}

std::set<std::string> JetMatchingAlpgen::capabilities() const
{
	std::set<std::string> result;
	result.insert("psFinalState");
	result.insert("hepevt");
	result.insert("pythia6");
	// we could remove Pythia6 dependency and run on all hepevt
	// generators (actually Pythia6 + Herwig6) since Alpgen actually
	// supports both
	return result;
}

// implements the Alpgen MLM method - use alpg_match.F

void JetMatchingAlpgen::init(const boost::shared_ptr<LHERunInfo> &runInfo)
{
	// read Alpgen run card

	std::vector<std::string> headerLines =
				runInfo->findHeader("AlpgenUnwParFile");
	if (headerLines.empty())
		throw cms::Exception("Generator|LHEInterface")
			<< "In order to use Alpgen jet matching, "
			   "the input file has to contain the corresponding "
			   "Alpgen headers." << std::endl;

	header.parse(headerLines.begin(), headerLines.end());

	// process the header information
	// I don't want to print this right now.
	
// 	std::cout << "Alpgen header" << std::endl;
// 	std::cout << "========================" << std::endl;
// 	std::cout << "\tihrd = " << header.ihrd << std::endl;
// 	std::cout << "\tmc = " << header.masses[AlpgenHeader::mc]
// 	          << ", mb = " << header.masses[AlpgenHeader::mb]
// 	          << ", mt = " << header.masses[AlpgenHeader::mt]
// 	          << ", mw = " << header.masses[AlpgenHeader::mw]
// 	          << ", mz = " << header.masses[AlpgenHeader::mz]
// 	          << ", mh = " << header.masses[AlpgenHeader::mh]
// 	          << std::endl;
// 	for(std::map<AlpgenHeader::Parameter, double>::const_iterator iter =
// 		header.params.begin(); iter != header.params.end(); ++iter)
// 		std::cout << "\t" << AlpgenHeader::parameterName(iter->first)
// 		          << " = " << iter->second << std::endl;
// 	std::cout << "\txsec = " << header.xsec
// 	          << " +-" << header.xsecErr << std::endl;
// 	std::cout << "========================" << std::endl;
	
	// Here we pass a few header variables to common block and 
	// call Alpgen init routine to do the rest.
	// The variables passed first are the ones directly that
	// need to be set up "manually": IHRD and the masses.
	// (ebeam is set just to mimic the original code)
	// Afterwards, we pass the full spectrum of Alpgen
	// parameters directly into the AHPARS structure, to be
	// treated by AHSPAR which is called inside alsetp_().

	std::copy(header.masses, header.masses + AlpgenHeader::MASS_MAX,
	          ahppara_.masses);
	ahppara_.ihrd = header.ihrd;
	ahppara_.ebeam = header.ebeam;
	for(std::map<AlpgenHeader::Parameter, double>::const_iterator iter =
		header.params.begin(); iter != header.params.end(); ++iter) {

		if (iter->first <= 0 || iter->first >= (int)AHPARS::nparam - 1)
			continue;

		ahpars_.parval[(int)iter->first - 1] = iter->second;
	}

	// Run the rest of the setup.
	alsetp_();

	runInitialized = true;
}

void JetMatchingAlpgen::beforeHadronisation(
				const boost::shared_ptr<LHEEvent> &event)
{
	if (!runInitialized)
		throw cms::Exception("Generator|LHEInterface")
			<< "Run not initialized in JetMatchingAlpgen"
			<< std::endl;

	// we are called just after LHEInterface has filled in
	// the Fortran common block (and Pythia6 called UPEVNT)

	// possibly not interesting for us
	// (except perhaps for debugging?)
	//pyupre_();
	//dbpart_();
	
	eventInitialized = true;
}

double JetMatchingAlpgen::match(const HepMC::GenEvent *partonLevel,
                                  const HepMC::GenEvent *finalState,
                                  bool showeredFinalState)
{
	if (!showeredFinalState)
		throw cms::Exception("Generator|LHEInterface")
			<< "Alpgen matching expected parton shower "
			   "final state." << std::endl;

	if (!runInitialized)
		throw cms::Exception("Generator|LHEInterface")
			<< "Run not initialized in JetMatchingAlpgen"
			<< std::endl;

	if (!eventInitialized)
		throw cms::Exception("Generator|LHEInterface")
			<< "Event not initialized in JetMatchingAlpgen"
			<< std::endl;

	// If matching not required (e.g., icckw = 0), don't run the
	// FORTRAN veto code.
	if(!applyMatching) return 1.0;

	// Call the Fortran veto code and set veto to 1 if vetoed.
	int veto = 0;
	alveto_(&veto);

	eventInitialized = false;

	return veto ? 0.0 : 1.0;
}

void alshcd_(char csho[3])
{
	std::strncpy(csho, "PYT", 3);	// or "HER"
}

void alshen_()
{
}

DEFINE_LHE_JETMATCHING_PLUGIN(JetMatchingAlpgen);
