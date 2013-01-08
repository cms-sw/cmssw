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
#include "GeneratorInterface/PartonShowerVeto/interface/JetMatchingAlpgen.h"

namespace gen {

extern "C" {
    // Put here all the functions for interfacing Fortran code.
    
//	extern void alinit_();
	extern void alsetp_();
//	extern void alevnt_();
	extern void alveto_(int *ipveto);

      extern void dbpart_();
      extern void pyupre_();

	void alshcd_(char csho[3]);
	void alshen_();
} // extern "C"

// Constructor
JetMatchingAlpgen::JetMatchingAlpgen(const edm::ParameterSet &params) :
  JetMatching(params),
  applyMatching(params.getParameter<bool>("applyMatching")),
  runInitialized(false)
{
  ahopts_.etclus = params.getParameter<double>("etMin");
  ahopts_.rclus = params.getParameter<double>("drMin");
  ahopts_.iexc = params.getParameter<bool>("exclusive");
}

// Destructor   
JetMatchingAlpgen::~JetMatchingAlpgen()
{
}

std::set<std::string> JetMatchingAlpgen::capabilities() const
{
	std::set<std::string> result;
	result.insert("psFinalState");
	result.insert("hepevt");
	result.insert("pythia6");
	return result;
}

// Implements the Alpgen MLM method - use alpg_match.F

void JetMatchingAlpgen::init(const lhef::LHERunInfo* runInfo)
{

  // Read Alpgen run card stored in the LHERunInfo object.
  std::vector<std::string> headerLines = runInfo->findHeader("AlpgenUnwParFile");
  if (headerLines.empty())
    throw cms::Exception("Generator|PartonShowerVeto")
      << "In order to use Alpgen jet matching, "
      "the input file has to contain the corresponding "
      "Alpgen headers." << std::endl;
  
  // Parse the header using its bultin function.
  header.parse(headerLines.begin(), headerLines.end());

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

  std::copy(header.masses, header.masses + AlpgenHeader::MASS_MAX, ahppara_.masses);
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
  
  // When we reach this point, the run is fully initialized.
  runInitialized = true;
}

void JetMatchingAlpgen::beforeHadronisation(const lhef::LHEEvent* event)
{
  // We can't continue if the run has not been initialized.
  if (!runInitialized)
    throw cms::Exception("Generator|PartonShowerVeto")
      << "Run not initialized in JetMatchingAlpgen"
      << std::endl;

  // We are called just after LHEInterface has filled in
  // the Fortran common block (and Pythia6 called UPEVNT).
  
  // Possibly not interesting for us.
  // (except perhaps for debugging?)
  //  pyupre_();
  //  dbpart_();
  eventInitialized = true;
}

/*
int JetMatchingAlpgen::match(const HepMC::GenEvent *partonLevel,
			     const HepMC::GenEvent *finalState,
			     bool showeredFinalState)
*/
int JetMatchingAlpgen::match( const lhef::LHEEvent* partonLevel, const std::vector<fastjet::PseudoJet>* jetInput )
{

/*
  if (!showeredFinalState)
    throw cms::Exception("Generator|PartonShowerVeto")
      << "Alpgen matching expected parton shower "
      "final state." << std::endl;
*/

  if (!runInitialized)
    throw cms::Exception("Generator|PartonShowerVeto")
      << "Run not initialized in JetMatchingAlpgen"
      << std::endl;

  if (!eventInitialized)
    throw cms::Exception("Generator|PartonShowerVeto")
      << "Event not initialized in JetMatchingAlpgen"
      << std::endl;

  // If matching not required (e.g., icckw = 0), don't run the
  // FORTRAN veto code.
  if(!applyMatching) return 0;
  
  // Call the Fortran veto code. 
  int veto = 0;
  alveto_(&veto);
  
  eventInitialized = false;

  // If event was vetoed, the variable veto will contain the number 1. 
  // In this case, we must return 1 - that will be used as the return value from UPVETO.
  // If event was accepted, the variable veto will contain the number 0.
  // In this case, we must return 0 - that will be used as the return value from UPVETO.
  return veto ? 1 : 0;
}

void alshcd_(char csho[3])
{
	std::strncpy(csho, "PYT", 3);	// or "HER"
}

void alshen_()
{
}

} // end namespace gen
