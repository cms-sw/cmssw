/* \class HiggsAnalysisSkimming
 *
 * Consult header file for description
 *
 * author:  Dominique Fortin - UC Riverside
 *
 */


// system include files
#include <HiggsAnalysis/Skimming/interface/HiggsAnalysisSkimming.h>

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>

#include <HiggsAnalysis/Skimming/interface/HiggsAnalysisSkimType.h>
#include <HiggsAnalysis/Skimming/interface/HiggsAnalysisSkimmingPluginFactory.h>

#include <FWCore/Utilities/interface/Exception.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

// C++
#include <memory>
#include <vector>

using namespace std;
using namespace edm;


// Constructor
HiggsAnalysisSkimming::HiggsAnalysisSkimming(const edm::ParameterSet& pset) {

  // Local Debug flag
  debug = pset.getParameter<bool>("DebugHiggsAnalysisSkimming");

  // Find out which skim filter to use:
  int chosenSkim = pset.getParameter<int>("skim_type");
    
  // Find appropriate ParameterSets this skim type
  std::vector<edm::ParameterSet> allSkimPSets = pset.getParameter<std::vector<edm::ParameterSet> >("skim_psets");

  // Find name of class for this skim  
  std::string skimName = allSkimPSets[chosenSkim].getParameter<std::string>("skim_name");

  // Store all parameters for this skim
  std::vector<edm::ParameterSet> skimPSets = allSkimPSets[chosenSkim].getParameter<std::vector<edm::ParameterSet> >("skim_psets");
        
  skimFilter = HiggsAnalysisSkimmingPluginFactory::get()->create(skimName, skimPSets[0]);
 
  nevt = 0;
  n1e  = 0;
  n2e  = 0;
  n1m  = 0;
  n2m  = 0;
  n4l  = 0;

}


// Destructor
HiggsAnalysisSkimming::~HiggsAnalysisSkimming() {

std::cout << "************* Selection efficiency **************" << std::endl;
std::cout << "Number of evts: " << nevt << std::endl;
std::cout << "Number of 1 e : " << n1e  << std::endl;
std::cout << "Number of 2 e : " << n2e  << std::endl;
std::cout << "Number of 1 mu: " << n1m  << std::endl;
std::cout << "Number of 2 mu: " << n2m  << std::endl;
std::cout << "Number of 4 l : " << n4l  << std::endl;

}


// Filter
bool HiggsAnalysisSkimming::filter(edm::Event& event, const edm::EventSetup& setup) {

  bool keepEvent = false;

  int whichTrig = 0;

  keepEvent = skimFilter->skim( event, setup, whichTrig );
  nevt++;
  if ( keepEvent             ) n4l++;
  if ( whichTrig%10000 > 999 ) n2m++;
  if ( whichTrig%1000  > 99  ) n1m++;
  if ( whichTrig%100   > 9   ) n2e++;
  if ( whichTrig%10    > 0   ) n1e++;

  return keepEvent;
}


