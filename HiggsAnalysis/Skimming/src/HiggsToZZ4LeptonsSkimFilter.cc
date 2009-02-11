
/* \class HiggsTo4LeptonsSkimFilter
 *
 * Consult header file for description
 *
 * author:  Dominique Fortin - UC Riverside
 * modified by N. De Filippis - LLR - Ecole Polytechnique
 */


// system include files
#include <HiggsAnalysis/Skimming/interface/HiggsToZZ4LeptonsSkimFilter.h>

// User include files
#include <FWCore/ParameterSet/interface/ParameterSet.h>

// C++
#include <iostream>
#include <vector>

using namespace std;
using namespace edm;

// Constructor
HiggsToZZ4LeptonsSkimFilter::HiggsToZZ4LeptonsSkimFilter(const edm::ParameterSet& pset) {

  // Local Debug flag
  HLTinst_            = pset.getParameter<string>("HLTinst");
  HLTflag_            = pset.getParameter<vector<string> >("HLTflag");
  Skiminst_           = pset.getParameter<string>("Skiminst");
  Skimflag_           = pset.getParameter<string>("Skimflag");
  nSelectedEvents.clear();
  nSelectedSkimEvents = 0;

}


// Destructor
HiggsToZZ4LeptonsSkimFilter::~HiggsToZZ4LeptonsSkimFilter() {

  std::cout << "HiggsToZZ4LeptonsSkimFilter: \n" 
  << " N_events_read= " << nSelectedEvents.at(0)
  << " N_events_HLT= " <<  nSelectedEvents.at(1)
  << " N_events_Skim= " << nSelectedSkimEvents
  << "     RelEfficiency4lFilter= "     << double(nSelectedSkimEvents)/double(nSelectedEvents.at(1)) 
  << "     AbsEfficiencySkim4lFilter= " << double(nSelectedSkimEvents)/double(nSelectedEvents.at(0))            
  << std::endl;
}



// Filter event
bool HiggsToZZ4LeptonsSkimFilter::filter(edm::Event& event, const edm::EventSetup& setup ) {

  nSelectedEvents.resize(HLTflag_.size());

  // Get the HLT flag
  for (size_t i=0;i<HLTflag_.size();i++){
   edm::Handle<bool> HLTboolHandle;
   event.getByLabel(HLTinst_.c_str(),HLTflag_.at(i).c_str(), HLTboolHandle);
  
   if ( *HLTboolHandle.product()==1 )  {
         nSelectedEvents.at(i)++;
   }
   else {
      return false;
   }
  }

  edm::Handle<bool> SkimboolHandle;
  event.getByLabel(Skiminst_.c_str(),Skimflag_.c_str(), SkimboolHandle);

  if ( *SkimboolHandle.product()==1 )  {
	nSelectedSkimEvents++;
  }	
  else {
      return false;
  }
  
  return true;

}


