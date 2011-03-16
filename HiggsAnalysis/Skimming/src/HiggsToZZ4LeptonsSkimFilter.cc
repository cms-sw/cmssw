
/* \class HiggsTo4LeptonsSkimFilter
 *
 * Consult header file for description
 *
 * author:  N. De Filippis - Politecnico and INFN Bari
 *
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

  // HLT
  useHLT              = pset.getUntrackedParameter<bool>("useHLT");
  HLTinst_            = pset.getParameter<string>("HLTinst");
  HLTflag_            = pset.getParameter<vector<string> >("HLTflag");

  // DiLepton
  useDiLeptonSkim     = pset.getUntrackedParameter<bool>("useDiLeptonSkim");
  SkimDiLeptoninst_   = pset.getParameter<string>("SkimDiLeptoninst");
  SkimDiLeptonflag_   = pset.getParameter<string>("SkimDiLeptonflag");

  // TriLepton
  useTriLeptonSkim    = pset.getUntrackedParameter<bool>("useTriLeptonSkim");
  SkimTriLeptoninst_  = pset.getParameter<string>("SkimTriLeptoninst");
  SkimTriLeptonflag_  = pset.getParameter<string>("SkimTriLeptonflag");

  nSelectedEvents.clear();
  nSelectedSkimEvents = 0;
  nDiLeptonSkimEvents = 0;
  nTriLeptonSkimEvents = 0;

}


// Destructor
HiggsToZZ4LeptonsSkimFilter::~HiggsToZZ4LeptonsSkimFilter() {

  std::cout << "HiggsToZZ4LeptonsSkimFilter: \n" 
  << " N_events_read  = " << nSelectedEvents.at(0)
  << " N_events_HLT   = " << nSelectedEvents.at(1)
  << " N_events_DiLeptonSkim   = "  << nDiLeptonSkimEvents
  << " N_events_TriLeptonSkim   = " << nTriLeptonSkimEvents
  << " N_events_Skim  = " << nSelectedSkimEvents
  << std::endl;
}



// Filter event
bool HiggsToZZ4LeptonsSkimFilter::filter(edm::Event& event, const edm::EventSetup& setup ) {

  nSelectedEvents.resize(HLTflag_.size());

  size_t HLTsize=1;

  if (useHLT){   
	HLTsize=HLTflag_.size();
  }
  
  // Get the HLT flag
  for (size_t i=0;i<HLTsize;i++){
    edm::Handle<bool> HLTboolHandle;
    event.getByLabel(HLTinst_.c_str(),HLTflag_.at(i).c_str(), HLTboolHandle);
  
    if ( *HLTboolHandle.product()==1 )  {
         nSelectedEvents.at(i)++;
    }
    else {
      return false;
    }
  }
  
  // DiLepton
  bool DiLeptonpassed=false;
  bool TriLeptonpassed=false;

  if (useDiLeptonSkim){ 
    edm::Handle<bool> SkimDiboolHandle;
    event.getByLabel(SkimDiLeptoninst_.c_str(),SkimDiLeptonflag_.c_str(), SkimDiboolHandle);
    if ( *SkimDiboolHandle.product()==1 )  {
      DiLeptonpassed=true;
      nDiLeptonSkimEvents++;
      if (!useTriLeptonSkim) nSelectedSkimEvents++;    
    }
    else {
      if (!useTriLeptonSkim)  return false;
    }
  } 
 
  // TriLepton
  if (useTriLeptonSkim){ 
    edm::Handle<bool> SkimTriboolHandle;
    event.getByLabel(SkimTriLeptoninst_.c_str(),SkimTriLeptonflag_.c_str(), SkimTriboolHandle);
    
    if ( *SkimTriboolHandle.product()==1 )  {
      TriLeptonpassed=true;
      nTriLeptonSkimEvents++;
      if (!useDiLeptonSkim) nSelectedSkimEvents++;
    }	
    else {
      if (!useDiLeptonSkim)  return false;
    }
  }
  
  // DiLepton OR TriLepton
  if (useDiLeptonSkim && useTriLeptonSkim) {
    if (DiLeptonpassed ||  TriLeptonpassed){
      nSelectedSkimEvents++;
    }
    else return false;
  }

  return true;

}


