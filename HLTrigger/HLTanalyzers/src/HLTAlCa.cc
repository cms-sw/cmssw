#include <iostream> 
#include <sstream> 
#include <istream> 
#include <fstream> 
#include <iomanip> 
#include <string> 
#include <cmath> 
#include <functional> 
#include <stdlib.h> 
#include <string.h> 

#include "HLTrigger/HLTanalyzers/interface/HLTAlCa.h"

HLTAlCa::HLTAlCa() {
  evtCounter=0;

  //set parameter defaults 
  _Monte=false;
  _Debug=false;
}

/*  Setup the analysis to put the branch-variables into the tree. */
void HLTAlCa::setup(const edm::ParameterSet& pSet, TTree* HltTree) {
  
  edm::ParameterSet myEmParams = pSet.getParameter<edm::ParameterSet>("RunParameters") ;
  std::vector<std::string> parameterNames = myEmParams.getParameterNames() ;
  
  for ( std::vector<std::string>::iterator iParam = parameterNames.begin();
	iParam != parameterNames.end(); iParam++ ){
    if  ( (*iParam) == "Monte" ) _Monte =  myEmParams.getParameter<bool>( *iParam );
    else if ( (*iParam) == "Debug" ) _Debug =  myEmParams.getParameter<bool>( *iParam );
  }
  
  // AlCa-specific branches of the tree 
  HltTree->Branch("ohHighestEnergyEERecHit",&ohHighestEnergyEERecHit,"ohHighestEnergyEERecHit/F");
  HltTree->Branch("ohHighestEnergyEBRecHit",&ohHighestEnergyEBRecHit,"ohHighestEnergyEBRecHit/F"); 
  HltTree->Branch("ohHighestEnergyHBHERecHit",&ohHighestEnergyHBHERecHit,"ohHighestEnergyHBHERecHit/F");  
  HltTree->Branch("ohHighestEnergyHORecHit",&ohHighestEnergyHORecHit,"ohHighestEnergyHORecHit/F");   
  HltTree->Branch("ohHighestEnergyHFRecHit",&ohHighestEnergyHFRecHit,"ohHighestEnergyHFRecHit/F");   
}

/* **Analyze the event** */
void HLTAlCa::analyze(const edm::Handle<EBRecHitCollection>             & ebrechits, 
		      const edm::Handle<EERecHitCollection>             & eerechits, 
		      const edm::Handle<HBHERecHitCollection>           & hbherechits,
		      const edm::Handle<HORecHitCollection>             & horechits,
		      const edm::Handle<HFRecHitCollection>             & hfrechits,
		      TTree* HltTree) {

  //std::cout << " Beginning HLTAlCa " << std::endl;

  ohHighestEnergyEERecHit = -1.0;
  ohHighestEnergyEBRecHit = -1.0;
  ohHighestEnergyHBHERecHit = -1.0; 
  ohHighestEnergyHORecHit = -1.0; 
  ohHighestEnergyHFRecHit = -1.0; 

  int neerechits = 0;
  int nebrechits = 0;
  int nhbherechits = 0; 
  int nhorechits = 0; 
  int nhfrechits = 0; 

  if (ebrechits.isValid()) {
    EBRecHitCollection myebrechits;
    myebrechits = * ebrechits;
    
    nebrechits = myebrechits.size();
    float ebrechitenergy = -1.0;

    typedef EBRecHitCollection::const_iterator ebrechititer;

    for (ebrechititer i=myebrechits.begin(); i!=myebrechits.end(); i++) {
      ebrechitenergy = i->energy();
      if(ebrechitenergy > ohHighestEnergyEBRecHit)
	ohHighestEnergyEBRecHit = ebrechitenergy;
    }
  }
  else {nebrechits = 0;}

  if (eerechits.isValid()) { 
    EERecHitCollection myeerechits; 
    myeerechits = * eerechits; 
 
    neerechits = myeerechits.size(); 
    float eerechitenergy = -1.0; 

    typedef EERecHitCollection::const_iterator eerechititer; 
 
    for (eerechititer i=myeerechits.begin(); i!=myeerechits.end(); i++) { 
      eerechitenergy = i->energy(); 
      if(eerechitenergy > ohHighestEnergyEERecHit) 
        ohHighestEnergyEERecHit = eerechitenergy; 
    } 
  } 
  else {neerechits = 0;} 

  if (hbherechits.isValid()) {  
    HBHERecHitCollection myhbherechits;  
    myhbherechits = * hbherechits;  
  
    nhbherechits = myhbherechits.size();  
    float hbherechitenergy = -1.0;  
 
    typedef HBHERecHitCollection::const_iterator hbherechititer;  
  
    for (hbherechititer i=myhbherechits.begin(); i!=myhbherechits.end(); i++) {  
      hbherechitenergy = i->energy();  
      if(hbherechitenergy > ohHighestEnergyHBHERecHit)  
        ohHighestEnergyHBHERecHit = hbherechitenergy;  
    }  
  }  
  else {nhbherechits = 0;}  

  if (horechits.isValid()) {   
    HORecHitCollection myhorechits;   
    myhorechits = * horechits;   
   
    nhorechits = myhorechits.size();   
    float horechitenergy = -1.0;   
  
    typedef HORecHitCollection::const_iterator horechititer;   
   
    for (horechititer i=myhorechits.begin(); i!=myhorechits.end(); i++) {   
      horechitenergy = i->energy();   
      if(horechitenergy > ohHighestEnergyHORecHit)   
        ohHighestEnergyHORecHit = horechitenergy;   
    }   
  }   
  else {nhorechits = 0;}   

  if (hfrechits.isValid()) {   
    HFRecHitCollection myhfrechits;   
    myhfrechits = * hfrechits;   
   
    nhfrechits = myhfrechits.size();   
    float hfrechitenergy = -1.0;   
  
    typedef HFRecHitCollection::const_iterator hfrechititer;   
   
    for (hfrechititer i=myhfrechits.begin(); i!=myhfrechits.end(); i++) {   
      hfrechitenergy = i->energy();   
      if(hfrechitenergy > ohHighestEnergyHFRecHit)   
        ohHighestEnergyHFRecHit = hfrechitenergy;   
    }   
  }   
  else {nhfrechits = 0;}   


  //////////////////////////////////////////////////////////////////////////////



}
