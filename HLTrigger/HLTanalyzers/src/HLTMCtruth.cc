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

#include "HLTrigger/HLTAnalyzers/interface/HLTMCtruth.h"

HLTMCtruth::HLTMCtruth() {

  //set parameter defaults 
  _Monte=false;
  _Debug=false;
}

/*  Setup the analysis to put the branch-variables into the tree. */
void HLTMCtruth::setup(const edm::ParameterSet& pSet, TTree* HltTree) {

  edm::ParameterSet myMCParams = pSet.getParameter<edm::ParameterSet>("RunParameters") ;
  vector<std::string> parameterNames = myMCParams.getParameterNames() ;
  
  for ( vector<std::string>::iterator iParam = parameterNames.begin();
	iParam != parameterNames.end(); iParam++ ){
    if  ( (*iParam) == "Monte" ) _Monte =  myMCParams.getParameter<bool>( *iParam );
    else if ( (*iParam) == "Debug" ) _Debug =  myMCParams.getParameter<bool>( *iParam );
  }

  const int kMaxMcTruth = 10000;
  mcpid = new float[kMaxMcTruth];
  mcvx = new float[kMaxMcTruth];
  mcvy = new float[kMaxMcTruth];
  mcvz = new float[kMaxMcTruth];
  mcpt = new float[kMaxMcTruth];

  // MCtruth-specific branches of the tree 
  HltTree->Branch("NobjMCPart",&nmcpart,"NobjMCPart/I");
  HltTree->Branch("MCPid",mcpid,"MCPid[NobjMCPart]/I");
  HltTree->Branch("MCVtxX",mcvx,"MCVtxX[NobjMCPart]/F");
  HltTree->Branch("MCVtxY",mcvy,"MCVtxY[NobjMCPart]/F");
  HltTree->Branch("MCVtxZ",mcvz,"MCVtxZ[NobjMCPart]/F");
  HltTree->Branch("MCPt",mcpt,"MCPt[NobjMCPart]/F");

}

/* **Analyze the event** */
void HLTMCtruth::analyze(const HepMC::GenEvent mctruth,
			 TTree* HltTree) {

  //std::cout << " Beginning HLTMCtruth " << std::endl;

  if (_Monte) {
    int nmc = 0;
    if (&mctruth){
      for (HepMC::GenEvent::particle_const_iterator partIter = mctruth.particles_begin();
	   partIter != mctruth.particles_end();++partIter) {
	CLHEP::HepLorentzVector creation = (*partIter)->CreationVertex();
	CLHEP::HepLorentzVector momentum = (*partIter)->Momentum();
	HepPDT::ParticleID id = (*partIter)->particleID();  // electrons and positrons are 11 and -11
	mcpid[nmc] = id.pid(); 
	mcvx[nmc] = creation.x();  
	mcvy[nmc] = creation.y();
	mcvz[nmc] = creation.z();
	mcpt[nmc] = momentum.perp();
	nmc++;
      }
    }
    else {std::cout << "%HLTMCtruth -- No MC truth information" << std::endl;}
    nmcpart = nmc;

  }

}
