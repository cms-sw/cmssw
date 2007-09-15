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

#include "HLTrigger/HLTanalyzers/interface/HLTMCtruth.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

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
  mcpid = new int[kMaxMcTruth];
  mcvx = new float[kMaxMcTruth];
  mcvy = new float[kMaxMcTruth];
  mcvz = new float[kMaxMcTruth];
  mcpt = new float[kMaxMcTruth];

  // MCtruth-specific branches of the tree 
  HltTree->Branch("NMCPart",&nmcpart,"NMCPart/I");
  HltTree->Branch("MCPid",mcpid,"MCPid[NMCPart]/I");
  HltTree->Branch("MCVtxX",mcvx,"MCVtxX[NMCPart]/F");
  HltTree->Branch("MCVtxY",mcvy,"MCVtxY[NMCPart]/F");
  HltTree->Branch("MCVtxZ",mcvz,"MCVtxZ[NMCPart]/F");
  HltTree->Branch("MCPt",mcpt,"MCPt[NMCPart]/F");
  HltTree->Branch("MCPtHat",&pthat,"MCPtHat/F");
  HltTree->Branch("MCmu3",&nmu3,"MCmu3/I");
  HltTree->Branch("MCbb",&nbb,"MCbb/I");
  HltTree->Branch("MCab",&nab,"MCab/I");

}

/* **Analyze the event** */
void HLTMCtruth::analyze(const CandidateCollection& mctruth,
			 TTree* HltTree) {

  //std::cout << " Beginning HLTMCtruth " << std::endl;

  if (_Monte) {
    int nmc = 0;
    int mu3 = 0;
    int mab = 0;
    int mbb = 0;

    if (&mctruth){
      //pthat = mctruth.event_scale(); // Pt-hat of the event

      for (size_t i = 0; i < mctruth.size(); ++ i) {
	const Candidate & p = (mctruth)[i];
	mcpid[nmc] = p.pdgId();
	mcpt[nmc] = p.pt();
// 	= p.eta();
// 	= p.phi();
// 	= p.mass();
	mcvx[nmc] = p.vx();
	mcvy[nmc] = p.vy();
	mcpt[nmc] = p.vz();
	if (((mcpid[nmc]==13)||(mcpid[nmc]==-13))&&(mcpt[nmc]>3.)) {mu3 += 1;} // Flag for muons with pT > 3 GeV/c
	if (mcpid[nmc]==-5) {mab += 1;} // Flag for bbar
	if (mcpid[nmc]==5) {mbb += 1;} // Flag for b
	nmc++;
      }

    }
    else {std::cout << "%HLTMCtruth -- No MC truth information" << std::endl;}
    nmcpart = nmc;
    nmu3 = mu3;
    nbb = mbb;
    nab = mab;

  }

}
