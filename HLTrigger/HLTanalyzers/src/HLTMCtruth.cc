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
  mceta = new float[kMaxMcTruth];
  mcphi = new float[kMaxMcTruth];

  // MCtruth-specific branches of the tree 
  HltTree->Branch("NMCpart",&nmcpart,"NMCpart/I");
  HltTree->Branch("MCpid",mcpid,"MCpid[NMCpart]/I");
  HltTree->Branch("MCvtxX",mcvx,"MCvtxX[NMCpart]/F");
  HltTree->Branch("MCvtxY",mcvy,"MCvtxY[NMCpart]/F");
  HltTree->Branch("MCvtxZ",mcvz,"MCvtxZ[NMCpart]/F");
  HltTree->Branch("MCpt",mcpt,"MCpt[NMCpart]/F");
  HltTree->Branch("MCeta",mceta,"MCeta[NMCpart]/F");
  HltTree->Branch("MCphi",mcphi,"MCphi[NMCpart]/F");
  HltTree->Branch("MCPtHat",&pthat,"MCPtHat/F");
  HltTree->Branch("MCmu3",&nmu3,"MCmu3/I");
  HltTree->Branch("MCel1",&nel1,"MCel1/I");
  HltTree->Branch("MCbb",&nbb,"MCbb/I");
  HltTree->Branch("MCab",&nab,"MCab/I");

}

/* **Analyze the event** */
void HLTMCtruth::analyze(const CandidateCollection& mctruth,
			 const HepMC::GenEvent hepmc,
			 TTree* HltTree) {

  //std::cout << " Beginning HLTMCtruth " << std::endl;

  if (_Monte) {
    int nmc = 0;
    int mu3 = 0;
    int el1 = 0;
    int mab = 0;
    int mbb = 0;

    if (&hepmc){
      pthat = hepmc.event_scale(); // Pt-hat of the event
    }

     if (&mctruth){

      for (size_t i = 0; i < mctruth.size(); ++ i) {
	const Candidate & p = (mctruth)[i];
	mcpid[nmc] = p.pdgId();
	mcpt[nmc] = p.pt();
	mceta[nmc] = p.eta();
	mcphi[nmc] = p.phi();
// 	= p.mass();
	mcvx[nmc] = p.vx();
	mcvy[nmc] = p.vy();
	mcvz[nmc] = p.vz();
	if (((mcpid[nmc]==13)||(mcpid[nmc]==-13))&&(mcpt[nmc]>2.5)) {mu3 += 1;} // Flag for muons with pT > 2.5 GeV/c
	if (((mcpid[nmc]==11)||(mcpid[nmc]==-11))&&(mcpt[nmc]>1.)) {el1 += 1;} // Flag for electrons with pT > 1 GeV/c
	if (mcpid[nmc]==-5) {mab += 1;} // Flag for bbar
	if (mcpid[nmc]==5) {mbb += 1;} // Flag for b
	nmc++;
      }

    }
    else {std::cout << "%HLTMCtruth -- No MC truth information" << std::endl;}

    nmcpart = nmc;
    nmu3 = mu3;
    nel1 = el1;
    nbb = mbb;
    nab = mab;

  }

}
