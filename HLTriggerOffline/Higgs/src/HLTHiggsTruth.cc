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

#include "HLTriggerOffline/Higgs/interface/HLTHiggsTruth.h"

HLTHiggsTruth::HLTHiggsTruth() {

  //set parameter defaults 
  _Monte=false;
  _Debug=false;
  isvisible=false;
}

/*  Setup the analysis to put the branch-variables into the tree. */
void HLTHiggsTruth::setup(const edm::ParameterSet& pSet, TTree* HltTree) {

  edm::ParameterSet myMCParams = pSet.getParameter<edm::ParameterSet>("RunParameters") ;
  vector<std::string> parameterNames = myMCParams.getParameterNames() ;
  
  for ( vector<std::string>::iterator iParam = parameterNames.begin();
	iParam != parameterNames.end(); iParam++ ){
    if  ( (*iParam) == "Monte" ) _Monte =  myMCParams.getParameter<bool>( *iParam );
    else if ( (*iParam) == "Debug" ) _Debug =  myMCParams.getParameter<bool>( *iParam );
  }

//  const int kMaxMcTruth = 50000;
//  mcpid = new float[kMaxMcTruth];
//  mcvx = new float[kMaxMcTruth];
//  mcvy = new float[kMaxMcTruth];
//  mcvz = new float[kMaxMcTruth];
//  mcpt = new float[kMaxMcTruth];

  // MCtruth-specific branches of the tree 
//  HltTree->Branch("NobjMCPart",&nmcpart,"NobjMCPart/I");
//  HltTree->Branch("MCPid",mcpid,"MCPid[NobjMCPart]/I");
//  HltTree->Branch("MCVtxX",mcvx,"MCVtxX[NobjMCPart]/F");
//  HltTree->Branch("MCVtxY",mcvy,"MCVtxY[NobjMCPart]/F");
//  HltTree->Branch("MCVtxZ",mcvz,"MCVtxZ[NobjMCPart]/F");
//  HltTree->Branch("MCPt",mcpt,"MCPt[NobjMCPart]/F");

}

/* **Analyze the event** */
void HLTHiggsTruth::analyzeHWW2l(const CandidateCollection& mctruth,TTree* HltTree) {
  if (_Monte) {
//    int nleptons=0;
    if (&mctruth){
      double maxpt_pos = 0.0;
      double maxpt_neg = 0.0;
      for (size_t i = 0; i < mctruth.size(); ++ i) {
	const Candidate & p = (mctruth)[i];
        int status = p.status();
	if (status==1) {
          int pid=p.pdgId();
	  bool ismuon = abs(pid)==13;
	  bool iselectron = abs(pid)==11;
	  bool islepton = ismuon || iselectron;
// old         bool inacceptance = (abs(p.eta()) < 2.0);
          bool inacceptance = (abs(p.eta()) < 2.5);
// should request only 1 lepton >20, the other above 10. Take 2 hardest ones.
	  bool aboveptcut = (p.pt() > 10.0);
//	  if (islepton && inacceptance && aboveptcut) {
//	    if (nleptons==0) {
//	      if (pid<0) {
//	        nleptons=-1;
//              } else {
//	        nleptons=1;
//              }
//            } else if (abs(nleptons)==1) {
//	      if (nleptons*pid<0) {
//	        nleptons=2;
//            }
//          }
//        }
          if (islepton && inacceptance && aboveptcut) {
	    if (pid<0 && p.pt() > maxpt_neg) {
	      maxpt_neg = p.pt();
	    } else if (pid>0 && p.pt() > maxpt_pos) {
	      maxpt_pos = p.pt();
            }
	  }    
        }
      }
//      isvisible = nleptons==2; 
      isvisible = (maxpt_neg>20.0 || maxpt_pos>20.0);
    }
    else {std::cout << "%HLTHiggsTruth -- No MC truth information" << std::endl;}
//    std::cout << "nleptons = " << nleptons << " " << isvisible_WW << std::endl;
  }
}

void HLTHiggsTruth::analyzeHZZ4l(const CandidateCollection& mctruth,TTree* HltTree) {
  if (_Monte) {
    int nmuons=0;
    int nelectrons=0;
    int nmupair=0;
    int nepair=0;
    if (&mctruth){
      for (size_t i = 0; i < mctruth.size(); ++ i) {
	const Candidate & p = (mctruth)[i];
        int status = p.status();
	if (status==1) {
          int pid=p.pdgId();
	  bool ismuon = abs(pid)==13;
	  bool iselectron = abs(pid)==11;
	  bool istau = abs(pid)==15;
	  if (istau) std::cout << " Found unexpacted tau " << std::endl;
          bool inacceptance = (abs(p.eta()) < 2.4 && ismuon) || (abs(p.eta()) < 2.5 && iselectron);
	  bool aboveptcut = (p.pt() > 3.0 && ismuon) || (p.pt() > 5.0 && iselectron);
	  if (inacceptance && aboveptcut) {
	    // have to look for 2 pairs of opposite charge, either mu:s or e:s
	    if (iselectron) {
	      if (nelectrons>0 && pid<0) {
	        nepair++;
		nelectrons--;
              } else if (nelectrons<0 && pid>0) {
	        nepair++;
		nelectrons++;
              } else {
                nelectrons=nelectrons + int(pid/abs(pid));
              }
            } else if (ismuon) {
	      if (nmuons>0 && pid<0) {
	        nmupair++;
		nmuons--;
              } else if (nmuons<0 && pid>0) {
	        nmupair++;
		nmuons++;
              } else {
                nmuons=nmuons + int(pid/abs(pid));
              }
            }
          }
        }
      }
      // request 2+2, 4+0 or 0+4  opposite charge leptons, i.e. 2 pairs
      isvisible = (nmupair + nepair) >= 2; 
    }
    else {std::cout << "%HLTHiggsTruth -- No MC truth information" << std::endl;}
//    std::cout << "nepair, nmupair = " << nepair << ", " << nmupair << " " << isvisible << std::endl;
  }
}

void HLTHiggsTruth::analyzeHgg(const CandidateCollection& mctruth,TTree* HltTree) {
  if (_Monte) {
    int nphotons=0;
    if (&mctruth){
      for (size_t i = 0; i < mctruth.size(); ++ i) {
	const Candidate & p = (mctruth)[i];
        int status = p.status();
	if (status==1) {
          int pid=p.pdgId();
	  bool isphoton = pid==22;
          bool inacceptance = (abs(p.eta()) < 2.5);
	  bool aboveptcut = (p.pt() > 5.0);
	  if (inacceptance && aboveptcut && isphoton) {
	      nphotons++;
          }
        }
      }
      // request 2 photons (or more)
      isvisible = nphotons > 1; 
    }
    else {std::cout << "%HLTHiggsTruth -- No MC truth information" << std::endl;}
//    std::cout << "nphotons = " << nphotons << " " << isvisible_gg << std::endl;
  }
}

void HLTHiggsTruth::analyzeA2mu(const CandidateCollection& mctruth,TTree* HltTree) {
  if (_Monte) {
    int nmuons=0;
    if (&mctruth){
      for (size_t i = 0; i < mctruth.size(); ++ i) {
	const Candidate & p = (mctruth)[i];
        int status = p.status();
	if (status==1) {
          int pid=p.pdgId();
	  bool ismuon = abs(pid)==13;
          bool inacceptance = (abs(p.eta()) < 2.4);
	  bool aboveptcut = (p.pt() > 3.0);
	  if (inacceptance && aboveptcut && ismuon) {
	    if (nmuons==0) {
	      nmuons=int(pid/abs(pid));
	    } else if (pid<0 && nmuons==1) {
	      nmuons=2;
	    } else if (pid>0 && nmuons==-1) {
	      nmuons=2;
            }
          }
        }
      }
      // request 2  opposite charge muons
      isvisible = nmuons==2; 
    }
    else {std::cout << "%HLTHiggsTruth -- No MC truth information" << std::endl;}
//    std::cout << "nmuons = " << nmuons << " " << isvisible_2mu << std::endl;
  }
}


void HLTHiggsTruth::analyzeH2tau(const CandidateCollection& mctruth,TTree* HltTree) {
  if (_Monte) {
    int ntaus=0;
    if (&mctruth){
      for (size_t i = 0; i < mctruth.size(); ++ i) {
	const Candidate & p = (mctruth)[i];
        int status = p.status();
	if (status==1 || status==2) {
          int pid=p.pdgId();
	  bool istau = abs(pid)==15;
          bool inacceptance = (abs(p.eta()) < 2.4);
	  bool aboveptcut = (p.pt() > 3.0);
	  if (inacceptance && aboveptcut && istau) {
	    if (ntaus==0) {
	      ntaus=int(pid/abs(pid));
	    } else if (pid<0 && ntaus==1) {
	      ntaus=2;
	    } else if (pid>0 && ntaus==-1) {
	      ntaus=2;
            }
          }
        }
      }
      // request 2  opposite charge taus
      isvisible = ntaus==2; 
    }
    else {std::cout << "%HLTHiggsTruth -- No MC truth information" << std::endl;}
//    std::cout << "ntaus = " << ntaus << " " << isvisible << std::endl;
  }
}

void HLTHiggsTruth::analyzeHtaunu(const CandidateCollection& mctruth,TTree* HltTree) {
  if (_Monte) {
    int ntaus=0;
    if (&mctruth){
      for (size_t i = 0; i < mctruth.size(); ++ i) {
	const Candidate & p = (mctruth)[i];
        int status = p.status();
	if (status==1 || status==2) {
          int pid=p.pdgId();
	  bool istau = abs(pid)==15;
          bool inacceptance = (abs(p.eta()) < 4.5);
	  bool aboveptcut = (p.pt() > 100.0);
	  if (inacceptance && aboveptcut && istau) {
	      ntaus++;
          }
        }
      }
      // request 1 tau
      isvisible = ntaus >= 1; 
    }
    else {std::cout << "%HLTHiggsTruth -- No MC truth information" << std::endl;}
//    std::cout << "ntaus = " << ntaus << " " << isvisible_taunu << std::endl;
  }
}

void HLTHiggsTruth::analyzeHinv(const CandidateCollection& mctruth,TTree* HltTree) {
  if (_Monte) {
    if (&mctruth){
      isvisible = true; 
    } else {
      std::cout << "%HLTHiggsTruth -- No MC truth information" << std::endl;
    }
//    std::cout << "Invisible: MC exists, accept " << std::endl;
  }
}


