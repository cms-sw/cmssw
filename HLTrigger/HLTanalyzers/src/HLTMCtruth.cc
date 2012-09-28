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
  std::vector<std::string> parameterNames = myMCParams.getParameterNames() ;
  
  for ( std::vector<std::string>::iterator iParam = parameterNames.begin();
	iParam != parameterNames.end(); iParam++ ){
    if  ( (*iParam) == "Monte" ) _Monte =  myMCParams.getParameter<bool>( *iParam );
    else if ( (*iParam) == "Debug" ) _Debug =  myMCParams.getParameter<bool>( *iParam );
  }

  const int kMaxMcTruth = 10000;
  mcpid = new int[kMaxMcTruth];
  mcstatus = new int[kMaxMcTruth];
  mcvx = new float[kMaxMcTruth];
  mcvy = new float[kMaxMcTruth];
  mcvz = new float[kMaxMcTruth];
  mcpt = new float[kMaxMcTruth];
  mceta = new float[kMaxMcTruth];
  mcphi = new float[kMaxMcTruth];

  // MCtruth-specific branches of the tree 
  HltTree->Branch("NMCpart",&nmcpart,"NMCpart/I");
  HltTree->Branch("MCpid",mcpid,"MCpid[NMCpart]/I");
  HltTree->Branch("MCstatus",mcstatus,"MCstatus[NMCpart]/I");
  HltTree->Branch("MCvtxX",mcvx,"MCvtxX[NMCpart]/F");
  HltTree->Branch("MCvtxY",mcvy,"MCvtxY[NMCpart]/F");
  HltTree->Branch("MCvtxZ",mcvz,"MCvtxZ[NMCpart]/F");
  HltTree->Branch("MCpt",mcpt,"MCpt[NMCpart]/F");
  HltTree->Branch("MCeta",mceta,"MCeta[NMCpart]/F");
  HltTree->Branch("MCphi",mcphi,"MCphi[NMCpart]/F");
  HltTree->Branch("MCPtHat",&pthatf,"MCPtHat/F");
  HltTree->Branch("MCmu3",&nmu3,"MCmu3/I");
  HltTree->Branch("MCel3",&nel3,"MCel3/I");
  HltTree->Branch("MCbb",&nbb,"MCbb/I");
  HltTree->Branch("MCab",&nab,"MCab/I");
  HltTree->Branch("MCWenu",&nwenu,"MCWenu/I");
  HltTree->Branch("MCWmunu",&nwmunu,"MCmunu/I");
  HltTree->Branch("MCZee",&nzee,"MCZee/I");
  HltTree->Branch("MCZmumu",&nzmumu,"MCZmumu/I");
  HltTree->Branch("MCptEleMax",&ptEleMax,"MCptEleMax/F");
  HltTree->Branch("MCptMuMax",&ptMuMax,"MCptMuMax/F");
  HltTree->Branch("NPUTrueBX0",&npubx0, "NPUTrueBX0/I");
}

/* **Analyze the event** */
void HLTMCtruth::analyze(const edm::Handle<reco::CandidateView> & mctruth,
			 const double        & pthat,
			 const edm::Handle<std::vector<SimTrack> > & simTracks,
			 const edm::Handle<std::vector<SimVertex> > & simVertices,
			 const edm::Handle<std::vector< PileupSummaryInfo > > & PupInfo,
			 TTree* HltTree) {

  //std::cout << " Beginning HLTMCtruth " << std::endl;

  if (_Monte) {
    int nmc = 0;
    int mu3 = 0;
    int el3 = 0;
    int mab = 0;
    int mbb = 0;
    int wel = 0;
    int wmu = 0;
    int zee = 0;
    int zmumu = 0;

    ptEleMax = -999.0;
    ptMuMax  = -999.0;    
    pthatf   = pthat;
    npubx0  = 0.0;

    int npvtrue = 0; 

    if((simTracks.isValid())&&(simVertices.isValid())){
      for (unsigned int j=0; j<simTracks->size(); j++) {
	int pdgid = simTracks->at(j).type();
	if (abs(pdgid)!=13) continue;
	double pt = simTracks->at(j).momentum().pt();
	if (pt<2.5) continue;
	double eta = simTracks->at(j).momentum().eta();
	if (abs(eta)>2.5) continue;
	if (simTracks->at(j).noVertex()) continue;
	int vertIndex = simTracks->at(j).vertIndex();
	double x = simVertices->at(vertIndex).position().x();
	double y = simVertices->at(vertIndex).position().y();
	double r = sqrt(x*x+y*y);
	if (r>150.) continue; // I think units are cm here
	double z = simVertices->at(vertIndex).position().z();
	if (abs(z)>300.) continue; // I think units are cm here
	mu3 += 1;
	break;
      }


      std::vector<PileupSummaryInfo>::const_iterator PVI;  
      for(PVI = PupInfo->begin(); PVI != PupInfo->end(); ++PVI) {  
	
	int BX = PVI->getBunchCrossing();  
	npvtrue = PVI->getTrueNumInteractions();  
	if(BX == 0)  
	  {  
	    npubx0+=npvtrue;  
	  }  
      }  
      
    }

    if (mctruth.isValid()){

      for (size_t i = 0; i < mctruth->size(); ++ i) {
	const reco::Candidate & p = (*mctruth)[i];

	mcpid[nmc] = p.pdgId();
	mcstatus[nmc] = p.status();
	mcpt[nmc] = p.pt();
	mceta[nmc] = p.eta();
	mcphi[nmc] = p.phi();
	mcvx[nmc] = p.vx();
	mcvy[nmc] = p.vy();
	mcvz[nmc] = p.vz();

	if ((mcpid[nmc]==24)||(mcpid[nmc]==-24)) { // Checking W -> e/mu nu
	  size_t idg = p.numberOfDaughters();
	  for (size_t j=0; j != idg; ++j){
	    const reco::Candidate & d = *p.daughter(j);
	    if ((d.pdgId()==11)||(d.pdgId()==-11)){wel += 1;}
	    if ((d.pdgId()==13)||(d.pdgId()==-13)){wmu += 1;}
// 	    if ( (abs(d.pdgId())!=24) && ((mcpid[nmc])*(d.pdgId())>0) ) 
// 	      {std::cout << "Wrong sign between mother-W and daughter !" << std::endl;}
	  }
	}
	if (mcpid[nmc]==23) { // Checking Z -> 2 e/mu
	  size_t idg = p.numberOfDaughters();
	  for (size_t j=0; j != idg; ++j){
	    const reco::Candidate & d = *p.daughter(j);
	    if (d.pdgId()==11){zee += 1;}
	    if (d.pdgId()==-11){zee += 2;}
	    if (d.pdgId()==13){zmumu += 1;}
	    if (d.pdgId()==-13){zmumu += 2;}
	  }
	}

	// Set-up flags, based on Pythia-generator information, for avoiding double-counting events when
	// using both pp->{e,mu}X AND QCD samples
// 	if (((mcpid[nmc]==13)||(mcpid[nmc]==-13))&&(mcpt[nmc]>2.5)) {mu3 += 1;} // Flag for muons with pT > 2.5 GeV/c
	if (((mcpid[nmc]==11)||(mcpid[nmc]==-11))&&(mcpt[nmc]>2.5)) {el3 += 1;} // Flag for electrons with pT > 2.5 GeV/c

	if (mcpid[nmc]==-5) {mab += 1;} // Flag for bbar
	if (mcpid[nmc]==5) {mbb += 1;} // Flag for b

	if ((mcpid[nmc]==13)||(mcpid[nmc]==-13))
	  {if (p.pt()>ptMuMax) {ptMuMax=p.pt();} } // Save max pt of generated Muons
	if ((mcpid[nmc]==11)||(mcpid[nmc]==-11))
	  {if (p.pt() > ptEleMax) ptEleMax=p.pt();} // Save max pt of generated Electrons

	nmc++;
      }

    }
    //    else {std::cout << "%HLTMCtruth -- No MC truth information" << std::endl;}

    nmcpart = nmc;
    nmu3 = mu3;
    nel3 = el3;
    nbb = mbb;
    nab = mab;
    nwenu = wel;
    nwmunu = wmu;
    if((zee%3)==0){nzee = zee/3;}
//     else {std::cout << "Z does not decay in e+ e- !" << std::endl;}
    if ((zmumu%3)==0){nzmumu = zmumu/3;}
//     else {std::cout << "Z does not decay in mu+ mu- !" << std::endl;}

  }

}
