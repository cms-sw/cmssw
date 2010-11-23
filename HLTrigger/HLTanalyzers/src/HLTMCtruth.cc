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
  _DoHeavyIon = false;
  _DoParticles = true;
  _DoRapidity = false;
  _DoVerticesByParticle = true;

}

/*  Setup the analysis to put the branch-variables into the tree. */
void HLTMCtruth::setup(const edm::ParameterSet& pSet, TTree* HltTree) {

  edm::ParameterSet myMCParams = pSet.getParameter<edm::ParameterSet>("RunParameters") ;
  std::vector<std::string> parameterNames = myMCParams.getParameterNames() ;
  
  for ( std::vector<std::string>::iterator iParam = parameterNames.begin();
	iParam != parameterNames.end(); iParam++ ){
     if  ( (*iParam) == "Monte" ) _Monte =  myMCParams.getParameter<bool>( *iParam );
     if ( (*iParam) == "Debug" ) _Debug =  myMCParams.getParameter<bool>( *iParam );
     if  ( (*iParam) == "DoParticles" ) _DoParticles =  myMCParams.getUntrackedParameter<bool>( *iParam ,true);     
     if  ( (*iParam) == "DoRapidity" ) _DoRapidity =  myMCParams.getUntrackedParameter<bool>( *iParam ,false);
     if  ( (*iParam) == "DoVerticesByParticle" ) _DoVerticesByParticle =  myMCParams.getUntrackedParameter<bool>( *iParam ,true);
     if  ( (*iParam) == "DoHeavyIon" ) _DoHeavyIon =  myMCParams.getUntrackedParameter<bool>( *iParam, false );

  }

  const int kMaxMcTruth = 1000000;
  mcpid = new int[kMaxMcTruth];
  mcPhotonIso = new int[kMaxMcTruth];
  mcPromptPhoton = new int[kMaxMcTruth];
  mcstatus = new int[kMaxMcTruth];
  mcvx = new float[kMaxMcTruth];
  mcvy = new float[kMaxMcTruth];
  mcvz = new float[kMaxMcTruth];
  mcpt = new float[kMaxMcTruth];
  mceta = new float[kMaxMcTruth];
  mcphi = new float[kMaxMcTruth];
  mcy = new float[kMaxMcTruth]; // hi changed
  mcDaughterEta1 = new float[kMaxMcTruth];
  mcDaughterEta2 = new float[kMaxMcTruth];


  // MCtruth-specific branches of the tree 
  HltTree->Branch("NMCpart",&nmcpart,"NMCpart/I");
  HltTree->Branch("MCpid",mcpid,"MCpid[NMCpart]/I");
  HltTree->Branch("MCPromptPhoton",mcPromptPhoton,"MCPromptPhoton[NMCpart]/I");
  HltTree->Branch("MCPhotonIso",mcPhotonIso,"MCPhotonIso[NMCpart]/I");
  HltTree->Branch("MCstatus",mcstatus,"MCstatus[NMCpart]/I");

  if(_DoVerticesByParticle){
     HltTree->Branch("MCvtxX",mcvx,"MCvtxX[NMCpart]/F");
     HltTree->Branch("MCvtxY",mcvy,"MCvtxY[NMCpart]/F");
     HltTree->Branch("MCvtxZ",mcvz,"MCvtxZ[NMCpart]/F");
  }else{
     HltTree->Branch("MCvtxX",mcvx,"MCvtxX[1]/F");
     HltTree->Branch("MCvtxY",mcvy,"MCvtxY[1]/F");
     HltTree->Branch("MCvtxZ",mcvz,"MCvtxZ[1]/F");
  }

  HltTree->Branch("MCpt",mcpt,"MCpt[NMCpart]/F");
  HltTree->Branch("MCeta",mceta,"MCeta[NMCpart]/F");
  HltTree->Branch("MCphi",mcphi,"MCphi[NMCpart]/F");
  HltTree->Branch("MCy",mcy,"MCy[NMCpart]/F"); // hi changed
  HltTree->Branch("MCPtHat",&pthatf,"MCPtHat/F");
  HltTree->Branch("MCmu3",&nmu3,"MCmu3/I");
  HltTree->Branch("MCel3",&nel3,"MCel3/I");
  HltTree->Branch("MCbb",&nbb,"MCbb/I");
  HltTree->Branch("MCab",&nab,"MCab/I");
  HltTree->Branch("MCWenu",&nwenu,"MCWenu/I");
  HltTree->Branch("MCWmunu",&nwmunu,"MCmunu/I");
  HltTree->Branch("MCZee",&nzee,"MCZee/I");
  HltTree->Branch("MCJpsimumu",&njpsimumu,"MCJpsimumu/I");
  HltTree->Branch("MCUpsilonmumu",&nupsimumu,"MCUpsilonmumu/I");
  HltTree->Branch("MCZmumu",&nzmumu,"MCZmumu/I");
  HltTree->Branch("MCptEleMax",&ptEleMax,"MCptEleMax/F");
  HltTree->Branch("MCptMuMax",&ptMuMax,"MCptMuMax/F");
  HltTree->Branch("MCetPhotonMax",&etPhotonMax,"MCetPhotonMax/F");
  HltTree->Branch("MCDaughterEta1",mcDaughterEta1,"MCDaughterEta1[NMCpart]/F");
  HltTree->Branch("MCDaughterEta2",mcDaughterEta2,"MCDaughterEta2[NMCpart]/F");
}

/* **Analyze the event** */
void HLTMCtruth::analyze(
			 const edm::Handle<reco::GenParticleCollection> & mctruth,
			 const double        & pthat,
			 const edm::Handle<std::vector<SimTrack> > & simTracks,
			 const edm::Handle<std::vector<SimVertex> > & simVertices,
			 TTree* HltTree) {

  HiPhotonType hiPhotonType(mctruth);

   if(_Debug) std::cout << " Beginning HLTMCtruth " << std::endl;

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
    int jmumu = 0;
    int umumu = 0;

    ptEleMax = -999.0;
    ptMuMax  = -999.0;    
    etPhotonMax  = -999.0;    
    pthatf   = pthat;

    if((simTracks.isValid())&&(simVertices.isValid())){
      for (unsigned int j=0; j<simTracks->size(); j++) {
	int pdgid = simTracks->at(j).type();
	if (std::abs(pdgid)!=13) continue;
	double pt = simTracks->at(j).momentum().pt();
	if (pt<2.5) continue;
	double eta = simTracks->at(j).momentum().eta();
	if (std::abs(eta)>2.5) continue;
	if (simTracks->at(j).noVertex()) continue;
	int vertIndex = simTracks->at(j).vertIndex();
	double x = simVertices->at(vertIndex).position().x();
	double y = simVertices->at(vertIndex).position().y();
	double r = sqrt(x*x+y*y);
	if (r>150.) continue; // I think units are cm here
	double z = simVertices->at(vertIndex).position().z();
	if (std::abs(z)>300.) continue; // I think units are cm here
	mu3 += 1;
	break;
      }
    }

    if (mctruth.isValid()){

      for (size_t i = 0; i < mctruth->size(); ++ i) {
	    const reco::GenParticle & p = (*mctruth)[i];

        mcDaughterEta1[nmc] = -1000;
        mcDaughterEta2[nmc] = -1000;

        // only keep interesting particles
        if (!(     p.pdgId() ==22||
	      fabs(p.pdgId())==11||
	      fabs(p.pdgId())==13||
	      fabs(p.pdgId())==23||
	      fabs(p.pdgId())==24||
	      fabs(p.pdgId())==443||
	      fabs(p.pdgId())==553||
	      fabs(p.pdgId())==5
	   )) continue;

        if (p.pdgId()==22&&p.pt()<1) continue;   // cut on photon Et > 1 GeV
        mcpid[nmc] = p.pdgId();
        mcstatus[nmc] = p.status();
        mcpt[nmc] = p.pt();
        mceta[nmc] = p.eta();
        mcphi[nmc] = p.phi();
        mcy[nmc] = p.rapidity(); // hi changed

	if(_DoVerticesByParticle){
	   mcvx[nmc] = p.vx();
	   mcvy[nmc] = p.vy();
	   mcvz[nmc] = p.vz();
	}else{
	   if( 1 ){
	      mcvx[0] = p.vx();
	      mcvy[0] = p.vy();
	      mcvz[0] = p.vz();
	   }
	}

        if ((mcpid[nmc]==24)||(mcpid[nmc]==-24)) { // Checking W -> e/mu nu
          size_t idg = p.numberOfDaughters();
          for (size_t j=0; j != idg; ++j){
            const reco::Candidate & d = *p.daughter(j);
            if ((d.pdgId()==11)||(d.pdgId()==-11)){wel += 1;}
            if ((d.pdgId()==13)||(d.pdgId()==-13)){wmu += 1;}
            //if ( (std::abs(d.pdgId())!=24) && ((mcpid[nmc])*(d.pdgId())>0) ) 
            //  {cout << "Wrong sign between mother-W and daughter !" << endl;}
          }
        }
        if (mcpid[nmc]==23) { // Checking Z -> 2 e/mu
          size_t idg = p.numberOfDaughters();
          for (size_t j=0; j != idg; ++j){
            const reco::Candidate & d = *p.daughter(j);
            if (d.pdgId()==11){zee += 1;}
            if (d.pdgId()==-11){zee += 2;}
            if (d.pdgId()==13){zmumu += 1; mcDaughterEta1[nmc] = d.eta();}
            if (d.pdgId()==-13){zmumu += 2; mcDaughterEta2[nmc] = d.eta();}
          }
        }
        // added by moon for Jpsi -> 2 muons
        if (mcpid[nmc]==443) { // Checking Jpsi -> 2 mu
            size_t idg = p.numberOfDaughters();
            for (size_t j=0; j != idg; ++j){
	      const reco::Candidate & d = *p.daughter(j);
                if (d.pdgId()==13){jmumu += 1; mcDaughterEta1[nmc] = d.eta();}
                if (d.pdgId()==-13){jmumu += 2; mcDaughterEta2[nmc] = d.eta();}
            }
        }
        // added by moon for Upsilon -> 2 muons
        if (mcpid[nmc]==553) { // Checking Upsilon -> 2 mu
            size_t idg = p.numberOfDaughters();
            for (size_t j=0; j != idg; ++j){
	      const reco::Candidate & d = *p.daughter(j);
                if (d.pdgId()==13){umumu += 1; mcDaughterEta1[nmc] = d.eta();}
                if (d.pdgId()==-13){umumu += 2; mcDaughterEta2[nmc] = d.eta();}
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
	if (mcpid[nmc]==22)
	{
           if (p.pt() > etPhotonMax) etPhotonMax=p.pt(); // Save max et of generated Photons
	   mcPromptPhoton[nmc] = hiPhotonType.IsIsolated(p);
	   mcPhotonIso[nmc] = hiPhotonType.IsPrompt(p);
        } else {
           mcPromptPhoton[nmc] = 0;
           mcPhotonIso[nmc] = 0;
        }

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
    
    // number of jpsi, upsilon z >> 2 muons
    njpsimumu = jmumu;
    nupsimumu = umumu;
    nzmumu = zmumu;

  }

}
