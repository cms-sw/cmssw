#define OHltTree_cxx
#include "OHltTree.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLeaf.h>
#include <TFormula.h>
#include <TMath.h>

#include <iostream>
#include <iomanip>
#include <string>

using namespace std;

void OHltTree::Loop(vector<int> * iCount, vector<int> * sPureCount, vector<int> * pureCount
		    ,vector< vector<int> > * overlapCount
		    ,vector<TString> trignames
		    ,map<TString,int> map_pathHLTPrescl
		    ,int NEntries
		    ,bool doMuonCut,bool doElecCut
		    ,double muonPt, double muonDr) {
  //       To read only selected branches, Insert statements like:
  // METHOD1:
  //    fChain->SetBranchStatus("*",0);  // disable all branches
  //    fChain->SetBranchStatus("branchname",1);  // activate branchname
  // METHOD2: replace line
  //    fChain->GetEntry(jentry);       //read all branches
  //by  b_branchname->GetEntry(ientry); //read only this branch
  if (fChain == 0) return;

  Long64_t nentries = (Long64_t)NEntries; 
  if (NEntries <= 0)
    nentries = fChain->GetEntries();

  vector<int> iCountNoPrescale;
  for (int it = 0; it < Ntrig; it++){
    iCountNoPrescale.push_back(0);
  }
   
  Long64_t nbytes = 0, nb = 0;

  //int tempFlag;
  //TBranch *tempBranch;

  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // if (Cut(ientry) < 0) continue;

    if (jentry%10000 == 0) cout<<"Processing entry "<<jentry<<"/"<<nentries<<"\r"<<flush<<endl;

    // 1. Loop to check which Bit fired
    // Triggernames are assigned to trigger cuts in unambigous way!
    // If you define a new trigger also define a new unambigous name!
    SetMapBitOfStandardHLTPath();

    // Cut on muon quality
    // init
    for (int i=0;i<10;i++) {
      NL1OpenMu = 0;
      L1OpenMuPt[i] = -999.;
      L1OpenMuE[i] = -999.;
      L1OpenMuEta[i] = -999.;
      L1OpenMuPhi[i] = -999.;
      L1OpenMuIsol[i] = -999;
      L1OpenMuMip[i] = -999;
      L1OpenMuFor[i] = -999;
      L1OpenMuRPC[i] = -999;
      L1OpenMuQal[i] = -999;     
    }
    for (int i=0;i<NL1Mu;i++) {
      if ( L1MuQal[i]==2 || L1MuQal[i]==3 || L1MuQal[i]==4 ||
	   L1MuQal[i]==5 || L1MuQal[i]==6 || L1MuQal[i]==7 ) {
	L1OpenMuPt[NL1OpenMu] = L1MuPt[i];
	L1OpenMuE[NL1OpenMu] = L1MuE[i];
	L1OpenMuEta[NL1OpenMu] = L1MuEta[i];
	L1OpenMuPhi[NL1OpenMu] = L1MuPhi[i];
	L1OpenMuIsol[NL1OpenMu] = L1MuIsol[i];
	L1OpenMuMip[NL1OpenMu] = L1MuMip[i];
	L1OpenMuFor[NL1OpenMu] = L1MuFor[i];
	L1OpenMuRPC[NL1OpenMu] = L1MuRPC[i];
	L1OpenMuQal[NL1OpenMu] = L1MuQal[i];
	NL1OpenMu++;
      }
    }
    // init
    for (int i=0;i<10;i++) {
      NL1GoodSingleMu = 0;
      L1GoodSingleMuPt[i] = -999.;
      L1GoodSingleMuE[i] = -999.;
      L1GoodSingleMuEta[i] = -999.;
      L1GoodSingleMuPhi[i] = -999.;
      L1GoodSingleMuIsol[i] = -999;
      L1GoodSingleMuMip[i] = -999;
      L1GoodSingleMuFor[i] = -999;
      L1GoodSingleMuRPC[i] = -999;
      L1GoodSingleMuQal[i] = -999;     
    }
    // Cut on muon quality      
    for (int i=0;i<NL1Mu;i++) {
      if ( L1MuQal[i]==4 || L1MuQal[i]==5 || L1MuQal[i]==6 || L1MuQal[i]==7 ) {
	L1GoodSingleMuPt[NL1GoodSingleMu] = L1MuPt[i];
	L1GoodSingleMuE[NL1GoodSingleMu] = L1MuE[i];
	L1GoodSingleMuEta[NL1GoodSingleMu] = L1MuEta[i];
	L1GoodSingleMuPhi[NL1GoodSingleMu] = L1MuPhi[i];
	L1GoodSingleMuIsol[NL1GoodSingleMu] = L1MuIsol[i];
	L1GoodSingleMuMip[NL1GoodSingleMu] = L1MuMip[i];
	L1GoodSingleMuFor[NL1GoodSingleMu] = L1MuFor[i];
	L1GoodSingleMuRPC[NL1GoodSingleMu] = L1MuRPC[i];
	L1GoodSingleMuQal[NL1GoodSingleMu] = L1MuQal[i];
	NL1GoodSingleMu++;
      }
    }

    // init
    for (int i=0;i<10;i++) {
      NL1GoodDoubleMu = 0;
      L1GoodDoubleMuPt[i] = -999.;
      L1GoodDoubleMuE[i] = -999.;
      L1GoodDoubleMuEta[i] = -999.;
      L1GoodDoubleMuPhi[i] = -999.;
      L1GoodDoubleMuIsol[i] = -999;
      L1GoodDoubleMuMip[i] = -999;
      L1GoodDoubleMuFor[i] = -999;
      L1GoodDoubleMuRPC[i] = -999;
      L1GoodDoubleMuQal[i] = -999;     
    }
    // Cut on muon quality
    for (int i=0;i<NL1Mu;i++) {
      if ( L1MuQal[i]==3 || L1MuQal[i]==5 || L1MuQal[i]==6 || L1MuQal[i]==7 ) {
	L1GoodDoubleMuPt[NL1GoodDoubleMu] = L1MuPt[i];
	L1GoodDoubleMuE[NL1GoodDoubleMu] = L1MuE[i];
	L1GoodDoubleMuEta[NL1GoodDoubleMu] = L1MuEta[i];
	L1GoodDoubleMuPhi[NL1GoodDoubleMu] = L1MuPhi[i];
	L1GoodDoubleMuIsol[NL1GoodDoubleMu] = L1MuIsol[i];
	L1GoodDoubleMuMip[NL1GoodDoubleMu] = L1MuMip[i];
	L1GoodDoubleMuFor[NL1GoodDoubleMu] = L1MuFor[i];
	L1GoodDoubleMuRPC[NL1GoodDoubleMu] = L1MuRPC[i];
	L1GoodDoubleMuQal[NL1GoodDoubleMu] = L1MuQal[i];
	NL1GoodDoubleMu++;
      }
    }
    
    //////////////////////////////////////////////////////////////////
    // Loop over HLT paths and do rate counting
    //////////////////////////////////////////////////////////////////
    for (int it = 0; it < Ntrig; it++){	
      triggerBit[it] = false;
      triggerBitNoPrescale[it] = false;
      previousBitsFired[it] = false;
      allOtherBitsFired[it] = false;
      if ( doMuonCut && MCmu3!=0 ) continue;
      if( doElecCut && MCel3!=0 ) continue;
      
      //////////////////////////////////////////////////////////////////
      // Standard paths
      if ( (map_BitOfStandardHLTPath.find(trignames[it])->second==1) ) { 
	triggerBitNoPrescale[it] = true;
	if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) { 
	  triggerBit[it] = true; 
	} 
      }
      //////////////////////////////////////////////////////////////////
      // All others incl. OpenHLT from here:
      
      /* ***************************** */
      /* ****** Taus start here ****** */
      /* ***************************** */
      else if (trignames[it].CompareTo("OpenHLT2TauPixel") == 0) {
	if ( L1_DoubleTauJet40==1 ) { // L1 Seed
	  if(OpenHltTauPassed(15.,5.,3.,1,0.,0)>=2) {	      
	    triggerBitNoPrescale[it] = true;	      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;
	    } 
	  }
	}
      }
      else if (trignames[it].CompareTo("HighPtTauMET") == 0) {
	if ( L1_SingleTauJet80==1 ) { // L1 Seed
	  if(OpenHltTauPassed(20.,5.,3.,1,20.,0)>=1  && recoMetCal>=65.)  {
	    triggerBitNoPrescale[it] = true;	      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;
	    } 
	  }
	}
      }
      else if (trignames[it].CompareTo("TauMET") == 0) {
	if ( L1_TauJet30_ETM30==1 ) { // L1 Seed
	  if(OpenHltTauPassed(20.,5.,3.,1,15.,0)>=1  && recoMetCal>=35.)  {
	    triggerBitNoPrescale[it] = true;	      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;
	    } 
	  }
	}
      }
      else if (trignames[it].CompareTo("TauMET_NoSi") == 0) {
	if ( L1_TauJet30_ETM30==1 ) { // L1 Seed
	  if(OpenHltTauPassed(20.,5.,0.,0,0.,0)>=1  && recoMetCal>=35.)  {
	    triggerBitNoPrescale[it] = true;	      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;
	    } 
	  }
	}
      }
      else if (trignames[it].CompareTo("DiTau") == 0) {
	if ( L1_DoubleTauJet40==1 ) { // L1 Seed
	  //PrintOhltVariables(3,tau);
	  if(OpenHltTauPassed(15.,5.,3.,1,0.,0)>=2)  {
	    triggerBitNoPrescale[it] = true;	      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;
	    } 
	  }
	}
      }
      else if (trignames[it].CompareTo("DiTau_NoSi") == 0) {
	if ( L1_DoubleTauJet40==1 ) { // L1 Seed
	  //PrintOhltVariables(3,tau);
	  if(OpenHltTauPassed(15.,5.,0.,0,0.,0)>=2)  {
	    triggerBitNoPrescale[it] = true;	      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;
	    } 
	  }
	}
      }
      else if (trignames[it].CompareTo("MuonTau") == 0) {
	if ( L1_Mu5_TauJet20==1 ) { // L1 Seed
	  if(OpenHltTauPassed(15.,5.,3.,1,0.,0)>=1 && OpenHlt1MuonPassed(5.,0.,0.,2.,1)>=1 )  {
	    triggerBitNoPrescale[it] = true;	      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;
	    } 
	  }
	}
      }
      else if (trignames[it].CompareTo("MuonTau_NoSi") == 0) {
	if ( L1_Mu5_TauJet20==1 ) { // L1 Seed
	  if(OpenHltTauPassed(15.,5.,0.,0,0.,0)>=1 && OpenHlt1MuonPassed(5.,0.,0.,2.,1)>=1 )  {
	    triggerBitNoPrescale[it] = true;	      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;
	    } 
	  }
	}
      }
      else if (trignames[it].CompareTo("MuonTau_NoL1") == 0) {
	if ( L1_Mu5_Jet15==1 ) { // L1 Seed
	  if(OpenHltTauPassed(15.,5.,3.,1,0.,0)>=1 && OpenHlt1MuonPassed(5.,0.,0.,2.,1)>=1 )  {
	    triggerBitNoPrescale[it] = true;	      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;
	    } 
	  }
	}
      }
      else if (trignames[it].CompareTo("MuonTau_NoL2") == 0) {
	if ( L1_Mu5_TauJet20==1 ) { // L1 Seed
	  if(OpenHltTauPassed(0.,999.,3.,1,0.,0)>=1 && OpenHlt1MuonPassed(5.,0.,0.,2.,1)>=1 )  {
	    triggerBitNoPrescale[it] = true;	      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;
	    } 
	  }
	}
      }
      else if (trignames[it].CompareTo("MuonTau_NoL25") == 0) {
	if ( L1_Mu5_TauJet20==1 ) { // L1 Seed
	  if(OpenHltTauPassed(15.,5.,0.,0,0.,0)>=1 && OpenHlt1MuonPassed(5.,0.,5.,2.,1)>=1 )  {
	    triggerBitNoPrescale[it] = true;	      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;
	    } 
	  }
	}
      }
      else if (trignames[it].CompareTo("ElectronTau") == 0) {
	if ( L1_IsoEG10_TauJet20==1 ) { // L1 Seed
	  if(OpenHltTauPassed(15.,5.,3.,1,0.,0)>=1 && OpenHlt1ElectronPassed(10.,1,0.06,3.)>=1 )  {
	    triggerBitNoPrescale[it] = true;	      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;
	    } 
	  }
	}
      }
      else if (trignames[it].CompareTo("ElectronTau_NoL1") == 0) {
	if ( L1_IsoEG10_Jet15==1 ) { // L1 Seed
	  if(OpenHltTauPassed(15.,5.,3.,1,0.,0)>=1 && OpenHlt1ElectronPassed(10.,1,0.06,3.)>=1 )  {
	    triggerBitNoPrescale[it] = true;	      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;
	    } 
	  }
	}
      }
      else if (trignames[it].CompareTo("ElectronTau_NoL2") == 0) {
	if ( L1_IsoEG10_TauJet20==1 ) { // L1 Seed
	  if(OpenHltTauPassed(0.,999.,3.,1,0.,0)>=1 && OpenHlt1ElectronPassed(10.,1,0.06,3.)>=1 )  {
	    triggerBitNoPrescale[it] = true;	      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;
	    } 
	  }
	}
      }
      else if (trignames[it].CompareTo("ElectronTau_NoL25") == 0) {
	if ( L1_IsoEG10_TauJet20==1 ) { // L1 Seed
	  if(OpenHltTauPassed(15.,5.,0.,0,0.,0)>=1 && OpenHlt1ElectronPassed(10.,1,0.06,3.)>=1 )  {
	    triggerBitNoPrescale[it] = true;	      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;
	    } 
	  }
	}
      }
      else if (trignames[it].CompareTo("ElectronTau_NoSi") == 0) {
	if ( L1_IsoEG10_TauJet20==1 ) { // L1 Seed
	  if(OpenHltTauPassed(15.,5.,0.,0,0.,0)>=1 && OpenHlt1ElectronPassed(10.,1,0.06,3.)>=1 )  {
	    triggerBitNoPrescale[it] = true;	      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;
	    } 
	  }
	}
      }
      else if (trignames[it].CompareTo("ElectronMET") == 0) {
	if ( L1_SingleIsoEG12==1 ) { // L1 Seed
	  if(OpenHlt1ElectronPassed(10.,1,0.06,3.)>=1 && recoMetCal>=35.)  {
	    triggerBitNoPrescale[it] = true;	      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;
	    } 
	  }
	}
      }
      /* ****** Taus end here ****** */



      /* ********************************** */
      /* ****** Electrons start here ****** */
      /* ********************************** */
      else if (trignames[it].CompareTo("OpenHLT1Electron") == 0) {
	if ( L1_SingleIsoEG12==1 ) { // L1 Seed
	  //PrintOhltVariables(3,electron);
	  if(OpenHlt1ElectronPassed(15.,1,0.06,3.)>=1) {
	    triggerBitNoPrescale[it] = true;
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {
	      triggerBit[it] = true;
	    }
	  }
	} 
      }
      /* ****** Electrons end here ****** */

	
      /* ******************************** */
      /* ****** Photons start here ****** */
      /* ******************************** */
      else if (trignames[it].CompareTo("OpenHLT1Photon") == 0) {
	if ( L1_SingleIsoEG12==1 ) {	  // L1 Seed				
	  //PrintOhltVariables(3,photon);
	  if(OpenHlt1PhotonPassed(30.,1,0,1.5,6.,4.)>=1) {
	    triggerBitNoPrescale[it] = true;
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {
	      triggerBit[it] = true;
	    }
	  }
	} 
      } 
      /* ****** Photons end here ****** */

      /* ******************************** */ 
      /* ****** Muons start here ****** */ 
      /* ******************************** */ 
      else if (trignames[it].CompareTo("OpenHLT1MuonNonIso") == 0) {
	if( L1_SingleMu7==1) {      // L1 Seed
	  //PrintOhltVariables(1,muon);
	  //PrintOhltVariables(2,muon);
	  //PrintOhltVariables(3,muon);
	  if(OpenHlt1MuonPassed(7.,16.,16.,2.,0)>=1) {
	    triggerBitNoPrescale[it] = true; 
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) { 
	      triggerBit[it] = true;
	    }
	  }
	}
      }

      else if (trignames[it].CompareTo("OpenHLT1MuonIso") == 0) { 
        if( L1_SingleMu7==1) {      // L1 Seed 
          //PrintOhltVariables(1,muon); 
          //PrintOhltVariables(2,muon); 
          //PrintOhltVariables(3,muon); 
          if(OpenHlt1MuonPassed(7.,11.,11.,2.,1)>=1) { 
            triggerBitNoPrescale[it] = true;  
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {  
	      triggerBit[it] = true;
            } 
	  }
        } 
      } 
      else if (trignames[it].CompareTo("OpenHLT2MuonNonIso") == 0) {
	if( L1_SingleMu3==1) {      // L1 Seed
	  if(OpenHlt2MuonPassed(3.,3.,3.,2.,0)>1) {
	    triggerBitNoPrescale[it] = true;
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {
	      triggerBit[it] = true;
	    }
	  }
	}
      }
      /* ****** Muons end here ****** */ 


      /* ******************************** */  
      /* ******   Jets start here  ****** */  
      /* ******************************** */  
      else if (trignames[it].CompareTo("OpenHLT1jet") == 0) { 
        if( L1_SingleJet150==1) {      // L1 Seed 
          if(OpenHlt1JetPassed(200)>=1) { 
            triggerBitNoPrescale[it] = true;  
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {  
	      triggerBit[it] = true;
            } 
          } 
        } 
      } 
      /* ****** Jets end here ****** */  
    }    


	    
    /* ******************************** */
    // 2. Loop to check overlaps
    for (int it = 0; it < Ntrig; it++){
      if (triggerBitNoPrescale[it]) {
	(iCountNoPrescale[it])++;
      }
      if (triggerBit[it]) {
	(iCount->at(it))++;
	for (int it2 = 0; it2 < Ntrig; it2++){
	  if ( (it2<it) && triggerBit[it2] )
	    previousBitsFired[it] = true;
	  if ( (it2!=it) && triggerBit[it2] )
	    allOtherBitsFired[it] = true;
	  if (triggerBit[it2])
	    (overlapCount->at(it))[it2] += 1;
	}
	if (not previousBitsFired[it])
	  (sPureCount->at(it))++;
	if (not allOtherBitsFired[it])
	  (pureCount->at(it))++;
      }
    }
    /* ******************************** */
    
  }
}



void OHltTree::PrintOhltVariables(int level, int type)
{
  cout << "Run " << Run <<", Event " << Event << endl;
  switch(type)
    {	
    case muon:

      if(level == 3) {

	cout << "Level 3: number of muons = " << NohMuL3 << endl;

	for (int i=0;i<NohMuL3;i++) {
	  cout << "ohMuL3Pt["<<i<<"] = " << ohMuL3Pt[i] << endl;
	  cout << "ohMuL3PtErr["<<i<<"] = " << ohMuL3PtErr[i] << endl;
	  cout << "ohMuL3Pt+Err["<<i<<"] = " << ohMuL3Pt[i]+2.2*ohMuL3PtErr[i]*ohMuL3Pt[i] << endl;
	  cout << "ohMuL3Phi["<<i<<"] = " << ohMuL3Phi[i] << endl;
	  cout << "ohMuL3Eta["<<i<<"] = " << ohMuL3Eta[i] << endl;
	  cout << "ohMuL3Chg["<<i<<"] = " << ohMuL3Chg[i] << endl;
	  cout << "ohMuL3Iso["<<i<<"] = " << ohMuL3Iso[i] << endl;
	  cout << "ohMuL3Dr["<<i<<"] = " << ohMuL3Dr[i] << endl;
	  cout << "ohMuL3Dz["<<i<<"] = " << ohMuL3Dz[i] << endl;
	  cout << "ohMuL3L2idx["<<i<<"] = " << ohMuL3L2idx[i] << endl;
	}
      }
      else if(level == 2) {
	cout << "Level 2: number of muons = " << NohMuL2 << endl;
	for (int i=0;i<NohMuL2;i++) {
	  cout << "ohMuL2Pt["<<i<<"] = " << ohMuL2Pt[i] << endl;
	  cout << "ohMuL2PtErr["<<i<<"] = " << ohMuL2PtErr[i] << endl;
	  cout << "ohMuL2Pt+Err["<<i<<"] = " << ohMuL2Pt[i]+3.9*ohMuL2PtErr[i]*ohMuL2Pt[i] << endl;
	  cout << "ohMuL2Phi["<<i<<"] = " << ohMuL2Phi[i] << endl;
	  cout << "ohMuL2Eta["<<i<<"] = " << ohMuL2Eta[i] << endl;
	  cout << "ohMuL2Chg["<<i<<"] = " << ohMuL2Chg[i] << endl;
	  cout << "ohMuL2Iso["<<i<<"] = " << ohMuL2Iso[i] << endl;
	  cout << "ohMuL2Dr["<<i<<"] = " << ohMuL2Dr[i] << endl;
	  cout << "ohMuL2Dz["<<i<<"] = " << ohMuL2Dz[i] << endl;
	}
      }
      else if(level == 1) {
	for(int i=0;i<NL1OpenMu;i++) {
	  cout << "L1MuPt["<<i<<"] = " << L1MuPt[i] << endl; 
	  cout << "L1MuEta["<<i<<"] = " << L1MuEta[i] << endl;  
	  cout << "L1MuPhi["<<i<<"] = " << L1MuPhi[i] << endl;  
	  cout << "L1MuIsol["<<i<<"] = " << L1MuIsol[i] << endl;  
	  cout << "L1MuQal["<<i<<"] = " << L1MuQal[i] << endl;  
	}
      }
      else {
	cout << "PrintOhltVariables: Ohlt has Muon variables only for L1, 2, and 3. Must provide one." << endl;
      }
      break;

    case electron:
      cout << "oh: number of electrons = " << NohEle << endl;
      for (int i=0;i<NohEle;i++) {
	cout << "ohEleEt["<<i<<"] = " << ohEleEt[i] << endl;
	cout << "ohElePhi["<<i<<"] = " << ohElePhi[i] << endl;
	cout << "ohEleEta["<<i<<"] = " << ohEleEta[i] << endl;
	cout << "ohEleE["<<i<<"] = " << ohEleE[i] << endl;
	cout << "ohEleP["<<i<<"] = " << ohEleP[i] << endl;
	cout << "ohElePt["<<i<<"] =" <<  ohEleP[i] * TMath::Sin(2*TMath::ATan(TMath::Exp(-1*ohEleEta[i]))) << endl;
	cout << "ohEleHiso["<<i<<"] = " << ohEleHiso[i] << endl;
	cout << "ohEleTiso["<<i<<"] = " << ohEleTiso[i] << endl;
	cout << "ohEleL1iso["<<i<<"] = " << ohEleL1iso[i] << endl;
	cout << "recoElecE["<<i<<"] = " << recoElecE[i] << endl;
	cout << "recoElecEt["<<i<<"] = " << recoElecEt[i] << endl;
	cout << "recoElecPt["<<i<<"] = " << recoElecPt[i] << endl;
	cout << "recoElecPhi["<<i<<"] = " << recoElecPhi[i] << endl;
	cout << "recoElecEta["<<i<<"] = " << recoElecEta[i] << endl;

      }
      break;

    case photon:

      cout << "oh: number of photons = " << NohPhot << endl;
      for (int i=0;i<NohPhot;i++) {
	cout << "ohPhotEt["<<i<<"] = " << ohPhotEt[i] << endl;
	cout << "ohPhotPhi["<<i<<"] = " << ohPhotPhi[i] << endl;
	cout << "ohPhotEta["<<i<<"] = " << ohPhotEta[i] << endl;
	cout << "ohPhotEiso["<<i<<"] = " << ohPhotEiso[i] << endl;
	cout << "ohPhotHiso["<<i<<"] = " << ohPhotHiso[i] << endl;
	cout << "ohPhotTiso["<<i<<"] = " << ohPhotTiso[i] << endl;
	cout << "ohPhotL1iso["<<i<<"] = " << ohPhotL1iso[i] << endl;
	cout << "recoPhotE["<<i<<"] = " << recoPhotE[i] << endl;
	cout << "recoPhotEt["<<i<<"] = " << recoPhotEt[i] << endl;
	cout << "recoPhotPt["<<i<<"] = " << recoPhotPt[i] << endl;
	cout << "recoPhotPhi["<<i<<"] = " << recoPhotPhi[i] << endl;
	cout << "recoPhotEta["<<i<<"] = " << recoPhotEta[i] << endl;

      }
      break;

    case jet:
      cout << "oh: number of recoJetCal = " << NrecoJetCal << endl;
      for (int i=0;i<NrecoJetCal;i++) {
	cout << "recoJetCalE["<<i<<"] = " << recoJetCalE[i] << endl;
	cout << "recoJetCalEt["<<i<<"] = " << recoJetCalEt[i] << endl;
	cout << "recoJetCalPt["<<i<<"] = " << recoJetCalPt[i] << endl;
	cout << "recoJetCalPhi["<<i<<"] = " << recoJetCalPhi[i] << endl;
	cout << "recoJetCalEta["<<i<<"] = " << recoJetCalEta[i] << endl;
      }
      break;

    case tau:
      cout << "oh: number of taus = " << NohTau1 << endl;
      for (int i=0;i<NohTau1;i++) {
	cout<<"ohTauEt["<<i<<"] = " <<ohTau1Pt[i]<<endl;
	cout<<"ohTauEiso["<<i<<"] = " <<ohTau1Eiso[i]<<endl;
	cout<<"ohTauL25Tpt["<<i<<"] = " <<ohTau1L25Tpt[i]<<endl;
	cout<<"ohTauL25Tiso["<<i<<"] = " <<ohTau1L25Tiso[i]<<endl;
	cout<<"ohTauL3Tpt["<<i<<"] = " <<ohTau1L3Tpt[i]<<endl;
	cout<<"ohTauL3Tiso["<<i<<"] = " <<ohTau1L3Tiso[i]<<endl;
      }
      break;

      
    default:

      cout << "PrintOhltVariables: You did not provide correct object type." <<endl;
      break;
    }
}

int OHltTree::OpenHltTauPassed(float Et,float Eiso, float L25Tpt, int L25Tiso, float L3Tpt, int L3Tiso)
{
  int rc = 0;
  // Loop over all oh electrons
  for (int i=0;i<NohTau1;i++) {
    if (ohTau1Pt[i] >= Et) {
      if (ohTau1Eiso[i] <= Eiso)
	if (ohTau1L25Tpt[i] >= L25Tpt)
	  if (ohTau1L25Tiso[i] >= L25Tiso)
	    if (ohTau1L3Tpt[i] >= L3Tpt)
	      if (ohTau1L3Tiso[i] >= L3Tiso)
		rc++;      
    }
  }
  
  return rc;
}


int OHltTree::OpenHlt1ElectronPassed(float Et, int L1iso, float Tiso, float Hiso)
{
  int rc = 0;
  // Loop over all oh electrons
  for (int i=0;i<NohEle;i++) {
    if ( ohEleEt[i] > Et) {
      if ( ohEleHiso[i] < Hiso || ohEleHiso[i]/ohEleEt[i] < 0.05)
	if (ohEleNewSC[i]==1)
	  if (ohElePixelSeeds[i]>0)
	    if ( ohEleTiso[i] < Tiso && ohEleTiso[i] != -999.)
	      if ( ohEleL1iso[i] >= L1iso )   // L1iso is 0 or 1
		rc++;      
    }
  }
  
  return rc;
}

int  OHltTree::OpenHlt1PhotonPassed(float Et, int L1iso, float Tiso, float Eiso, float HisoBR, float HisoEC)
{
  int rc = 0;
  // Loop over all oh photons
  for (int i=0;i<NohPhot;i++) {
    if ( ohPhotEt[i] > Et)
      if ( ohPhotL1iso[i] >= L1iso )
	if( ohPhotTiso[i]<=Tiso )
	  if( ohPhotEiso[i] < Eiso ) {
	    if( (TMath::Abs(ohPhotEta[i]) < 1.5 && ohPhotHiso[i] < HisoBR )  ||
		(1.5 < TMath::Abs(ohPhotEta[i]) && TMath::Abs(ohPhotEta[i]) < 2.5 && ohPhotHiso[i] < HisoEC ) )
	      rc++;
	  }
  }
  return rc;
}


int OHltTree::OpenHlt1MuonPassed(double ptl1, double ptl2, double ptl3, double dr, int iso)
{
  // This example implements the new (CMSSW_2_X) flat muon pT cuts.
  // To emulate the old behavior, the cuts should be written
  // L2:        ohMuL2Pt[i]+3.9*ohMuL2PtErr[i]*ohMuL2Pt[i]
  // L3:        ohMuL3Pt[i]+2.2*ohMuL3PtErr[i]*ohMuL3Pt[i]

  int rcL1 = 0; int rcL2 = 0; int rcL3 = 0; int rcL1L2L3 = 0;
  int NL1Mu = 8;
  int L1MinimalQuality = 4;
  int L1MaximalQuality = 7;

  // Loop over all oh L3 muons and apply cuts
  for (int i=0;i<NohMuL3;i++) {  
    int bestl1l2drmatchind = -1;
    double bestl1l2drmatch = 999.0; 

    if( fabs(ohMuL3Eta[i]) < 2.5 ) { // L3 eta cut  
      if(ohMuL3Pt[i] > ptl3)  {  // L3 pT cut        
	if(ohMuL3Dr[i] < dr)  {  // L3 DR cut
	  if(ohMuL3Iso[i] >= iso)  {  // L3 isolation
	    rcL3++;

	    // Begin L2 muons here. 
	    // Get best L2<->L3 match, then 
	    // begin applying cuts to L2
	    int j = ohMuL3L2idx[i];  // Get best L2<->L3 match

	    if ( (fabs(ohMuL2Eta[j])<2.5) ) {  // L2 eta cut
	      if( ohMuL2Pt[j] > ptl2 ) { // L2 pT cut
		rcL2++;

		// Begin L1 muons here.
		// Require there be an L1Extra muon Delta-R
		// matched to the L2 candidate, and that it have 
		// good quality and pass nominal L1 pT cuts 
		for(int k = 0;k < NL1Mu;k++) {
		  if( (L1MuPt[k] < ptl1) ) // L1 pT cut
		    continue;

		  double deltaphi = fabs(ohMuL2Phi[j]-L1MuPhi[k]); 
		  if(deltaphi > 3.14159) 
		    deltaphi = (2.0 * 3.14159) - deltaphi; 
 
		  double deltarl1l2 = sqrt((ohMuL2Eta[j]-L1MuEta[k])*(ohMuL2Eta[j]-L1MuEta[k]) +   
					   (deltaphi*deltaphi)); 
		  if(deltarl1l2 < bestl1l2drmatch)  
		    {  
		      bestl1l2drmatchind = k;  
		      bestl1l2drmatch = deltarl1l2;  
		    }  
		} // End loop over L1Extra muons
		
		// Cut on L1<->L2 matching and L1 quality
		if((bestl1l2drmatch > 0.3) || (L1MuQal[bestl1l2drmatchind] < L1MinimalQuality) || (L1MuQal[bestl1l2drmatchind] > L1MaximalQuality))  
		  {  
		    rcL1 = 0; 
		  }
		else
		  {
		    rcL1++;
		    rcL1L2L3++;
		  } // End L1 matching and quality cuts	      
	      } // End L2 pT cut
	    } // End L2 eta cut
	  } // End L3 isolation cut
	} // End L3 DR cut
      } // End L3 pT cut
    } // End L3 eta cut
  } // End loop over L3 muons		      

  return rcL1L2L3;
}

int OHltTree::OpenHlt2MuonPassed(double ptl1, double ptl2, double ptl3, double dr, int iso) 
{ 
  // Note that the dimuon paths generally have different L1 requirements than 
  // the single muon paths. Therefore this example is implemented in a separate
  // function.
  //
  // This example implements the new (CMSSW_2_X) flat muon pT cuts. 
  // To emulate the old behavior, the cuts should be written 
  // L2:        ohMuL2Pt[i]+3.9*ohMuL2PtErr[i]*ohMuL2Pt[i] 
  // L3:        ohMuL3Pt[i]+2.2*ohMuL3PtErr[i]*ohMuL3Pt[i] 
 
  int rcL1 = 0; int rcL2 = 0; int rcL3 = 0; int rcL1L2L3 = 0; 
  int NL1Mu = 8; 
  int L1MinimalQuality = 3; 
  int L1MaximalQuality = 7; 
 
  // Loop over all oh L3 muons and apply cuts 
  for (int i=0;i<NohMuL3;i++) {   
    int bestl1l2drmatchind = -1; 
    double bestl1l2drmatch = 999.0;  
 
    if( fabs(ohMuL3Eta[i]) < 2.5 ) { // L3 eta cut   
      if(ohMuL3Pt[i] > ptl3) {  // L3 pT cut         
        if(ohMuL3Dr[i] < dr) {  // L3 DR cut 
          if(ohMuL3Iso[i] >= iso) {  // L3 isolation 
            rcL3++; 
 
            // Begin L2 muons here.  
            // Get best L2<->L3 match, then  
            // begin applying cuts to L2 
            int j = ohMuL3L2idx[i];  // Get best L2<->L3 match 
 
            if ( (fabs(ohMuL2Eta[j])<2.5) ) {  // L2 eta cut 
              if( ohMuL2Pt[j] > ptl2 ) { // L2 pT cut 
                rcL2++; 
 
                // Begin L1 muons here. 
                // Require there be an L1Extra muon Delta-R 
                // matched to the L2 candidate, and that it have  
                // good quality and pass nominal L1 pT cuts  
                for(int k = 0;k < NL1Mu;k++) { 
                  if( (L1MuPt[k] < ptl1) ) // L1 pT cut 
                    continue; 
 
                  double deltaphi = fabs(ohMuL2Phi[j]-L1MuPhi[k]);  
                  if(deltaphi > 3.14159)  
                    deltaphi = (2.0 * 3.14159) - deltaphi;  
  
                  double deltarl1l2 = sqrt((ohMuL2Eta[j]-L1MuEta[k])*(ohMuL2Eta[j]-L1MuEta[k]) +    
                                           (deltaphi*deltaphi));  
                  if(deltarl1l2 < bestl1l2drmatch)   
                    {   
                      bestl1l2drmatchind = k;   
                      bestl1l2drmatch = deltarl1l2;   
                    }   
                } // End loop over L1Extra muons 
                 
                // Cut on L1<->L2 matching and L1 quality 
                if((bestl1l2drmatch > 0.3) || (L1MuQal[bestl1l2drmatchind] < L1MinimalQuality) || (L1MuQal[bestl1l2drmatchind] > L1MaximalQuality))   
                  {   
                    rcL1 = 0;  
                  } 
                else 
                  { 
                    rcL1++; 
                    rcL1L2L3++; 
                  } // End L1 matching and quality cuts        
              } // End L2 pT cut 
            } // End L2 eta cut 
          } // End L3 isolation cut 
        } // End L3 DR cut 
      } // End L3 pT cut 
    } // End L3 eta cut 
  } // End loop over L3 muons                  
 
  return rcL1L2L3; 
} 


int OHltTree::OpenHlt1JetPassed(double pt)
{
  int rc = 0;

  // Loop over all oh photons 
  for (int i=0;i<NrecoJetCal;i++) {
    if(recoJetCalPt[i]>pt) {  // Jet pT cut
      rc++;
    }
  }

  return rc;
}
