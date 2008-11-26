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

void OHltTree::Loop( vector<int> * iCount, vector<int> * sPureCount, vector<int> * pureCount
                    ,vector< vector<int> > * overlapCount
                    ,vector<TString> trignames
                    //		    ,map<TString,int> map_pathHLTPrescl
                    ,map<TString,int> map_L1Prescl
                    ,map<TString,int> map_pathHLTPrescl
                    ,map<TString,int> map_MultEle,map<TString,int> map_MultPho,map<TString,int> map_MultMu
                    ,map<TString,int> map_MultJets, map<TString,int> map_MultMET
                    , SampleDiagnostics& primaryDatasetsDiagnostics  //SAK
                    ,int NEntries
                    ,bool doMuonCut,bool doElecCut
                    ,double muonPt, double muonDr
                    ,int NObjects, int MaxMult, int ip, int RateOnly
                    ,vector <TH1F*> &Num_pt,vector <TH1F*> &Num_eta,vector <TH1F*> &Num_phi
                    ,vector <TH1F*> &Den_pt,vector <TH1F*> &Den_eta,vector <TH1F*> &Den_phi
                    ,vector <TH1F*> &DenwrtL1_pt,vector <TH1F*> &DenwrtL1_eta,vector <TH1F*> &DenwrtL1_phi
                    ) 
{
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

  //for (int it = 0; it < NL1trig; it++) {
  for (int it = 0; it < 128; it++) {
    iCountL1NoPrescale.push_back(0);
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

    // We're running on unskimmed, unprescaled MC. _After_ getting the event but _before_ setting the 
    // L1->HLT association with SetMapL1BitOfStandardHLTPath, apply prescales to L1.
    // Leave room for a switch in case we don't want to do prescaling (i.e. for data).
    int doMCPrescales = 1;
    if(doMCPrescales == 1)
    {
      // Third argument should be set to 1 to apply deterministic prescales. Currrently this is the only option 
      // implemented, but we may want to leave room for other prescaling algorithms (i.e. random prescales).
      ApplyL1Prescales(map_L1Prescl,jentry,1);
    }

    // 1. Loop to check which Bit fired
    // Triggernames are assigned to trigger cuts in unambigous way!
    // If you define a new trigger also define a new unambigous name!
    SetMapBitOfStandardHLTPath();
    SetMapL1BitOfStandardHLTPath();

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
      L1AssHLTBit[it]=false;
      previousBitsFired[it] = false;
      allOtherBitsFired[it] = false;
      if ( doMuonCut && MCmu3!=0 ) continue;
      if( doElecCut && MCel3!=0 ) continue;

      //////////////////////////////////////////////////////////////////
      // Standard paths
      if ( (map_BitOfStandardHLTPath.find(trignames[it])->second==1) ) { 
        // JJH - first check L1 bit 
        if (map_L1BitOfStandardHLTPath.find(trignames[it])->second==1) { 
          triggerBitNoPrescale[it] = true;
          if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {
            triggerBit[it] = true; 
          }
        }
      }

      //      if ( (map_L1BitOfStandardHLTPath.find(trignames[it])->second!=0) ) { 
      if ( (map_L1BitOfStandardHLTPath.find(trignames[it])->second==1) ) {  
        L1AssHLTBit[it] = true;
      }

      //////////////////////////////////////////////////////////////////
      // All others incl. OpenHLT from here:

      /* ******************************** */    
      /* ** "Lean" triggers start here ** */    
      /* ******************************** */    

      //------------------8E29 menu triggers-----------------------------------

      //-- SAK ----------------------------------------------------------------
      else if (trignames[it].CompareTo("OpenHLT_DiJetAve15") == 0) {   
        if( L1_SingleJet15==1) {      // L1 Seed   
          L1AssHLTBit[it] = true;  
	  if(OpenHltDiJetAvePassed(15)>=1) {   
            triggerBitNoPrescale[it] = true;    
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
              triggerBit[it] = true;  
            }   
          }   
        }   
      }   
      //-----------------------------------------------------------------------

      else if (trignames[it].CompareTo("OpenHLT_DoubleEle10_LW_OnlyPixelM_L1R") == 0) {   
        if ( L1_DoubleEG5==1 ) { // L1 Seed       
          L1AssHLTBit[it] = true;   

          if(OpenHlt1LWElectronPassed(10.,1,9999.,9999.)>=2) {       
            triggerBitNoPrescale[it] = true;       
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {       
              triggerBit[it] = true;       
            }       
          }       
          //	  PrintOhltVariables(3,electron);	  
        }
      }   

      else if (trignames[it].CompareTo("OpenHLT_DoubleEle5_SW_L1R") == 0) {      
        if ( L1_DoubleEG5==1 ) { // L1 Seed      
          L1AssHLTBit[it] = true;   

          if(OpenHlt1ElectronPassed(5.,0,9999.,9999.)>=2) {       
            triggerBitNoPrescale[it] = true;      
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {      
              triggerBit[it] = true;      
            }      
          }      
          //	      PrintOhltVariables(3,electron);          
        }
      }  

      else if (trignames[it].CompareTo("OpenHLT_DoubleEle5_LW_L1R") == 0) {      
	if ( L1_DoubleEG5==1 ) { // L1 Seed      
          L1AssHLTBit[it] = true;   

	  if(OpenHlt1LWElectronPassed(5.,0,9999.,9999.)>=2) {       
	    triggerBitNoPrescale[it] = true;      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {      
	      triggerBit[it] = true;      
	    }      
	  }      
	  //      PrintOhltVariables(3,electron);          
	}
      }  

      else if (trignames[it].CompareTo("OpenHLT_DoubleLooseIsoTau") == 0) { 
        if ( L1_DoubleTauJet40==1 ) { // L1 Seed 
          L1AssHLTBit[it] = true; 
          //PrintOhltVariables(3,tau); 
          if(OpenHltTauPassed(15.,5.,0.,0,0.,0)>=2) { 
            triggerBitNoPrescale[it] = true; 
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) { 
              triggerBit[it] = true; 
            } 
          } 
        } 
      } 

      else if (trignames[it].CompareTo("OpenHLT_Ele10_SW_L1R") == 0) {     
        if ( L1_SingleEG8==1 ) { // L1 Seed     
          L1AssHLTBit[it] = true;   

          if(OpenHlt1ElectronPassed(10.,0,9999.,9999.)>=1) {
            triggerBitNoPrescale[it] = true;     
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {     
              triggerBit[it] = true;     
            }     
          }     
          //	  PrintOhltVariables(3,electron);
        }
      } 

      else if (trignames[it].CompareTo("OpenHLT_FwdJet20") == 0) {      
        if( L1_IsoEG10_Jet15_ForJet10==1) {      // L1 Seed      
          L1AssHLTBit[it] = true;     
          if(OpenHltFwdJetPassed(20.)>=1) {      
            triggerBitNoPrescale[it] = true;       
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {       
              triggerBit[it] = true;     
            }      
          }      
        }      
      }      

      else if (trignames[it].CompareTo("OpenHLT_IsoPhoton10_L1R") == 0) {     
        if ( L1_SingleEG8==1 ) {      // L1 Seed                                  
          L1AssHLTBit[it] = true;   

          if(OpenHlt1PhotonPassed(10.,0,1.,1.5,6.,4.)>=1) {      
            triggerBitNoPrescale[it] = true;      
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {      
              triggerBit[it] = true;      
            }      
          }
        }
      }     

      else if (trignames[it].CompareTo("OpenHLT_Jet30") == 0) {   
        if( L1_SingleJet15==1) {      // L1 Seed   
          L1AssHLTBit[it] = true;  
	  if(OpenHlt1JetPassed(30)>=1) {   
	  //	  if(OpenHlt1CorJetPassed(30)>=1) {
            triggerBitNoPrescale[it] = true;    
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
              triggerBit[it] = true;  
            }   
          }   
        }   
      }   

      else if (trignames[it].CompareTo("OpenHLT_Jet50") == 0) {    
        if( L1_SingleJet30==1) {      // L1 Seed    
          L1AssHLTBit[it] = true;   
	  if(OpenHlt1JetPassed(50)>=1) {    
	  //	  if(OpenHlt1CorJetPassed(50)>=1) {
            triggerBitNoPrescale[it] = true;     
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {     
              triggerBit[it] = true;   
            }    
          }    
        }    
      }    

      else if (trignames[it].CompareTo("OpenHLT_Jet80") == 0) {     
        if( L1_SingleJet50==1) {      // L1 Seed     
          L1AssHLTBit[it] = true;    
	  if(OpenHlt1JetPassed(80)>=1) {     
	  //	  if(OpenHlt1CorJetPassed(80)>=1) {
            triggerBitNoPrescale[it] = true;      
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {      
              triggerBit[it] = true;    
            }     
          }     
        }     
      }     

      else if (trignames[it].CompareTo("OpenHLT_L1Jet15") == 0) {    
        if( L1_SingleJet15==1) {      // L1 Seed    
          L1AssHLTBit[it] = true;   
          if(1) {    
            triggerBitNoPrescale[it] = true;     
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {     
              triggerBit[it] = true;   
            }    
          }    
        }    
      }    

      else if (trignames[it].CompareTo("OpenHLT_L1MET20") == 0) {       
        if( L1_ETM20==1) {      // L1 Seed       
          L1AssHLTBit[it] = true;      
          if(1) {       
            triggerBitNoPrescale[it] = true;        
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {        
              triggerBit[it] = true;      
            }       
          }       
        }       
      }       

      else if (trignames[it].CompareTo("OpenHLT_L1Mu") == 0) {        
        if( (L1_SingleMu7==1) || (L1_DoubleMu3==1)) {      // L1 Seed        
          L1AssHLTBit[it] = true;       
          if(1) {        
            triggerBitNoPrescale[it] = true;         
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {         
              triggerBit[it] = true;       
            }        
          }        
        }        
      }        

      else if (trignames[it].CompareTo("OpenHLT_L1MuOpen") == 0) {         
        if( (L1_SingleMuOpen==1) || (L1_SingleMu3==1) || (L1_SingleMu5==1)) {      // L1 Seed         
          L1AssHLTBit[it] = true;        
          if(1) {         
            triggerBitNoPrescale[it] = true;          
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {          
              triggerBit[it] = true;        
            }         
          }         
        }         
      }         

      else if (trignames[it].CompareTo("OpenHLT_L2Mu9") == 0) {          
        if((L1_SingleMu7==1)) {      // L1 Seed          
          L1AssHLTBit[it] = true;         
          int rc = 0;
          for(int i = 0; i < NohMuL2; i++) {
            if(ohMuL2Pt[i] > 9.) {
              rc++;
            }
          }
          if(rc>0) {
            triggerBitNoPrescale[it] = true;           
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {           
              triggerBit[it] = true;         
            }          
          }          
        }          
      }          

      else if (trignames[it].CompareTo("OpenHLT_LooseIsoTau_MET30") == 0) {        
        if(L1_SingleTauJet80==1) {      // L1 Seed        
          L1AssHLTBit[it] = true;       
          if(OpenHltTauPassed(15.,5.,0.,0,0.,0)>=1  && recoMetCal>=30.) { 
            triggerBitNoPrescale[it] = true;         
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {         
              triggerBit[it] = true;       
            }        
          }        
        }        
      } 

      else if (trignames[it].CompareTo("OpenHLT_LooseIsoTau_MET30_L1MET") == 0) {         
        if(L1_TauJet30_ETM30==1) {      // L1 Seed
          L1AssHLTBit[it] = true;        
          if(OpenHltTauPassed(15.,5.,0.,0,0.,0)>=1  && recoMetCal>=30.) {  
            triggerBitNoPrescale[it] = true;          
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {          
              triggerBit[it] = true;        
            }         
          }         
        }         
      }  

      else if (trignames[it].CompareTo("OpenHLT_MET35") == 0) {         
        if( L1_ETM30==1) {      // L1 Seed         
          L1AssHLTBit[it] = true;        
          if(recoMetCal > 35.) {         
            triggerBitNoPrescale[it] = true;          
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {          
              triggerBit[it] = true;        
            }         
          }         
        }         
      }         

      else if (trignames[it].CompareTo("OpenHLT_Mu3") == 0) {  
        if( L1_SingleMu3==1) {      // L1 Seed  
          L1AssHLTBit[it] = true;  
          if(OpenHlt1MuonPassed(3.,3.,3.,2.,0)>=1) {  
            triggerBitNoPrescale[it] = true;   
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {   
              triggerBit[it] = true;  
            }  
          }  
        }  
      }  

      else if (trignames[it].CompareTo("OpenHLT_DoubleMu3") == 0) {   
        if( L1_DoubleMu3==1) {      // L1 Seed   
          L1AssHLTBit[it] = true;   
          if(OpenHlt2MuonPassed(3.,3.,3.,2.,0)>=2) {   
            triggerBitNoPrescale[it] = true;    
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
              triggerBit[it] = true;   
            }   
          }   
        }   
      }   


      else if (trignames[it].CompareTo("OpenHLT_Photon15_L1R") == 0) {    
        if ( L1_SingleEG12==1 ) {      // L1 Seed                                 
          L1AssHLTBit[it] = true;   

          if(OpenHlt1PhotonPassed(15.,0,999.,999.,999.,999.)>=1) {     
            triggerBitNoPrescale[it] = true;     
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {     
              triggerBit[it] = true;     
            }     
          }     
        }
      }    

      //------------------triggers added for 1E31------------------------------- 

      //else if (trignames[it].CompareTo("OpenHLT_DoubleEle10_LW_OnlyPixelM_L1R") == 0) {   
      else if (trignames[it].CompareTo("OpenHLT_DoubleEle10_LW_L1R") == 0) {   
	if ( L1_DoubleEG5==1 ) { // L1 Seed       
          L1AssHLTBit[it] = true;   

	  if(OpenHlt1LWElectronPassed(10.,1,9999.,9999.)>=2) {       
	    triggerBitNoPrescale[it] = true;       
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {       
	      triggerBit[it] = true;       
	    }       
	  }       
	}
      }   

      else if(trignames[it].CompareTo("OpenHLT_Ele15_LW_L1R") == 0) {
        if ( L1_SingleEG10=1 ) { // L1 Seed
          if(OpenHlt1LWElectronPassed(15.,1,9999.,9999.)>=1) {
            triggerBitNoPrescale[it] = true;
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {
              triggerBit[it] = true;
            }
          }
        }
      }

      else if (trignames[it].CompareTo("OpenHLT_LooseIsoEle15_LW_L1R") == 0) {      
        if ( L1_SingleEG12==1 ) { // L1 Seed      
          if(OpenHlt1LWElectronPassed(15.,0,0.12,6.)>=1) {      
            triggerBitNoPrescale[it] = true;      
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {      
              triggerBit[it] = true;      
            }      
          }      
          //  PrintOhltVariables(3,electron); 
        } 
      }  

      else if (trignames[it].CompareTo("OpenHLT_IsoEle18_L1R") == 0) {       
        if ( L1_SingleEG15==1 ) { // L1 Seed       
          L1AssHLTBit[it] = true;   

          if(OpenHlt1ElectronPassed(18.,1,0.06,3.)>=1) {       
            triggerBitNoPrescale[it] = true;       
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {       
              triggerBit[it] = true;       
            }       
          }       
          //  PrintOhltVariables(3,electron);  
        }  
      }   

      else if (trignames[it].CompareTo("OpenHLT_Photon25_L1R") == 0) {     
        if ( L1_SingleEG15==1 ) {      // L1 Seed                                  
          L1AssHLTBit[it] = true;   

          if(OpenHlt1PhotonPassed(25.,0,999.,999.,999.,999.)>=1) {      
            triggerBitNoPrescale[it] = true;      
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {      
              triggerBit[it] = true;      
            }      
          }      
        } 
      }     

      else if (trignames[it].CompareTo("OpenHLT_IsoPhoton15_L1R") == 0) {     
	if ( L1_SingleEG12==1 ) {      // L1 Seed                                  
          L1AssHLTBit[it] = true;   

	  if(OpenHlt1PhotonPassed(15.,0,1.,1.5,6.,4.)>=1) {      
	    triggerBitNoPrescale[it] = true;      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {      
	      triggerBit[it] = true;      
	    }      
	  }
	}
      }     
    
      else if (trignames[it].CompareTo("OpenHLT_IsoPhoton20_L1R") == 0) {     
	if ( L1_SingleEG12==1 ) {      // L1 Seed                                  
          L1AssHLTBit[it] = true;   

	  if(OpenHlt1PhotonPassed(20.,0,1.,1.5,6.,4.)>=1) {      
	    triggerBitNoPrescale[it] = true;      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {      
	      triggerBit[it] = true;      
	    }      
	  }
	}
      }     

      else if (trignames[it].CompareTo("OpenHLT_Jet110") == 0) {     
	if( L1_SingleJet70==1) {      // L1 Seed     
	  L1AssHLTBit[it] = true;    
	  if(OpenHlt1JetPassed(110)>=1) {     
	    triggerBitNoPrescale[it] = true;      
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {      
	      triggerBit[it] = true;    
	    }     
	  }     
	}     
      }     

      else if (trignames[it].CompareTo("OpenHLT_Jet180") == 0) {
        if( L1_SingleJet70==1) {      // L1 Seed
          L1AssHLTBit[it] = true;
          if(OpenHlt1JetPassed(180)>=1) {
            triggerBitNoPrescale[it] = true;
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {
              triggerBit[it] = true;
            }
          }
        }
      }

      else if (trignames[it].CompareTo("OpenHLT_DiJetAve30") == 0) {   
	if( L1_SingleJet30==1) {      // L1 Seed   
	  L1AssHLTBit[it] = true;  
	  if(OpenHltDiJetAvePassed(30)>=1) {   
	    triggerBitNoPrescale[it] = true;    
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;  
	    }   
	  }   
	}   
      }   

      else if (trignames[it].CompareTo("OpenHLT_DiJetAve50") == 0) {   
	if( L1_SingleJet50==1) {      // L1 Seed   
	  L1AssHLTBit[it] = true;  
	  if(OpenHltDiJetAvePassed(50)>=1) {   
	    triggerBitNoPrescale[it] = true;    
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;  
	    }   
	  }   
	}   
      }   

      else if (trignames[it].CompareTo("OpenHLT_DiJetAve70") == 0) {   
	if( L1_SingleJet70==1) {      // L1 Seed   
	  L1AssHLTBit[it] = true;  
	  if(OpenHltDiJetAvePassed(50)>=1) {   
	    triggerBitNoPrescale[it] = true;    
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;  
	    }   
	  }   
	}   
      }   

      else if (trignames[it].CompareTo("OpenHLT_DiJetAve130") == 0) {
        if( L1_SingleJet70==1) {      // L1 Seed
          L1AssHLTBit[it] = true;
          if(OpenHltDiJetAvePassed(130)>=1) {
            triggerBitNoPrescale[it] = true;
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {
              triggerBit[it] = true;
            }
          }
        }
      }
            
      else if (trignames[it].CompareTo("OpenHLT_DiJetAve15_NoL1") == 0) {   
	if(true) {      // L1 Seed   
	  L1AssHLTBit[it] = true;  
	  if(OpenHltDiJetAvePassed(15)>=1) {   
	    triggerBitNoPrescale[it] = true;    
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;  
	    }   
	  }   
	}   
      }   

      else if (trignames[it].CompareTo("OpenHLT_DiJetAve15_NoL1") == 0) {   
	if(true) {      // L1 Seed   
	  L1AssHLTBit[it] = true;  
	  if(OpenHltDiJetAvePassed(15)>=1) {   
	    triggerBitNoPrescale[it] = true;    
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;  
	    }   
	  }   
	}   
      }   

      else if (trignames[it].CompareTo("OpenHLT_DiJetAve30_NoL1") == 0) {   
	if(true) {      // L1 Seed   
	  L1AssHLTBit[it] = true;  
	  if(OpenHltDiJetAvePassed(30)>=1) {   
	    triggerBitNoPrescale[it] = true;    
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;  
	    }   
	  }   
	}   
      }   

      else if (trignames[it].CompareTo("OpenHLT_DiJetAve50_NoL1") == 0) {   
	if(true) {      // L1 Seed   
	  L1AssHLTBit[it] = true;  
	  if(OpenHltDiJetAvePassed(50)>=1) {   
	    triggerBitNoPrescale[it] = true;    
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;  
	    }   
	  }   
	}   
      }   

      else if (trignames[it].CompareTo("OpenHLT_DiJetAve70_NoL1") == 0) {   
	if(true) {      // L1 Seed   
	  L1AssHLTBit[it] = true;  
	  if(OpenHltDiJetAvePassed(70)>=1) {   
	    triggerBitNoPrescale[it] = true;    
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;  
	    }   
	  }   
	}   
      }   

      else if (trignames[it].CompareTo("OpenHLT_DiJetAve130_NoL1") == 0) {   
	if(true) {      // L1 Seed   
	  L1AssHLTBit[it] = true;  
	  if(OpenHltDiJetAvePassed(130)>=1) {   
	    triggerBitNoPrescale[it] = true;    
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
	      triggerBit[it] = true;  
	    }   
	  }   
	}   
      }   

      else if (trignames[it].CompareTo("OpenHLT_QuadJet30") == 0) {
	if(L1_QuadJet15==1) { // L1 Seed
          L1AssHLTBit[it] = true;   
          if(OpenHltQuadJetPassed(30.)>=1) {    
            triggerBitNoPrescale[it] = true;     
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {     
              triggerBit[it] = true;   
            }    
          }    
	}
      }

      else if (trignames[it].CompareTo("OpenHLT_MET25") == 0) { 
        if( L1_ETM20==1) {      // L1 Seed 
          L1AssHLTBit[it] = true; 
          if(recoMetCal > 25.) { 
            triggerBitNoPrescale[it] = true; 
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) { 
              triggerBit[it] = true; 
            } 
          } 
        } 
      } 

      else if (trignames[it].CompareTo("OpenHLT_MET50") == 0) {
        if( L1_ETM40==1) {      // L1 Seed
          L1AssHLTBit[it] = true;
          if(recoMetCal > 50.) {
            triggerBitNoPrescale[it] = true;
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {
              triggerBit[it] = true;
            }
          }
        }
      }

      else if (trignames[it].CompareTo("OpenHLT_MET65") == 0) {
        if( L1_ETM50==1) {      // L1 Seed
          L1AssHLTBit[it] = true;
          if(recoMetCal > 65.) {
            triggerBitNoPrescale[it] = true;
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {
              triggerBit[it] = true;
            }
          }
        }
      }

      else if (trignames[it].CompareTo("OpenHLT_Mu5") == 0) {  
	if( L1_SingleMu5==1) {      // L1 Seed  
	  L1AssHLTBit[it] = true;  
	  if(OpenHlt1MuonPassed(5.,3.,5.,2.,0)>=1) {  
	    triggerBitNoPrescale[it] = true;   
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {   
	      triggerBit[it] = true;  
	    }  
	  }  
	}  
      }  
      
      else if (trignames[it].CompareTo("OpenHLT_Mu7") == 0) {  
	if( L1_SingleMu5==1) {      // L1 Seed  
	  L1AssHLTBit[it] = true;  
	  if(OpenHlt1MuonPassed(7.,5.,7.,2.,0)>=1) {  
	    triggerBitNoPrescale[it] = true;   
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {   
	      triggerBit[it] = true;  
	    }  
	  }  
	}  
      }  
      
      else if (trignames[it].CompareTo("OpenHLT_Mu9") == 0) {  
	if( L1_SingleMu7==1) {      // L1 Seed  
	  L1AssHLTBit[it] = true;  
	  if(OpenHlt1MuonPassed(7.,7.,9.,2.,0)>=1) {  
	    triggerBitNoPrescale[it] = true;   
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {   
	      triggerBit[it] = true;  
	    }  
	  }  
	}  
      }

      else if (trignames[it].CompareTo("OpenHLT_Mu11") == 0) {   
        if( L1_SingleMu7==1) {      // L1 Seed   
          L1AssHLTBit[it] = true;   
          if(OpenHlt1MuonPassed(7.,9.,11.,2.,0)>=1) {   
            triggerBitNoPrescale[it] = true;    
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {    
              triggerBit[it] = true;   
            }   
          }   
        }   
      } 


      else if (trignames[it].CompareTo("OpenHLT_L1Photon5") == 0) {    
	if ( L1_SingleEG5==1 ) {      // L1 Seed                                 
	  if(true) { // passthrough     
	    triggerBitNoPrescale[it] = true;     
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {     
	      triggerBit[it] = true;     
	    }     
	  }     
	}
      }    
      
      else if (trignames[it].CompareTo("OpenHLT_Photon10_L1R") == 0) {    
	if ( L1_SingleEG8==1 ) {      // L1 Seed                                 
          L1AssHLTBit[it] = true;   
	  if(OpenHlt1PhotonPassed(15.,0,999.,999.,999.,999.)>=1) {     
	    triggerBitNoPrescale[it] = true;     
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {     
	      triggerBit[it] = true;     
	    }     
	  }     
	}
      }    
      
      else if (trignames[it].CompareTo("OpenHLT_Photon15_L1R") == 0) {    
	if ( L1_SingleEG12==1 ) {      // L1 Seed                                 
          L1AssHLTBit[it] = true;   
	  if(OpenHlt1PhotonPassed(15.,0,999.,999.,999.,999.)>=1) {     
	    triggerBitNoPrescale[it] = true;     
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {     
		triggerBit[it] = true;     
	    }     
	  }     
	}
      }    
      
      else if (trignames[it].CompareTo("OpenHLT_Photon20_L1R") == 0) {    
	if ( L1_SingleEG15==1 ) {      // L1 Seed                                 
          L1AssHLTBit[it] = true;   
	  if(OpenHlt1PhotonPassed(20.,0,999.,999.,999.,999.)>=1) {     
	    triggerBitNoPrescale[it] = true;     
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {     
	      triggerBit[it] = true;     
	    }     
	  }     
	}
      }    
      
      else if (trignames[it].CompareTo("OpenHLT_DoublePhoton10_L1R") == 0) {    
	if ( L1_SingleEG8==1 ) {      // L1 Seed                                 
          L1AssHLTBit[it] = true;   
	  if(OpenHlt1PhotonPassed(10.,0,999.,999.,999.,999.)>=2) {     
	    triggerBitNoPrescale[it] = true;     
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {     
	      triggerBit[it] = true;     
	    }     
	    }     
	}
      }    

      else if (trignames[it].CompareTo("OpenHLT_DoubleIsoPhoton20_L1R") == 0) {     
        if ( L1_DoubleEG10==1 ) {      // L1 Seed                                  
          L1AssHLTBit[it] = true;    
          if(OpenHlt1PhotonPassed(20.,0,1.,1.5,6.,4.)>=2) {      
	    triggerBitNoPrescale[it] = true;      
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {      
              triggerBit[it] = true;      
            }      
	  }      
        } 
      }     


      else if (trignames[it].CompareTo("OpenHLT_BTagMu_Jet20_Calib") == 0) {
        if ( L1_Mu5_Jet15==1 ) {      // L1 Seed
          L1AssHLTBit[it] = true;   

          int rc = 0; 
	  int max =  (NohBJetL2 > 2) ? 2 : NohBJetL2;
          for(int i = 0; i < max; i++) { 
            if(ohBJetL2CorrectedEt[i] > 20.) { // ET cut
	      if(ohBJetPerfL25Discriminator[i] > 0.5) { // Level 2.5 b tag
		if(ohBJetPerfL3Discriminator[i] > 0.5) { // Level 3 b tag
		  rc++; 
		} 
	      }
	    }
	  }
	  if(rc >= 1) { 
	    triggerBitNoPrescale[it] = true; 
	    if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) { 
	      triggerBit[it] = true; 
	    } 
	  }
	}
      }

      //------------------bonus triggers not in the core menus----------------------------  
      else if (trignames[it].CompareTo("OpenHLT_IsoEle20_LW_L1R") == 0) {      
        if ( L1_SingleIsoEG15==1 ) { // L1 Seed      
          L1AssHLTBit[it] = true;   

          if(OpenHlt1LWElectronPassed(20.,0,0.06,3.)>=1) {      
            triggerBitNoPrescale[it] = true;      
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {      
              triggerBit[it] = true;      
            }      
          }      
          //  PrintOhltVariables(3,electron); 
        } 
      }  
 
      else if (trignames[it].CompareTo("OpenHLT_IsoEle15_LW_L1I") == 0) {      
        if ( L1_SingleIsoEG12==1 ) { // L1 Seed      
          L1AssHLTBit[it] = true;   

          if(OpenHlt1LWElectronPassed(15.,1,0.06,3.)>=1) {      
            triggerBitNoPrescale[it] = true;      
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {      
              triggerBit[it] = true;      
            }      
          }      
          //  PrintOhltVariables(3,electron); 
        } 
      }  
       
      else if (trignames[it].CompareTo("OpenHLT_IsoEle20_LW_L1I") == 0) {      
        if ( L1_SingleIsoEG15==1 ) { // L1 Seed      
          L1AssHLTBit[it] = true;   

          if(OpenHlt1LWElectronPassed(20.,1,0.06,3.)>=1) {      
            triggerBitNoPrescale[it] = true;      
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {      
              triggerBit[it] = true;      
            }      
          }      
          //  PrintOhltVariables(3,electron); 
        } 
      }  

      else if (trignames[it].CompareTo("OpenHLT_Ele200_LW_L1R") == 0) {      
        if ( L1_SingleEG8==1 ) { // L1 Seed      
          L1AssHLTBit[it] = true;   

          if(OpenHlt1LWElectronPassed(20.,0,9999.,9999.)>=1) { 
            triggerBitNoPrescale[it] = true;      
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {      
              triggerBit[it] = true;      
            }      
          }      
          //  PrintOhltVariables(3,electron); 
        } 
      }  
       
      else if (trignames[it].CompareTo("OpenHLT_Ele10_SW_L1R") == 0) {      
        if ( L1_SingleEG8==1 ) { // L1 Seed      
          L1AssHLTBit[it] = true;   

          if(OpenHlt1ElectronPassed(10.,0,9999.,9999.)>=1) { 
            triggerBitNoPrescale[it] = true;      
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {      
              triggerBit[it] = true;      
            }      
          }      
          //  PrintOhltVariables(3,electron); 
        } 
      }  

      else if (trignames[it].CompareTo("OpenHLT_Ele15_SW_L1R") == 0) {      
        if ( L1_SingleEG12==1 ) { // L1 Seed      
          L1AssHLTBit[it] = true;   

          if(OpenHlt1ElectronPassed(15.,0,9999.,9999.)>=1) { 
            triggerBitNoPrescale[it] = true;      
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {      
              triggerBit[it] = true;      
            }      
          }      
          //  PrintOhltVariables(3,electron); 
        } 
      }  
       
      else if (trignames[it].CompareTo("OpenHLT_Ele20_SW_L1R") == 0) {      
        if ( L1_SingleEG15==1 ) { // L1 Seed      
          L1AssHLTBit[it] = true;   

          if(OpenHlt1ElectronPassed(15.,0,9999.,9999.)>=1) { 
            triggerBitNoPrescale[it] = true;      
            if ((iCountNoPrescale[it]) % map_pathHLTPrescl.find(trignames[it])->second == 0) {      
              triggerBit[it] = true;      
            }      
          }      
          //  PrintOhltVariables(3,electron); 
        } 
      }  

      /* *** "Lean" triggers end here *** */     
    }
    
    /* ******************************** */
    // 2. Loop to check overlaps
    for (int it = 0; it < Ntrig; it++){
      //      if (map_L1BitOfStandardHLTPath.find(trignames[it])->second==1) { // Checking the L1 bit before counting
      if(1) { // JJH
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
    }

    
    /* ******************************** */
    primaryDatasetsDiagnostics.fill(triggerBit);  //SAK -- record primary datasets decisions


    /////////////////////////////////////////////////////////////////
    // Filling Eff Histos

    if(RateOnly==0){
      int loopObject=0;
      int multObject=-1;
      Float_t ObjectToFillPt[MaxMult];
      Float_t ObjectToFillEta[MaxMult];
      Float_t ObjectToFillPhi[MaxMult];
      int numbhist=Num_pt.size()/(ip+1);

      int multele=-1;
      int multpho=-1;
      int multmu=-1;
      int multjets=-1;
      int multmet=-1;

      int CountObjects=0;
      for (int it = 0; it < Ntrig; it++){
        multele = map_MultEle.find(trignames[it])->second;
        multpho = map_MultPho.find(trignames[it])->second;
        multmu = map_MultMu.find(trignames[it])->second;
        multjets = map_MultJets.find(trignames[it])->second;
        multmet = map_MultMET.find(trignames[it])->second;
        for(int i=0; i<NObjects;i++){
          for(int imult=0;imult<MaxMult;imult++){
            ObjectToFillPt[imult]=-999;
            ObjectToFillEta[imult]=-999;
            ObjectToFillPhi[imult]=-999;
          }
          if (i==0){
            multObject=multele;
            loopObject=multele;
            if(NrecoElec<multele)  loopObject=NrecoElec;
            for(int j=0;j<loopObject;j++){
              ObjectToFillPt[j]=recoElecPt[j];
              ObjectToFillEta[j]=recoElecEta[j];
              ObjectToFillPhi[j]=recoElecPhi[j];
            }
          }
          if (i==1){
            multObject=multpho;
            loopObject=multpho;
            if(NrecoPhot<multpho)  loopObject=NrecoPhot;
            for(int j=0;j<loopObject;j++){
              ObjectToFillPt[j]=recoPhotPt[j];
              ObjectToFillEta[j]=recoPhotEta[j];
              ObjectToFillPhi[j]=recoPhotPhi[j];
            }
          }
          if (i==2){
            multObject=multmu;
            loopObject=multmu;
            if(NrecoMuon<multmu)  loopObject=NrecoMuon;
            for(int j=0;j<loopObject;j++){
              ObjectToFillPt[j]=recoMuonPt[j];
              ObjectToFillEta[j]=recoMuonEta[j];
              ObjectToFillPhi[j]=recoMuonPhi[j];
            }
          }
          if (i==3){
            multObject=multjets;
            loopObject=multjets;
            if(NrecoJetCal<multjets)  loopObject=NrecoJetCal;
            for(int j=0;j<loopObject;j++){ 
              ObjectToFillPt[j]=recoJetCalPt[j];
              ObjectToFillEta[j]=recoJetCalEta[j];
              ObjectToFillPhi[j]=recoJetCalPhi[j];
            }
          }
          if (i==4){
            multObject=multmet;
            loopObject=multmet;
            for(int j=0;j<loopObject;j++){
              ObjectToFillPt[j]=recoMetCal;
              ObjectToFillEta[j]=-9999;
              ObjectToFillPhi[j]=recoMetCalPhi;
            }
          }
          for (int imult=0;imult<multObject;imult++){
            Den_pt[CountObjects+imult+ip*numbhist]->Fill(ObjectToFillPt[imult]);
            Den_eta[CountObjects+imult+ip*numbhist]->Fill(ObjectToFillEta[imult]);
            Den_phi[CountObjects+imult+ip*numbhist]->Fill(ObjectToFillPhi[imult]);
            if(L1AssHLTBit[it]==1){
              DenwrtL1_pt[CountObjects+imult+ip*numbhist]->Fill(ObjectToFillPt[imult]);
              DenwrtL1_eta[CountObjects+imult+ip*numbhist]->Fill(ObjectToFillEta[imult]);
              DenwrtL1_phi[CountObjects+imult+ip*numbhist]->Fill(ObjectToFillPhi[imult]);
            }
            if(triggerBitNoPrescale[it]==1){
              Num_pt[CountObjects+imult+ip*numbhist]->Fill(ObjectToFillPt[imult]);
              Num_eta[CountObjects+imult+ip*numbhist]->Fill(ObjectToFillEta[imult]);
              Num_phi[CountObjects+imult+ip*numbhist]->Fill(ObjectToFillPhi[imult]);
            }
          }
          CountObjects+=multObject;
        }
      }
      /* ******************************** */
    }
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
      cout << "ohEleHiso["<<i<<"]/ohEleEt["<<i<<"] = " << ohEleHiso[i]/ohEleEt[i] << endl;
      cout << "ohEleNewSC["<<i<<"] = " << ohEleNewSC[i] << endl; 
      cout << "ohElePixelSeeds["<<i<<"] = " << ohElePixelSeeds[i] << endl;

      cout << "recoElecE["<<i<<"] = " << recoElecE[i] << endl;
      cout << "recoElecEt["<<i<<"] = " << recoElecEt[i] << endl;
      cout << "recoElecPt["<<i<<"] = " << recoElecPt[i] << endl;
      cout << "recoElecPhi["<<i<<"] = " << recoElecPhi[i] << endl;
      cout << "recoElecEta["<<i<<"] = " << recoElecEta[i] << endl;

    }
    cout << "oh: number of electrons = " << NohEleLW << endl; 
    for (int i=0;i<NohEleLW;i++) { 
      cout << "ohEleEtLW["<<i<<"] = " << ohEleEtLW[i] << endl; 
      cout << "ohElePhiLW["<<i<<"] = " << ohElePhiLW[i] << endl; 
      cout << "ohEleEtaLW["<<i<<"] = " << ohEleEtaLW[i] << endl; 
      cout << "ohEleELW["<<i<<"] = " << ohEleELW[i] << endl; 
      cout << "ohElePLW["<<i<<"] = " << ohElePLW[i] << endl; 
      cout << "ohElePtLW["<<i<<"] =" <<  ohElePLW[i] * TMath::Sin(2*TMath::ATan(TMath::Exp(-1*ohEleEtaLW[i]))) << endl; 
      cout << "ohEleHisoLW["<<i<<"] = " << ohEleHisoLW[i] << endl; 
      cout << "ohEleTisoLW["<<i<<"] = " << ohEleTisoLW[i] << endl; 
      cout << "ohEleL1isoLW["<<i<<"] = " << ohEleL1isoLW[i] << endl; 
      cout << "ohEleHisoLW["<<i<<"]/ohEleEtLW["<<i<<"] = " << ohEleHisoLW[i]/ohEleEtLW[i] << endl; 
      cout << "ohEleNewSCLW["<<i<<"] = " << ohEleNewSCLW[i] << endl;  
      cout << "ohElePixelSeedsLW["<<i<<"] = " << ohElePixelSeedsLW[i] << endl; 
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
    cout << "oh: number of taus = " << NohTau << endl;
    for (int i=0;i<NohTau;i++) {
      cout<<"ohTauEt["<<i<<"] = " <<ohTauPt[i]<<endl;
      cout<<"ohTauEiso["<<i<<"] = " <<ohTauEiso[i]<<endl;
      cout<<"ohTauL25Tpt["<<i<<"] = " <<ohTauL25Tpt[i]<<endl;
      cout<<"ohTauL25Tiso["<<i<<"] = " <<ohTauL25Tiso[i]<<endl;
      cout<<"ohTauL3Tpt["<<i<<"] = " <<ohTauL3Tpt[i]<<endl;
      cout<<"ohTauL3Tiso["<<i<<"] = " <<ohTauL3Tiso[i]<<endl;
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
  for (int i=0;i<NohTau;i++) {
    if (ohTauPt[i] >= Et) {
      if (ohTauEiso[i] <= Eiso)
        if (ohTauL25Tpt[i] >= L25Tpt)
          if (ohTauL25Tiso[i] >= L25Tiso)
            if (ohTauL3Tpt[i] >= L3Tpt)
              if (ohTauL3Tiso[i] >= L3Tiso)
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

int OHltTree::OpenHlt1LWElectronPassed(float Et, int L1iso, float Tiso, float Hiso) 
{ 
  int rc = 0; 
  // Loop over all oh LW electrons 
  for (int i=0;i<NohEleLW;i++) { 
    if ( ohEleEtLW[i] > Et) { 
      if ( ohEleHisoLW[i] < Hiso || ohEleHisoLW[i]/ohEleEtLW[i] < 0.05) 
        if (ohEleNewSCLW[i]==1) 
          if (ohElePixelSeedsLW[i]>0) 
            if ( ohEleTisoLW[i] < Tiso && ohEleTisoLW[i] != -999.) 
              if ( ohEleL1isoLW[i] >= L1iso )   // L1iso is 0 or 1 
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
    if ( ohPhotEt[i] > Et) {
      if ( ohPhotL1iso[i] >= L1iso ) {
        if( ohPhotTiso[i]<Tiso ) {
          if( ohPhotEiso[i] < Eiso ) {
            if( (TMath::Abs(ohPhotEta[i]) < 1.5 && ohPhotHiso[i] < HisoBR )  ||
              (1.5 < TMath::Abs(ohPhotEta[i]) && TMath::Abs(ohPhotEta[i]) < 2.5 && ohPhotHiso[i] < HisoEC ) || 
              (ohPhotHiso[i]/ohPhotEt[i] < 0.05) ) {
                rc++;
            }
          }
        }
      }
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
  int doL1L2matching = 0;

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

                if(doL1L2matching == 1) 
                {
                  // Cut on L1<->L2 matching and L1 quality
                  if((bestl1l2drmatch > 0.3) || (L1MuQal[bestl1l2drmatchind] < L1MinimalQuality) || (L1MuQal[bestl1l2drmatchind] > L1MaximalQuality))  
                  {  
                    rcL1 = 0; 
                    cout << "Failed L1-L2 match/quality" << endl;
                    cout << "L1-L2 delta-eta = " << L1MuEta[bestl1l2drmatchind] << ", " << ohMuL2Eta[j] << endl; 
                    cout << "L1-L2 delta-pho = " << L1MuPhi[bestl1l2drmatchind] << ", " << ohMuL2Phi[j] << endl;  
                    cout << "L1-L2 delta-R = " << bestl1l2drmatch << endl;
                  }
                  else
                  {
                    cout << "Passed L1-L2 match/quality" << endl;
                    rcL1++;
                    rcL1L2L3++;
                  } // End L1 matching and quality cuts	      
                }
                else
                {
                  rcL1L2L3++;
                }
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
  int doL1L2matching = 0; 

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

		if(doL1L2matching == 1)  
		  { 
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
		  }
		else
		  {
		    rcL1L2L3++;  
		  }
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

  // Loop over all oh jets 
  for (int i=0;i<NrecoJetCal;i++) {
    if(recoJetCalPt[i]>pt) {  // Jet pT cut
      rc++;
    }
  }

  return rc;
}

int OHltTree::OpenHlt1CorJetPassed(double pt)
{
  int rc = 0;

  // Loop over all oh corrected jets
  for (int i=0;i<NrecoJetCorCal;i++) {
    if(recoJetCorCalPt[i]>pt) {  // Jet pT cut
      rc++;
    }
  }

  return rc;
}


int OHltTree::OpenHltDiJetAvePassed(double pt)
{
  int rc = 0;

  // Loop over all oh jets, select events where the *average* pT of a pair is above threshold
  for (int i=0;i<NrecoJetCal;i++) { 
    for (int j=0;j<NrecoJetCal && j!=i;j++) {      
      if((recoJetCalPt[i]+recoJetCalPt[j])/2.0 > pt) {  // Jet pT cut 
        rc++; 
      }
    } 
  }  
  return rc; 
}

int OHltTree::OpenHltCorDiJetAvePassed(double pt) 
{ 
  int rc = 0; 
 
  // Loop over all oh jets, select events where the *average* pT of a pair is above threshold 
  for (int i=0;i<NrecoJetCorCal;i++) {  
    for (int j=0;j<NrecoJetCorCal && j!=i;j++) {       
      if((recoJetCorCalPt[i]+recoJetCorCalPt[j])/2.0 > pt) {  // Jet pT cut  
        rc++;  
      } 
    }  
  }   
  return rc;  
} 

int OHltTree::OpenHltQuadJetPassed(double pt)
{
  int njet = 0;
  int rc = 0;
  
  // Loop over all oh jets, select events where the *average* pT of a pair is above threshold
  for (int i=0;i<NrecoJetCorCal;i++) {
    for (int j=0;j<NrecoJetCorCal && j!=i;j++) {
      if(recoJetCorCalPt[i] > pt) {  // Jet pT cut
	njet++;
      }
    }
  }

  if(njet >= 4)
    rc = 1;

  return rc;
}


int OHltTree::OpenHltFwdJetPassed(double esum)
{
  int rc = 0; 
  double gap = 0.; 

  // Loop over all oh jets, count the sum of energy deposited in HF 
  for (int i=0;i<NrecoJetCal;i++) {   
    if(((recoJetCalEta[i] > 3.0 && recoJetCalEta[i] < 5.0) || (recoJetCalEta[i] < -3.0 && recoJetCalEta[i] > -5.0))) { 
      gap+=recoJetCalE[i]; 
    }   
  }    

  // Backward FWD physics logic - we want to select the events *without* large jet energy in HF 
  if(gap < esum) 
    rc = 1; 
  else 
    rc = 0; 

  return rc;  
}
