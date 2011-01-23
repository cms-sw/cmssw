//////////////////////////////////////////////////////////////////
// OpenHLT definitions
//////////////////////////////////////////////////////////////////

#define OHltTreeOpen_cxx

#include "TVector2.h"
#include "OHltTree.h"

using namespace std;

void OHltTree::CheckOpenHlt(OHltConfig *cfg,OHltMenu *menu,OHltRateCounter *rcounter,int it) 
{
  //////////////////////////////////////////////////////////////////
  // Check OpenHLT L1 bits for L1 rates

  if (menu->GetTriggerName(it).CompareTo("OpenL1_ZeroBias") == 0) {     
    if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenL1_EG5_HTT100") == 0) { 
    if(map_BitOfStandardHLTPath.find("OpenL1_EG5_HTT100")->second == 1) {   
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }   
    }   
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenL1_Mu3EG5") == 0) {
    if(map_BitOfStandardHLTPath.find("OpenL1_Mu3EG5")->second == 1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenL1_QuadJet8U") == 0) { 
    if(map_BitOfStandardHLTPath.find("OpenL1_QuadJet8U")->second == 1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
    } 
  } 
  
  //////////////////////////////////////////////////////////////////
  // Example for pass through triggers, e.g. to be used for L1 seed rates ...

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1Seed1") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }   
    }    
  }    
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1Seed2") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }   
    }    
  }    

  // Activity
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Activity_Ecal_SC7") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	float Et=7.0;
	int nsc=0;
	for (int i=0;i<NohEle;i++) {   
	  if ( ohEleEt[i] > Et) {
	    nsc++;
	  }
	}
	if (nsc>0)
	  triggerBit[it] = true; 
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Activity_Phot_SC7") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	float Et=7.0;
	int nsc=0;
	for (int i=0;i<NohPhot;i++) {   
	  if ( ohPhotEt[i] > Et) {
	    nsc++;
	  }
	}
	if (nsc>0)
	  triggerBit[it] = true; 
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Activity_Ecal_SC0") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	float Et=0.01;
	int nsc=0;
	for (int i=0;i<NohEle;i++) {   
	  if ( ohEleEt[i] > Et) {
	    nsc++;
	  }
      }
	if (nsc>0)
	  triggerBit[it] = true; 
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Activity_Ecal_SC15") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	float Et=15.0;
	int nsc=0;
	for (int i=0;i<NohEle;i++) {   
	  if ( ohEleEt[i] > Et) {
	    nsc++;
	  }
	}
	if (nsc>0)
	  triggerBit[it] = true; 
      }
    }
  }
  //////////////////////////////////////////////////////////////////
  // Check OpenHLT trigger

  /* Single Jet */
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1SingleCenJet") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	triggerBit[it] = true; 
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1SingleForJet") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	triggerBit[it] = true; 
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1SingleTauJet") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	triggerBit[it] = true; 
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1Jet6") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	triggerBit[it] = true; 
      }
    }    
  }    
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1Jet10") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	for(int i=0;i<NL1CenJet;i++) if(L1CenJetEt[i] >= 10.0) rc++;
	for(int i=0;i<NL1ForJet;i++) if(L1ForJetEt[i] >= 10.0) rc++;
	for(int i=0;i<NL1Tau   ;i++) if(L1TauEt   [i] >= 10.0) rc++;
	if(rc > 0)
	  triggerBit[it] = true; 
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1Jet15") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	triggerBit[it] = true; 
      }
    }    
  }    
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet15U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1JetPassed(15.)>=1) {
	  triggerBit[it] = true; 
	}
      }    
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet30U") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1JetPassed(30.)>=1) {
	  triggerBit[it] = true; 
	}
      }    
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet50U") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1JetPassed(50.)>=1) {
	  triggerBit[it] = true; 
	}
      }    
    }
  }
  ////////////////ADDED THIS //////////////// //////////////// ////////////////
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet60U") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1JetPassed(60.)>=1) {
	  triggerBit[it] = true; 
	}
      }    
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet70U") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1JetPassed(70.)>=1) {
	  triggerBit[it] = true; 
	}
      }    
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet100U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1JetPassed(100.)>=1) {
	  triggerBit[it] = true; 
	}
      }    
    }
  }
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet140U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1JetPassed(140.)>=1) {
	  triggerBit[it] = true; 
	}
      }    
    }
  }
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet180U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1JetPassed(180.)>=1) {
	  triggerBit[it] = true; 
	}
      }    
    }
  }
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet300U") == 0) { 
   if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
     if (prescaleResponse(menu,cfg,rcounter,it)) { 
       if(OpenHlt1JetPassed(300.)>=1) { 
	 triggerBit[it] = true;  
       } 
     }     
   } 
 } 
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet400U") == 0) {  
   if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {   
     if (prescaleResponse(menu,cfg,rcounter,it)) {  
       if(OpenHlt1JetPassed(400.)>=1) {  
         triggerBit[it] = true;   
       }  
     }      
   }  
 }  
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet500U") == 0) {  
   if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {   
     if (prescaleResponse(menu,cfg,rcounter,it)) {  
       if(OpenHlt1JetPassed(500.)>=1) {  
         triggerBit[it] = true;   
       }  
     }      
   }  
 }  
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet600U") == 0) {  
   if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {   
     if (prescaleResponse(menu,cfg,rcounter,it)) {  
       if(OpenHlt1JetPassed(600.)>=1) {  
         triggerBit[it] = true;   
       }  
     }      
   }  
 }  
  //////////////// //////////////// //////////////// //////////////// ////////////////
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet30") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1CorJetPassed(30)>=1) { 
	  triggerBit[it] = true; 
	}
      }    
    }    
  }    
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet50") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1CorJetPassed(50)>=1) { 
	  triggerBit[it] = true; 
	}
      }     
    }     
  }     
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet80") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1CorJetPassed(80)>=1) { 
	  triggerBit[it] = true; 
	}
      }      
    }      
  }      
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet110") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1CorJetPassed(110)>=1) {      
	  triggerBit[it] = true; 
	}
      }      
    }      
  }      
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet140") == 0) {       
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1CorJetPassed(140)>=1) {       
	  triggerBit[it] = true; 
	}
      }       
    }       
  }       
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet180") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1CorJetPassed(180)>=1) { 
	  triggerBit[it] = true; 
	}
      } 
    } 
  } 

  /* Double Jet */
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleJet15U_ForwardBackward") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	int rc1 = 0;
	int rc2 = 0;
	
	// Loop over all oh jets, select events where both pT of a pair are above threshold and in HF+ and HF-
	for (int i=0;i<NrecoJetCal;i++) {
	  if(recoJetCalPt[i]/0.7 > 15.0 && recoJetCalEta[i] > 3.0 && recoJetCalEta[i] < 5.1) {  // Jet pT/eta cut
	    ++rc1;
	  }
	  if(recoJetCalPt[i]/0.7 > 15.0 && recoJetCalEta[i] > -5.1 && recoJetCalEta[i] < -3.0) {  // Jet pT/eta cut
	    ++rc2;
	}
	}
	if (rc1!=0 && rc2!=0) rc=1;
	if(rc > 0)
	  {
	    triggerBit[it] = true; 
	  }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleJet20U_ForwardBackward") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	int rc1 = 0;
	int rc2 = 0;
	
	// Loop over all oh jets, select events where both pT of a pair are above threshold and in HF+ and HF-
	for (int i=0;i<NrecoJetCal;i++) {
	  if(recoJetCalPt[i]/0.7 > 20.0 && recoJetCalEta[i] > 3.0 && recoJetCalEta[i] < 5.1) {  // Jet pT/eta cut
	    ++rc1;
	  }
	  if(recoJetCalPt[i]/0.7 > 20.0 && recoJetCalEta[i] > -5.1 && recoJetCalEta[i] < -3.0) {  // Jet pT/eta cut
	    ++rc2;
	  }
	}
	if (rc1!=0 && rc2!=0) rc=1;
	if(rc > 0)
	  {
	    triggerBit[it] = true; 
	  }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleJet25U_ForwardBackward") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	int rc1 = 0;
	int rc2 = 0;
	
	// Loop over all oh jets, select events where both pT of a pair are above threshold and in HF+ and HF-
	for (int i=0;i<NrecoJetCal;i++) {
	  if(recoJetCalPt[i]/0.7 > 25.0 && recoJetCalEta[i] > 3.0 && recoJetCalEta[i] < 5.1) {  // Jet pT/eta cut
	    ++rc1;
	  }
	  if(recoJetCalPt[i]/0.7 > 25.0 && recoJetCalEta[i] > -5.1 && recoJetCalEta[i] < -3.0) {  // Jet pT/eta cut
	    ++rc2;
	  }
	}
	if (rc1!=0 && rc2!=0) rc=1;
	if(rc > 0)
	  {
	    triggerBit[it] = true; 
	  }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleJet35U_ForwardBackward") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	int rc1 = 0;
	int rc2 = 0;
	
	// Loop over all oh jets, select events where both pT of a pair are above threshold and in HF+ and HF-
	for (int i=0;i<NrecoJetCal;i++) {
	  if(recoJetCalPt[i]/0.7 > 35.0 && recoJetCalEta[i] > 3.0 && recoJetCalEta[i] < 5.1) {  // Jet pT/eta cut
	    ++rc1;
	  }
	  if(recoJetCalPt[i]/0.7 > 35.0 && recoJetCalEta[i] > -5.1 && recoJetCalEta[i] < -3.0) {  // Jet pT/eta cut
	    ++rc2;
	  }
	}
	if (rc1!=0 && rc2!=0) rc=1;
	if(rc > 0)
	  {
	    triggerBit[it] = true; 
	  }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleJet45U_ForwardBackward") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) { 
        int rc = 0; 
        int rc1 = 0; 
        int rc2 = 0; 
         
        // Loop over all oh jets, select events where both pT of a pair are above threshold and in HF+ and HF- 
        for (int i=0;i<NrecoJetCal;i++) { 
          if(recoJetCalPt[i]/0.7 > 45.0 && recoJetCalEta[i] > 3.0 && recoJetCalEta[i] < 5.1) {  // Jet pT/eta cut 
            ++rc1; 
          } 
          if(recoJetCalPt[i]/0.7 > 45.0 && recoJetCalEta[i] > -5.1 && recoJetCalEta[i] < -3.0) {  // Jet pT/eta cut 
            ++rc2; 
          } 
        } 
        if (rc1!=0 && rc2!=0) rc=1; 
        if(rc > 0) 
          { 
            triggerBit[it] = true;  
          } 
      } 
    } 
  } 
  //Corrected jets
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleJet15_ForwardBackward") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
      
	// Loop over all oh jets, select events where both pT of a pair are above threshold and in HF+ and HF-
	for (int i=0;i<NrecoJetCal;i++) {
	  if(((recoJetCalPt[i]/0.7) > 15.0) && (recoJetCalEta[i] > 3.0) && (recoJetCalEta[i] < 5.1)) {  // Jet pT/eta cut
	    for (int j=0;j<NrecoJetCal && j!=i;j++) {
	      if(((recoJetCalPt[j]/0.7) > 15.0) && (recoJetCalEta[j] > -5.1) && (recoJetCalEta[j] < -3.0)) {  // Jet pT/eta cut
		rc++;
	      }
	    }
	  }      
	  if(((recoJetCalPt[i]/0.7) > 15.0) && (recoJetCalEta[i] > -5.1) && (recoJetCalEta[i] < -3.0)) {  // Jet pT/eta cut 
	    for (int j=0;j<NrecoJetCal && j!=i;j++) { 
	      if(((recoJetCalPt[j]/0.7) > 15.0) && (recoJetCalEta[j] > 3.0) && (recoJetCalEta[j] < 5.1)) {  // Jet pT/eta cut 
		rc++; 
	      }
	    }
	  }
	}
	if(rc > 0)
	  {
	    triggerBit[it] = true; 
	  }
      }
    }
  }
/* DiJetAve */
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DiJetAve15") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltDiJetAvePassed(15.)>=1) {   
	  triggerBit[it] = true; 
	}
      }   
    }   
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DiJetAve30") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltDiJetAvePassed(30.)>=1) {   
	  triggerBit[it] = true; 
	}
      }   
    }   
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DiJetAve50") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltDiJetAvePassed(50.)>=1) {    
	  triggerBit[it] = true; 
	}
      }    
    }    
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DiJetAve70") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltDiJetAvePassed(70.)>=1) {    
	  triggerBit[it] = true; 
	}
      }    
    }    
  } 
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DiJetAve100") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltDiJetAvePassed(100.)>=1) {    
	  triggerBit[it] = true; 
	}
      }    
    }    
  } 
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DiJetAve140") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltDiJetAvePassed(140.)>=1) {    
	  triggerBit[it] = true; 
	}
      }    
    }    
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DiJetAve130U") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltDiJetAvePassed(130.)>=1) {    
	  triggerBit[it] = true; 
	}
      }    
    }    
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DiJetAve180U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHltDiJetAvePassed(180.)>=1) {
          triggerBit[it] = true;
        }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DiJetAve300U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHltDiJetAvePassed(300.)>=1) {
          triggerBit[it] = true;
        }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DiJetAve400U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHltDiJetAvePassed(400.)>=1) {
          triggerBit[it] = true;
        }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DiJetAve500U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHltDiJetAvePassed(500.)>=1) {
          triggerBit[it] = true;
        }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DiJetAve600U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHltDiJetAvePassed(600.)>=1) {
          triggerBit[it] = true;
        }
      }
    }
  }

/* Forward & MultiJet */
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_FwdJet20U") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltFwdJetPassed(20.)>=1) {      
	  triggerBit[it] = true; 
	}
      }      
    }      
  }      
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_ExclDiJet30U") == 0) {       
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) { 

	int rcDijetCand = 0; 
	double rcHFplusEnergy = 0;
	double rcHFminusEnergy = 0;

	// First loop over all jets and find a pair above threshold and with DeltaPhi/pi > 0.5
	for (int i=0;i<NrecoJetCal;i++) { 
	  if(recoJetCalPt[i]>30.0) {  // Jet pT cut 
	    for (int j=0;j<NrecoJetCal && j!=i;j++) {  
	      if(recoJetCalPt[j]>30.0) {
		double Dphi=fabs(recoJetCalPhi[i]-recoJetCalPhi[j]);
		if(Dphi>3.14159) 
		  Dphi=2.0*(3.14159)-Dphi;
		if(Dphi>0.5*3.14159) {
		  rcDijetCand++; 
		}
	      } 
	    }
	  }
	}
	
	// Now ask for events with HF energy below threshold
	if(rcDijetCand > 0) {
	  for(int i=0; i < NrecoTowCal;i++)
	    {
	      if((recoTowEta[i] > 3.0) && (recoTowE[i] > 4.0))
		rcHFplusEnergy += recoTowE[i];
	      if((recoTowEta[i] < -3.0) && (recoTowE[i] > 4.0))
		rcHFminusEnergy += recoTowE[i];	      
	    }
	}


	// Now put them together
	if((rcDijetCand > 0) && (rcHFplusEnergy < 50) && (rcHFminusEnergy < 50))
	  triggerBit[it] = true;  
      } 
    }    
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_ExclDiJet30U_HFOR") == 0) {        
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {   
      if (prescaleResponse(menu,cfg,rcounter,it)) {  
 
        int rcDijetCand = 0;  
        double rcHFplusEnergy = 0; 
        double rcHFminusEnergy = 0; 
 
        // First loop over all jets and find a pair above threshold and with DeltaPhi/pi > 0.5 
        for (int i=0;i<NrecoJetCal;i++) {  
          if(recoJetCalPt[i]>30.0) {  // Jet pT cut  
            for (int j=0;j<NrecoJetCal && j!=i;j++) {   
              if(recoJetCalPt[j]>30.0) { 
                double Dphi=fabs(recoJetCalPhi[i]-recoJetCalPhi[j]); 
                if(Dphi>3.14159)  
                  Dphi=2.0*(3.14159)-Dphi; 
                if(Dphi>0.5*3.14159) { 
                  rcDijetCand++;  
                } 
              }  
            } 
          } 
        } 
         
        // Now ask for events with HF energy below threshold 
        if(rcDijetCand > 0) { 
          for(int i=0; i < NrecoTowCal;i++) 
            { 
              if((recoTowEta[i] > 3.0) && (recoTowE[i] > 4.0)) 
                rcHFplusEnergy += recoTowE[i]; 
              if((recoTowEta[i] < -3.0) && (recoTowE[i] > 4.0)) 
                rcHFminusEnergy += recoTowE[i];        
            } 
        } 
 
 
        // Now put them together 
        if((rcDijetCand > 0) && ((rcHFplusEnergy < 50) || (rcHFminusEnergy < 50))) 
          triggerBit[it] = true;   
      }  
    }     
  } 

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_QuadJet15U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltQuadJetPassed(15.)>=1) {    
	  triggerBit[it] = true; 
	}
      }    
    }    
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_QuadJet20U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltQuadJetPassed(20.)>=1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_QuadJet25U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltQuadJetPassed(25.)>=1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_QuadJet40U") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltQuadJetPassed(40.)>=1) {     
	  triggerBit[it] = true; 
	}
      }     
    }     
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_QuadJet50U") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltQuadJetPassed(50.)>=1) {     
	  triggerBit[it] = true; 
	}
      }     
    }     
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_FwdJet40") == 0) {       
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltFwdCorJetPassed(40.)>=1) {       
	  triggerBit[it] = true; 
	}
      }       
    }       
  }       
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_QuadJet30") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltQuadCorJetPassed(30.)>=1) {     
	  triggerBit[it] = true; 
	}
      }     
    }     
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_QuadJet40") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltQuadCorJetPassed(40.)>=1) {      
	  triggerBit[it] = true; 
	}
      }      
    }      
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_QuadJet50") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltQuadCorJetPassed(50.)>=1) {      
	  triggerBit[it] = true; 
	}
      }      
    }      
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_PentaJet25U20U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if((OpenHlt1JetPassed(25)>=4) && 
	   (OpenHlt1JetPassed(25)>=5))
          triggerBit[it] = true;
      }
    }
  }

/* MET */
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1MET20") == 0) {       
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	triggerBit[it] = true; 
      }
    }     
  }       
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET25") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(recoMetCal > 25.) {  
	  triggerBit[it] = true; 
	}
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET35") == 0) {         
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(recoMetCal > 35.) {         
	  triggerBit[it] = true; 
	}
      }         
    }         
  }         
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET45") == 0) {          
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(recoMetCal > 45.) {          
	  triggerBit[it] = true; 
	}
      }          
    }          
  }          
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET50") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(recoMetCal > 50.) { 
	  triggerBit[it] = true; 
	}
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET60") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(recoMetCal > 60.) {  
	  triggerBit[it] = true; 
	}
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET65") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(recoMetCal > 65.) { 
	  triggerBit[it] = true; 
	}
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET80") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(recoMetCal > 80.) {  
	  triggerBit[it] = true; 
	}
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET100") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(recoMetCal > 100.) { 
	  triggerBit[it] = true; 
	}
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET120") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) { 
        if(recoMetCal > 120.) {  
          triggerBit[it] = true;  
        } 
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET180") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(recoMetCal > 180.) {
          triggerBit[it] = true;
        }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET200") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(recoMetCal > 200.) {
          triggerBit[it] = true;
        }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_pfMHT50") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(pfMHT > 50.) {
	  triggerBit[it] = true;
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_pfMHT70") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(pfMHT > 70.) {
          triggerBit[it] = true;
        }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_pfMHT90") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(pfMHT > 90.) {
          triggerBit[it] = true;
        }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_pfMHT110") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(pfMHT > 110.) {
          triggerBit[it] = true;
        }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_pfMHT150") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) { 
        if(pfMHT > 150.) { 
          triggerBit[it] = true; 
        } 
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT240U_MHT50") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) { 
        if(OpenHltMHT(50., 30.)==1 && (OpenHltSumHTPassed(240., 30.) == 1)) { 
          triggerBit[it] = true;  
        } 
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT300_MHT100") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	//if(recoHTCal > 100. && (OpenHltSumHTPassed(300., 30.) == 1)) {
	//std:: cout << "OpenHltMHT(100., 30.) = " << OpenHltMHT(100., 30.) << std::endl;
	//std:: cout << "OpenHltSumHTPassed(300., 30.) = " << OpenHltSumHTPassed(300., 30.) << std::endl;
	if(OpenHltMHT(100., 30.)==1 && (OpenHltSumHTPassed(300., 30.) == 1)) {
	  triggerBit[it] = true; 
	}
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT100U_MHT45U") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltMHT(45., 20.)==1 && (OpenHltSumHTPassed(100., 20.) == 1)) {
	  triggerBit[it] = true; 
	}
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Meff150U") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltMeffU(150., 20.)==1) {
	  triggerBit[it] = true; 
	}
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Meff175U") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltMeffU(175., 20.)==1) {
	  triggerBit[it] = true; 
	}
      } 
    } 
  } 
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Meff180U") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltMeffU(180., 20.)==1) {
	  triggerBit[it] = true; 
	}
      } 
    } 
  } 
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Meff300U") == 0) {     
   if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {   
     if (prescaleResponse(menu,cfg,rcounter,it)) {  
       if(OpenHltMeffU(300., 20.)==1) {  
         triggerBit[it] = true;   
       }  
     }   
   }   
 }   
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Meff420U") == 0) {    
   if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
     if (prescaleResponse(menu,cfg,rcounter,it)) { 
       if(OpenHltMeffU(420., 20.)==1) { 
	 triggerBit[it] = true;  
       } 
     }  
   }  
 }  
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Meff460U") == 0) {
   if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
     if (prescaleResponse(menu,cfg,rcounter,it)) {
       if(OpenHltMeffU(460., 20.)==1) {
         triggerBit[it] = true;
       }
     }
   }
 }
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_PT12U_50") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltPT12U(50., 50.)==1) {
	  triggerBit[it] = true; 
	}
      } 
    } 
  } 
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_PT12U_60") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltPT12U(60., 50.)==1) {
	  triggerBit[it] = true; 
	}
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT50U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHltSumHTPassed(50., 20.) == 1) {
          triggerBit[it] = true;
        }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT100U_JT20_Eta3") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {   
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(100., 20., 3) == 1) {
	  triggerBit[it] = true; 
	}
      }   
    }   
  }   
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT140U_JT20_Eta3") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {   
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(140., 20., 3) == 1) {
	  triggerBit[it] = true; 
	}
      }   
    }   
  }   
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT150U_Eta3") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {   
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(150., 20., 3) == 1) {
	  triggerBit[it] = true; 
	}
      }   
    }   
  }   
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT160U_Eta3") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {   
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(160., 20., 3) == 1) {
	  triggerBit[it] = true; 
	}
      }   
    }   
  }   

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT100U_JT20_Eta3_NJ2") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {   
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(100., 20., 3., 2) == 1) { 
	  triggerBit[it] = true; 
	}
      }   
    }   
  }   
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT100U_JT15_NJ3") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(100., 15., 10., 3) == 1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT100U_JT15_Eta3_NJ3") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(100., 15., 3., 3) == 1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT120U_JT15_Eta3_NJ3") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(120., 15., 3., 3) == 1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT140U_JT15_Eta3_NJ3") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(140., 15., 3., 3) == 1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT100U_JT20_NJ3") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(100., 20., 10., 3) == 1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT100U_JT20_Eta3_NJ3") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {   
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(100., 20., 3., 3) == 1) { 
	  triggerBit[it] = true; 
	}
      }   
    }   
  }   
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT120U_JT20_Eta3_NJ3") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {   
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(120., 20., 3., 3) == 1) { 
	  triggerBit[it] = true; 
	}
      }   
    }   
  }   
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT140U_JT20_Eta3_NJ3") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {   
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(140., 20., 3., 3) == 1) { 
	  triggerBit[it] = true; 
	}
      }   
    }   
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT100U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {   
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(100., 20.) == 1) { 
	  triggerBit[it] = true; 
	}
      }   
    }   
  }   
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT120U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(120., 20.) == 1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT130U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(130., 20.) == 1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT140U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(140., 20.) == 1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT150U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(150., 20.) == 1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT160U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(160., 20.) == 1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT200U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {   
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(200., 20.) == 1) { 
	  triggerBit[it] = true; 
	}
      }   
    }   
  }   
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT250U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHltSumHTPassed(250., 20.) == 1) {
          triggerBit[it] = true;
        }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT320U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {    
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(320., 30.) == 1) {  
	  triggerBit[it] = true; 
	}
      }    
    }    
  }    
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT200_JT40") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {    
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(200., 40.) == 1) {  
	  triggerBit[it] = true; 
	}
      }    
    }    
  }    
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT250_JT20") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {     
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(250., 20.) == 1) { 
	  triggerBit[it] = true; 
	}
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT250") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {     
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(250., 30.) == 1) {  
	  triggerBit[it] = true; 
	}
      }   
    }   
  }   
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT250_JT40") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {     
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(250., 40.) == 1) {  
	  triggerBit[it] = true; 
	}
      }   
    }   
  }   

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_SumET120") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(recoMetCalSum > 120.) {
	  triggerBit[it] = true; 
	}
      }       
    }       
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_R0.10U_MR50U") ==0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHltRPassed(0.10,50,false,7,30.)>0) { 
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  } 
 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_R0.30U_MR100U") ==0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHltRPassed(0.30,100,false,7,30.)>0) { 
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_R0.32U_MR100U") ==0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if(OpenHltRPassed(0.32,100,false,7,30.)>0) {  
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }  
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_R0.33U_MR100U") ==0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHltRPassed(0.33,100,false,7,30.)>0) { 
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_R0.35U_MR100U") ==0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if(OpenHltRPassed(0.35,100,false,7,30.)>0) {
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_RP0.25U_MR70U") ==0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHltRPassed(0.25,70,true,7,30.)>0) { 
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  } 


  /* Muons */
  else if (menu->GetTriggerName(it).CompareTo("OpenAlCa_RPCMuonNormalisation") == 0) {          
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	for(int i=0;i<NL1OpenMu;i++) { 
	  if(L1OpenMuEta[i] > -1.6 && L1OpenMuEta[i] < 1.6) 
	    rc++;
	}
	if(rc > 0)
	  triggerBit[it] = true; 
      }
    }
  }          
  else if (menu->GetTriggerName(it).CompareTo("OpenAlCa_RPCMuonNoHits") == 0) {          
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      int rc = 0;
      for(int i=0;i<NL1OpenMu;i++) {  
        if(L1OpenMuEta[i] > -1.6 && L1OpenMuEta[i] < 1.6)  
	  if(L1OpenMuQal[i] == 6)
	    rc++; 
      } 
      if(rc > 0) 
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }         
    }          
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1MuOpen") == 0) {         
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	triggerBit[it] = true; 
      }
    }         
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu") == 0) {        
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	triggerBit[it] = true; 
      }
    }        
  }     
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu20") == 0) {         
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	triggerBit[it] = true; 
      }
    }         
  }    
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu25") == 0) {            
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;   
	for(int i=0;i<NL1OpenMu;i++) {   
	  if(L1OpenMuPt[i] > 25.0)  
	    rc++;   
	}   
	if(rc > 0)   
	  triggerBit[it] = true; 
      }
    }            
  }           
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu30") == 0) {           
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;  
	for(int i=0;i<NL1OpenMu;i++) {  
	  if(L1OpenMuPt[i] > 30.0) 
	    rc++;  
	}  
	if(rc > 0)  
	  triggerBit[it] = true; 
      }
    }           
  }          
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu20HQ") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {   
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;   
	for(int i=0;i<NL1Mu;i++) {   
	  if(L1MuPt[i] > 20.0)  
	    if(L1MuQal[i] == 7) 
	      rc++;   
	}   
	if(rc > 0)   
	  triggerBit[it] = true; 
      }
    }            
  } 
// JH - no tracking paths
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu0") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(0.,0.,-1.,9999.,0)>=1) {  
	  triggerBit[it] = true; 
	}
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu3") == 0) {          
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1L2MuonPassed(0.0, 3.0, 9999.0) > 0)
	  triggerBit[it] = true; 
      }
    }
  }
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu5") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(3.,5.,-1.,9999.,0)>=1) {  
	  triggerBit[it] = true; 
	}
      }  
    }  
  }  
// JH
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2DoubleMu0") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1L2MuonPassed(0.,0.,9999.)>=2) {  
	  triggerBit[it] = true; 
	}
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2DoubleMu20") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1L2MuonPassed(0.,20.,9999.)>=2) {   
	  triggerBit[it] = true; 
	}
      }   
    }   
  }   

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu9") == 0) {           
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1L2MuonPassed(7.,9.,9999.)>=1) {    
	  triggerBit[it] = true; 
	}
      }    
    }           
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu11") == 0) {            
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {   
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1L2MuonPassed(7.,11.,9999.)>=1) {     
	  triggerBit[it] = true; 
	}
      }     
    }            
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu15") == 0) {             
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {    
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1L2MuonPassed(7.,15.,9999.)>=1) {      
	  triggerBit[it] = true; 
	}
      }      
    }             
  }   
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu25") == 0) {              
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {     
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1L2MuonPassed(7.,25.,9999.)>=1) {       
	  triggerBit[it] = true; 
	}
      }       
    }              
  }    
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu30") == 0) {               
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {      
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1L2MuonPassed(7.,30.,9999.)>=1) {        
	  triggerBit[it] = true; 
	}
      }        
    }               
  }     

////////////////////////////////////////////////////////////////////////
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu3") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(0.,0.,3.,2.,0)>=1) {  
	  triggerBit[it] = true; 
	}
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu0_L1MuOpen") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(0.,0.,0.,2.,0)>=1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu0_v1") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(0.,0.,0.,2.,0)>=1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu3_L1MuOpen") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(0.,3.,3.,2.,0)>=1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_L1MuOpen") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu3Mu0") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt2MuonPassed(0.,0.,0.,2.,0)>=2 && OpenHlt1MuonPassed(0.,3.,3.,2.,0)>=1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Onia") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	//       cout << "checking for Onia " << endl;
	//variables for pixel cuts
	double ptPix = 0.;
	double pPix = 3.;
	double etaPix = 999.;
	double DxyPix = 999.;
	double DzPix = 999.;
	int NHitsPix = 3;
	double normChi2Pix = 999999999.;
	double massMinPix[2] = {2.6, 7.5};
	double massMaxPix[2] = {3.6, 12.0};
	double DzMuonPix = 999.;
	bool   checkChargePix = false;
	//variables for tracker track cuts
	double ptTrack = 0.;
	double pTrack = 3.;
	double etaTrack = 999.;
	double DxyTrack = 999.;
	double DzTrack = 999.;
	int NHitsTrack = 5;
	double normChi2Track = 999999999.;
	double massMinTrack[2] = {2.8, 8.5};
	double massMaxTrack[2] = {3.4, 11.0};
	double DzMuonTrack = 0.5;
	bool   checkChargeTrack = true;
	if((OpenHlt1MuonPassed(0.,3.,3.,2.,0)>=1) && //check the L3 muon
	   OpenHltMuPixelPassed(ptPix, pPix, etaPix, DxyPix, DzPix, NHitsPix, normChi2Pix, massMinPix, massMaxPix, DzMuonPix, checkChargePix) && //check the L3Mu + pixel
	   OpenHltMuTrackPassed(ptTrack, pTrack, etaTrack, DxyTrack, DzTrack, NHitsTrack, normChi2Track, massMinTrack, massMaxTrack, DzMuonTrack, checkChargeTrack)) { //check the L3Mu + tracker track
	  triggerBit[it] = true; 
	}
	//        if (GetIntRandom() % menu->GetPrescale(it) == 0) { triggerBit[it] = true; }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu0_Track0_Ups") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	//       cout << "checking for Onia " << endl;
	//variables for pixel cuts
	double ptPix = 0.;
	double pPix = 3.;
	double etaPix = 999.;
	double DxyPix = 999.;
	double DzPix = 999.;
	int NHitsPix = 3;
	double normChi2Pix = 999999999.;
	double massMinPix[1] = {7.5};
	double massMaxPix[1] = {12.0};
	double DzMuonPix = 999.;
	bool   checkChargePix = false;
	//variables for tracker track cuts
	double ptTrack = 0.;
	double pTrack = 3.;
	double etaTrack = 999.;
	double DxyTrack = 999.;
	double DzTrack = 999.;
	int NHitsTrack = 5;
	double normChi2Track = 999999999.;
	double massMinTrack[1] = {8.5};
	double massMaxTrack[1] = {11.0};
	double DzMuonTrack = 0.5;
	bool   checkChargeTrack = true;
	if((OpenHlt1MuonPassed(0.,0.,0.,2.,0)>=1) && //check the L3 muon
	   OpenHltMuPixelPassed_Ups(ptPix, pPix, etaPix, DxyPix, DzPix, NHitsPix, normChi2Pix, massMinPix, massMaxPix, DzMuonPix, checkChargePix, 7) && //check the L3Mu + pixel
	   OpenHltMuTrackPassed_Ups(ptTrack, pTrack, etaTrack, DxyTrack, DzTrack, NHitsTrack, normChi2Track, massMinTrack, massMaxTrack, DzMuonTrack, checkChargeTrack, 7)) {
	  //check the L3Mu + tracker track
	  //        if (GetIntRandom() % menu->GetPrescale(it) == 0) { triggerBit[it] = true; }
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu3_Track0_Ups") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	//       cout << "checking for Onia " << endl;
	//variables for pixel cuts
	double ptPix = 0.;
	double pPix = 3.;
	double etaPix = 999.;
	double DxyPix = 999.;
	double DzPix = 999.;
	int NHitsPix = 3;
	double normChi2Pix = 999999999.;
	double massMinPix[1] = {7.5};
	double massMaxPix[1] = {12.0};
	double DzMuonPix = 999.;
	bool   checkChargePix = false;
	//variables for tracker track cuts
	double ptTrack = 0.;
	double pTrack = 3.;
	double etaTrack = 999.;
	double DxyTrack = 999.;
	double DzTrack = 999.;
	int NHitsTrack = 5;
	double normChi2Track = 999999999.;
	double massMinTrack[1] = {8.5};
	double massMaxTrack[1] = {11.0};
	double DzMuonTrack = 0.5;
	bool   checkChargeTrack = true;
	if((OpenHlt1MuonPassed(0.,3.,3.,2.,0)>=1) && //check the L3 muon
	   OpenHltMuPixelPassed_Ups(ptPix, pPix, etaPix, DxyPix, DzPix, NHitsPix, normChi2Pix, massMinPix, massMaxPix, DzMuonPix, checkChargePix, 8) && //check the L3Mu + pixel
	   OpenHltMuTrackPassed_Ups(ptTrack, pTrack, etaTrack, DxyTrack, DzTrack, NHitsTrack, normChi2Track, massMinTrack, massMaxTrack, DzMuonTrack, checkChargeTrack, 8)) {
	  //check the L3Mu + tracker track
	  //        if (GetIntRandom() % menu->GetPrescale(it) == 0) { triggerBit[it] = true; }
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_Track0_Ups") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	//       cout << "checking for Onia " << endl;
	//variables for pixel cuts
	double ptPix = 0.;
	double pPix = 3.;
	double etaPix = 999.;
	double DxyPix = 999.;
	double DzPix = 999.;
	int NHitsPix = 3;
	double normChi2Pix = 999999999.;
	double massMinPix[1] = {7.5};
	double massMaxPix[1] = {12.0};
	double DzMuonPix = 999.;
	bool   checkChargePix = false;
	//variables for tracker track cuts
	double ptTrack = 0.;
	double pTrack = 3.;
	double etaTrack = 999.;
	double DxyTrack = 999.;
	double DzTrack = 999.;
	int NHitsTrack = 5;
	double normChi2Track = 999999999.;
	double massMinTrack[1] = {8.5};
	double massMaxTrack[1] = {11.0};
	double DzMuonTrack = 0.5;
	bool   checkChargeTrack = true;
	if((OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1) && //check the L3 muon
	   OpenHltMuPixelPassed_Ups(ptPix, pPix, etaPix, DxyPix, DzPix, NHitsPix, normChi2Pix, massMinPix, massMaxPix, DzMuonPix, checkChargePix, 9) && //check the L3Mu + pixel
	   OpenHltMuTrackPassed_Ups(ptTrack, pTrack, etaTrack, DxyTrack, DzTrack, NHitsTrack, normChi2Track, massMinTrack, massMaxTrack, DzMuonTrack, checkChargeTrack, 9)) { 
	  //check the L3Mu + tracker track
	  //	if (GetIntRandom() % menu->GetPrescale(it) == 0) { triggerBit[it] = true; }
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu0_Track0_Jpsi") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	//       cout << "checking for Onia " << endl;
	//variables for pixel cuts
	double ptPix = 0.;
	double pPix = 3.;
	double etaPix = 999.;
	double DxyPix = 999.;
	double DzPix = 999.;
	int NHitsPix = 3;
	double normChi2Pix = 999999999.;
	double massMinPix[1] = {2.6};
	double massMaxPix[1] = {3.6};
	double DzMuonPix = 999.;
	bool   checkChargePix = false;
	//variables for tracker track cuts
	double ptTrack = 0.;
	double pTrack = 3.;
	double etaTrack = 999.;
	double DxyTrack = 999.;
	double DzTrack = 999.;
	int NHitsTrack = 5;
	double normChi2Track = 999999999.;
	double massMinTrack[1] = {2.8};
	double massMaxTrack[1] = {3.4};
	double DzMuonTrack = 0.5;
	bool   checkChargeTrack = true;
	if((OpenHlt1MuonPassed(0.,0.,0.,2.,0)>=1) && //check the L3 muon
	   OpenHltMuPixelPassed_JPsi(ptPix, pPix, etaPix, DxyPix, DzPix, NHitsPix, normChi2Pix, massMinPix, massMaxPix, DzMuonPix, checkChargePix, 0) && //check the L3Mu + pixel
	   OpenHltMuTrackPassed_JPsi(ptTrack, pTrack, etaTrack, DxyTrack, DzTrack, NHitsTrack, normChi2Track, massMinTrack, massMaxTrack, DzMuonTrack, checkChargeTrack, 0)) { 
	  //check the L3Mu + tracker track
	  //	if (GetIntRandom() % menu->GetPrescale(it) == 0) { triggerBit[it] = true; }
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu3_Track0_Jpsi") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	//       cout << "checking for Onia " << endl;
	//variables for pixel cuts
	double ptPix = 0.;
	double pPix = 3.;
	double etaPix = 999.;
	double DxyPix = 999.;
	double DzPix = 999.;
	int NHitsPix = 3;
	double normChi2Pix = 999999999.;
	double massMinPix[1] = {2.6};
	double massMaxPix[1] = {3.6};
	double DzMuonPix = 999.;
	bool   checkChargePix = false;
	//variables for tracker track cuts
	double ptTrack = 0.;
	double pTrack = 3.;
	double etaTrack = 999.;
	double DxyTrack = 999.;
	double DzTrack = 999.;
	int NHitsTrack = 5;
	double normChi2Track = 999999999.;
	double massMinTrack[1] = {2.8};
	double massMaxTrack[1] = {3.4};
	double DzMuonTrack = 0.5;
	bool   checkChargeTrack = true;
	if((OpenHlt1MuonPassed(0.,3.,3.,2.,0)>=1) && //check the L3 muon
	   OpenHltMuPixelPassed_JPsi(ptPix, pPix, etaPix, DxyPix, DzPix, NHitsPix, normChi2Pix, massMinPix, massMaxPix, DzMuonPix, checkChargePix, 5) && //check the L3Mu + pixel
	   OpenHltMuTrackPassed_JPsi(ptTrack, pTrack, etaTrack, DxyTrack, DzTrack, NHitsTrack, normChi2Track, massMinTrack, massMaxTrack, DzMuonTrack, checkChargeTrack,5)) { 
	  //check the L3Mu + tracker track
	  //	if (GetIntRandom() % menu->GetPrescale(it) == 0) { triggerBit[it] = true; }
	  triggerBit[it] = true; 
	}
      }
    }
  }

 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu3_Track5_Jpsi") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	//       cout << "checking for Onia " << endl;
	//variables for pixel cuts
	double ptPix = 0.;
	double pPix = 3.;
	double etaPix = 999.;
	double DxyPix = 999.;
	double DzPix = 999.;
	int NHitsPix = 3;
	double normChi2Pix = 999999999.;
	double massMinPix[1] = {2.6};
	double massMaxPix[1] = {3.6};
	double DzMuonPix = 999.;
	bool   checkChargePix = false;
	//variables for tracker track cuts
	double ptTrack = 5.;
	double pTrack = 3.;
	double etaTrack = 999.;
	double DxyTrack = 999.;
	double DzTrack = 999.;
	int NHitsTrack = 5;
	double normChi2Track = 999999999.;
	double massMinTrack[1] = {2.8};
	double massMaxTrack[1] = {3.4};
	double DzMuonTrack = 0.5;
	bool   checkChargeTrack = true;
	if((OpenHlt1MuonPassed(0.,3.,3.,2.,0)>=1) && //check the L3 muon
	   OpenHltMuPixelPassed_JPsi(ptPix, pPix, etaPix, DxyPix, DzPix, NHitsPix, normChi2Pix, massMinPix, massMaxPix, DzMuonPix, checkChargePix, 5) && //check the L3Mu + pixel
	   OpenHltMuTrackPassed_JPsi(ptTrack, pTrack, etaTrack, DxyTrack, DzTrack, NHitsTrack, normChi2Track, massMinTrack, massMaxTrack, DzMuonTrack, checkChargeTrack,5)) { 
	  //check the L3Mu + tracker track
	  //	if (GetIntRandom() % menu->GetPrescale(it) == 0) { triggerBit[it] = true; }
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_Track0_Jpsi") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	//       cout << "checking for Onia " << endl;
	//variables for pixel cuts
	double ptPix = 0.;
	double pPix = 3.;
	double etaPix = 999.;
	double DxyPix = 999.;
	double DzPix = 999.;
	int NHitsPix = 3;
	double normChi2Pix = 999999999.;
	double massMinPix[1] = {2.6};
	double massMaxPix[1] = {3.6};
	double DzMuonPix = 999.;
	bool   checkChargePix = false;
	//variables for tracker track cuts
	double ptTrack = 0.;
	double pTrack = 3.;
	double etaTrack = 999.;
	double DxyTrack = 999.;
	double DzTrack = 999.;
	int NHitsTrack = 5;
	double normChi2Track = 999999999.;
	double massMinTrack[1] = {2.8};
	double massMaxTrack[1] = {3.4};
	double DzMuonTrack = 0.5;
	bool   checkChargeTrack = true;
	if((OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1) && //check the L3 muon
	   OpenHltMuPixelPassed_JPsi(ptPix, pPix, etaPix, DxyPix, DzPix, NHitsPix, normChi2Pix, massMinPix, massMaxPix, DzMuonPix, checkChargePix, 6) && //check the L3Mu + pixel
	   OpenHltMuTrackPassed_JPsi(ptTrack, pTrack, etaTrack, DxyTrack, DzTrack, NHitsTrack, normChi2Track, massMinTrack, massMaxTrack, DzMuonTrack, checkChargeTrack, 6)) { 
	  //check the L3Mu + tracker track
	  //	if (GetIntRandom() % menu->GetPrescale(it) == 0) { triggerBit[it] = true; }
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(3.,3.,5.,2.,0)>=1) {   
	  triggerBit[it] = true; 
	}
      }   
    }   
  }   
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu7") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(5.,5.,7.,2.,0)>=1) {  
	  triggerBit[it] = true; 
	}
      }  
    }  
  }  
      
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu9") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(7.,7.,9.,2.,0)>=1) {  
	  triggerBit[it] = true; 
	}
      }  
    }  
  }  
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu10") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(7.,7.,10.,2.,0)>=1) {  
	  triggerBit[it] = true; 
	}
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu11") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(7.,7.,11.,2.,0)>=1) {   
	  triggerBit[it] = true; 
	}
      }   
    }   
  }   
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu12") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) { 
        if(OpenHlt1MuonPassed(7.,7.,12.,2.,0)>=1) {    
          triggerBit[it] = true;  
        } 
      }    
    }    
  }    
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu13") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(7.,7.,13.,2.,0)>=1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu15") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(10.,10.,15.,2.,0)>=1) {    
	  triggerBit[it] = true; 
	}
      }    
    }    
  }    
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu20") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) { 
        if(OpenHlt1MuonPassed(12.,12.,20.,2.,0)>=1) { 
          triggerBit[it] = true;  
        } 
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu24") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {  
        if(OpenHlt1MuonPassed(12.,12.,24.,2.,0)>=1) {  
          triggerBit[it] = true;   
        }  
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu30") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {  
        if(OpenHlt1MuonPassed(12.,12.,30.,2.,0)>=1) {  
          triggerBit[it] = true;   
        }  
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu20_NoVertex") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(7.,15.,20.,9999.,0)>=1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu30_NoVertex") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(7.,15.,30.,9999.,0)>=1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu50_NoVertex") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(7.,15.,50.,9999.,0)>=1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleMu3") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt2MuonPassed(0.,0.,3.,2.,0)>=2) {   
	  triggerBit[it] = true; 
	}
      }   
    }   
  }
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleMu3_HT100U") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt2MuonPassed(0.,0.,3.,2.,0)>=2 && OpenHltSumHTPassed(100,20)>0) {   
	  triggerBit[it] = true; 
	}
      }   
    }   
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleMu4_Exclusive") == 0) {    
    int rc = 0;
    float ptl2 = 3.0;
    float ptl3 = 4.0;
    float drl3 = 2.0;
    float etal3 = 2.4;
    float etal2 = 2.4;
    float deltaphil3 = 0.3;
    

    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) { 
	for (int i=0;i<NohMuL3;i++) {   
	  for(int j=i+1;j<NohMuL3;j++) {
	    if( fabs(ohMuL3Eta[i]) < etal3 && fabs(ohMuL3Eta[j]) < etal3 ) { // L3 eta cut   
	      if(ohMuL3Pt[i] > ptl3 && ohMuL3Pt[j] > ptl3) {  // L3 pT cut         
		if(ohMuL3Dr[i] < drl3 && ohMuL3Dr[j] < drl3) {  // L3 DR cut 
		  if((ohMuL3Chg[i] * ohMuL3Chg[j]) < 0) { // opposite charge
		    float deltaphi = fabs(ohMuL3Phi[i]-ohMuL3Phi[j]); 
		    if(deltaphi > 3.14159) 
		      deltaphi = (2.0 * 3.14159) - deltaphi;  

		    deltaphi = 3.14159 - deltaphi; 
		    if(deltaphi < deltaphil3) {
		      int l2match1 = ohMuL3L2idx[i];
		      int l2match2 = ohMuL3L2idx[j];
		      
		      if ( (fabs(ohMuL2Eta[l2match1]) < etal2) && (fabs(ohMuL2Eta[l2match2]) < etal2)) {  // L2 eta cut  
			if( (ohMuL2Pt[l2match1] > ptl2) && (ohMuL2Pt[l2match2] > ptl2) ) { // L2 pT cut  
			  rc++;
			}
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
    if(rc >=1)
      triggerBit[it] = true;  
  
  } 
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleMu5") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt2MuonPassed(0.,0.,5.,2.,0)>=2) {   
	  triggerBit[it] = true; 
	}
      }   
    }   
 }
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleMu6") == 0) {    
   if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
     if (prescaleResponse(menu,cfg,rcounter,it)) { 
       if(OpenHlt2MuonPassed(3.,3.,6.,2.,0)>=2) {    
	 triggerBit[it] = true;  
       } 
     }    
   }    
 } 
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleMu7") == 0) {    
   if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
     if (prescaleResponse(menu,cfg,rcounter,it)) { 
       if(OpenHlt2MuonPassed(3.,3.,7.,2.,0)>=2) {    
	 triggerBit[it] = true;  
       } 
     }    
   }    
 } 
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1DoubleMuOpen") == 0) {    
   if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
     if (prescaleResponse(menu,cfg,rcounter,it)) {
       if(1) {    // Pass through
	 triggerBit[it] = true; 
       }
     }    
   } 
 }
 else if(menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleMu0") == 0) {     
   if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt2MuonPassed(0.,0.,0.,2.,0)>=2) {     
	  triggerBit[it] = true; 
	}
      }
    }     
  }     
 else if(menu->GetTriggerName(it).CompareTo("OpenHLT_TripleMu5") == 0) {      
   if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
     if (prescaleResponse(menu,cfg,rcounter,it)) { 
       if(OpenHlt2MuonPassed(3.,3.,5.,2.,0)>=3) {      
	 triggerBit[it] = true;  
       } 
     } 
   }      
 }      

 else if(menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleMu0_Quarkonium") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      TLorentzVector mu1;  
      TLorentzVector mu2;  
      TLorentzVector diMu; 
      const double muMass = 0.105658367;
      int rc = 0; 
      for (int i=0;i<NohMuL3;i++) { 
	for (int j=0;j<NohMuL3 && j != i;j++) {  
	  
	  mu1.SetPtEtaPhiM(ohMuL3Pt[i],ohMuL3Eta[i],ohMuL3Phi[i],muMass); 
	  mu2.SetPtEtaPhiM(ohMuL3Pt[j],ohMuL3Eta[j],ohMuL3Phi[j],muMass); 
	  diMu = mu1 + mu2; 
	  int dimuCharge = (int) (ohMuL3Chg[i] + ohMuL3Chg[j]);
	  float diMuMass = diMu.M();
 	  if(diMuMass > 2.5 && diMuMass < 14.5 && dimuCharge == 0)
	    rc++;
	}
      }
      if(rc >= 1) {  
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }  
      }
    }
  }     
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleMu0_Quarkonium_LS") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      TLorentzVector mu1;  
      TLorentzVector mu2;  
      TLorentzVector diMu; 
      const double muMass = 0.105658367;
      int rc = 0; 
      for (int i=0;i<NohMuL3;i++) { 
	for (int j=0;j<NohMuL3 && j != i;j++) {  
	  
	  mu1.SetPtEtaPhiM(ohMuL3Pt[i],ohMuL3Eta[i],ohMuL3Phi[i],muMass); 
	  mu2.SetPtEtaPhiM(ohMuL3Pt[j],ohMuL3Eta[j],ohMuL3Phi[j],muMass); 
	  diMu = mu1 + mu2; 
	  int dimuCharge = (int) (ohMuL3Chg[i] + ohMuL3Chg[j]);
	  float diMuMass = diMu.M();
 	  if(diMuMass > 2.5 && diMuMass < 14.5 && dimuCharge != 0)
	    rc++;
	}
      }
      if(rc >= 1) {  
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }  
      }
    }
  }     

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleMu5_Ele8") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt2MuonPassed(0.,0.,5.,2.,0)>=2&&OpenHlt1ElectronPassed(8.,0.,9999.,9999.)>=1) {   
	  triggerBit[it] = true; 
	}
      }   
    }   
  }
 

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoMu3") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(3.,3.,3.,2.,1)>=1) {   
	  triggerBit[it] = true; 
	}
      }   
    }   
  }     
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoMu9") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(7.,7.,9.,2.,1)>=1) {   
	  triggerBit[it] = true; 
	}
      }   
    }   
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoMu11") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(7.,7.,11.,2.,1)>=1) {   
	  triggerBit[it] = true; 
	}
      }   
    }   
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoMu12") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) { 
        if(OpenHlt1MuonPassed(7.,7.,12.,2.,1)>=1) {    
          triggerBit[it] = true;  
        } 
      }    
    }    
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoMu13") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(7.,7.,13.,2.,1)>=1) {   
	  triggerBit[it] = true; 
	}
      }   
    }   
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoMu15") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) { 
        if(OpenHlt1MuonPassed(10.,10.,15.,2.,1)>=1) {    
          triggerBit[it] = true;  
        } 
      }    
    }    
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoMu17") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {   
      if (prescaleResponse(menu,cfg,rcounter,it)) {  
        if(OpenHlt1MuonPassed(10.,10.,17.,2.,1)>=1) {     
          triggerBit[it] = true;   
        }  
      }     
    }     
  }   
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoMu30") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {    
      if (prescaleResponse(menu,cfg,rcounter,it)) {   
        if(OpenHlt1MuonPassed(12.,12.,30.,2.,1)>=1) {      
          triggerBit[it] = true;    
        }   
      }      
    }      
  }    
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoEle12_IsoMu12") == 0) {       
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1ElectronPassed(12.,1,0.06,3.)>=1 && OpenHlt1MuonPassed(7.,7.,12.,2.,1)>=1) {       
	  triggerBit[it] = true; 
	}
      }       
    }       
  }


  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu14") == 0) {         
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	triggerBit[it] = true; 
      }
    }            
  }         
  
/* Electrons */
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1SingleEG2") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(true) { // passthrough      
	  triggerBit[it] = true; 
	}
      }      
    }      
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1SingleEG5") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(true) { // passthrough      
	  triggerBit[it] = true; 
	}
      }      
    }      
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1SingleEG8") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(true) { // passthrough       
	  triggerBit[it] = true; 
	}
      }       
    }       
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1DoubleEG5") == 0) {       
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(true) { // passthrough        
	  triggerBit[it] = true; 
	}
      }        
    }        
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele17_SW_TighterEleIdIsol_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHlt1ElectronSamHarperPassed(17.,0,          // ET, L1isolation
                                         999., 999.,       // Track iso barrel, Track iso endcap
                                         0.15, 0.10,        // Track/pT iso barrel, Track/pT iso endcap
                                         999., 0.05,       // H/ET iso barrel, H/ET iso endcap
                                         0.125, 0.075,       // E/ET iso barrel, E/ET iso endcap
                                         0.05, 0.05,       // H/E barrel, H/E endcap
                                         0.011, 0.031,       // cluster shape barrel, cluster shape endcap
                                         0.98, 1.0,       // R9 barrel, R9 endcap
                                         0.008, 0.007,       // Deta barrel, Deta endcap
                                         0.1, 0.1        // Dphi barrel, Dphi endcap
                                         )>=1) {
          triggerBit[it] = true;
        }
      }
    }
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele22_SW_TighterEleId_L1R") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) { 
        if(OpenHlt1ElectronSamHarperPassed(22.,0,          // ET, L1isolation 
					   999., 999.,       // Track iso barrel, Track iso endcap 
					   999., 999.,        // Track/pT iso barrel, Track/pT iso endcap 
					   999., 999.,       // H/ET iso barrel, H/ET iso endcap 
					   999., 999.,       // E/ET iso barrel, E/ET iso endcap 
					   0.05, 0.05,       // H/E barrel, H/E endcap 
					   0.011, 0.031,       // cluster shape barrel, cluster shape endcap 
					   0.98, 1.0,       // R9 barrel, R9 endcap 
					   0.008, 0.007,       // Deta barrel, Deta endcap 
					   0.1, 0.1        // Dphi barrel, Dphi endcap 
					   )>=1) { 
          triggerBit[it] = true; 
        } 
      } 
    } 
  } 
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele22_SW_TighterCaloIdIsol_L1R") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {  
        if(OpenHlt1ElectronSamHarperPassed(22.,0,          // ET, L1isolation  
                                           999., 999.,       // Track iso barrel, Track iso endcap  
                                           0.15, 0.10,        // Track/pT iso barrel, Track/pT iso endcap  
                                           999., 0.05,       // H/ET iso barrel, H/ET iso endcap  
                                           0.125, 0.075,       // E/ET iso barrel, E/ET iso endcap  
                                           0.05, 0.05,       // H/E barrel, H/E endcap  
                                           0.011, 0.031,       // cluster shape barrel, cluster shape endcap  
                                           0.98, 1.0,       // R9 barrel, R9 endcap  
                                           999., 999.,       // Deta barrel, Deta endcap  
                                           999., 999.        // Dphi barrel, Dphi endcap  
                                           )>=1) {  
          triggerBit[it] = true;  
        }  
      }  
    }  
  }  
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele32_SW_TighterEleId_L1R") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {  
        if(OpenHlt1ElectronSamHarperPassed(32.,0,          // ET, L1isolation  
                                           999., 999.,       // Track iso barrel, Track iso endcap  
                                           999., 999.,        // Track/pT iso barrel, Track/pT iso endcap  
                                           999., 999.,       // H/ET iso barrel, H/ET iso endcap  
                                           999., 999.,       // E/ET iso barrel, E/ET iso endcap  
                                           0.05, 0.05,       // H/E barrel, H/E endcap  
                                           0.011, 0.031,       // cluster shape barrel, cluster shape endcap  
                                           1.0, 1.0,       // R9 barrel, R9 endcap  
                                           0.008, 0.007,       // Deta barrel, Deta endcap  
                                           0.1, 0.1        // Dphi barrel, Dphi endcap  
                                           )>=1) {  
          triggerBit[it] = true;  
        }  
      }  
    }  
  }  
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele17_Ele8") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) { 
        if(OpenHlt2ElectronsAsymSamHarperPassed(17.,0,          // ET, L1isolation  
                                                999., 999.,       // Track iso barrel, Track iso endcap  
                                                999., 999.,        // Track/pT iso barrel, Track/pT iso endcap  
                                                999., 999.,       // H/ET iso barrel, H/ET iso endcap  
                                                999., 999.,       // E/ET iso barrel, E/ET iso endcap  
                                                0.15, 0.15,       // H/E barrel, H/E endcap  
                                                999., 999.,       // cluster shape barrel, cluster shape endcap  
                                                999., 999.,       // R9 barrel, R9 endcap  
                                                999., 999.,       // Deta barrel, Deta endcap  
                                                999., 999.,        // Dphi barrel, Dphi endcap  
                                                8.,0,          // ET, L1isolation 
                                                999., 999.,       // Track iso barrel, Track iso endcap 
                                                999., 999.,        // Track/pT iso barrel, Track/pT iso endcap 
                                                999., 999.,       // H/ET iso barrel, H/ET iso endcap 
                                                999., 999.,       // E/ET iso barrel, E/ET iso endcap 
                                                0.15, 0.15,       // H/E barrel, H/E endcap 
                                                999., 999.,       // cluster shape barrel, cluster shape endcap 
                                                999., 999.,       // R9 barrel, R9 endcap 
                                                999., 999.,       // Deta barrel, Deta endcap 
                                                999., 999.        // Dphi barrel, Dphi endcap 
                                                )>=1) { 
          triggerBit[it] = true; 
        } 
      } 
    } 
  } 

  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele17_Isol_Ele8") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHlt2ElectronsAsymSamHarperPassed(17.,0,          // ET, L1isolation 
						999., 999.,       // Track iso barrel, Track iso endcap 
						0.15, 0.10,        // Track/pT iso barrel, Track/pT iso endcap 
						999., 0.05,       // H/ET iso barrel, H/ET iso endcap 
						0.125, 0.075,       // E/ET iso barrel, E/ET iso endcap 
						0.15, 0.15,       // H/E barrel, H/E endcap 
						999., 999.,       // cluster shape barrel, cluster shape endcap 
						999., 999.,       // R9 barrel, R9 endcap 
						999., 999.,       // Deta barrel, Deta endcap 
						999., 999.,        // Dphi barrel, Dphi endcap 
						8.,0,          // ET, L1isolation
						999., 999.,       // Track iso barrel, Track iso endcap
						999., 999.,        // Track/pT iso barrel, Track/pT iso endcap
						999., 999.,       // H/ET iso barrel, H/ET iso endcap
						999., 999.,       // E/ET iso barrel, E/ET iso endcap
						0.15, 0.15,       // H/E barrel, H/E endcap
						999., 999.,       // cluster shape barrel, cluster shape endcap
						999., 999.,       // R9 barrel, R9 endcap
						999., 999.,       // Deta barrel, Deta endcap
						999., 999.        // Dphi barrel, Dphi endcap
						)>=1) {
          triggerBit[it] = true;
        }
      }
    }
  }

  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele17_VeryLooseIsol_Ele8") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHlt2ElectronsAsymSamHarperPassed(17.,0,          // ET, L1isolation 
						999., 999.,       // Track iso barrel, Track iso endcap 
						0.3, 0.3,        // Track/pT iso barrel, Track/pT iso endcap 
						999., 0.3,       // H/ET iso barrel, H/ET iso endcap 
						0.3, 0.3,       // E/ET iso barrel, E/ET iso endcap 
						0.15, 0.15,       // H/E barrel, H/E endcap 
						999., 999.,       // cluster shape barrel, cluster shape endcap 
						999., 999.,       // R9 barrel, R9 endcap 
						999., 999.,       // Deta barrel, Deta endcap 
						999., 999.,        // Dphi barrel, Dphi endcap 
						8.,0,          // ET, L1isolation
						999., 999.,       // Track iso barrel, Track iso endcap
						999., 999.,        // Track/pT iso barrel, Track/pT iso endcap
						999., 999.,       // H/ET iso barrel, H/ET iso endcap
						999., 999.,       // E/ET iso barrel, E/ET iso endcap
						0.15, 0.15,       // H/E barrel, H/E endcap
						999., 999.,       // cluster shape barrel, cluster shape endcap
						999., 999.,       // R9 barrel, R9 endcap
						999., 999.,       // Deta barrel, Deta endcap
						999., 999.        // Dphi barrel, Dphi endcap
						)>=1) {
          triggerBit[it] = true;
        }
      }
    }
  }

  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele17_Ele8_Isol") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHlt2ElectronsAsymSamHarperPassed(17.,0,          // ET, L1isolation 
						999., 999.,       // Track iso barrel, Track iso endcap 
						999., 999.,        // Track/pT iso barrel, Track/pT iso endcap 
						999., 999.,       // H/ET iso barrel, H/ET iso endcap 
						999., 999.,       // E/ET iso barrel, E/ET iso endcap 
						0.15, 0.15,       // H/E barrel, H/E endcap 
						999., 999.,       // cluster shape barrel, cluster shape endcap 
						999., 999.,       // R9 barrel, R9 endcap 
						999., 999.,       // Deta barrel, Deta endcap 
						999., 999.,        // Dphi barrel, Dphi endcap 
						8.,0,          // ET, L1isolation
						999., 999.,       // Track iso barrel, Track iso endcap
						0.15, 0.10,        // Track/pT iso barrel, Track/pT iso endcap
						999., 0.05,       // H/ET iso barrel, H/ET iso endcap
						0.125, 0.075,       // E/ET iso barrel, E/ET iso endcap
						0.15, 0.15,       // H/E barrel, H/E endcap
						999., 999.,       // cluster shape barrel, cluster shape endcap
						999., 999.,       // R9 barrel, R9 endcap
						999., 999.,       // Deta barrel, Deta endcap
						999., 999.        // Dphi barrel, Dphi endcap
						)>=1) {
          triggerBit[it] = true;
        }
      }
    }
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele17_Ele8_VeryLooseIsol") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHlt2ElectronsAsymSamHarperPassed(17.,0,          // ET, L1isolation 
						999., 999.,       // Track iso barrel, Track iso endcap 
						999., 999.,        // Track/pT iso barrel, Track/pT iso endcap 
						999., 999.,       // H/ET iso barrel, H/ET iso endcap 
						999., 999.,       // E/ET iso barrel, E/ET iso endcap 
						0.15, 0.15,       // H/E barrel, H/E endcap 
						999., 999.,       // cluster shape barrel, cluster shape endcap 
						999., 999.,       // R9 barrel, R9 endcap 
						999., 999.,       // Deta barrel, Deta endcap 
						999., 999.,        // Dphi barrel, Dphi endcap 
						8.,0,          // ET, L1isolation
						999., 999.,       // Track iso barrel, Track iso endcap
						0.3, 0.3,        // Track/pT iso barrel, Track/pT iso endcap
						999., 0.3,       // H/ET iso barrel, H/ET iso endcap
						0.3, 0.3,       // E/ET iso barrel, E/ET iso endcap
						0.15, 0.15,       // H/E barrel, H/E endcap
						999., 999.,       // cluster shape barrel, cluster shape endcap
						999., 999.,       // R9 barrel, R9 endcap
						999., 999.,       // Deta barrel, Deta endcap
						999., 999.        // Dphi barrel, Dphi endcap
						)>=1) {
          triggerBit[it] = true;
        }
      }
    }
  }

  //    # Grid of conditions on both legs

  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele17_EleId_Ele8") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHlt2ElectronsAsymSamHarperPassed(17.,0,          // ET, L1isolation 
						999., 999.,       // Track iso barrel, Track iso endcap 
						999., 999.,        // Track/pT iso barrel, Track/pT iso endcap 
						999., 999.,       // H/ET iso barrel, H/ET iso endcap 
						999., 999.,       // E/ET iso barrel, E/ET iso endcap 
						0.15, 0.15,       // H/E barrel, H/E endcap 
						0.014, 0.035,       // cluster shape barrel, cluster shape endcap 
						999., 999.,       // R9 barrel, R9 endcap 
						0.01, 0.01,       // Deta barrel, Deta endcap 
						0.08, 0.08,        // Dphi barrel, Dphi endcap 
						8.,0,          // ET, L1isolation
						999., 999.,       // Track iso barrel, Track iso endcap
						999., 999.,        // Track/pT iso barrel, Track/pT iso endcap
						999., 999.,       // H/ET iso barrel, H/ET iso endcap
						999., 999.,       // E/ET iso barrel, E/ET iso endcap
						0.15, 0.15,       // H/E barrel, H/E endcap
						999., 999.,       // cluster shape barrel, cluster shape endcap
						999., 999.,       // R9 barrel, R9 endcap
						999., 999.,       // Deta barrel, Deta endcap
						999., 999.        // Dphi barrel, Dphi endcap
						)>=1) {
          triggerBit[it] = true;
        }
      }
    }
  }

  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele17_EleId_Isol_Ele8") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHlt2ElectronsAsymSamHarperPassed(17.,0,          // ET, L1isolation 
						999., 999.,       // Track iso barrel, Track iso endcap 
						0.15, 0.10,        // Track/pT iso barrel, Track/pT iso endcap 
						999., 0.05,       // H/ET iso barrel, H/ET iso endcap 
						0.125, 0.075,       // E/ET iso barrel, E/ET iso endcap 
						0.15, 0.15,       // H/E barrel, H/E endcap 
						0.014, 0.035,       // cluster shape barrel, cluster shape endcap 
						999., 999.,       // R9 barrel, R9 endcap 
						0.01, 0.01,       // Deta barrel, Deta endcap 
						0.08, 0.08,        // Dphi barrel, Dphi endcap 
						8.,0,          // ET, L1isolation
						999., 999.,       // Track iso barrel, Track iso endcap
						999., 999.,        // Track/pT iso barrel, Track/pT iso endcap
						999., 999.,       // H/ET iso barrel, H/ET iso endcap
						999., 999.,       // E/ET iso barrel, E/ET iso endcap
						0.15, 0.15,       // H/E barrel, H/E endcap
						999., 999.,       // cluster shape barrel, cluster shape endcap
						999., 999.,       // R9 barrel, R9 endcap
						999., 999.,       // Deta barrel, Deta endcap
						999., 999.        // Dphi barrel, Dphi endcap
						)>=1) {
          triggerBit[it] = true;
        }
      }
    }
  }

  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele17_EleId_Isol_Ele8_EleId_Isol") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHlt2ElectronsAsymSamHarperPassed(17.,0,          // ET, L1isolation 
						999., 999.,       // Track iso barrel, Track iso endcap 
						0.15, 0.10,        // Track/pT iso barrel, Track/pT iso endcap 
						999., 0.05,       // H/ET iso barrel, H/ET iso endcap 
						0.125, 0.075,       // E/ET iso barrel, E/ET iso endcap 
						0.15, 0.15,       // H/E barrel, H/E endcap 
						0.014, 0.035,       // cluster shape barrel, cluster shape endcap 
						999., 999.,       // R9 barrel, R9 endcap 
						0.01, 0.01,       // Deta barrel, Deta endcap 
						0.08, 0.08,        // Dphi barrel, Dphi endcap 
						8.,0,          // ET, L1isolation
						999., 999.,       // Track iso barrel, Track iso endcap
						0.15, 0.10,        // Track/pT iso barrel, Track/pT iso endcap
						999., 0.05,       // H/ET iso barrel, H/ET iso endcap
						0.125, 0.075,       // E/ET iso barrel, E/ET iso endcap
						0.15, 0.15,       // H/E barrel, H/E endcap
						0.014, 0.035,       // cluster shape barrel, cluster shape endcap
						999., 999.,       // R9 barrel, R9 endcap
						0.01, 0.01,       // Deta barrel, Deta endcap
						0.08, 0.08        // Dphi barrel, Dphi endcap
						)>=1) {
          triggerBit[it] = true;
        }
      }
    }
  }
  // ########################################################
  // # End of long sequence of asym DiEle
  // ########################################################

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleEle5_SW_Jpsi_L1R") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt2ElectronMassWinPassed(5.0, 0, 9.0, 2.0, 4.5)>=1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleEle5_SW_Upsilon_L1R") == 0) {  
      if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) { 
	if (prescaleResponse(menu,cfg,rcounter,it)) {
	  if(OpenHlt2ElectronMassWinPassed(5.0, 0, 9.0, 8.0, 11.0)>=1) {
	    triggerBit[it] = true; 
	  }
	}
      }
  }  
  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleEle5_SW_L1R") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1ElectronPassed(5.,0,9999.,9999.)>=2) {       
	  triggerBit[it] = true; 
	}
      }      
    }      
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele10_SW_L1R") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1ElectronPassed(10.,0,9999.,9999.)>=1) { 
	  triggerBit[it] = true; 
	}
      }      
    }      
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele15_SW_L1R") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1ElectronPassed(15.,0,9999.,9999.)>=1) { 
	  triggerBit[it] = true; 
	}
      }      
    }      
  }        
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele20_SW_L1R") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1ElectronPassed(20.,0,9999.,9999.)>=1) { 
	  triggerBit[it] = true; 
	}
      }      
    }      
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele25_SW_L1R") == 0) {       
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1ElectronPassed(25.,0,9999.,9999.)>=1) {  
	  triggerBit[it] = true; 
	}
      }       
    }       
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele70_SW_L1R") == 0) {       
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1ElectronPassed(70.,0,9999.,9999.)>=1) {  
	  triggerBit[it] = true; 
	}
      }       
    }       
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele15_SW_LooseTrackIso_L1R") == 0) {       
    float Et = 15.;  
    int L1iso = 0;   
    float Tiso = 8.0;   
    float Hiso = 9999.; 
    float Tisoratio = 0.5; 
    int rc = 0; 
    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {       
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	for (int i=0;i<NohEle;i++) {  
	  if ( ohEleEt[i] > Et) {  
	    if( TMath::Abs(ohEleEta[i]) < 2.65 )
	      if ( ohEleHiso[i] < Hiso || ohEleHiso[i]/ohEleEt[i] < 0.05)  
		if (ohEleNewSC[i]==1)  
		  if (ohElePixelSeeds[i]>0)  
		    if ( (ohEleTiso[i] < Tisoratio || (ohEleTiso[i]*ohEleEt[i]) < Tiso) && ohEleTiso[i] != -999.)  
		      if ( ohEleL1iso[i] >= L1iso )   // L1iso is 0 or 1  
			if( ohEleL1Dupl[i] == false) // remove double-counted L1 SCs    
			  rc++;        
	  }  
	}  
      
	if(rc > 0) 
	  triggerBit[it] = true; 
      }
    }        
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoEle18_L1R") == 0) {       
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1ElectronPassed(18.,1,0.06,3.)>=1) {       
	  triggerBit[it] = true; 
	}
      }       
    }       
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleEle10_SW_L1R") == 0) {       
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1ElectronPassed(10.,0,9999.,9999.)>=2) {        
	  triggerBit[it] = true; 
	}
      }       
    }       
  } 
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleEle20_SW_L1R") == 0) {       
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1ElectronPassed(20.,0,9999.,9999.)>=2) {        
	  triggerBit[it] = true; 
	}
      }       
    }       
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleEle10_SW_1EleId_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt2Electron1LegIdPassed(10.,0,9999.,9999.)>=2) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleEle12_SW_1EleId_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt2Electron1LegIdPassed(12.,0,9999.,9999.)>=2) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleEle8_SW_1EleId_L1R_MET20") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt2Electron1LegIdPassed(8.,0,9999.,9999.)>=2 && recoMetCal>20) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleEle17_SW_1EleId_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt2Electron1LegIdPassed(17.,0,9999.,9999.)>=2) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele20_SC15_SW_L1R") == 0) {         
    float Et = 20.;  
    int L1iso = 0;   
    float Hiso = 9999.;  
    int rc = 0;  
 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	for (int i=0;i<NohEle;i++) {   
	  if ( ohEleEt[i] > Et) {   
	    if( TMath::Abs(ohEleEta[i]) < 2.65 ) {
	      if ( ohEleHiso[i] < Hiso || ohEleHiso[i]/ohEleEt[i] < 0.05) {  
		if (ohEleNewSC[i]==1) {  
		  if (ohElePixelSeeds[i]>0) {  
		    if ( ohEleL1iso[i] >= L1iso ) {  // L1iso is 0 or 1   
		      for(int j=0;j<NohEle && j != i;j++) {  
			if(ohEleEt[j] > 15.) {  
			  if(TMath::Abs(ohEleEt[j]) < 2.65 ) {
			    if( ohEleL1Dupl[i] == false && ohEleL1Dupl[j] == false) // remove double-counted L1 SCs  
			      rc++;         
			  }  
			}
		      }  
		    }  
		  }
		}  
	      }  
	    }  
	  }  
	} 
      } 
      if(rc >= 1) {  
	triggerBit[it] = true; 
      }
    } 
  }  
   
/* Photons */
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon32_SC32_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHlt1PhotonPassedRA3(32.,0,999.,999.,999.,999.,0.075,0.075,0.98,1.0)>=2) { 
          triggerBit[it] = true; 
        }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon32_SC26_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHlt1PhotonPassedRA3(32.,0,999.,999.,999.,999.,0.075,0.075,0.98,1.0)>=1  && OpenHlt1PhotonPassedRA3(26.,0,999.,999.,999.,999.,0.075,0.075,0.98,1.0)>=2) { 
          triggerBit[it] = true; 
        }
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon28_SC26_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHlt1PhotonPassedRA3(28.,0,999.,999.,999.,999.,0.15,0.15,0.98,1.0)>=1  && OpenHlt1PhotonPassedRA3(26.,0,999.,999.,999.,999.,0.15,0.15,0.98,1.0)>=2) { 
          triggerBit[it] = true; 
        }
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon70_HT200_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHlt1PhotonPassedRA3(70.,0,999.,999.,999.,999.,0.075,0.075,0.98,1.0)>=1  && OpenHltSumHTPassed(200.,30.)>=1) { 
          triggerBit[it] = true; 
        }
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon70_HT300_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHlt1PhotonPassedRA3(70.,0,999.,999.,999.,999.,0.075,0.075,0.98,1.0)>=1  && OpenHltSumHTPassed(300.,30.)>=1) { 
          triggerBit[it] = true; 
        }
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon60_HT200_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHlt1PhotonPassedRA3(60.,0,999.,999.,999.,999.,0.075,0.075,0.98,1.0)>=1  && OpenHltSumHTPassed(200.,30.)>=1) { 
          triggerBit[it] = true; 
        }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon70_MHT30_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHlt1PhotonPassedRA3(70.,0,999.,999.,999.,999.,0.075,0.075,0.98,1.0)>=1  && OpenHltMHT(30., 20.)==1) { 
          triggerBit[it] = true; 
        }
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon70_MHT50_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHlt1PhotonPassedRA3(70.,0,999.,999.,999.,999.,0.075,0.075,0.98,1.0)>=1  && OpenHltMHT(50., 20.)==1) { 
          triggerBit[it] = true; 
        }
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_EgammaSuperClusterOnly_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1PhotonPassed(5.,0,999.,999.,999.,999.)>=1) { // added track iso!
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon20_L1R") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1PhotonPassed(20.,0,999.,999.,999.,999.)>=1) {     
	  triggerBit[it] = true; 
	}
      }     
    }     
  }      
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon25_L1R") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1PhotonPassed(25.,0,999.,999.,999.,999.)>=1) {       
	  triggerBit[it] = true; 
	}
      }       
    }       
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon30_L1R") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1PhotonPassed(30.,0,999.,999.,999.,999.)>=1) {      
	  triggerBit[it] = true; 
	}
      }      
    }
  }       
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon50_Cleaned_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1PhotonPassed(50.,0,999.,999.,999.,999.)>=1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Photon65_CaloEleId_Isol_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHlt1PhotonSamHarperPassed(65.,0,          // ET, L1isolation
                                         999., 999.,       // Track iso barrel, Track iso endcap
                                         0.2, 0.2,        // Track/pT iso barrel, Track/pT iso endcap
                                         0.2, 0.2,       // H/ET iso barrel, H/ET iso endcap
                                         0.2, 0.2,       // E/ET iso barrel, E/ET iso endcap
                                         0.15, 0.15,       // H/E barrel, H/E endcap
                                         0.014, 0.035,       // cluster shape barrel, cluster shape endcap
                                         0.98, 999.,       // R9 barrel, R9 endcap
                                         999., 999.,       // Deta barrel, Deta endcap
                                         999., 999.        // Dphi barrel, Dphi endcap
                                         )>=1) {
          triggerBit[it] = true;
        }
      }
    }
  }

  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Photon70_Cleaned_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1PhotonPassed(70.,0,999.,999.,999.,999.)>=1) {       
	  triggerBit[it] = true; 
	}
      }       
    }       
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_DoublePhoton22_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
        if(OpenHlt1PhotonSamHarperPassed(22.,0,          // ET, L1isolation
                                         999., 999.,       // Track iso barrel, Track iso endcap
                                         999., 999.,        // Track/pT iso barrel, Track/pT iso endcap
                                         999., 999.,       // H/ET iso barrel, H/ET iso endcap
                                         999., 999.,       // E/ET iso barrel, E/ET iso endcap
                                         0.15, 0.15,       // H/E barrel, H/E endcap
                                         999., 999.,       // cluster shape barrel, cluster shape endcap
                                         999., 999.,       // R9 barrel, R9 endcap
                                         999., 999.,       // Deta barrel, Deta endcap
                                         999., 999.        // Dphi barrel, Dphi endcap
                                         )>=2) {
          triggerBit[it] = true;
        }
      }
    }
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Photon100_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1PhotonPassed(100.,0,999.,999.,999.,999.)>=1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Photon350_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1PhotonPassed(350.,0,999.,999.,999.,999.)>=1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Photon500_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1PhotonPassed(500.,0,999.,999.,999.,999.)>=1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }

/* Taus */
    
/*muon-Tau triggers*/
        else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Mu11_PFIsoTau15") == 0) {        
        if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
            if (prescaleResponse(menu,cfg,rcounter,it)) {
                if(OpenHlt1MuonPassed(7.,7.,11.,2.,0)>=1)
                if(OpenHltPFTauPassedNoMuon(15.,1.,1,1)>=1) { 
                    triggerBit[it] = true; 
                }
            }        
        }        
    }
          
        else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Mu15_PFIsoTau15") == 0) {        
            if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
                if (prescaleResponse(menu,cfg,rcounter,it)) {
                    if(OpenHlt1MuonPassed(7.,7.,15.,2.,0)>=1)
                        if(OpenHltPFTauPassedNoMuon(15.,1.,1,1)>=1) { 
                            triggerBit[it] = true; 
                        }
                }        
            }        
        }
        else if(menu->GetTriggerName(it).CompareTo("OpenHLT_IsoMu11_PFIsoTau15") == 0) {        
            if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
                if (prescaleResponse(menu,cfg,rcounter,it)) {
                    if(OpenHlt1MuonPassed(7.,7.,11.,2.,1)>=1)
                        if(OpenHltPFTauPassedNoMuon(15.,1.,1,1)>=1) { 
                            triggerBit[it] = true; 
                        }
                }        
            }        
        }
    
    
    /*Ele-Tau triggers*/
        else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele12_PFIsoTau15") == 0) {        
            if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
                if (prescaleResponse(menu,cfg,rcounter,it)) {
                    //PUT HERE THE ELECTRON PART
                        if(OpenHltPFTauPassedNoEle(15.,1.,1,1)>=1) { 
                            triggerBit[it] = true; 
                        }
                }        
            }        
        }
    
    
    else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleIsoTau15_Trk5") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltTauL2SCPassed(15.,5.,0,0.,1,14.,30.)>=2) { //Thresholds are for UNcorrected L1 jets in 8E29
	  triggerBit[it] = true; 
	}
      }
    }
  }
    else if (menu->GetTriggerName(it).CompareTo("OpenHLT_SingleIsoTau20_Trk15_MET25") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltTauL2SCMETPassed(20.,15.,0,0.,1,25.,20.,30.)>=1) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
    
    else if (menu->GetTriggerName(it).CompareTo("OpenHLT_SingleIsoTau30_Trk15_MET25") == 0) {
        if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
            if (prescaleResponse(menu,cfg,rcounter,it)) {
                if(OpenHltTauL2SCMETPassed(30.,15.,0,0.,1,25.,20.,30.)>=1) {
                    triggerBit[it] = true; 
                }
            }
        }
    }
    
    else if (menu->GetTriggerName(it).CompareTo("OpenHLT_SingleIsoTau30_Trk15_MET35") == 0) {
        if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
            if (prescaleResponse(menu,cfg,rcounter,it)) {
                if(OpenHltTauL2SCMETPassed(30.,15.,0,0.,1,35.,20.,30.)>=1) {
                    triggerBit[it] = true; 
                }
            }
        }
    }

    else if (menu->GetTriggerName(it).CompareTo("OpenHLT_SingleIsoTau35_Trk15_MET35") == 0) {
        if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
            if (prescaleResponse(menu,cfg,rcounter,it)) {
                if(OpenHltTauL2SCMETPassed(35.,15.,0,0.,1,35.,20.,30.)>=1) {
                    triggerBit[it] = true; 
                }
            }
        }
    }

    else if (menu->GetTriggerName(it).CompareTo("OpenHLT_SingleIsoTau30_Trk15_MET40") == 0) {
        if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
            if (prescaleResponse(menu,cfg,rcounter,it)) {
                if(OpenHltTauL2SCMETPassed(30.,15.,0,0.,1,40.,20.,30.)>=1) {
                    triggerBit[it] = true; 
                }
            }
        }
    }
    else if (menu->GetTriggerName(it).CompareTo("OpenHLT_SingleIsoTau40_Trk15_MET40") == 0) {
        if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
            if (prescaleResponse(menu,cfg,rcounter,it)) {
                if(OpenHltTauL2SCMETPassed(40.,15.,0,0.,1,40.,30.,40.)>=1) {
                    triggerBit[it] = true; 
                }
            }
        }
    }
    else if (menu->GetTriggerName(it).CompareTo("OpenHLT_SingleIsoTau30_Trk15_MET50") == 0) {
        if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
            if (prescaleResponse(menu,cfg,rcounter,it)) {
                if(OpenHltTauL2SCMETPassed(30.,15.,0,0.,1,50.,30.,40.)>=1) {
                    triggerBit[it] = true; 
                }
            }
        }
    }
    else if (menu->GetTriggerName(it).CompareTo("OpenHLT_SingleIsoTau30_Trk0_MET50") == 0) {
        if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
            if (prescaleResponse(menu,cfg,rcounter,it)) {
                if(OpenHltTauL2SCMETPassed(30.,0.,0,0.,1,50.,30.,40.)>=1) {
                    triggerBit[it] = true; 
                }
            }
        }
    }
    else if (menu->GetTriggerName(it).CompareTo("OpenHLT_SingleTau5_Trk0_MET50") == 0) {
        if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
            if (prescaleResponse(menu,cfg,rcounter,it)) {
                if(OpenHltTauL2SCMETPassed(5.,0.,0,0.,0,50.,30.,40.)>=1) {
                    triggerBit[it] = true; 
                }
            }
        }
    }
    else if (menu->GetTriggerName(it).CompareTo("OpenHLT_SingleTau5_Trk0_MET50_Level1_10") == 0) {
        if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
            if (prescaleResponse(menu,cfg,rcounter,it)) {
                if( recoMetCal>50){
                //                if(OpenHltTauL2SCMETPassed(5.,0.,0,0.,0,50.,10.,10.)>=1) {
                    triggerBit[it] = true; 
                }
            }
        }
    }
    else if (menu->GetTriggerName(it).CompareTo("OpenHLT_SinglePFIsoTau50_Trk15_PFMHT40") == 0) {
        if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
            if (prescaleResponse(menu,cfg,rcounter,it)) {
                if(OpenHltPFTauPassedNoMuon(50.,15.,1,1)>=1) {
                    if( pfMHT > 40)
                        triggerBit[it] = true; 
                }
            }
        }
    }
    else if (menu->GetTriggerName(it).CompareTo("OpenHLT_SinglePFIsoTau30_Trk15_PFMHT50") == 0) {
        if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
            if (prescaleResponse(menu,cfg,rcounter,it)) {
                if(OpenHltPFTauPassedNoMuon(30.,15.,1,1)>=1) {
                    if( pfMHT > 50)
                        triggerBit[it] = true; 
                }
            }
        }
    }
    
   /* End: Taus */

/* BTag */
else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_Jet10") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0; 
	int max =  (NohBJetL2Corrected > 2) ? 2 : NohBJetL2Corrected;
	for(int i = 0; i < max; i++) { 
	  if(ohBJetL2CorrectedEt[i] > 10.) { // ET cut
	    if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	      if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag
		rc++; 
	      } 
	    }
	  }
	}
	if(rc >= 1) { 
	  triggerBit[it] = true; 
	}
      } 
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_Jet20") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0; 
	int max =  (NohBJetL2Corrected > 2) ? 2 : NohBJetL2Corrected;
	for(int i = 0; i < max; i++) { 
	  if(ohBJetL2CorrectedEt[i] > 20.) { // ET cut
	    if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	      if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag
		rc++; 
	      } 
	    }
	  }
	}
	if(rc >= 1) { 
	  triggerBit[it] = true; 
	}
      } 
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_Jet10U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	  int rc = 0; 
	  int njets = 0; 
        
	  // apply L2 cut on jets 
	  for(int i = 0; i < NohBJetL2; i++)   
	    if(ohBJetL2Et[i] > 10. && abs(ohBJetL2Eta[i]) < 3.0)  // change this ET cut to 20 for the 20U patath 
	      njets++; 
	  
	  // apply b-tag cut 
	  int max =  (NohBJetL2 > 4) ? 4 : NohBJetL2;  
          for(int i = 0; i < max; i++) {
	    if(ohBJetL2Et[i] > 10.) { // keep this at 10 even for the 20UU path - also, no eta cut here 
	      if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag 
		if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag 
		  rc++; 
		} 
	      } 
	    } 
	  } 
	  if(rc >= 1 && njets>=1) { 
	    triggerBit[it] = true; 
	  } 
	}
      }
    }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_Jet20U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) { 
          int rc = 0;  
          int njets = 0;  
         
          // apply L2 cut on jets  
          for(int i = 0; i < NohBJetL2; i++)  
            if(ohBJetL2Et[i] > 20. && abs(ohBJetL2Eta[i]) < 3.0)  // change this ET cut to 20 for the 20U patath  
              njets++;  

          // apply b-tag cut  
	  int max =  (NohBJetL2 > 4) ? 4 : NohBJetL2; 
	  for(int i = 0; i < max; i++) {  
            if(ohBJetL2Et[i] > 10.) { // keep this at 10 even for the 20UU path - also, no eta cut here  
              if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag  
                if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag  
                  rc++;  
                }
              } 
            }
          }  
          if((rc >= 1) && (njets>=1)) {  
            triggerBit[it] = true;  
          }  
      } 
    } 
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_DiJet10U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
     if (prescaleResponse(menu,cfg,rcounter,it)) {
       int rc = 0;
       int njets = 0;
       
       // apply L2 cut on jets
       for(int i = 0; i < NohBJetL2; i++)
	 if(ohBJetL2Et[i] > 10. && abs(ohBJetL2Eta[i]) < 3.0)  // change this ET cut to 20 for the 20U patath
	   njets++;
       
       // apply b-tag cut
       for(int i = 0; i < NohBJetL2; i++) {
	 if(ohBJetL2Et[i] > 10.) { // keep this at 10 even for the 20UU path - also, no eta cut here
	   if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	     if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag
		   rc++;
	     }
	   }
	 }
       }
       if(rc >= 1 && njets>=2) {
	 triggerBit[it] = true;
       }
     }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_DiJet20U") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	int njets = 0;

	// apply L2 cut on jets
	for(int i = 0; i < NohBJetL2; i++)
	  if(ohBJetL2Et[i] > 20. && abs(ohBJetL2Eta[i]) < 3.0)  // change this ET cut to 20 for the 20U patath
	    njets++;

	// apply b-tag cut
	for(int i = 0; i < NohBJetL2; i++) {
	  if(ohBJetL2Et[i] > 10.) { // keep this at 10 even for the 20UU path - also, no eta cut here
	    if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	      if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag
		rc++;
	      }
	    }
	  }
	}
	if(rc >= 1 && njets>=2) {
	  triggerBit[it] = true;
	}
      }
    }
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_DiJet30U") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	int njets = 0;

	// apply L2 cut on jets
	for(int i = 0; i < NohBJetL2; i++)
	  if(ohBJetL2Et[i] > 30. && abs(ohBJetL2Eta[i]) < 3.0)  // change this ET cut to 20 for the 20U patath
	    njets++;

	// apply b-tag cut
	for(int i = 0; i < NohBJetL2; i++) {
	  if(ohBJetL2Et[i] > 10.) { // keep this at 10 even for the 20UU path - also, no eta cut here
	    if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	      if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag
		rc++;
	      }
	    }
	  }
	}
	if(rc >= 1 && njets>=2) {
	  triggerBit[it] = true;
	}
      }
    }
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_DiJet40U") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	int njets = 0;

	// apply L2 cut on jets
	for(int i = 0; i < NohBJetL2; i++)
	  if(ohBJetL2Et[i] > 40. && abs(ohBJetL2Eta[i]) < 3.0)
	    njets++;

	// apply b-tag cut
	for(int i = 0; i < NohBJetL2; i++) {
	  if(ohBJetL2Et[i] > 10.) { // keep this at 10 even for the 20UU path - also, no eta cut here
	    if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	      if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag
		rc++;
	      }
	    }
	  }
	}
	if(rc >= 1 && njets>=2) {
	  triggerBit[it] = true;
	}
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_DiJet50U") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	int njets = 0;

	// apply L2 cut on jets
	for(int i = 0; i < NohBJetL2; i++)
	  if(ohBJetL2Et[i] > 50. && abs(ohBJetL2Eta[i]) < 3.0)
	    njets++;

	// apply b-tag cut
	for(int i = 0; i < NohBJetL2; i++) {
	  if(ohBJetL2Et[i] > 10.) { // keep this at 10 even for the 20UU path - also, no eta cut here
	    if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	      if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag
		rc++;
	      }
	    }
	  }
	}
	if(rc >= 1 && njets>=2) {
	  triggerBit[it] = true;
	}
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_DiJet70U") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	int njets = 0;

	// apply L2 cut on jets
	for(int i = 0; i < NohBJetL2; i++)
	  if(ohBJetL2Et[i] > 70. && abs(ohBJetL2Eta[i]) < 3.0)
	    njets++;

	// apply b-tag cut
	for(int i = 0; i < NohBJetL2; i++) {
	  if(ohBJetL2Et[i] > 10.) { // keep this at 10 even for the 20UU path - also, no eta cut here
	    if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	      if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag
		rc++;
	      }
	    }
	  }
	}
	if(rc >= 1 && njets>=2) {
	  triggerBit[it] = true;
	}
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_DiJet100U") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	int njets = 0;

	// apply L2 cut on jets
	for(int i = 0; i < NohBJetL2; i++)
	  if(ohBJetL2Et[i] > 100. && abs(ohBJetL2Eta[i]) < 3.0)
	    njets++;

	// apply b-tag cut
	for(int i = 0; i < NohBJetL2; i++) {
	  if(ohBJetL2Et[i] > 10.) { // keep this at 10 even for the 20UU path - also, no eta cut here
	    if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	      if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag
		rc++;
	      }
	    }
	  }
	}
	if(rc >= 1 && njets>=2) {
	  triggerBit[it] = true;
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_DiJet10U_Mu5") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
     if (prescaleResponse(menu,cfg,rcounter,it)) {
       int rc = 0;
       int njets = 0;
       
       // apply L2 cut on jets
       for(int i = 0; i < NohBJetL2; i++)
	 if(ohBJetL2Et[i] > 10. && abs(ohBJetL2Eta[i]) < 3.0) 
	   njets++;
       
       // apply b-tag cut
       for(int i = 0; i < NohBJetL2; i++) {
	 if(ohBJetL2Et[i] > 10.) { // keep this at 10 even for all btag mu paths
	   if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	     if(OpenHlt1L3MuonPassed(5.0, 5.0) >=1 ) {//require at least one L3 muon
	       if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag
		 rc++;
	       }
	     }
	   }
	 }
       }
       if(rc >= 1 && njets>=2) {
	 triggerBit[it] = true;
       }
     }
    }
  }
  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_DiJet30U_Mu5") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	int njets = 0;
	
	// apply L2 cut on jets
	for(int i = 0; i < NohBJetL2; i++)
	  if(ohBJetL2Et[i] > 30. && abs(ohBJetL2Eta[i]) < 3.0)
	    njets++;
	
	// apply b-tag cut
	for(int i = 0; i < NohBJetL2; i++) {
	  if(ohBJetL2Et[i] > 10.) { // keep this at 10 for all btag mu paths - also, no eta cut here
	    if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	      if(OpenHlt1L3MuonPassed(5.0, 5.0) >=1 ) {//require at least one L3 muon
		if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag
		  rc++;
		}
	      }
	    }
	  }
	}
	if(rc >= 1 && njets>=2) {
	  triggerBit[it] = true;
	}
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_DiJet30U_Mu7") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	int njets = 0;
	
	// apply L2 cut on jets
	for(int i = 0; i < NohBJetL2; i++)
	  if(ohBJetL2Et[i] > 30. && abs(ohBJetL2Eta[i]) < 3.0)
	    njets++;
	
	// apply b-tag cut
	for(int i = 0; i < NohBJetL2; i++) {
	  if(ohBJetL2Et[i] > 10.) { // keep this at 10 for all btag mu paths - also, no eta cut here
	    if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	      if(OpenHlt1L3MuonPassed(7.0, 5.0) >=1 ) {//require at least one L3 muon
		if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag
		  rc++;
		}
	      }
	    }
	  }
	}
	if(rc >= 1 && njets>=2) {
	  triggerBit[it] = true;
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_DiJet50U_Mu5") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	int njets = 0;
	
	// apply L2 cut on jets
	for(int i = 0; i < NohBJetL2; i++)
	  if(ohBJetL2Et[i] > 50. && abs(ohBJetL2Eta[i]) < 3.0)
	    njets++;
	
	// apply b-tag cut
	for(int i = 0; i < NohBJetL2; i++) {
	  if(ohBJetL2Et[i] > 10.) { // keep this at 10 for all btag mu paths - also, no eta cut here
	    if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	      if(OpenHlt1L3MuonPassed(5.0, 5.0) >=1 ) {//require at least one L3 muon
		if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag
		  rc++;
		}
	      }
	    }
	  }
	}
	if(rc >= 1 && njets>=2) {
	  triggerBit[it] = true;
	}
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_DiJet50U_Mu7") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	int njets = 0;
	
	// apply L2 cut on jets
	for(int i = 0; i < NohBJetL2; i++)
	  if(ohBJetL2Et[i] > 50. && abs(ohBJetL2Eta[i]) < 3.0)
	    njets++;
	
	// apply b-tag cut
	for(int i = 0; i < NohBJetL2; i++) {
	  if(ohBJetL2Et[i] > 10.) { // keep this at 10 for all btag mu paths - also, no eta cut here
	    if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	      if(OpenHlt1L3MuonPassed(7.0, 5.0) >=1 ) {//require at least one L3 muon
		if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag
		  rc++;
		}
	      }
	    }
	  }
	}
	if(rc >= 1 && njets>=2) {
	  triggerBit[it] = true;
	}
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_DiJet50U_Mu9") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	int njets = 0;
	
	// apply L2 cut on jets
	for(int i = 0; i < NohBJetL2; i++)
	  if(ohBJetL2Et[i] > 50. && abs(ohBJetL2Eta[i]) < 3.0)
	    njets++;
	
	// apply b-tag cut
	for(int i = 0; i < NohBJetL2; i++) {
	  if(ohBJetL2Et[i] > 10.) { // keep this at 10 for all btag mu paths - also, no eta cut here
	    if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	      if(OpenHlt1L3MuonPassed(9.0, 5.0) >=1 ) {//require at least one L3 muon
		if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag
		  rc++;
		}
	      }
	    }
	  }
	}
	if(rc >= 1 && njets>=2) {
	  triggerBit[it] = true;
	}
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_DiJet70U_Mu9") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	int njets = 0;
	
	// apply L2 cut on jets
	for(int i = 0; i < NohBJetL2; i++)
	  if(ohBJetL2Et[i] > 70. && abs(ohBJetL2Eta[i]) < 3.0)
	    njets++;
	
	// apply b-tag cut
	for(int i = 0; i < NohBJetL2; i++) {
	  if(ohBJetL2Et[i] > 10.) { // keep this at 10 for all btag mu paths - also, no eta cut here
	    if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	      if(OpenHlt1L3MuonPassed(9.0, 5.0) >=1 ) {//require at least one L3 muon
		if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag
		  rc++;
		}
	      }
	    }
	  }
	}
	if(rc >= 1 && njets>=2) {
	  triggerBit[it] = true;
	}
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_DiJet70U_Mu11") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	int njets = 0;
	
	// apply L2 cut on jets
	for(int i = 0; i < NohBJetL2; i++)
	  if(ohBJetL2Et[i] > 70. && abs(ohBJetL2Eta[i]) < 3.0)
	    njets++;
	
	// apply b-tag cut
	for(int i = 0; i < NohBJetL2; i++) {
	  if(ohBJetL2Et[i] > 10.) { // keep this at 10 for all btag mu paths - also, no eta cut here
	    if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	      if(OpenHlt1L3MuonPassed(11.0, 5.0) >=1 ) {//require at least one L3 muon
		if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag
		  rc++;
		}
	      }
	    }
	  }
	}
	if(rc >= 1 && njets>=2) {
	  triggerBit[it] = true;
	}
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_DiJet70U_Mu15") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	int njets = 0;
	
	// apply L2 cut on jets
	for(int i = 0; i < NohBJetL2; i++)
	  if(ohBJetL2Et[i] > 70. && abs(ohBJetL2Eta[i]) < 3.0)
	    njets++;
	
	// apply b-tag cut
	for(int i = 0; i < NohBJetL2; i++) {
	  if(ohBJetL2Et[i] > 10.) { // keep this at 10 for all btag mu paths - also, no eta cut here
	    if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	      if(OpenHlt1L3MuonPassed(15.0, 5.0) >=1 ) {//require at least one L3 muon
		if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag
		  rc++;
		}
	      }
	    }
	  }
	}
	if(rc >= 1 && njets>=2) {
	  triggerBit[it] = true;
	}
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_DiJet100U_Mu11") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	int njets = 0;
	
	// apply L2 cut on jets
	for(int i = 0; i < NohBJetL2; i++)
	  if(ohBJetL2Et[i] > 100. && abs(ohBJetL2Eta[i]) < 3.0)
	    njets++;
	
	// apply b-tag cut
	for(int i = 0; i < NohBJetL2; i++) {
	  if(ohBJetL2Et[i] > 10.) { // keep this at 10 for all btag mu paths - also, no eta cut here
	    if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	      if(OpenHlt1L3MuonPassed(11.0, 5.0) >=1 ) {//require at least one L3 muon
		if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag
		  rc++;
		}
	      }
	    }
	  }
	}
	if(rc >= 1 && njets>=2) {
	  triggerBit[it] = true;
	}
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_DiJet100U_Mu15") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	int njets = 0;
	
	// apply L2 cut on jets
	for(int i = 0; i < NohBJetL2; i++)
	  if(ohBJetL2Et[i] > 100. && abs(ohBJetL2Eta[i]) < 3.0)
	    njets++;
	
	// apply b-tag cut
	for(int i = 0; i < NohBJetL2; i++) {
	  if(ohBJetL2Et[i] > 10.) { // keep this at 10 for all btag mu paths - also, no eta cut here
	    if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	      if(OpenHlt1L3MuonPassed(15.0, 5.0) >=1 ) {//require at least one L3 muon
		if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag
		  rc++;
		}
	      }
	    }
	  }
	}
	if(rc >= 1 && njets>=2) {
	  triggerBit[it] = true;
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagIP_Jet50") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;  
	int max =  (NohBJetL2Corrected > 2) ? 2 : NohBJetL2Corrected; 
	for(int i = 0; i < max; i++) {  
	  if(ohBJetL2CorrectedEt[i] > 50.) { // ET cut 
	    if(ohBJetIPL25Tag[i] > 2.5) { // Level 2.5 b tag 
	      if(ohBJetIPL3Tag[i] > 3.5) { // Level 3 b tag 
		rc++;  
	      }  
	    } 
	  } 
	} 
	if(rc >= 1) {  
	  triggerBit[it] = true; 
	}
      }
    }  
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagIP_Jet80") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;  
	int max =  (NohBJetL2Corrected > 2) ? 2 : NohBJetL2Corrected; 
	for(int i = 0; i < max; i++) {  
	  if(ohBJetL2CorrectedEt[i] > 80.) { // ET cut 
	    if(ohBJetIPL25Tag[i] > 2.5) { // Level 2.5 b tag 
	      if(ohBJetIPL3Tag[i] > 3.5) { // Level 3 b tag 
		rc++;  
	      }  
	    } 
	  } 
	} 
	if(rc >= 1) {  
	  triggerBit[it] = true; 
	}
      }
    }  
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagIP_Jet120") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;   
	int max =  (NohBJetL2Corrected > 2) ? 2 : NohBJetL2Corrected;  
	for(int i = 0; i < max; i++) {   
	  if(ohBJetL2CorrectedEt[i] > 120.) { // ET cut  
	    if(ohBJetIPL25Tag[i] > 2.5) { // Level 2.5 b tag  
	      if(ohBJetIPL3Tag[i] > 3.5) { // Level 3 b tag  
		rc++;   
	      }   
	    }  
	  }  
	}  
	if(rc >= 1) {   
	  triggerBit[it] = true; 
	}
      } 
    }   
  }
 

/* Minbias */
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MinBiasHcal") == 0) {         
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(true) { // passthrough       
	  triggerBit[it] = true; 
	}
      }       
    }         
  }         
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MinBiasEcal") == 0) {          
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(true) { // passthrough        
	  triggerBit[it] = true; 
	}
      }        
    }          
  }          
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_MinBiasBSC_OR") == 0) { 
    bool techTriggerBSCOR = (bool) L1Tech_BSC_minBias_OR_v0;
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(techTriggerBSCOR)
	  triggerBit[it] = true; 
      }
    } 
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_MinBiasBSC") == 0) {
    bool techTriggerBSC1 = (bool) L1Tech_BSC_minBias_threshold1_v0;
    bool techTriggerBSC2 = (bool) L1Tech_BSC_minBias_threshold2_v0;
    bool techTriggerBS3 = (bool) L1Tech_BSC_minBias_inner_threshold1_v0;
    bool techTriggerBS4 = (bool) L1Tech_BSC_minBias_inner_threshold2_v0;

    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(techTriggerBSC1 || techTriggerBSC2 || techTriggerBS3 || techTriggerBS4)
	  triggerBit[it] = true; 
      }
    }
  }
 else if(menu->GetTriggerName(it).CompareTo("OpenHLT_MinBiasPixel_SingleTrack") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1PixelTrackPassed(0.0, 1.0, 0.0)>=1){
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_MinBiasPixel_DoubleTrack") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1PixelTrackPassed(0.0, 1.0, 0.0)>=2){
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_MinBiasPixel_DoubleIsoTrack5") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1PixelTrackPassed(5.0, 1.0, 1.0)>=0){
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_ZeroBias") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(map_BitOfStandardHLTPath.find("OpenL1_ZeroBias")->second == 1)
	  triggerBit[it] = true; 
      }
    }
  }
  
  /* AlCa */
  else if (menu->GetTriggerName(it).CompareTo("OpenAlCa_HcalPhiSym") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(ohHighestEnergyHFRecHit > 0 || ohHighestEnergyHBHERecHit > 0) {
	  // Require one RecHit with E > 0 MeV in any HCAL subdetector
	  triggerBit[it] = true; 
	}
      }
    }
  }

 // else if (menu->GetTriggerName(it).CompareTo("OpenAlCa_EcalPi0") == 0) { 
//     if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 

//       TLorentzVector gamma1; 
//       TLorentzVector gamma2; 
//       TLorentzVector meson;
//       TLorentzVector gammaiso;
      
//       int rc = 0;

//       for(int i = 0; i < Nalcapi0clusters; i++) {
// 	for(int j = i+1;j < Nalcapi0clusters && j != i; j++) {

// 	  //if one is in barrel, the other one is in endcap, check next pair 
// 	  if( (TMath::Abs(ohAlcapi0etaClusAll[i]) < 1.479 && TMath::Abs(ohAlcapi0etaClusAll[j]) > 1.479	    )
// 	      || (TMath::Abs(ohAlcapi0etaClusAll[i]) > 1.479 && TMath::Abs(ohAlcapi0etaClusAll[j]) < 1.479  )
// 	      ){
// 	    continue; 
// 	  }
	  

// 	  gamma1.SetPtEtaPhiM(ohAlcapi0ptClusAll[i],ohAlcapi0etaClusAll[i],ohAlcapi0phiClusAll[i],0.0);
// 	  gamma2.SetPtEtaPhiM(ohAlcapi0ptClusAll[j],ohAlcapi0etaClusAll[j],ohAlcapi0phiClusAll[j],0.0);
// 	  meson = gamma1 + gamma2;
	  
//           float mesonpt = meson.Pt(); 
//           float mesoneta = meson.Eta(); 
//           float mesonmass = meson.M(); 

// 	  float iso = 0.0;
// 	  float dr = 0.0;
// 	  float deta = 0.0;
	  
	  
// 	  for(int k = 0;k < Nalcapi0clusters && k != i && k != j; k++) { 
// 	    gammaiso.SetPtEtaPhiM(ohAlcapi0ptClusAll[k],ohAlcapi0etaClusAll[k],ohAlcapi0phiClusAll[k],0.0);
// 	    dr = gammaiso.DeltaR(meson);
// 	    deta = TMath::Abs(ohAlcapi0etaClusAll[k] - mesoneta);
	    
// 	    if((dr < 0.2) && (deta < 0.05) && ohAlcapi0ptClusAll[k] > 0.5 ) {
// 	      iso = iso + ohAlcapi0ptClusAll[k];
// 	    }
// 	  }
// 	  iso /= mesonpt; 
	  
// 	  if(TMath::Abs(ohAlcapi0etaClusAll[i]) < 1.479 && TMath::Abs(ohAlcapi0etaClusAll[j]) < 1.479) {  
// 	    //pi0 barrel
// 	    if(ohAlcapi0ptClusAll[i] > 1.3 && ohAlcapi0ptClusAll[j] > 1.3 && 
// 	       ohAlcapi0s4s9ClusAll[i] > 0.83 && ohAlcapi0s4s9ClusAll[j] > 0.83 &&  
// 	       iso < 0.5 &&  
// 	       mesonpt > 2.6 && 
// 	       mesonmass > 0.04 && mesonmass < 0.23) 
// 	      rc++; 
// 	  }
// 	  if(TMath::Abs(ohAlcapi0etaClusAll[i]) > 1.479 && TMath::Abs(ohAlcapi0etaClusAll[j]) > 1.479)  {   
	    
// 	    if( ohAlcapi0s4s9ClusAll[i] > 0.9 && ohAlcapi0s4s9ClusAll[j] > 0.9 && iso <0.5 && 
// 		mesonmass > 0.05 && mesonmass < 0.3){
	      
// 	      if( (fabs(mesoneta)<2.0 && ohAlcapi0ptClusAll[i] > 0.6 && ohAlcapi0ptClusAll[j] >0.6 && mesonpt > 2.5) ||
// 		  (fabs(mesoneta)>2.0 && fabs(mesoneta)<2.5 && ohAlcapi0ptClusAll[i] > 0.6 && ohAlcapi0ptClusAll[j] >0.6 && mesonpt > 2.5) ||
// 		  (fabs(mesoneta)>2.5 && ohAlcapi0ptClusAll[i] > 0.5 && ohAlcapi0ptClusAll[j] >0.5 && mesonpt >1.0 && mesonpt < 2.5) ){
// 		rc ++; 
// 	      }
// 	    }
// 	  }//pi0 endcap
	  
// 	}
//       }
      
//       if(rc > 0) {
// 	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }          
//       }
//     } 
//   } 

//   else if (menu->GetTriggerName(it).CompareTo("OpenAlCa_EcalEta") == 0) {  
//     if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      
      
//       TLorentzVector gamma1;  
//       TLorentzVector gamma2;  
//       TLorentzVector meson; 
//       TLorentzVector gammaiso;
      
//       ///pi0 veto only for barrel 
//       std::vector<int> indpi0cand; 
//       for(int i = 0; i < Nalcapi0clusters; i++) { 

// 	if(  TMath::Abs(ohAlcapi0etaClusAll[i]) > 1.479) continue; 

//         for(int j = i+1;j < Nalcapi0clusters && j != i; j++) { 
// 	  if( TMath::Abs(ohAlcapi0etaClusAll[j]) > 1.479  )  continue; 
	  
// 	  gamma1.SetPtEtaPhiM(ohAlcapi0ptClusAll[i],ohAlcapi0etaClusAll[i],ohAlcapi0phiClusAll[i],0.0); 
//           gamma2.SetPtEtaPhiM(ohAlcapi0ptClusAll[j],ohAlcapi0etaClusAll[j],ohAlcapi0phiClusAll[j],0.0); 
//           meson = gamma1 + gamma2; 	  
// 	  float mesonmass = meson.M(); 
	  
// 	  if( mesonmass > 0.084 && mesonmass < 0.156){
	    
// 	    int tmp[2] = {i,j};
// 	    for(int n=0; n<2; n++){
// 	      vector<int>::iterator it = find(indpi0cand.begin(),indpi0cand.end(),tmp[n]); 
// 	      if( it == indpi0cand.end()){
// 		indpi0cand.push_back(tmp[n]);
// 	      }
// 	    }
// 	  }
	  
// 	}
//       }
            
      
//       int rc = 0; 
 
//       for(int i = 0; i < Nalcapi0clusters; i++) { 

// 	if(TMath::Abs(ohAlcapi0etaClusAll[i]) < 1.479){
// 	  vector<int>::iterator it = find(indpi0cand.begin(),indpi0cand.end(),i); 
// 	  if( it != indpi0cand.end() ) continue; 
// 	}
	
	
//         for(int j = i+1;j < Nalcapi0clusters && j != i; j++) { 

// 	  if(TMath::Abs(ohAlcapi0etaClusAll[j]) < 1.479){
// 	    vector<int>::iterator it = find(indpi0cand.begin(),indpi0cand.end(),j); 
// 	    if( it != indpi0cand.end() ) continue; 
// 	  }
	  
// 	  //if one is in barrel, the other one is in endcap, check next pair 
// 	  if( (TMath::Abs(ohAlcapi0etaClusAll[i]) < 1.479 && TMath::Abs(ohAlcapi0etaClusAll[j]) > 1.479	    )
// 	      || (TMath::Abs(ohAlcapi0etaClusAll[i]) > 1.479 && TMath::Abs(ohAlcapi0etaClusAll[j]) < 1.479  )
// 	      ){
// 	    continue; 
// 	  }
	  
	  
// 	  gamma1.SetPtEtaPhiM(ohAlcapi0ptClusAll[i],ohAlcapi0etaClusAll[i],ohAlcapi0phiClusAll[i],0.0); 
//           gamma2.SetPtEtaPhiM(ohAlcapi0ptClusAll[j],ohAlcapi0etaClusAll[j],ohAlcapi0phiClusAll[j],0.0); 
//           meson = gamma1 + gamma2; 
//           float mesonpt = meson.Pt(); 
//           float mesoneta = meson.Eta(); 
//           float mesonmass = meson.M(); 

//           float iso = 0.0; 
//           float dr = 0.0; 
//           float deta = 0.0; 
//           for(int k = 0;k < Nalcapi0clusters && k != i && k != j; k++) {  
//             gammaiso.SetPtEtaPhiM(ohAlcapi0ptClusAll[k],ohAlcapi0etaClusAll[k],ohAlcapi0phiClusAll[k],0.0); 
//             dr = gammaiso.DeltaR(meson); 
//             deta = TMath::Abs(ohAlcapi0etaClusAll[k] - mesoneta); 
// 	    if((dr < 0.3) && (deta < 0.1) && ohAlcapi0ptClusAll[k] > 0.5 ) {  
//               iso = iso + ohAlcapi0ptClusAll[k]; 
//             } 
//           } 
// 	  iso /= mesonpt; 
	  
	  
// 	  if(TMath::Abs(ohAlcapi0etaClusAll[i]) < 1.479 && TMath::Abs(ohAlcapi0etaClusAll[j]) < 1.479) { 
// 	    //eta barrel  
// 	    if(ohAlcapi0ptClusAll[i] > 1.2 && ohAlcapi0ptClusAll[j] > 1.2 &&    
//                ohAlcapi0s4s9ClusAll[i] > 0.87 && ohAlcapi0s4s9ClusAll[j] > 0.87 &&     
// 	       ohAlcapi0s9s25ClusAll[i] > 0.8 && ohAlcapi0s9s25ClusAll[j] > 0.8 &&     
// 	       iso < 0.5 && 
//                mesonpt > 4.0 &&    
// 	       mesonmass > 0.3 && mesonmass < 0.8)    
// 	      rc++;    
// 	  } 
	  
// 	  //eta endcap
// 	  if(TMath::Abs(ohAlcapi0etaClusAll[i]) > 1.479 && TMath::Abs(ohAlcapi0etaClusAll[j]) > 1.479) {  
	    
// 	    if( ohAlcapi0s4s9ClusAll[i] > 0.9 && ohAlcapi0s4s9ClusAll[j] > 0.9 && iso <0.5 && 
// 		ohAlcapi0s9s25ClusAll[i] > 0.85 && ohAlcapi0s9s25ClusAll[j] > 0.85 &&
// 		mesonmass > 0.2 && mesonmass < 0.9){
	      
// 	      if( (fabs(mesoneta)<2.0 && ohAlcapi0ptClusAll[i] > 1.0 && ohAlcapi0ptClusAll[j] >1.0 && mesonpt > 3.0 ) ||
// 		  (fabs(mesoneta)>2.0 && fabs(mesoneta)<2.5 && ohAlcapi0ptClusAll[i] > 1.0 && ohAlcapi0ptClusAll[j] >1.0 && mesonpt > 3.0) ||
// 		  (fabs(mesoneta)>2.5 && ohAlcapi0ptClusAll[i] > 0.7 && ohAlcapi0ptClusAll[j] >0.7 && mesonpt > 3.0) ){
// 		rc ++; 
// 	      }
// 	    }
	    
// 	  }
// 	} 
//       }
//       if(rc > 0) { 
// 	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }           
//       } 
//     }  
//   }
  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoTrackHB") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	
	bool passL2=false;
	for(int itrk=0; itrk<ohIsoPixelTrackHBL2N; itrk++){
	  if( ohIsoPixelTrackHBL2P[itrk]>8.0               &&
	      TMath::Abs(ohIsoPixelTrackHBL2Eta[itrk])>0.0 &&
	      TMath::Abs(ohIsoPixelTrackHBL2Eta[itrk])<1.3 &&
	      ohIsoPixelTrackHBL2MaxNearP[itrk]<2.0)
	    passL2=true;
	}
	
	bool passL3=false;
	for(int itrk=0; itrk<ohIsoPixelTrackHBL3N; itrk++){
	  if( ohIsoPixelTrackHBL3P[itrk]>20.0              &&
	      TMath::Abs(ohIsoPixelTrackHBL3Eta[itrk])>0.0 &&
	      TMath::Abs(ohIsoPixelTrackHBL3Eta[itrk])<1.3 &&
	      ohIsoPixelTrackHBL3MaxNearP[itrk]<2.0)
	    passL3=true;
	}
	
	if(passL2 && passL3) {
	  triggerBit[it] = true; 
	}
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoTrackHE") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	
	bool passL2=false;
	for(int itrk=0; itrk<ohIsoPixelTrackHEL2N; itrk++){
	  if( ohIsoPixelTrackHEL2P[itrk]>12.0              &&
	      TMath::Abs(ohIsoPixelTrackHEL2Eta[itrk])>0.0 &&
	      TMath::Abs(ohIsoPixelTrackHEL2Eta[itrk])<2.2 &&
	      ohIsoPixelTrackHEL2MaxNearP[itrk]<2.0)
	    passL2=true;
	}
	
	bool passL3=false;
	for(int itrk=0; itrk<ohIsoPixelTrackHEL3N; itrk++){
	  if( ohIsoPixelTrackHEL3P[itrk]>38.0              &&
	      TMath::Abs(ohIsoPixelTrackHEL3Eta[itrk])>0.0 &&
	      TMath::Abs(ohIsoPixelTrackHEL3Eta[itrk])<2.2 &&
	      ohIsoPixelTrackHEL3MaxNearP[itrk]<2.0)
	    passL3=true;
	}
	
	if(passL2 && passL3) {
	  triggerBit[it] = true; 
	}
      }
    }
  }

  /*New cross-triggers, Jan. 2011 */ 
  
/* New cross-triggers, Sept. 2010 */
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele10_MET45") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(recoMetCal > 45 && OpenHlt1ElectronPassed(10.,0.,9999.,9999.)>=1) { 
	  triggerBit[it] = true; 
	}  
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele12_MET45") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(recoMetCal > 45 && OpenHlt1ElectronPassed(12.,0.,9999.,9999.)>=1) { 
	  triggerBit[it] = true; 
	}  
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_HT50U") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) { 
	if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1 && OpenHltSumHTPassed(50,20)>0) { 
	  triggerBit[it] = true; 
	}  
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_HT70U") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1 && OpenHltSumHTPassed(70,20)>0) { 
	  triggerBit[it] = true; 
	}  
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_MET20") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) { 
	if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1 && recoMetCal>45) { 
	  triggerBit[it] = true; 
	}  
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_MET45") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) { 
	if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1 && recoMetCal>45) { 
	  triggerBit[it] = true; 
	}  
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_Jet30U") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1 && OpenHlt1JetPassed(30)>=1) { 
	  triggerBit[it] = true; 
	}  
      }  
    }  
  }  
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_Jet35U") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1 && OpenHlt1JetPassed(35)>=1) { 
	  triggerBit[it] = true; 
	}  
      }  
    }  
  }  
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_Jet50U") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1 && OpenHlt1JetPassed(50)>=1) { 
	  triggerBit[it] = true; 
	}  
      }  
    }  
  } 

 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET65_CenJet50U") == 0) {        
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {           
      if(OpenHlt1JetPassed(50,2.6)>=1 && recoMetCal>=65) {                                    
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }            
      }                                                                                   
    }                                                                                     
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET65_CenJet50U_EMF") == 0) {        
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {           
      if(OpenHlt1JetPassed(50,2.6,0.02,0.98)>=1 && recoMetCal>=65) {                                    
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }            
      }                                                                                   
    }                                                                                     
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET80_CenJet50U") == 0) {        
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {           
      if(OpenHlt1JetPassed(50,2.6)>=1 && recoMetCal>=80) {                                    
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }            
      }                                                                                   
    }                                                                                     
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET80_CenJet50U_EMF") == 0) {        
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {           
      if(OpenHlt1JetPassed(50,2.6,0.02,0.98)>=1 && recoMetCal>=80) {                                    
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }            
      }                                                                                   
    }                                                                                     
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET100_CenJet50U") == 0) {        
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {           
      if(OpenHlt1JetPassed(50,2.6)>=1 && recoMetCal>=100) {                                    
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }            
      }                                                                                   
    }                                                                                     
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET160_CenJet50U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if(OpenHlt1JetPassed(50,2.6)>=1 && recoMetCal>=160) {
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET100_CenJet50U_EMF") == 0) {        
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {           
      if(OpenHlt1JetPassed(50,2.6,0.02,0.98)>=1 && recoMetCal>=100) {                                    
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }            
      }                                                                                   
    }                                                                                     
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET45_HT100U") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) { 
	if(OpenHltSumHTPassed(100,20)>=1 && recoMetCal>=45) { 
	  triggerBit[it] = true; 
	}  
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET45_DiJet30U") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {   
      if (prescaleResponse(menu,cfg,rcounter,it)) {  
        if(OpenHlt1JetPassed(30.)>=2 && recoMetCal>=45) {  
          triggerBit[it] = true;  
        }   
      }   
    }   
  }   
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET45_HT120U") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) { 
	if(OpenHltSumHTPassed(120,20)>=1 && recoMetCal>=45) { 
	  triggerBit[it] = true; 
	}  
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_Jet70") == 0){
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1JetPassed(70.)>=1) {
	  if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1)
	    triggerBit[it] = true; 
	}
      }
    }
  }
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT140_Eta3_J30") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHltHTJetNJPassed(140,30,3.,0.)>=1) {
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  } 

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele10_HT70U") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1ElectronHTPassed(10.,70.,20.,0.,9999.,9999.,0.5)>=1) {
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  }  

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele10_HT100U") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1ElectronHTPassed(10.,100.,20.,0.,9999.,9999.,0.5)>=1) {
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  }  

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele10_EleId_HT70U") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1ElectronEleIDHTPassed(10.,70.,20.,0.,9999.,9999.,0.5)>=1) {
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  }   
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele10_EleId_HT150U") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1ElectronEleIDHTPassed(10.,70.,20.,0.,9999.,9999.,0.5)>=1) {
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  }  
  // //Not requesting Ele10+MET45, but the module exists, so the code is here in 
  //case you want to estimate the rate
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele10_MET45") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if(recoMetCal>=45&&OpenHlt1ElectronPassed(10.,0.,9999.,9999.)>=1){
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu8_Ele8") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if(OpenHlt1MuonPassed(5.,5.,8.,2.,0)>=1&&OpenHlt1ElectronPassed(8.,0.,9999.,9999.)>=1){
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_NewMu8_Ele8") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if((OpenHlt1MuonPassed(5.,5.,8.,2.,0)>=1)&&
	 (OpenHlt1ElectronSamHarperPassed(17.,0,          // ET, L1isolation
                                         999., 999.,       // Track iso barrel, Track iso endcap
                                         999., 999.,        // Track/pT iso barrel, Track/pT iso endcap
                                         999., 999.,       // H/ET iso barrel, H/ET iso endcap
                                         999., 999.,       // E/ET iso barrel, E/ET iso endcap
                                         0.15, 0.15,       // H/E barrel, H/E endcap
                                         999., 999.,       // cluster shape barrel, cluster shape endcap
                                         1.0, 1.0,       // R9 barrel, R9 endcap
                                         999., 999.,       // Deta barrel, Deta endcap
                                         999., 999.        // Dphi barrel, Dphi endcap
					  )>=1)) {
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu8_Ele12") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1MuonPassed(5.,5.,8.,2.,0)>=1&&OpenHlt1ElectronPassed(12.,0.,9999.,9999.)>=1){ 
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }  
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu10_Ele10") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if(OpenHlt1MuonPassed(5.,5.,10.,2.,0)>=1&&OpenHlt1ElectronPassed(10.,0.,9999.,9999.)>=1){  
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }   
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu11_Ele8") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if(OpenHlt1MuonPassed(9.,9.,11.,2.,0)>=1&&OpenHlt1ElectronPassed(8.,0.,9999.,9999.)>=1){
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu12_Ele8") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1MuonPassed(9.,9.,12.,2.,0)>=1&&OpenHlt1ElectronPassed(8.,0.,9999.,9999.)>=1){ 
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }  
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu13_Ele8") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if(OpenHlt1MuonPassed(9.,9.,13.,2.,0)>=1&&OpenHlt1ElectronPassed(8.,0.,9999.,9999.)>=1){
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_Ele13") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1&&OpenHlt1ElectronPassed(13.,0.,9999.,9999.)>=1){
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_Ele15") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1&&OpenHlt1ElectronPassed(15.,0.,9999.,9999.)>=1){
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu15_Ele15") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if(OpenHlt1MuonPassed(9.,9.,15.,2.,0)>=1&&OpenHlt1ElectronPassed(15.,0.,9999.,9999.)>=1){
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      }
    }
  }
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_DoubleEle8_SW_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1&&OpenHlt1ElectronPassed(8.,0,9999.,9999.)>=2){
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_MET45x") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1 && recoMetCal>45) {
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu7_MET20") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1MuonPassed(3.,4.,7.,2.,0)>=1 && recoMetCal>20) {
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  } 
  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_HT70U") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1 && OpenHltSumHTPassed(70,20)>0) {
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_HT100U") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1 && OpenHltSumHTPassed(100,20)>0) {
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_HT120U") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1 && OpenHltSumHTPassed(120,20)>0) {
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_HT140U") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1 && OpenHltSumHTPassed(140,20)>0) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }  
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu20_CentralJet20U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if(OpenHlt1MuonPassed(7.,7.,20.,2.,0)>=1 && OpenHlt1JetPassed(20,2.6)>=1) {
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu17_TripleCentralJet20U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if(OpenHlt1MuonPassed(7.,7.,17.,2.,0)>=1 && OpenHlt1JetPassed(20,2.6)>=3) {
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_Jet50U") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1 && OpenHlt1JetPassed(50)>=1) {
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_Jet70U") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1 && OpenHlt1JetPassed(70)>=1) {
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  } 


/* Cross Triggers (approved in Jan 2009) */

// SGL - lepton+jet cross-triggers. These are for 1E31, so the *corrected* 
// jets are used at HLT. 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu9_1JetU15") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1CorJetPassed(30)>=1){ // Require 1 corrected jet above threshold
	  if(OpenHlt1L2MuonPassed(7.,9.,9999.)>=1) {
	    triggerBit[it] = true; 
	  }
	}           
      } 
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele10_SW_L1R_3Jet30_3JetL1") == 0){
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	if(OpenHlt1CorJetPassed(30)>=3){
	  if(OpenHlt1ElectronPassed(10.,0,9999.,9999.)>=1) {
	    if(rc>0) {  
	      triggerBit[it] = true; 
	    }
	  }
	}
      }
    }
  }
// John Paul Chou - e(gamma) + mu cross-trigger. 
// One non-isolated photon plus one non-isolated L2 muons.
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu5_Photon9_L1R") == 0){
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1L2MuonPassed(3.,5.,2.)>=1 && OpenHlt1PhotonPassed(9.,0,9999.,9999.,9999.,9999.)>=1)
	  triggerBit[it] = true; 
      }
    } 
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu8_Photon9_L1R") == 0){
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1L2MuonPassed(3.,8.,2.)>=1 && OpenHlt1PhotonPassed(9.,0,9999.,9999.,9999.,9999.)>=1)
	  triggerBit[it] = true; 
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu10_Photon9_L1R") == 0){
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1L2MuonPassed(3.,10.,2.)>=1 && OpenHlt1PhotonPassed(9.,0,9999.,9999.,9999.,9999.)>=1)
	  triggerBit[it] = true; 
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu5_Photon11_L1R") == 0){
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1L2MuonPassed(5.,5.,2.)>=1 && OpenHlt1PhotonPassed(11.,0,9999.,9999.,9999.,9999.)>=1)
	  triggerBit[it] = true; 
      }
    } 
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu5_Photon13_L1R") == 0){
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1L2MuonPassed(5.,5.,2.)>=1 && OpenHlt1PhotonPassed(13.,0,9999.,9999.,9999.,9999.)>=1)
	  triggerBit[it] = true; 
      }
    } 
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_Photon9_L1R") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1 && OpenHlt1PhotonPassed(9.,0,9999.,9999.,9999.,9999.)>=1) 
	  triggerBit[it] = true; 
      }
    }  
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_Photon11_L1R") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1 && OpenHlt1PhotonPassed(11.,0,9999.,9999.,9999.,9999.)>=1) 
	  triggerBit[it] = true; 
      }
    }  
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu7_Photon11_L1R") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(5.,5.,7.,2.,0)>=1 && OpenHlt1PhotonPassed(11.,0,9999.,9999.,9999.,9999.)>=1) 
	  triggerBit[it] = true; 
      }
    }  
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu7_Photon13_L1R") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(5.,5.,7.,2.,0)>=1 && OpenHlt1PhotonPassed(13.,0,9999.,9999.,9999.,9999.)>=1) 
	  triggerBit[it] = true; 
      }
    }  
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu15_Photon15_L1R") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(5.,5.,15.,2.,0)>=1 && OpenHlt1PhotonPassed(15.,0,9999.,9999.,9999.,9999.)>=1) 
	  triggerBit[it] = true; 
      }
    }  
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu15_Photon20_L1R") == 0){  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {   
      if (prescaleResponse(menu,cfg,rcounter,it)) { 
        if(OpenHlt1MuonPassed(5.,5.,15.,2.,0)>=1 && OpenHlt1PhotonPassed(20.,0,9999.,9999.,9999.,9999.)>=1)  
          triggerBit[it] = true;  
      } 
    }   
  }  
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu15_DiPhoton15_L1R") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(5.,5.,15.,2.,0)>=1 && OpenHlt1PhotonPassed(15.,0,999.,999.,999.,999.)>=2) 
	  triggerBit[it] = true; 
      }
    }  
  } 

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_Ele9") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1 && OpenHlt1ElectronPassed(9.,0,9999.,9999.)>=1)
	  triggerBit[it] = true; 
      }
    }  
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_Ele5") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if( OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1 && OpenHlt1ElectronPassed(5.,0,9999.,9999.)>=1) 
	  triggerBit[it] = true; 
      }
    }  
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu7_Ele9") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1MuonPassed(5.,5.,7.,2.,0)>=1 && OpenHlt1ElectronPassed(9.,0,9999.,9999.)>=1)
	  triggerBit[it] = true; 
      }
    }  
  } 
 
// Exotica mu + e/gamma, mu + jet, and mu + MET L1-passthrough cross-triggers
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu14_L1SingleEG10") == 0){
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	triggerBit[it] = true; 
      }
    }
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu14_L1SingleJet6") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	triggerBit[it] = true; 
      }
    } 
  } 
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu14_L1SingleJet15") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	triggerBit[it] = true; 
      }
    } 
  } 
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu20_L1SingleJet15") == 0){
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	triggerBit[it] = true; 
      }
    }
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu30_L1SingleJet15") == 0){
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;
	for(int i=0;i<NL1Mu;i++) {
	  if(L1MuPt[i] > 30.0)
	    rc++;
	}
	if(rc > 0)
	  triggerBit[it] = true; 
      }
    }
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu14_L1SingleJet20") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int rc = 0;   
	for(int i=0;i<NL1CenJet;i++) if(L1CenJetEt[i] >= 20.0) rc++;   
	for(int i=0;i<NL1ForJet;i++) if(L1ForJetEt[i] >= 20.0) rc++;   
	for(int i=0;i<NL1Tau   ;i++) if(L1TauEt   [i] >= 20.0) rc++;   
	if(rc > 0)
	  triggerBit[it] = true; 
      }
    } 
  } 
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu14_L1ETM30") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	triggerBit[it] = true; 
      }
    } 
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu14_L1ETM40") == 0){  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	triggerBit[it] = true; 
      }
    }  
  } 
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_HT50") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {    
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(50., 30.) == 1) { 
	  if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1)     
	    triggerBit[it] = true; 
	}
      } 
    } 
  } 
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_HT80") == 0){   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {      
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(80., 30.) == 1) {   
	  if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1)       
	    triggerBit[it] = true; 
	}
      }   
    }   
  }   
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Mu8_HT50") == 0){  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {     
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(50., 30.) == 1) {  
	  if(OpenHlt1MuonPassed(3.,4.,8.,2.,0)>=1)      
	    triggerBit[it] = true; 
	}
      }  
    }  
  }  
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu8_HT50") == 0){   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {      
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(50., 30.) == 1) {   
	  if(OpenHlt1L2MuonPassed(7.,8.,9999.)>=1) { 
	    triggerBit[it] = true; 
	  }
	}   
      }   
    }   
  } 
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu10_HT50") == 0){
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHltSumHTPassed(50., 30.) == 1) {
	  if(OpenHlt1L2MuonPassed(7.,10.,9999.)>=1) { 
	    triggerBit[it] = true; 
	  }
	}
      }
    }
  }
  // triple jet b-tag trigger for top
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_BTagIP_TripleJet20U") == 0){
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	int njets = 0;
	int ntaggedjets = 0;
	int max =  (NohBJetL2 > 2) ? 2 : NohBJetL2; 
	for(int i = 0; i < max; i++) {  
	  if(ohBJetL2Et[i] > 20.) { // ET cut on uncorrected jets 
	    njets++;
	    if(ohBJetPerfL25Tag[i] > 0.5) { // Level 2.5 b tag 
	      if(ohBJetPerfL3Tag[i] > 0.5) { // Level 3 b tag 
		ntaggedjets++;
	      }  
	    } 
	  } 
	} 
	if(njets > 2 && ntaggedjets > 0) {  // Require >= 3 jets, and >= 1 tagged jet
	  triggerBit[it] = true; 
	}
      }  
    } 
  }
// Lepton+jet triggers for... top? exotica? b-tagging?
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu9_DiJet30") == 0) {
    int njetswithmu = 0;
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1L2MuonPassed(9.,9.,2.)>=1) {
	  for(int i = 0; i < NrecoJetCal; i++) {
	    if(recoJetCorCalPt[i] > 30.) { // Cut on corrected jet energy
	      njetswithmu++;
	    }
	  }
	}
	if(njetswithmu >= 2) // Require >= 2 jets above threshold
	  triggerBit[it] = true; 
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele10_SW_L1R_TripleJet30") == 0) {
    int njetswithe = 0; 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1ElectronPassed(10.,0,9999.,9999.)>=1) {
	  for(int i = 0; i < NrecoJetCal; i++) { 
	    if(recoJetCorCalPt[i] > 30.) { // Cut on corrected jet energy 
	      njetswithe++; 
	    } 
	  } 
	} 
	if(njetswithe >= 3) // Require >= 3 jets above threshold 
	  triggerBit[it] = true; 
      }
    } 
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele15_SW_L1R_TripleJet30") == 0) {
    int njetswithe = 0; 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) {
	if(OpenHlt1ElectronPassed(15.,0,9999.,9999.)>=1) {
	  for(int i = 0; i < NrecoJetCal; i++) { 
	    if(recoJetCorCalPt[i] > 30.) { // Cut on corrected jet energy 
	      njetswithe++; 
	    } 
	  } 
	} 
	if(njetswithe >= 3) // Require >= 3 jets above threshold 
	  triggerBit[it] = true; 
      }
    } 
  }
  else if(menu->GetTriggerName(it).BeginsWith("OpenHLT_Photon26")==1) {
    // Photon Paths (V. Rekovic)
    
    int lowerEt = 18;
    int upperEt = 26;
    
    // Names of Photon Paths With Mass Cut
    // -------------------------------------
    
    // No isol
    char pathNamePhotonPhoton[100];
    sprintf(pathNamePhotonPhoton,"OpenHLT_Photon%d_Photon%d_L1R",upperEt,lowerEt);
    // One leg Isol
    char pathNamePhotonIsolPhoton[100];
    sprintf(pathNamePhotonIsolPhoton,"OpenHLT_Photon%d_Isol_Photon%d_L1R",upperEt,lowerEt);
    // One leg Loose Isol
    char pathNamePhotonLooseIsolPhoton[100];
    sprintf(pathNamePhotonLooseIsolPhoton,"OpenHLT_Photon%d_LooseIsol_Photon%d_L1R",upperEt,lowerEt);
    // One leg Looser Isol
    char pathNamePhotonLooserIsolPhoton[100];
    sprintf(pathNamePhotonLooserIsolPhoton,"OpenHLT_Photon%d_LooserIsol_Photon%d_L1R",upperEt,lowerEt);
    // One leg CaloId
    char pathNamePhotonCaloIdPhoton[100];
    sprintf(pathNamePhotonCaloIdPhoton,"OpenHLT_Photon%d_CaloId_Photon%d_L1R",upperEt,lowerEt);
    // One leg Isol + CaloId
    char pathNamePhotonIsolCaloIdPhoton[100];
    sprintf(pathNamePhotonIsolCaloIdPhoton,"OpenHLT_Photon%d_Isol_CaloId_Photon%d_L1R",upperEt,lowerEt);
    // One leg Isol + CaloId + LooseHoverE
    char pathNamePhotonIsolCaloIdLooseHEPhoton[100];
    sprintf(pathNamePhotonIsolCaloIdLooseHEPhoton,"OpenHLT_Photon%d_Isol_CaloId_LooseHE_Photon%d_L1R",upperEt,lowerEt);
    // Both legs Isol
    char pathNamePhotonIsolPhotonIsol[100];
    sprintf(pathNamePhotonIsolPhotonIsol,"OpenHLT_Photon%d_Isol_Photon%d_Isol_L1R",upperEt,lowerEt);
    // Both legs Isol + CaloId
    char pathNamePhotonIsolCaloIdPhotonIsolCaloId[100];
    sprintf(pathNamePhotonIsolCaloIdPhotonIsolCaloId,"OpenHLT_Photon%d_Isol_CaloId_Photon%d_Isol_CaloId_L1R",upperEt,lowerEt);
    
    
    // Names of Photon Paths With Mass Cut
    // -----------------------------------
    
    // One leg Isol + Mass>60
    char pathNamePhotonIsolPhotonMass60[100];
    sprintf(pathNamePhotonIsolPhotonMass60,"OpenHLT_Photon%d_Isol_Photon%d_Mass60_L1R",upperEt,lowerEt);
    // One leg Isol + CaloId + Mass>60
    char pathNamePhotonIsolCaloIdPhotonMass60[100];
    sprintf(pathNamePhotonIsolCaloIdPhotonMass60,"OpenHLT_Photon%d_Isol_CaloId_Photon%d_Mass60_L1R",upperEt,lowerEt);
    // Both legs Isol  + Mass>60
    char pathNamePhotonIsolPhotonIsolMass60[100];
    sprintf(pathNamePhotonIsolPhotonIsolMass60,"OpenHLT_Photon%d_Isol_Photon%d_Isol_Mass60_L1R",upperEt,lowerEt);
    // Both legs Isol + CaloId + Mass>60
    char pathNamePhotonIsolCaloIdPhotonIsolCaloIdMass60[100];
    sprintf(pathNamePhotonIsolCaloIdPhotonIsolCaloIdMass60,"OpenHLT_Photon%d_Isol_CaloId_Photon%d_Isol_CaloId_Mass60_L1R",upperEt,lowerEt);

    if (menu->GetTriggerName(it).CompareTo(pathNamePhotonPhoton) == 0) {     
      if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) {
          std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(lowerEt,0,999,999,999,999,0.15,0.98);
	if(firstVector.size()>=2) {      
         std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(upperEt,0,999,999,999,999,0.15,0.98);
  	 if(secondVector.size()>=1) {      
  	  triggerBit[it] = true; 
  	 }
        }
       }      
      }      
    } 
    else if (menu->GetTriggerName(it).CompareTo(pathNamePhotonIsolPhoton) == 0) {     
      if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) {
          std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(lowerEt,0,999,999,999,999,0.15,0.98);
	if(firstVector.size()>=2) {      
         std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(upperEt,0,3,5,3,3,0.15,0.98);
         if(secondVector.size()>=1) {      
  	  triggerBit[it] = true; 
  	 }
        }
       }      
      }      
    } 
    else if (menu->GetTriggerName(it).CompareTo(pathNamePhotonLooseIsolPhoton) == 0) {     
      if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) {
          std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(lowerEt,0,999,999,999,999,0.15,0.98);
	if(firstVector.size()>=2) {      
          std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(upperEt,0,3.5,5.5,3.5,3.5,0.15,0.98);
  	 if(secondVector.size()>=1) {      
  	  triggerBit[it] = true; 
  	 }
        }
       }      
      }      
    } 
    else if (menu->GetTriggerName(it).CompareTo(pathNamePhotonLooserIsolPhoton) == 0) {     
      if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
       if (prescaleResponse(menu,cfg,rcounter,it)) {
        std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(lowerEt,0,999,999,999,999,0.15,0.98);
	if(firstVector.size()>=2) {      
          std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(upperEt,0,4.0,6.0,4.0,4.0,0.15,0.98);
  	 if(secondVector.size()>=1) {      
  	  triggerBit[it] = true; 
  	 }
        }
       }      
      }      
    } 
    else if (menu->GetTriggerName(it).CompareTo(pathNamePhotonIsolCaloIdPhoton) == 0) {     
      if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
       if (prescaleResponse(menu,cfg,rcounter,it)) {
        std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(lowerEt,0,999,999,999,999,0.15,0.98);
	if(firstVector.size()>=2) {      
  	 //if(OpenHlt1PhotonPassed(upperEt,0,3,5,3,3,0.15,0.98,0.014,0.035)>=1) {      
         std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(upperEt,0,3,5,3,3,0.15,0.98,0.014,0.035);
  	 if(secondVector.size()>=1) {      
  	  triggerBit[it] = true; 
  	 }
        }
       }      
      }      
    } 
    else if (menu->GetTriggerName(it).CompareTo(pathNamePhotonIsolCaloIdLooseHEPhoton) == 0) {     
      if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
       if (prescaleResponse(menu,cfg,rcounter,it)) {
        std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(lowerEt,0,999,999,999,999,0.15,0.98);
	if(firstVector.size()>=2) {      
  	 //if(OpenHlt1PhotonPassed(upperEt,0,3,5,3,3,0.15,999.98,0.014,0.035)>=1) {      
         std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(upperEt,0,3,5,3,3,0.15,999.98,0.014,0.035);
  	 if(secondVector.size()>=1) {      
  	  triggerBit[it] = true; 
  	 }
        }
       }      
      }      
    } 
    else if (menu->GetTriggerName(it).CompareTo(pathNamePhotonIsolPhotonIsol) == 0) {     
      if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) {
         std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(lowerEt,0,3,5,3,3,0.15,0.98);
  	 if(firstVector.size()>=2) {      
         std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(upperEt,0,3,5,3,3,0.15,0.98);
  	 if(secondVector.size()>=1) {      
  	  triggerBit[it] = true; 
  	 }
        }
       }      
      }      
    } 
    else if (menu->GetTriggerName(it).CompareTo(pathNamePhotonIsolCaloIdPhotonIsolCaloId) == 0) {     
      if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) {
         std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(lowerEt,0,3,5,3,3,0.15,0.98,0.014,0.035);
  	 if(firstVector.size()>=2) {      
         std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(upperEt,0,3,5,3,3,0.15,0.98,0.014,0.035);
  	 if(secondVector.size()>=1) {      
  	  triggerBit[it] = true; 
  	 }
        }
       }      
      }      
    } 

    // With Mass Cut
    else if (menu->GetTriggerName(it).CompareTo(pathNamePhotonIsolPhotonMass60) == 0) {     
      if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) {
          std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(lowerEt,0,999,999,999,999,0.15,0.98);
	if(firstVector.size()>=2) {      
          std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(upperEt,0,3,5,3,3,0.15,0.98);
  	 if(secondVector.size()>=1) {      

           TLorentzVector e1;  
           TLorentzVector e2;  
           TLorentzVector meson;
           float mass = 0.;
           for (unsigned int i=0;i<firstVector.size();i++) { 
             for (unsigned int j=0;j<secondVector.size() ;j++) {  	  

                 if(firstVector[i] == secondVector[j]) continue;
                 e1.SetPtEtaPhiM(ohPhotEt[firstVector[i]],ohPhotEta[firstVector[i]],ohPhotPhi[firstVector[i]],0.); 
                 e2.SetPtEtaPhiM(ohPhotEt[secondVector[j]],ohPhotEta[secondVector[j]],ohPhotPhi[secondVector[j]],0.); 
                 meson = e1 + e2; 
                 mass = meson.M();  
                 
                 if (mass>60) triggerBit[it] = true; 

             }
           }
  
  	 }
        }
       }      
      }      
    } 
    else if (menu->GetTriggerName(it).CompareTo(pathNamePhotonIsolCaloIdPhotonMass60) == 0) {     
      if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) {
          std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(lowerEt,0,999,999,999,999,0.15,0.98);
	if(firstVector.size()>=2) {      
          std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(upperEt,0,3,5,3,3,0.15,0.98,0.014,0.035);
  	 if(secondVector.size()>=1) {      

           TLorentzVector e1;  
           TLorentzVector e2;  
           TLorentzVector meson;
           float mass = 0.;
           for (unsigned int i=0;i<firstVector.size();i++) { 
             for (unsigned int j=0;j<secondVector.size() ;j++) {  	  

                 if(firstVector[i] == secondVector[j]) continue;
                 e1.SetPtEtaPhiM(ohPhotEt[firstVector[i]],ohPhotEta[firstVector[i]],ohPhotPhi[firstVector[i]],0.); 
                 e2.SetPtEtaPhiM(ohPhotEt[secondVector[j]],ohPhotEta[secondVector[j]],ohPhotPhi[secondVector[j]],0.); 
                 meson = e1 + e2; 
                 mass = meson.M();  
                 
                 if (mass>60) triggerBit[it] = true; 

             }
           }
  
  	 }
        }
       }      
      }      
    } 
    else if (menu->GetTriggerName(it).CompareTo(pathNamePhotonIsolPhotonIsolMass60) == 0) {     
      if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) {
         std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(lowerEt,0,3,5,3,3,0.15,0.98);
  	 if(firstVector.size()>=2) {      
         std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(upperEt,0,3,5,3,3,0.15,0.98);
  	 if(secondVector.size()>=1) {      

           TLorentzVector e1;  
           TLorentzVector e2;  
           TLorentzVector meson;
           float mass = 0.;
           for (unsigned int i=0;i<firstVector.size();i++) { 
             for (unsigned int j=0;j<secondVector.size() ;j++) {  	  

                 if(firstVector[i] == secondVector[j]) continue;
                 e1.SetPtEtaPhiM(ohPhotEt[firstVector[i]],ohPhotEta[firstVector[i]],ohPhotPhi[firstVector[i]],0.); 
                 e2.SetPtEtaPhiM(ohPhotEt[secondVector[j]],ohPhotEta[secondVector[j]],ohPhotPhi[secondVector[j]],0.); 
                 meson = e1 + e2; 
                 mass = meson.M();  

                 if (mass>60) triggerBit[it] = true; 

             }
           }
  

  	 }
        }
       }      
      }      
    } 
    else if (menu->GetTriggerName(it).CompareTo(pathNamePhotonIsolCaloIdPhotonIsolCaloIdMass60) == 0) {     
      if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) {
         std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(lowerEt,0,3,5,3,3,0.15,0.98,0.014,0.035);
  	 if(firstVector.size()>=2) {      
         std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(upperEt,0,3,5,3,3,0.15,0.98,0.014,0.035);
  	 if(secondVector.size()>=1) {      

           TLorentzVector e1;  
           TLorentzVector e2;  
           TLorentzVector meson;
           float mass = 0.;
           for (unsigned int i=0;i<firstVector.size();i++) { 
             for (unsigned int j=0;j<secondVector.size() ;j++) {  	  

                 if(firstVector[i] == secondVector[j]) continue;
                 e1.SetPtEtaPhiM(ohPhotEt[firstVector[i]],ohPhotEta[firstVector[i]],ohPhotPhi[firstVector[i]],0.); 
                 e2.SetPtEtaPhiM(ohPhotEt[secondVector[j]],ohPhotEta[secondVector[j]],ohPhotPhi[secondVector[j]],0.); 
                 meson = e1 + e2; 
                 mass = meson.M();  

                 if (mass>60) triggerBit[it] = true; 

             }
           }
  
  	 }
        }
       }      
      }      
    } 
  }
  else {
    if(nMissingTriggerWarnings < 10)
      cout << "Warning: the requested trigger " << menu->GetTriggerName(it) << " is not implemented in OHltTreeOpen. No rate will be calculated." << endl;
    nMissingTriggerWarnings++;
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
	for(int i=0;i<NL1Mu;i++) {
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
	float ohElePt = ohEleP[i] * TMath::Sin(2*TMath::ATan(TMath::Exp(-1*ohEleEta[i])));
	cout << "ohEleEt["<<i<<"] = " << ohEleEt[i] << endl;
	cout << "ohElePhi["<<i<<"] = " << ohElePhi[i] << endl;
	cout << "ohEleEta["<<i<<"] = " << ohEleEta[i] << endl;
	cout << "ohEleE["<<i<<"] = " << ohEleE[i] << endl;
	cout << "ohEleP["<<i<<"] = " << ohEleP[i] << endl;
	cout << "ohElePt["<<i<<"] =" <<  ohElePt << endl;
	cout << "ohEleHiso["<<i<<"] = " << ohEleHiso[i] << endl;
	cout << "ohEleTiso["<<i<<"] = " << ohEleTiso[i] << endl;
	cout << "ohEleL1iso["<<i<<"] = " << ohEleL1iso[i] << endl;
        cout << "ohEleHiso["<<i<<"]/ohEleEt["<<i<<"] = " << ohEleHiso[i]/ohEleEt[i] << endl; 
        cout << "ohEleEiso["<<i<<"]/ohEleEt["<<i<<"] = " << ohEleEiso[i]/ohEleEt[i] << endl; 
        cout << "ohEleTiso["<<i<<"]/ohEleEt["<<i<<"] = " << ohEleTiso[i]/ohEleEt[i] << endl;  
        cout << "ohEleHforHoverE["<<i<<"] = " << ohEleHforHoverE[i] << endl; 
        cout << "ohEleHforHoverE["<<i<<"]/ohEleE["<<i<<"] = " << ohEleHforHoverE[i]/ohEleE[i] << endl; 
        cout << "ohEleNewSC["<<i<<"] = " << ohEleNewSC[i] << endl;  
        cout << "ohElePixelSeeds["<<i<<"] = " << ohElePixelSeeds[i] << endl; 
        cout << "ohEleClusShap["<<i<<"] = " << ohEleClusShap[i] << endl; 
	cout << "ohEleR9["<<i<<"] = " << ohEleR9[i] << endl;  
        cout << "ohEleDeta["<<i<<"] = " << ohEleDeta[i] << endl; 
        cout << "ohEleDphi["<<i<<"] = " << ohEleDphi[i] << endl; 
      }

      for(int i=0;i<NL1IsolEm;i++) { 
	cout << "L1IsolEmEt["<<i<<"] = " << L1IsolEmEt[i] << endl;
	cout << "L1IsolEmE["<<i<<"] = " << L1IsolEmE[i] << endl;
	cout << "L1IsolEmEta["<<i<<"] = " << L1IsolEmEta[i] << endl;
	cout << "L1IsolEmPhi["<<i<<"] = " << L1IsolEmPhi[i] << endl;
      }
      for(int i=0;i<NL1NIsolEm;i++) {  
        cout << "L1NIsolEmEt["<<i<<"] = " << L1NIsolEmEt[i] << endl; 
        cout << "L1NIsolEmE["<<i<<"] = " << L1NIsolEmE[i] << endl; 
        cout << "L1NIsolEmEta["<<i<<"] = " << L1NIsolEmEta[i] << endl; 
        cout << "L1NIsolEmPhi["<<i<<"] = " << L1NIsolEmPhi[i] << endl; 
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
	cout << "ohPhotHiso["<<i<<"]/ohPhotEt["<<i<<"] = " << ohPhotHiso[i]/ohPhotEt[i] << endl;
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

// PFTau Leg
int OHltTree::OpenHltPFTauPassedNoMuon(float Et,float L25TrkPt, int L3TrkIso, int L3GammaIso)
{
    
    int rc = 0;    
    // Loop over all oh pfTaus
    for (int i=0;i < NohPFTau;i++) {    
//        if (pfTauJetPt[i] >= Et) 
                if (pfTauPt[i] >= Et) 
                    if(abs(pfTauEta[i])<2.5)
            if (pfTauLeadTrackPt[i] >= L25TrkPt)
                if (pfTauTrkIso[i] < L3TrkIso)
                    if (pfTauGammaIso[i] < L3GammaIso )   
		      // if(OpenHltTauPFToCaloMatching(pfTauEta[i], pfTauPhi[i]) == 1)
                            if (OpenHltTauMuonMatching(pfTauEta[i], pfTauPhi[i]) == 0)
                      //          if (OpenHltL1L2TauMatching(ohTauEta[i], ohTauPhi[i], 30, 40) == 1)
                                                               rc++;      
        
    }
    
    return rc;
}

int OHltTree::OpenHltPFTauPassedNoEle(float Et,float L25TrkPt, int L3TrkIso, int L3GammaIso)
{
    
    int rc = 0;    
    // Loop over all oh pfTaus
    for (int i=0;i < NohPFTau;i++) {    
        //        if (pfTauJetPt[i] >= Et) 
        if (pfTauPt[i] >= Et) 
            if(abs(pfTauEta[i])<2.5)
                if (pfTauLeadTrackPt[i] >= L25TrkPt)
                    if (pfTauTrkIso[i] < L3TrkIso)
                        if (pfTauGammaIso[i] < L3GammaIso )   
                            if(OpenHltTauPFToCaloMatching(pfTauEta[i], pfTauPhi[i]) == 1)
                                if (OpenHltTauEleMatching(pfTauEta[i], pfTauPhi[i]) == 0)
                                    rc++;      
        
    }
    
    return rc;
}

int OHltTree::OpenHltTauMuonMatching(float eta, float phi){
  for (int j=0;j<NohMuL2;j++) {
    double deltaphi = fabs(phi-ohMuL2Phi[j]);
    if(deltaphi > 3.14159) deltaphi = (2.0 * 3.14159) - deltaphi;
    double deltaeta = fabs(eta-ohMuL2Eta[j]);

    if (deltaeta<0.3 && deltaphi<0.3)
      return 1;
  }
  return 0;

}

int OHltTree::OpenHltTauEleMatching(float eta, float phi){
  for (int j=0;j<NohEle;j++) {
    double deltaphi = fabs(phi-ohElePhi[j]);
    if(deltaphi > 3.14159) deltaphi = (2.0 * 3.14159) - deltaphi;
    double deltaeta = fabs(eta-ohEleEta[j]);

    if (deltaeta<0.3 && deltaphi<0.3)
      return 1;
  }
  return 0;

}

int OHltTree::OpenHltTauPFToCaloMatching(float eta, float phi){
  for (int j=0;j<NrecoJetCal;j++) {
    if(recoJetCalPt[j]<8) continue;
    double deltaphi = fabs(phi-recoJetCalPhi[j]);
    if(deltaphi > 3.14159) deltaphi = (2.0 * 3.14159) - deltaphi;
    double deltaeta = fabs(eta-recoJetCalEta[j]);

    if (deltaeta<0.3 && deltaphi<0.3)
      return 1;
  }
  return 0;

}

int OHltTree::OpenHltL1L2TauMatching(float eta, float phi, float tauThr, float jetThr) {
  for (int j=0;j<NL1Tau;j++) {
    double deltaphi = fabs(phi-L1TauPhi[j]);
    if(deltaphi > 3.14159) deltaphi = (2.0 * 3.14159) - deltaphi;
    double deltaeta = fabs(eta-L1TauEta[j]);

    if (deltaeta<0.3 && deltaphi<0.3 && L1TauEt[j]>tauThr)
      return 1;
  }
  for (int j=0;j<NL1CenJet;j++) {
    double deltaphi = fabs(phi-L1CenJetPhi[j]);
    if(deltaphi > 3.14159) deltaphi = (2.0 * 3.14159) - deltaphi;
    double deltaeta = fabs(eta-L1CenJetEta[j]);

    if (deltaeta<0.3 && deltaphi<0.3 && L1CenJetEt[j]>jetThr)
      return 1;
  }
  return 0;
}

int OHltTree::OpenHltTauPassed(float Et,float Eiso, float L25Tpt, int L25Tiso, float L3Tpt, int L3Tiso,
				   float L1TauThr, float L1CenJetThr)
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
   		if (OpenHltL1L2TauMatching(ohTauEta[i], ohTauPhi[i], L1TauThr, L1CenJetThr) == 1)
		  rc++;      
    }
  }

  return rc;
}

// L2 Ecal sliding cut isolation
int OHltTree::OpenHltTauL2SCPassed(float Et,float L25Tpt, int L25Tiso, float L3Tpt, int L3Tiso,
				   float L1TauThr, float L1CenJetThr)
{
  int rc = 0;
    
  // Loop over all oh electrons
  for (int i=0;i<NohTau;i++) {
    if (ohTauPt[i] >= Et) {
      if (ohTauEiso[i] < (5 + 0.025*ohTauPt[i] + 0.0015*ohTauPt[i]*ohTauPt[i])) // sliding cut
        if (ohTauL25Tpt[i] >= L25Tpt)
          if (ohTauL25Tiso[i] >= L25Tiso)
            if (ohTauL3Tpt[i] >= L3Tpt)
              if (ohTauL3Tiso[i] >= L3Tiso)
		if (OpenHltL1L2TauMatching(ohTauEta[i], ohTauPhi[i], L1TauThr, L1CenJetThr) == 1)
		  rc++;      
    }
  }

  return rc;
}

int OHltTree::OpenHltTauL2SCMETPassed(float Et,float L25Tpt, int L25Tiso, float L3Tpt, int L3Tiso, float met,
		  float L1TauThr, float L1CenJetThr)
{
	  int rc = 0;

	  // Loop over all oh electrons
	  for (int i=0;i<NohTau;i++) {
	    if (ohTauPt[i]>= Et) {
	      if (ohTauEiso[i]<  (5 + 0.025*ohTauPt[i] + 0.0015*ohTauPt[i]*ohTauPt[i])) // sliding cut
		if (ohTauL25Tpt[i]>= L25Tpt)
		  if (ohTauL25Tiso[i]>= L25Tiso)
		    if (ohTauL3Tpt[i]>= L3Tpt)
		      if (ohTauL3Tiso[i]>= L3Tiso)
			if (OpenHltL1L2TauMatching(ohTauEta[i], ohTauPhi[i], L1TauThr, L1CenJetThr) == 1)
			  if(recoMetCal>  met)
			    rc++;
	    }
	  }
	  return rc;
}

int OHltTree::OpenHlt2Tau1LegL3IsoPassed(float Et,float L25Tpt, int L25Tiso, float L3Tpt,
					 float L1TauThr, float L1CenJetThr)
{
  int rc = 0; int l3iso = 0;
  
  // Loop over all oh taus
  for (int i=0;i<NohTau;i++) {
    if (ohTauPt[i] >= Et) {
      if (ohTauEiso[i] < (5 + 0.025*ohTauPt[i] + 0.0015*ohTauPt[i]*ohTauPt[i])) // sliding cut
	if (ohTauL25Tpt[i] >= L25Tpt)
	  if (ohTauL25Tiso[i] >= L25Tiso)
	    if (ohTauL3Tpt[i] >= L3Tpt)
	      if (OpenHltL1L2TauMatching(ohTauEta[i], ohTauPhi[i], L1TauThr, L1CenJetThr) == 1) {
		rc++;
		if (ohTauL3Tiso[i] >= 1) l3iso++;
	      }
    }
  }
  
  if (rc>=2) return l3iso;
  return 0;
}

int OHltTree::OpenHlt1PhotonSamHarperPassed(float Et, int L1iso, 
					    float Tisobarrel, float Tisoendcap, 
					    float Tisoratiobarrel, float Tisoratioendcap, 
					    float HisooverETbarrel, float HisooverETendcap, 
					    float EisooverETbarrel, float EisooverETendcap,
					    float hoverebarrel, float hovereendcap,
					    float clusshapebarrel, float clusshapeendcap, 
					    float r9barrel, float r9endcap,
					    float detabarrel, float detaendcap,
					    float dphibarrel, float dphiendcap)
{
  float barreleta = 1.479;
  float endcapeta = 2.65;
  
  int rc = 0;
  // Loop over all oh electrons
  for (int i=0;i<NohPhot;i++) {
    float ohPhotE = ohPhotEt[i] / (sin(2*atan(exp(-1.0*ohPhotEta[i]))));   
    float ohPhotHoverE = ohPhotHforHoverE[i]/ohPhotE;
    int isbarrel = 0;
    int isendcap = 0;
    if(TMath::Abs(ohPhotEta[i]) < barreleta)
      isbarrel = 1;
    if(barreleta < TMath::Abs(ohPhotEta[i]) && TMath::Abs(ohPhotEta[i]) < endcapeta)
      isendcap = 1;

    if ( ohPhotEt[i] > Et) {
      if( TMath::Abs(ohPhotEta[i]) < endcapeta ) {
	    if ( ohPhotL1iso[i] >= L1iso ) {  // L1iso is 0 or 1 
	      if( ohPhotL1Dupl[i] == false) { // remove double-counted L1 SCs 
		if ( (isbarrel && ((ohPhotHiso[i]/ohPhotEt[i]) < HisooverETbarrel)) ||
		     (isendcap && ((ohPhotHiso[i]/ohPhotEt[i]) < HisooverETendcap)) ) {
		  if ( (isbarrel && ((ohPhotEiso[i]/ohPhotEt[i]) < EisooverETbarrel)) ||
		       (isendcap && ((ohPhotEiso[i]/ohPhotEt[i]) < EisooverETendcap)) ) {
		    if ( ((isbarrel) && (ohPhotHoverE < hoverebarrel)) ||
			 ((isendcap) && (ohPhotHoverE < hovereendcap))) {
		      if ( (isbarrel && (((ohPhotTiso[i] < Tisobarrel && ohPhotTiso[i] != -999.) || (Tisobarrel == 999.)))) ||
			   (isendcap && (((ohPhotTiso[i] < Tisoendcap && ohPhotTiso[i] != -999.) || (Tisoendcap == 999.))))) {
			if (((isbarrel) && (ohPhotTiso[i]/ohPhotEt[i] < Tisoratiobarrel)) ||
			    ((isendcap) && (ohPhotTiso[i]/ohPhotEt[i] < Tisoratioendcap))) {
			  if ( (isbarrel && ohPhotClusShap[i] < clusshapebarrel) ||
			       (isendcap && ohPhotClusShap[i] < clusshapeendcap)) {
			    if ( (isbarrel && ohPhotR9[i] < r9barrel) || 
				 (isendcap && ohPhotR9[i] < r9endcap))  {
				  rc++;
			    }
			  }
			}
		      }
		    }
		  }
		}
	      }
	    }
      }
    }
  }
  
  return rc;
}

int OHltTree::OpenHlt1PhotonPassedRA3(float Et, int L1iso, 
				      float HisooverETbarrel, float HisooverETendcap, 
				      float EisooverETbarrel, float EisooverETendcap,
				      float hoverebarrel, float hovereendcap,
				      float r9barrel, float r9endcap)
{
  float barreleta = 1.479;
  float endcapeta = 2.65;
  
  int rc = 0;
  // Loop over all oh photons
  for (int i=0;i<NohPhot;i++) {
    float ohPhotE = ohPhotEt[i] / (sin(2*atan(exp(-1.0*ohPhotEta[i]))));   
    float ohPhotHoverE = ohPhotHforHoverE[i]/ohPhotE;
    int isbarrel = 0;
    int isendcap = 0;
    if(TMath::Abs(ohPhotEta[i]) < barreleta)
      isbarrel = 1;
    if(barreleta < TMath::Abs(ohPhotEta[i]) && TMath::Abs(ohPhotEta[i]) < endcapeta)
      isendcap = 1;
    
    if ( ohPhotEt[i] > Et) {
      if( TMath::Abs(ohPhotEta[i]) < endcapeta ) {
	if ( ohPhotL1iso[i] >= L1iso ) {  // L1iso is 0 or 1 
	  if( ohPhotL1Dupl[i] == false) { // remove double-counted L1 SCs 
	    if ( (isbarrel && ((ohPhotHiso[i]/ohPhotEt[i]) < HisooverETbarrel)) ||
		 (isendcap && ((ohPhotHiso[i]/ohPhotEt[i]) < HisooverETendcap)) ) {
	      if ( (isbarrel && ((ohPhotEiso[i]/ohPhotEt[i]) < EisooverETbarrel)) ||
		   (isendcap && ((ohPhotEiso[i]/ohPhotEt[i]) < EisooverETendcap)) ) {
		if ( ((isbarrel) && (ohPhotHoverE < hoverebarrel)) ||
		     ((isendcap) && (ohPhotHoverE < hovereendcap))) {
		  if ( (isbarrel && ohPhotR9[i] < r9barrel) || 
		       (isendcap && ohPhotR9[i] < r9endcap)) {
		    rc++;
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  return rc;
}


int OHltTree::OpenHlt1ElectronSamHarperPassed(float Et, int L1iso, 
					      float Tisobarrel, float Tisoendcap, 
					      float Tisoratiobarrel, float Tisoratioendcap, 
					      float HisooverETbarrel, float HisooverETendcap, 
					      float EisooverETbarrel, float EisooverETendcap,
					      float hoverebarrel, float hovereendcap,
					      float clusshapebarrel, float clusshapeendcap, 
					      float r9barrel, float r9endcap,
					      float detabarrel, float detaendcap,
					      float dphibarrel, float dphiendcap)
{
  float barreleta = 1.479;
  float endcapeta = 2.65;

  int rc = 0;
  // Loop over all oh electrons
  for (int i=0;i<NohEle;i++) {
    float ohEleHoverE = ohEleHforHoverE[i]/ohEleE[i];
    int isbarrel = 0;
    int isendcap = 0;
    if(TMath::Abs(ohEleEta[i]) < barreleta)
      isbarrel = 1;
    if(barreleta < TMath::Abs(ohEleEta[i]) && TMath::Abs(ohEleEta[i]) < endcapeta)
      isendcap = 1;

    if ( ohEleEt[i] > Et) {
      if( TMath::Abs(ohEleEta[i]) < endcapeta ) {
	if (ohEleNewSC[i]<=1) {
	  if (ohElePixelSeeds[i]>0) {
	    if ( ohEleL1iso[i] >= L1iso ) {  // L1iso is 0 or 1 
	      if( ohEleL1Dupl[i] == false) { // remove double-counted L1 SCs 
		if ( (isbarrel && ((ohEleHiso[i]/ohEleEt[i]) < HisooverETbarrel)) ||
		     (isendcap && ((ohEleHiso[i]/ohEleEt[i]) < HisooverETendcap)) ) {
		  if ( (isbarrel && ((ohEleEiso[i]/ohEleEt[i]) < EisooverETbarrel)) ||
		       (isendcap && ((ohEleEiso[i]/ohEleEt[i]) < EisooverETendcap)) ) {
		    if ( ((isbarrel) && (ohEleHoverE < hoverebarrel)) ||
			 ((isendcap) && (ohEleHoverE < hovereendcap))) {
		      if ( (isbarrel && (((ohEleTiso[i] < Tisobarrel && ohEleTiso[i] != -999.) || (Tisobarrel == 999.)))) ||
			   (isendcap && (((ohEleTiso[i] < Tisoendcap && ohEleTiso[i] != -999.) || (Tisoendcap == 999.))))) {
			if (((isbarrel) && (ohEleTiso[i]/ohEleEt[i] < Tisoratiobarrel)) ||
			    ((isendcap) && (ohEleTiso[i]/ohEleEt[i] < Tisoratioendcap))) {
			  if ( (isbarrel && ohEleClusShap[i] < clusshapebarrel) ||
			       (isendcap && ohEleClusShap[i] < clusshapeendcap) ) {
			    if ( (isbarrel && ohEleR9[i] < r9barrel) || 
				 (isendcap && ohEleR9[i] < r9endcap) ) {
			      if ( (isbarrel && TMath::Abs(ohEleDeta[i]) < detabarrel) ||
				   (isendcap && TMath::Abs(ohEleDeta[i]) < detaendcap) ) {
				if( (isbarrel && ohEleDphi[i] < dphibarrel) ||
				    (isendcap && ohEleDphi[i] < dphiendcap) ) {
				  rc++;
				}
			      }
			    }
			  }
			}
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  
  return rc;
}

int OHltTree::OpenHlt2ElectronsSamHarperPassed(float Et, int L1iso, 
					       float Tisobarrel, float Tisoendcap, 
					       float Tisoratiobarrel, float Tisoratioendcap, 
					       float HisooverETbarrel, float HisooverETendcap, 
					       float EisooverETbarrel, float EisooverETendcap,
					       float hoverebarrel, float hovereendcap,
					       float clusshapebarrel, float clusshapeendcap, 
					       float r9barrel, float r9endcap,
					       float detabarrel, float detaendcap,
					       float dphibarrel, float dphiendcap)
{
  float barreleta = 1.479;
  float endcapeta = 2.65;

  int rc = 0;
  int rcsconly = 0;

  // Loop over all oh electrons
  for (int i=0;i<NohEle;i++) {
    float ohEleHoverE = ohEleHforHoverE[i]/ohEleE[i];
    int isbarrel = 0;
    int isendcap = 0;
    if(TMath::Abs(ohEleEta[i]) < barreleta)
      isbarrel = 1;
    if(barreleta < TMath::Abs(ohEleEta[i]) && TMath::Abs(ohEleEta[i]) < endcapeta)
      isendcap = 1;

    if ( ohEleEt[i] > Et) {
      if( TMath::Abs(ohEleEta[i]) < endcapeta ) {
	if (ohEleNewSC[i]==1) {
	  if ( ohEleL1iso[i] >= L1iso ) {  // L1iso is 0 or 1 
	    if( ohEleL1Dupl[i] == false) { // remove double-counted L1 SCs 
	      if ( (isbarrel && ((ohEleEiso[i]/ohEleEt[i]) < EisooverETbarrel)) ||
		   (isendcap && ((ohEleEiso[i]/ohEleEt[i]) < EisooverETendcap)) ) {
		if ( (isbarrel && ((ohEleHiso[i]/ohEleEt[i]) < HisooverETbarrel)) ||
		     (isendcap && ((ohEleHiso[i]/ohEleEt[i]) < HisooverETendcap)) ) {
		  if ( ((isbarrel) && (ohEleHoverE < hoverebarrel)) || 
		       ((isendcap) && (ohEleHoverE < hovereendcap))) { 
		    if ( (isbarrel && ohEleClusShap[i] < clusshapebarrel) ||
			 (isendcap && ohEleClusShap[i] < clusshapeendcap) ) {
		      if ( (isbarrel && ohEleR9[i] < r9barrel) || 
			   (isendcap && ohEleR9[i] < r9endcap) ) {
			if (ohElePixelSeeds[i]>0) {
			  rcsconly++;
			}
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  
  if(rcsconly >= 2) {
    for (int i=0;i<NohEle;i++) {
      float ohEleHoverE = ohEleHforHoverE[i]/ohEleE[i];
      int isbarrel = 0;
      int isendcap = 0;
      if(TMath::Abs(ohEleEta[i]) < barreleta)
	isbarrel = 1;
      if(barreleta < TMath::Abs(ohEleEta[i]) && TMath::Abs(ohEleEta[i]) < endcapeta)
	isendcap = 1;
      
      if ( ohEleEt[i] > Et) {
	if( TMath::Abs(ohEleEta[i]) < endcapeta ) {
	  if (ohEleNewSC[i]<=1) {
	    if (ohElePixelSeeds[i]>0) {
	      if ( ohEleL1iso[i] >= L1iso ) {  // L1iso is 0 or 1 
		if( ohEleL1Dupl[i] == false) { // remove double-counted L1 SCs 
		  if ( (isbarrel && ((ohEleHiso[i]/ohEleEt[i]) < HisooverETbarrel)) ||
		       (isendcap && ((ohEleHiso[i]/ohEleEt[i]) < HisooverETendcap)) ) {
		    if ( (isbarrel && ((ohEleEiso[i]/ohEleEt[i]) < EisooverETbarrel)) ||
			 (isendcap && ((ohEleEiso[i]/ohEleEt[i]) < EisooverETendcap)) ) {
		      if ( ((isbarrel) && (ohEleHoverE < hoverebarrel)) ||
			   ((isendcap) && (ohEleHoverE < hovereendcap))) {
			if ( (isbarrel && (((ohEleTiso[i] < Tisobarrel && ohEleTiso[i] != -999.) || (Tisobarrel == 999.)))) ||
			     (isendcap && (((ohEleTiso[i] < Tisoendcap && ohEleTiso[i] != -999.) || (Tisoendcap == 999.))))) {
			  if (((isbarrel) && (ohEleTiso[i]/ohEleEt[i] < Tisoratiobarrel)) ||
			      ((isendcap) && (ohEleTiso[i]/ohEleEt[i] < Tisoratioendcap))) {
			    if ( (isbarrel && ohEleClusShap[i] < clusshapebarrel) ||
				 (isendcap && ohEleClusShap[i] < clusshapeendcap) ) {
			      if ( (isbarrel && ohEleR9[i] < r9barrel) || 
				   (isendcap && ohEleR9[i] < r9endcap) ) {
				if ( (isbarrel && TMath::Abs(ohEleDeta[i]) < detabarrel) ||
				     (isendcap && TMath::Abs(ohEleDeta[i]) < detaendcap) ) {
				  if( (isbarrel && ohEleDphi[i] < dphibarrel) ||
				      (isendcap && ohEleDphi[i] < dphiendcap) ) {
				    rc++;
				  }
				}
			      }
			    }
			  }
			}
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  
  return rc;
}

int OHltTree::OpenHltGetElectronsSamHarperPassed(int *Passed,
						 float Et, int L1iso, 
						 float Tisobarrel, float Tisoendcap, 
						 float Tisoratiobarrel, float Tisoratioendcap, 
						 float HisooverETbarrel, float HisooverETendcap, 
						 float EisooverETbarrel, float EisooverETendcap,
						 float hoverebarrel, float hovereendcap,
						 float clusshapebarrel, float clusshapeendcap, 
						 float r9barrel, float r9endcap,
						 float detabarrel, float detaendcap,
						 float dphibarrel, float dphiendcap)
{
  int NPassed = 0;

  float barreleta = 1.479;
  float endcapeta = 2.65;

  // First check if only one electron is going to pass the cuts using the one electron code
  // we use the one electron code to look to see if 0 or 1 electrons pass this condition
  int rc = 0;
  int tmpindex=-999;
  // Loop over all oh electrons
  for (int i=0;i<NohEle;i++) {
    //    float ohEleHoverE = ohEleHiso[i]/ohEleEt[i];
    float ohEleHoverE = ohEleHforHoverE[i]/ohEleE[i];
    int isbarrel = 0;
    int isendcap = 0;
    if(TMath::Abs(ohEleEta[i]) < barreleta)
      isbarrel = 1;
    if(barreleta < TMath::Abs(ohEleEta[i]) && TMath::Abs(ohEleEta[i]) < endcapeta)
      isendcap = 1;

    if ( ohEleEt[i] > Et) {
      if( TMath::Abs(ohEleEta[i]) < endcapeta ) {
	if (ohEleNewSC[i]<=1) {
	  if (ohElePixelSeeds[i]>0) {
	    if ( ohEleL1iso[i] >= L1iso ) {  // L1iso is 0 or 1 
	      if( ohEleL1Dupl[i] == false) { // remove double-counted L1 SCs 
		if ( (isbarrel && ((ohEleHiso[i]/ohEleEt[i]) < HisooverETbarrel)) ||
		     (isendcap && ((ohEleHiso[i]/ohEleEt[i]) < HisooverETendcap)) ) {
		  if ( (isbarrel && ((ohEleEiso[i]/ohEleEt[i]) < EisooverETbarrel)) ||
		       (isendcap && ((ohEleEiso[i]/ohEleEt[i]) < EisooverETendcap)) ) {
		    if ( ((isbarrel) && (ohEleHoverE < hoverebarrel)) ||
			 ((isendcap) && (ohEleHoverE < hovereendcap))) {
		      if ( (isbarrel && (((ohEleTiso[i] < Tisobarrel && ohEleTiso[i] != -999.) || (Tisobarrel == 999.)))) ||
			   (isendcap && (((ohEleTiso[i] < Tisoendcap && ohEleTiso[i] != -999.) || (Tisoendcap == 999.))))) {
			if (((isbarrel) && (ohEleTiso[i]/ohEleEt[i] < Tisoratiobarrel)) ||
			    ((isendcap) && (ohEleTiso[i]/ohEleEt[i] < Tisoratioendcap))) {
			  if ( (isbarrel && ohEleClusShap[i] < clusshapebarrel) ||
			       (isendcap && ohEleClusShap[i] < clusshapeendcap) ) {
			    if ( (isbarrel && ohEleR9[i] < r9barrel) || 
				 (isendcap && ohEleR9[i] < r9endcap) ) {
			      if ( (isbarrel && TMath::Abs(ohEleDeta[i]) < detabarrel) ||
				   (isendcap && TMath::Abs(ohEleDeta[i]) < detaendcap) ) {
				if( (isbarrel && ohEleDphi[i] < dphibarrel) ||
				    (isendcap && ohEleDphi[i] < dphiendcap) ) {
				  rc++; 
				  tmpindex=i;  // temporarily store the index of this event
				  if(rc >1) break; // If we have 2 electrons passing, we need the 2 ele code
				}
			      }
			    }
			  }
			}
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  //  cout << "rc = " << rc << endl;;
  if(rc == 0){
    return 0;
  }
  if(rc == 1){
    Passed[NPassed++]=tmpindex; // if only 1 ele matched we can use this result without looping on the 2 ele code
    return NPassed;
  }
  // otherwise, we use the 2 ele code:
  
  int rcsconly=0;
  std::vector<int> csPassedEle;
  // Loop over all oh electrons
  for (int i=0;i<NohEle;i++) {
    //float ohEleHoverE = ohEleHiso[i]/ohEleEt[i];
    float ohEleHoverE = ohEleHforHoverE[i]/ohEleE[i];
    int isbarrel = 0;
    int isendcap = 0;
    if(TMath::Abs(ohEleEta[i]) < barreleta)
      isbarrel = 1;
    if(barreleta < TMath::Abs(ohEleEta[i]) && TMath::Abs(ohEleEta[i]) < endcapeta)
      isendcap = 1;

    if ( ohEleEt[i] > Et) {
      if( TMath::Abs(ohEleEta[i]) < endcapeta ) {
	if (ohEleNewSC[i]==1) {
	  if ( ohEleL1iso[i] >= L1iso ) {  // L1iso is 0 or 1 
	    if( ohEleL1Dupl[i] == false) { // remove double-counted L1 SCs 
	      if ( (isbarrel && ((ohEleEiso[i]/ohEleEt[i]) < EisooverETbarrel)) ||
		   (isendcap && ((ohEleEiso[i]/ohEleEt[i]) < EisooverETendcap)) ) {
		if ( (isbarrel && ((ohEleHiso[i]/ohEleEt[i]) < HisooverETbarrel)) ||
		     (isendcap && ((ohEleHiso[i]/ohEleEt[i]) < HisooverETendcap)) ) {
		  if ( ((isbarrel) && (ohEleHoverE < hoverebarrel)) || 
		       ((isendcap) && (ohEleHoverE < hovereendcap))) { 
		    if ( (isbarrel && ohEleClusShap[i] < clusshapebarrel) ||
			 (isendcap && ohEleClusShap[i] < clusshapeendcap) ) {
		      if ( (isbarrel && ohEleR9[i] < r9barrel) || 
			   (isendcap && ohEleR9[i] < r9endcap) ) {
			if (ohElePixelSeeds[i]>0) {
			  rcsconly++;
			  csPassedEle.push_back(i);
			}
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  //  cout << "rcsconly = " << rcsconly << endl;
  if(rcsconly == 0){ // This really shouldn't happen, but included for safety
    return NPassed;
  }
  if(rcsconly == 1){ // ok, we only had 1 cs, but 2 eles were assigned to it
    Passed[NPassed++] = tmpindex;
    return NPassed;
  }

  if(rcsconly >= 2) {
    for (int i=0;i<NohEle;i++) {
      float ohEleHoverE = ohEleHforHoverE[i]/ohEleE[i];
      int isbarrel = 0;
      int isendcap = 0;
      if(TMath::Abs(ohEleEta[i]) < barreleta)
	isbarrel = 1;
      if(barreleta < TMath::Abs(ohEleEta[i]) && TMath::Abs(ohEleEta[i]) < endcapeta)
	isendcap = 1;
      
      if ( ohEleEt[i] > Et) {
	if( TMath::Abs(ohEleEta[i]) < endcapeta ) {
	  if (ohEleNewSC[i]<=1) {
	    if (ohElePixelSeeds[i]>0) {
	      if ( ohEleL1iso[i] >= L1iso ) {  // L1iso is 0 or 1 
		if( ohEleL1Dupl[i] == false) { // remove double-counted L1 SCs 
		  if ( (isbarrel && ((ohEleHiso[i]/ohEleEt[i]) < HisooverETbarrel)) ||
		       (isendcap && ((ohEleHiso[i]/ohEleEt[i]) < HisooverETendcap)) ) {
		    if ( (isbarrel && ((ohEleEiso[i]/ohEleEt[i]) < EisooverETbarrel)) ||
			 (isendcap && ((ohEleEiso[i]/ohEleEt[i]) < EisooverETendcap)) ) {
		      if ( ((isbarrel) && (ohEleHoverE < hoverebarrel)) ||
			   ((isendcap) && (ohEleHoverE < hovereendcap))) {
			if ( (isbarrel && (((ohEleTiso[i] < Tisobarrel && ohEleTiso[i] != -999.) || (Tisobarrel == 999.)))) ||
			     (isendcap && (((ohEleTiso[i] < Tisoendcap && ohEleTiso[i] != -999.) || (Tisoendcap == 999.))))) {
			  if (((isbarrel) && (ohEleTiso[i]/ohEleEt[i] < Tisoratiobarrel)) ||
			      ((isendcap) && (ohEleTiso[i]/ohEleEt[i] < Tisoratioendcap))) {
			    if ( (isbarrel && ohEleClusShap[i] < clusshapebarrel) ||
				 (isendcap && ohEleClusShap[i] < clusshapeendcap) ) {
			      if ( (isbarrel && ohEleR9[i] < r9barrel) || 
				   (isendcap && ohEleR9[i] < r9endcap) ) {
				if ( (isbarrel && TMath::Abs(ohEleDeta[i]) < detabarrel) ||
				     (isendcap && TMath::Abs(ohEleDeta[i]) < detaendcap) ) {
				  if( (isbarrel && ohEleDphi[i] < dphibarrel) ||
				      (isendcap && ohEleDphi[i] < dphiendcap) ) {
				    for(int j=0;j<csPassedEle.size();j++){
				      if(i == csPassedEle.at(j)){ // check if the electron is in the cs matching list
					Passed[NPassed++] = i;
					rc++;  // ok, don't really need this, but keeping for debugging
					break;
				      }
				    }
				  }
				}
			      }
			    }
			  }
			}
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  
  return NPassed;
}

int OHltTree::OpenHlt2ElectronsAsymSamHarperPassed(float Et1, int L1iso1, 
						   float Tisobarrel1, float Tisoendcap1, 
						   float Tisoratiobarrel1, float Tisoratioendcap1, 
						   float HisooverETbarrel1, float HisooverETendcap1, 
						   float EisooverETbarrel1, float EisooverETendcap1,
						   float hoverebarrel1, float hovereendcap1,
						   float clusshapebarrel1, float clusshapeendcap1, 
						   float r9barrel1, float r9endcap1,
						   float detabarrel1, float detaendcap1,
						   float dphibarrel1, float dphiendcap1,
						   float Et2, int L1iso2, 
						   float Tisobarrel2, float Tisoendcap2, 
						   float Tisoratiobarrel2, float Tisoratioendcap2, 
						   float HisooverETbarrel2, float HisooverETendcap2, 
						   float EisooverETbarrel2, float EisooverETendcap2,
						   float hoverebarrel2, float hovereendcap2,
						   float clusshapebarrel2, float clusshapeendcap2, 
						   float r9barrel2, float r9endcap2,
						   float detabarrel2, float detaendcap2,
						   float dphibarrel2, float dphiendcap2){
  //  cout << "AB" << endl;
  int FirstEle[8000],SecondEle[8000];
  // cout << "BA" << endl;  
  int NFirst = OpenHltGetElectronsSamHarperPassed(FirstEle, Et1,L1iso1,
						  Tisobarrel1,Tisoendcap1, 
						  Tisoratiobarrel1,Tisoratioendcap1,
						  HisooverETbarrel1,HisooverETendcap1,
						  EisooverETbarrel1,EisooverETendcap1,
						  hoverebarrel1,hovereendcap1,
						  clusshapebarrel1,clusshapeendcap1,
						  r9barrel1,r9endcap1,
						  detabarrel1,detaendcap1,
						  dphibarrel1,dphiendcap1);
int NSecond = OpenHltGetElectronsSamHarperPassed(SecondEle,Et2,L1iso2,
						 Tisobarrel2,Tisoendcap2, 
						 Tisoratiobarrel2,Tisoratioendcap2,
						 HisooverETbarrel2,HisooverETendcap2,
						 EisooverETbarrel2,EisooverETendcap2,
						 hoverebarrel2,hovereendcap2,
						 clusshapebarrel2,clusshapeendcap2,
						 r9barrel2,r9endcap2,
						 detabarrel2,detaendcap2,
						 dphibarrel2,dphiendcap2);
// std::cout << "ABBA " << NFirst << "  " << NSecond << endl;
  if(NFirst == 0 || NSecond == 0) return 0;  // if no eles passed one condition, fail
  if(NFirst == 1 && NSecond == 1 &&
     FirstEle[0] == SecondEle[0]) return 0; //only 1 electron passed both conditions
  return 1;  // in any other case, at least 1 unique electron passed each condition, so pass the event
  
}



int OHltTree::OpenHlt1ElectronPassed(float Et, int L1iso, float Tiso, float Hiso)
{
  int rc = 0;
  // Loop over all oh electrons
  for (int i=0;i<NohEle;i++) {
    if ( ohEleEt[i] > Et) {
      if( TMath::Abs(ohEleEta[i]) < 2.65 ) 
	//	if ( ohEleHiso[i] < Hiso || ohEleHiso[i]/ohEleEt[i] < 0.05)
        if (ohEleHiso[i]/ohEleEt[i] < 0.15) 
	  if (ohEleNewSC[i]==1)
	    if (ohElePixelSeeds[i]>0)
	      if ( (ohEleTiso[i] < Tiso && ohEleTiso[i] != -999.) || (Tiso == 9999.))
		if ( ohEleL1iso[i] >= L1iso )   // L1iso is 0 or 1
		  if( ohEleL1Dupl[i] == false) // remove double-counted L1 SCs  
		    rc++;      
    }
  }
  
  return rc;
}

int  OHltTree::OpenHlt1PhotonPassed(float Et, int L1iso, float Tiso, float Eiso, float HisoBR, float HisoEC) 
{ 
  float barreleta = 1.479; 
  float endcapeta = 2.65; 

  // Default cleaning cuts for all photon paths
  float r9barrel = 0.98;
  float r9endcap = 1.0;
  float hoverebarrel = 0.15;
  float hovereendcap = 0.15;
  int rc = 0; 

  // Loop over all oh photons 
  for (int i=0;i<NohPhot;i++) { 
    float ohPhotE = ohPhotEt[i] / (sin(2*atan(exp(-1.0*ohPhotEta[i]))));    
    float ohPhotHoverE = ohPhotHforHoverE[i]/ohPhotE; 
    int isbarrel = 0; 
    int isendcap = 0; 
    if(TMath::Abs(ohPhotEta[i]) < barreleta) 
      isbarrel = 1; 
    if(barreleta < TMath::Abs(ohPhotEta[i]) && TMath::Abs(ohPhotEta[i]) < endcapeta) 
      isendcap = 1; 

    if ( ohPhotEt[i] > Et) {   
      if ( ((isbarrel) && (ohPhotHoverE < hoverebarrel)) || 
	   ((isendcap) && (ohPhotHoverE < hovereendcap))) { 
	if ( (isbarrel && ohPhotR9[i] < r9barrel) ||  
	     (isendcap && ohPhotR9[i] < r9endcap) ) { 
	  if( ohPhotL1Dupl[i] == false) // remove double-counted L1 SCs    
	    rc++;  
	}
      }
    }
  }

  return rc; 
} 

int  OHltTree::OpenHlt2ElectronMassWinPassed(float Et, int L1iso, float Hiso, float massLow, float massHigh)
{
      TLorentzVector e1;   
      TLorentzVector e2;   
      TLorentzVector meson;  
       
      int rc = 0;  
       
      for (int i=0;i<NohEle;i++) {  
        for (int j=0;j<NohEle && j != i;j++) {   
          if ( ohEleEt[i] > Et && ohEleEt[j] > Et) {  
	    if( TMath::Abs(ohEleEta[i]) < 2.65 && TMath::Abs(ohEleEta[j]) < 2.65) { 	      
	      if ( ((ohEleHiso[i] < Hiso) || (ohEleHiso[i]/ohEleEt[i] < 0.2)) && ((ohEleHiso[j] < Hiso) || (ohEleHiso[j]/ohEleEt[j] < 0.2)) ){ 
		if (ohEleNewSC[i]==1 && ohEleNewSC[j]==1) { 
		  if (ohElePixelSeeds[i]>0 && ohElePixelSeeds[j]>0 ) { 
		    if ( ohEleL1iso[i] >= L1iso && ohEleL1iso[j] >= L1iso ) {  // L1iso is 0 or 1  
		      if( ohEleL1Dupl[i] == false && ohEleL1Dupl[j] == false) { // remove double-counted L1 SCs    
			e1.SetPtEtaPhiM(ohEleEt[i],ohEleEta[i],ohElePhi[i],0.0);  
			e2.SetPtEtaPhiM(ohEleEt[j],ohEleEta[j],ohElePhi[j],0.0);  
			meson = e1 + e2;  
			
			float mesonmass = meson.M();   
			if(mesonmass > massLow && mesonmass < massHigh) 
			  rc++; 
		      } 
		    } 
		  } 
		} 
	      } 
	    } 
	  } 
	} 
      }
      
      if (rc>0)
	return 1;
      else
	return 0;
}

int OHltTree::OpenHlt2Electron1LegIdPassed(float Et,int L1iso,float Tiso,float Hiso)
{
  int rc = 0;
  // Loop over all oh electrons
  for (int i=0;i<NohEle;i++) {
    if ( ohEleEt[i] > Et) {
      if( TMath::Abs(ohEleEta[i]) < 2.65 ) {
        if ( ohEleHiso[i] < Hiso || ohEleHiso[i]/ohEleEt[i] < 0.05) {
          if (ohEleNewSC[i]==1) {
            if (ohElePixelSeeds[i]>0) {
              if ( (ohEleTiso[i] < Tiso && ohEleTiso[i] != -999.) || (Tiso == 9999.)) {
                if ( ohEleL1iso[i] >= L1iso ) {  // L1iso is 0 or 1
                  if( ohEleL1Dupl[i] == false) { // remove double-counted L1 SCs
		    // Loop over all oh electrons
		    for (int i=0;i<NohEle;i++) {
		      if ( ohEleEt[i] > Et) {
			if( TMath::Abs(ohEleEta[i]) < 2.65 )
			  if ( ohEleHiso[i] < Hiso || ohEleHiso[i]/ohEleEt[i] < 0.05)
			    if (ohEleNewSC[i]==1)
			      if (ohElePixelSeeds[i]>0)
				if ( (ohEleTiso[i] < Tiso && ohEleTiso[i] != -999.) || (Tiso == 9999.) )
				  if ( ohEleL1iso[i] >= L1iso )   // L1iso is 0 or 1
				    if ( (TMath::Abs(ohEleEta[i]) < 1.479 && ohEleClusShap[i] < 0.015) ||
					 (1.479 < TMath::Abs(ohEleEta[i]) && TMath::Abs(ohEleEta[i]) < 2.65 && ohEleClusShap[i] < 0.04) )
				      if ( (ohEleDeta[i] < 0.008) && (ohEleDphi[i] < 0.1) )
					if( ohEleL1Dupl[i] == false) // remove double-counted L1 SCs
					  rc++;
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  
  return rc;
}

int OHltTree::OpenHlt1ElectronHTPassed(float Et, float HT,float jetThreshold, 
				       int L1iso, float Tiso, float Hiso, float dr)
{
  vector<int> PassedElectrons;
  int NPassedEle=0;
  for (int i=0;i<NohEle;i++) {
    if ( ohEleEt[i] > Et) {
      if( TMath::Abs(ohEleEta[i]) < 2.65 ) 
	//if ( ohEleHiso[i] < Hiso || ohEleHiso[i]/ohEleEt[i] < 0.05)
	if (ohEleHiso[i]/ohEleEt[i] < 0.15)
	  if (ohEleNewSC[i]==1)
	    if (ohElePixelSeeds[i]>0)
	      if ( (ohEleTiso[i] < Tiso && ohEleTiso[i] != -999.) || (Tiso == 9999.))
		if ( ohEleL1iso[i] >= L1iso )   // L1iso is 0 or 1
		  if( ohEleL1Dupl[i] == false){ // remove double-counted L1 SCs  
		    PassedElectrons.push_back(i);
		    NPassedEle++;
		  }
		  
      
    }
  }
  if(NPassedEle==0) return 0;
  
  float sumHT=0;
  for(int i=0;i<NrecoJetCal;i++){
    if(recoJetCalPt[i] < jetThreshold) continue;
    bool jetPass=true;
    for(unsigned int iEle = 0; iEle<PassedElectrons.size(); iEle++){
      float dphi = ohElePhi[PassedElectrons.at(iEle)] - recoJetCalPhi[i];
      float deta = ohEleEta[PassedElectrons.at(iEle)] - recoJetCalEta[i];
      if(dphi*dphi+deta*deta<dr*dr) // require electron not in any jet
	jetPass=false;
    }
    if(jetPass)
      sumHT+=(recoJetCalE[i]/cosh(recoJetCalEta[i]));
  }
  if(sumHT>HT) return 1;
  return 0;
}

int OHltTree::OpenHlt1ElectronEleIDHTPassed(float Et, float HT,float jetThreshold, 
				       int L1iso, float Tiso, float Hiso, float dr)
{
  vector<int> PassedElectrons;
  int NPassedEle=0;
  for (int i=0;i<NohEle;i++) {
    if ( ohEleEt[i] > Et) {
      if( TMath::Abs(ohEleEta[i]) < 2.65 ) 
	//if ( ohEleHiso[i] < Hiso || ohEleHiso[i]/ohEleEt[i] < 0.05)
	if (ohEleHiso[i]/ohEleEt[i] < 0.15)
	  if (ohEleNewSC[i]==1)
	    if (ohElePixelSeeds[i]>0)
	      if ( (ohEleTiso[i] < Tiso && ohEleTiso[i] != -999.) || (Tiso == 9999.))
		if ( ohEleL1iso[i] >= L1iso )   // L1iso is 0 or 1
		  if ( (TMath::Abs(ohEleEta[i]) < 1.479 && ohEleClusShap[i] < 0.015) ||  
		       (1.479 < TMath::Abs(ohEleEta[i]) && TMath::Abs(ohEleEta[i]) < 2.65 && ohEleClusShap[i] < 0.04) )   
		    if ( (ohEleDeta[i] < 0.008) && (ohEleDphi[i] < 0.1) )  
		      if( ohEleL1Dupl[i] == false){ // remove double-counted L1 SCs  
			PassedElectrons.push_back(i);
			NPassedEle++;
		      }
      
      
    }
  }
  if(NPassedEle==0) return 0;
  
  float sumHT=0;
  for(int i=0;i<NrecoJetCal;i++){
    if(recoJetCalPt[i] < jetThreshold) continue;
    bool jetPass=true;
    for(unsigned int iEle = 0; iEle<PassedElectrons.size(); iEle++){
      float dphi = ohElePhi[PassedElectrons.at(iEle)] - recoJetCalPhi[i];
      float deta = ohEleEta[PassedElectrons.at(iEle)] - recoJetCalEta[i];
      if(dphi*dphi+deta*deta<dr*dr) // require electron not in any jet
	jetPass=false;
    }
    if(jetPass)
      sumHT+=(recoJetCalE[i]/cosh(recoJetCalEta[i]));
  }
  if(sumHT>HT) return 1;
  return 0;
}
 


int OHltTree::OpenHltRPassed(float Rmin, float MRmin,bool MRP, int NJmax, float jetPt){
  //make a list of the vectors
  vector<TLorentzVector*> JETS;

  for(int i=0; i<NrecoJetCal; i++) {
    if(fabs(recoJetCalEta[i])>=3 || recoJetCalPt[i] < jetPt) continue; // require jets with eta<3
    TLorentzVector* tmp = new TLorentzVector();
    tmp->SetPtEtaPhiE(recoJetCalPt[i],recoJetCalEta[i],
		      recoJetCalPhi[i],recoJetCalE[i]);
    
    JETS.push_back(tmp);
  }
  
  int jetsSize = 0;
  jetsSize = JETS.size(); 
  
  //Now make the hemispheres
  //for this simulation, we will used TLorentzVectors, although this is probably not
  //possible online
  if(jetsSize<2) return 0;
  if(NJmax!=-1 && jetsSize > NJmax) return 1;
  int N_comb = 1; // compute the number of combinations of jets possible
  for(int i = 0; i < jetsSize; i++){ // this is code is kept as close as possible
    N_comb *= 2;                 //to Chris' code for validation
  }
  TLorentzVector j1,j2;
  double M_min = 9999999999.0;
  double dHT_min = 99999999.0;
  int j_count;
  for(int i=0;i<N_comb;i++){       
    TLorentzVector j_temp1, j_temp2;
    int itemp = i;
    j_count = N_comb/2;
    int count = 0;
    while(j_count > 0){
      if(itemp/j_count == 1){
	j_temp1 += *(JETS.at(count));
      } else {
	j_temp2 += *(JETS.at(count));
      }
      itemp -= j_count*(itemp/j_count);
      j_count /= 2;
      count++;
    }
    double M_temp = j_temp1.M2()+j_temp2.M2();
    if(M_temp < M_min){
      M_min = M_temp;
      j1= j_temp1;
      j2= j_temp2;
    }
    double dHT_temp = fabs(j_temp1.E()-j_temp2.E());
    if(dHT_temp < dHT_min){
      dHT_min = dHT_temp;
      //deltaHT = dHT_temp;
    }
  }
  
  j1.SetPtEtaPhiM(j1.Pt(),j1.Eta(),j1.Phi(),0.0);
  j2.SetPtEtaPhiM(j2.Pt(),j2.Eta(),j2.Phi(),0.0);
  
  if(j2.Pt() > j1.Pt()){
    TLorentzVector temp = j1;
    j1 = j2;
    j2 = temp;
  }
  //Done Calculating Hemispheres
  //Now we can check if the event is of type R or R'
  
  double num = j1.P()-j2.P();
  double den = j1.Pz()-j2.Pz();
  if(fabs(num)==fabs(den)) return 0; //ignore if beta=1
  if(fabs(num)<fabs(den) && MRP) return 0; //num<den ==> R event
  if(fabs(num)>fabs(den) && !MRP) return 0; // num>den ==> R' event
  
  //now we can calculate MTR
  TVector3 met;
  met.SetPtEtaPhi(recoMetCal,0,recoMetCalPhi);
  double MTR = sqrt(0.5*(met.Mag()*(j1.Pt()+j2.Pt()) - met.Dot(j1.Vect()+j2.Vect())));
  
  //calculate MR or MRP
  double MR=0;
  if(!MRP){    //CALCULATE MR
    double temp = (j1.P()*j2.Pz()-j2.P()*j1.Pz())*(j1.P()*j2.Pz()-j2.P()*j1.Pz());
    temp /= (j1.Pz()-j2.Pz())*(j1.Pz()-j2.Pz())-(j1.P()-j2.P())*(j1.P()-j2.P());    
    MR = 2.*sqrt(temp);
  }else{      //CALCULATE MRP   
    double jaP = j1.Pt()*j1.Pt() +j1.Pz()*j2.Pz()-j1.P()*j2.P();
    double jbP = j2.Pt()*j2.Pt() +j1.Pz()*j2.Pz()-j1.P()*j2.P();
    jbP *= -1.;
    double den = sqrt((j1.P()-j2.P())*(j1.P()-j2.P())-(j1.Pz()-j2.Pz())*(j1.Pz()-j2.Pz()));
    
    jaP /= den;
    jbP /= den;
    
    double temp = jaP*met.Dot(j2.Vect())/met.Mag() + jbP*met.Dot(j1.Vect())/met.Mag();
    temp = temp*temp;
    
    den = (met.Dot(j1.Vect()+j2.Vect())/met.Mag())*(met.Dot(j1.Vect()+j2.Vect())/met.Mag())-(jaP-jbP)*(jaP-jbP);
    
    if(den <= 0.0) return 0.;

    temp /= den;
    temp = 2.*sqrt(temp);
    
    double bR = (jaP-jbP)/(met.Dot(j1.Vect()+j2.Vect())/met.Mag());
    double gR = 1./sqrt(1.-bR*bR);
    
    temp *= gR;

    MR = temp;
  }
  if(MR<MRmin || float(MTR)/float(MR)<Rmin) return 0;
  
  return 1;
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

  for(int ic = 0; ic < 10; ic++)
    L3MuCandIDForOnia[ic] = -1;

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
		if(ohMuL2Iso[j] >= iso) { // L2 isolation
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
                          L3MuCandIDForOnia[rcL1L2L3] = i;
			  rcL1++;
			  rcL1L2L3++;
			} // End L1 matching and quality cuts	      
		    }
		  else
		    {
                      L3MuCandIDForOnia[rcL1L2L3] = i;
		      rcL1L2L3++;
		    }
		} // End L2 isolation cut 
	      } // End L2 eta cut
	    } // End L2 pT cut
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
		if(ohMuL2Iso[i] >= iso) {  // L2 isolation  
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
		} // End L2 isolation cut 
	      } // End L2 eta cut
	    } // End L2 pT cut 
          } // End L3 isolation cut 
        } // End L3 DR cut 
      } // End L3 pT cut 
    } // End L3 eta cut 
  } // End loop over L3 muons                  

  return rcL1L2L3; 
} 

int OHltTree::OpenHlt1L2MuonPassed(double ptl1, double ptl2, double dr) 
{ 
  // This is a modification of the standard Hlt1Muon code, which does not consider L3 information 
  
  int rcL1 = 0; int rcL2 = 0; int rcL1L2L3 = 0; 
  int NL1Mu = 8; 
  int L1MinimalQuality = 3; 
  int L1MaximalQuality = 7; 
  int doL1L2matching = 0; 
  
  // Loop over all oh L2 muons and apply cuts 
  for (int j=0;j<NohMuL2;j++) {   
    int bestl1l2drmatchind = -1; 
    double bestl1l2drmatch = 999.0;  
    
    if(fabs(ohMuL2Eta[j])>=2.5) continue;  // L2 eta cut 
    if( ohMuL2Pt[j] <= ptl2 ) continue; // L2 pT cut 
    rcL2++; 
    
    // Begin L1 muons here. 
    // Require there be an L1Extra muon Delta-R 
    // matched to the L2 candidate, and that it have  
    // good quality and pass nominal L1 pT cuts  
    for(int k = 0;k < NL1Mu;k++) { 
      if( (L1MuPt[k] < ptl1) ) continue; // L1 pT cut 
      
      double deltaphi = fabs(ohMuL2Phi[j]-L1MuPhi[k]);  
      if(deltaphi > 3.14159) deltaphi = (2.0 * 3.14159) - deltaphi;  
      double deltarl1l2 = sqrt((ohMuL2Eta[j]-L1MuEta[k])*(ohMuL2Eta[j]-L1MuEta[k]) + (deltaphi*deltaphi)); 
      if(deltarl1l2 < bestl1l2drmatch) { 
	bestl1l2drmatchind = k;   
	bestl1l2drmatch = deltarl1l2;   
      }   
    } // End loop over L1Extra muons 
    if(doL1L2matching == 1) { 
	// Cut on L1<->L2 matching and L1 quality 
	if((bestl1l2drmatch > 0.3) || (L1MuQal[bestl1l2drmatchind] < L1MinimalQuality) || (L1MuQal[bestl1l2drmatchind] > L1MaximalQuality)) { 
	    rcL1 = 0;  
	    cout << "Failed L1-L2 match/quality" << endl; 
	    cout << "L1-L2 delta-eta = " << L1MuEta[bestl1l2drmatchind] << ", " << ohMuL2Eta[j] << endl;  
	    cout << "L1-L2 delta-pho = " << L1MuPhi[bestl1l2drmatchind] << ", " << ohMuL2Phi[j] << endl;   
	    cout << "L1-L2 delta-R = " << bestl1l2drmatch << endl; 
	} else { 
	  cout << "Passed L1-L2 match/quality" << endl; 
	  rcL1++; 
	  rcL1L2L3++; 
	} // End L1 matching and quality cuts            
    } else { 
      rcL1L2L3++; 
    } 
  } // End L2 loop over muons 
  return rcL1L2L3; 
} 

int OHltTree::OpenHltMuPixelPassed(double ptPix, double pPix, double etaPix, double DxyPix, double DzPix, int NHitsPix, double normChi2Pix, double *massMinPix, double *massMaxPix, double DzMuonPix, bool checkChargePix)
{

  //   printf("\n\n");
  const double muMass = 0.105658367;
  TLorentzVector pix4Mom, mu4Mom, onia4Mom;
  int iNPix = 0;
  //reset counter variables:
  for(int iMu = 0; iMu < 10; iMu++){
    L3PixelCandIDForOnia[iMu] = -1;
    L3MuPixCandIDForOnia[iMu] = -1;
  }

  //0.) check how many L3 muons there are:
  int nMuons = 0;
  for(int iMu = 0; iMu < 10; iMu++)
    if(L3MuCandIDForOnia[iMu] > -1)
      nMuons++;

  //1.) loop over the Pixel tracks
  for(int iP = 0; iP < NohOniaPixel; iP++){

    //select those that survive the kinematical and
    //topological selection cuts
    if(fabs(ohOniaPixelEta[iP]) > etaPix) continue; //eta cut
    if(ohOniaPixelPt[iP] < ptPix) continue; //pT cut
    double momThisPix = ohOniaPixelPt[iP] * cosh(ohOniaPixelEta[iP]);
    if(momThisPix < pPix) continue; //momentum cut
    if(ohOniaPixelHits[iP] <  NHitsPix) continue; //min. nb. of hits
    if(ohOniaPixelNormChi2[iP] > normChi2Pix) continue; //chi2 cut
    if(fabs(ohOniaPixelDr[iP]) > DxyPix) continue; //Dr cut
    if(fabs(ohOniaPixelDz[iP]) > DzPix) continue;

    pix4Mom.SetPtEtaPhiM(ohOniaPixelPt[iP], ohOniaPixelEta[iP], ohOniaPixelPhi[iP], muMass);
    //2.) loop now over all L3 muons and check if they would give a
    //Onia (J/psi or upsilon) pair:
    for(int iMu = 0; iMu < nMuons; iMu++){
      mu4Mom.SetPtEtaPhiM(ohMuL3Pt[L3MuCandIDForOnia[iMu]],
                          ohMuL3Eta[L3MuCandIDForOnia[iMu]],
                          ohMuL3Phi[L3MuCandIDForOnia[iMu]], muMass);
      onia4Mom = pix4Mom + mu4Mom;

      double oniaMass = onia4Mom.M();
      if(oniaMass < massMinPix[0] || oniaMass > massMaxPix[1]) continue; //mass cut
      if(oniaMass > massMaxPix[0] && oniaMass < massMinPix[1]) continue; //mass cut
      if(checkChargePix)
        if(ohMuL3Chg[iMu] == ohOniaPixelChg[iP]) continue; //charge cut
      if(fabs(ohMuL3Dz[iMu] - ohOniaPixelDz[iP]) > DzMuonPix) continue;

      //store the surviving pixel-muon combinations:
      if(iNPix < 10){
        L3PixelCandIDForOnia[iNPix] = iP;
        L3MuPixCandIDForOnia[iNPix] = iMu;
        iNPix++;
      }
      //       printf("mu[%d]-pixel[%d] inv. mass %f\n",
      //           L3MuCandIDForOnia[iMu], iP, oniaMass);
    }
  }

  //   hNPixelCand->Fill(iNPix);
  return iNPix;
}

int OHltTree::OpenHltMuTrackPassed(double ptTrack, double pTrack, double etaTrack, double DxyTrack, double DzTrack, int NHitsTrack, double normChi2Track, double *massMinTrack, double *massMaxTrack, double DzMuonTrack, bool checkChargeTrack)
{

  double pixMatchingDeltaR = 0.03;
  const double muMass = 0.105658367;
  TLorentzVector track4Mom, mu4Mom, onia4Mom;
  int iNTrack = 0;

  //0.) check how many pixel-muon combinations there are:
  int nComb = 0;
  for(int iMu = 0; iMu < 10; iMu++)
    if(L3MuPixCandIDForOnia[iMu] > -1)
      nComb++;

  //   printf("OpenHltMuTrackPassed: %d incoming pixels and %d tracks\n", nComb, NohOniaTrack);

  //1.) loop over the Tracker tracks
  for(int iT = 0; iT < NohOniaTrack; iT++){

    //select those that survive the kinematical and
    //topological selection cuts
    if(fabs(ohOniaTrackEta[iT]) > etaTrack) continue; //eta cut
    if(ohOniaTrackPt[iT] < ptTrack) continue; //pT cut
    double momThisTrack = ohOniaTrackPt[iT] * cosh(ohOniaTrackEta[iT]);
    //     printf("track[%d] has eta %f, pT %f and mom %f\n",
    //         iT, ohOniaTrackEta[iT], ohOniaTrackPt[iT], momThisTrack);
    if(momThisTrack < pTrack) continue; //momentum cut
    if(ohOniaTrackHits[iT] <  NHitsTrack) continue; //min. nb. of hits
    if(ohOniaTrackNormChi2[iT] > normChi2Track) continue; //chi2 cut
    if(fabs(ohOniaTrackDr[iT]) > DxyTrack) continue; //Dr cut
    if(fabs(ohOniaTrackDz[iT]) > DzTrack) continue;

    //2.) loop over the pixels candidates to see whether the track
    //under investigation has a match to the pixel track
    bool trackMatched = false;
    for(int iPix = 0; iPix < nComb; iPix++){

      if(trackMatched) break; //in case the track was already matched
      if(L3PixelCandIDForOnia[iPix] < 0) continue; //in case the pixel has been matched to a previous track

      double deltaEta = ohOniaPixelEta[L3PixelCandIDForOnia[iPix]] - ohOniaTrackEta[iT];
      double deltaPhi = ohOniaPixelPhi[L3PixelCandIDForOnia[iPix]] - ohOniaTrackPhi[iT];
      double deltaR = sqrt(pow(deltaEta,2) + pow(deltaPhi,2));

      if(deltaR > pixMatchingDeltaR) continue;
      //       printf("track[%d], pixel[%d], delta R %f\n", iT, L3PixelCandIDForOnia[iPix], deltaR);

      trackMatched = true;
      L3PixelCandIDForOnia[iPix] = -1; //deactivate this candidate to not match it to any further track

      track4Mom.SetPtEtaPhiM(ohOniaTrackPt[iT], ohOniaTrackEta[iT], ohOniaTrackPhi[iT], muMass);
      //check if the matched tracker track combined with the
      //muon gives again an opposite sign onia:
      mu4Mom.SetPtEtaPhiM(ohMuL3Pt[L3MuPixCandIDForOnia[iPix]],
                          ohMuL3Eta[L3MuPixCandIDForOnia[iPix]],
                          ohMuL3Phi[L3MuPixCandIDForOnia[iPix]], muMass);
      onia4Mom = track4Mom + mu4Mom;

      double oniaMass = onia4Mom.M();
      //       printf("mu[%d]-track[%d] inv. mass %f\n",
      //           L3MuPixCandIDForOnia[iPix], iT, oniaMass);

      if(oniaMass < massMinTrack[0] || oniaMass > massMaxTrack[1]) continue; //mass cut
      if(oniaMass > massMaxTrack[0] && oniaMass < massMinTrack[1]) continue; //mass cut

      //       printf("surviving: mu[%d]-track[%d] inv. mass %f\n",
      //           L3MuPixCandIDForOnia[iPix], iT, oniaMass);

      if(checkChargeTrack)
        if(ohMuL3Chg[L3MuPixCandIDForOnia[iPix]] == ohOniaTrackChg[iT]) continue; //charge cut
      if(fabs(ohMuL3Dz[L3MuPixCandIDForOnia[iPix]] - ohOniaTrackDz[iT]) > DzMuonTrack) continue; //deltaZ cut

      //store the surviving track-muon combinations:
      if(iNTrack < 10)
        iNTrack++;

      break; //don't check further pixels... go to next track
    }
  }

  //   if(iNTrack > 0)
  //     printf("found %d final candidates!!!\n", iNTrack);
  return iNTrack;
}

int OHltTree::OpenHltMuPixelPassed_JPsi(double ptPix, double pPix, double etaPix, double DxyPix, double DzPix, int NHitsPix, double normChi2Pix, double *massMinPix, double *massMaxPix, double DzMuonPix, bool checkChargePix, int histIndex)
{
  //   printf("in OpenHltMuPixelPassed_JPsi \n\n");
  const double muMass = 0.105658367;
  TLorentzVector pix4Mom, mu4Mom, onia4Mom;
  int iNPix = 0;
  //reset counter variables:
  for(int iMu = 0; iMu < 10; iMu++){
    L3PixelCandIDForJPsi[iMu] = -1;
    L3MuPixCandIDForJPsi[iMu] = -1;
  }

  //0.) check how many L3 muons there are:
  int nMuons = 0;
  for(int iMu = 0; iMu < 10; iMu++)
    if(L3MuCandIDForOnia[iMu] > -1)
      nMuons++;

//   Int_t countCut = 0, countOniaCut = 0;
  //1.) loop over the Pixel tracks
  for(int iP = 0; iP < NohOniaPixel; iP++){

//     countCut = 0;
//     hEta[histIndex][0][countCut]->Fill(ohOniaPixelEta[iP]);
//     hPt[histIndex][0][countCut]->Fill(ohOniaPixelPt[iP]);
//     hHits[histIndex][0][countCut]->Fill(ohOniaPixelHits[iP]);
//     hNormChi2[histIndex][0][countCut]->Fill(ohOniaPixelNormChi2[iP]);
//     hDxy[histIndex][0][countCut]->Fill(ohOniaPixelDr[iP]);
//     hDz[histIndex][0][countCut]->Fill(ohOniaPixelDz[iP]);

    //select those that survive the kinematical and
    //topological selection cuts
    if(fabs(ohOniaPixelEta[iP]) > etaPix) continue; //eta cut
    if(ohOniaPixelPt[iP] < ptPix) continue; //pT cut

    double momThisPix = ohOniaPixelPt[iP] * cosh(ohOniaPixelEta[iP]);
//     hP[histIndex][0][countCut]->Fill(momThisPix);
//     countCut++;

    if(momThisPix < pPix) continue; //momentum cut
    if(ohOniaPixelHits[iP] <  NHitsPix) continue; //min. nb. of hits
    if(ohOniaPixelNormChi2[iP] > normChi2Pix) continue; //chi2 cut
    if(fabs(ohOniaPixelDr[iP]) > DxyPix) continue; //Dr cut
    if(fabs(ohOniaPixelDz[iP]) > DzPix) continue;

//     hEta[histIndex][0][countCut]->Fill(ohOniaPixelEta[iP]);
//     hPt[histIndex][0][countCut]->Fill(ohOniaPixelPt[iP]);
//     hHits[histIndex][0][countCut]->Fill(ohOniaPixelHits[iP]);
//     hNormChi2[histIndex][0][countCut]->Fill(ohOniaPixelNormChi2[iP]);
//     hDxy[histIndex][0][countCut]->Fill(ohOniaPixelDr[iP]);
//     hDz[histIndex][0][countCut]->Fill(ohOniaPixelDz[iP]);
//     hP[histIndex][0][countCut]->Fill(momThisPix);
//     countCut++;

    pix4Mom.SetPtEtaPhiM(ohOniaPixelPt[iP], ohOniaPixelEta[iP], ohOniaPixelPhi[iP], muMass);
    //2.) loop now over all L3 muons and check if they would give a
    //JPsi pair:
    for(int iMu = 0; iMu < nMuons; iMu++){
      mu4Mom.SetPtEtaPhiM(ohMuL3Pt[L3MuCandIDForOnia[iMu]],
                          ohMuL3Eta[L3MuCandIDForOnia[iMu]],
                          ohMuL3Phi[L3MuCandIDForOnia[iMu]], muMass);
      onia4Mom = pix4Mom + mu4Mom;

      double oniaMass = onia4Mom.M();
      //       printf("mu[%d]-pixel[%d] inv. mass %f\n",
      //           L3MuCandIDForOnia[iMu], iP, oniaMass);
//       countOniaCut = 0;
//       if(oniaMass > 5.0) continue; //Only JPsi 
//       hOniaEta[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta());
//       hOniaRap[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity());
//       hOniaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Pt());
//       hOniaP[histIndex][0][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       hOniaMass[histIndex][0][countOniaCut]->Fill(onia4Mom.M());
//       hOniaEtaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
//       hOniaRapP[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       countOniaCut++;

      if(oniaMass < massMinPix[0] || oniaMass > massMaxPix[0]) continue; //mass cut
//       hOniaEta[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta());
//       hOniaRap[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity());
//       hOniaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Pt());
//       hOniaP[histIndex][0][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       hOniaMass[histIndex][0][countOniaCut]->Fill(onia4Mom.M());
//       hOniaEtaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
//       hOniaRapP[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       countOniaCut++;

      if(checkChargePix)
        if(ohMuL3Chg[iMu] == ohOniaPixelChg[iP]) continue; //charge cut

//       hOniaEta[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta());
//       hOniaRap[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity());
//       hOniaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Pt());
//       hOniaMass[histIndex][0][countOniaCut]->Fill(onia4Mom.M());
//       hOniaP[histIndex][0][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       hOniaEtaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
//       hOniaRapP[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       countOniaCut++;

      if(fabs(ohMuL3Dz[iMu] - ohOniaPixelDz[iP]) > DzMuonPix) continue;

//       hOniaEta[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta());
//       hOniaRap[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity());
//       hOniaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Pt());
//       hOniaMass[histIndex][0][countOniaCut]->Fill(onia4Mom.M());
//       hOniaP[histIndex][0][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       hOniaEtaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
//       hOniaRapP[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));

      //store the surviving pixel-muon combinations:
      if(iNPix < 10){
        L3PixelCandIDForJPsi[iNPix] = iP;
	L3MuPixCandIDForJPsi[iNPix] = iMu;
        iNPix++;
      }
      //       printf("surviving: mu[%d]-pixel[%d] inv. mass %f\n",
      //           L3MuCandIDForOnia[iMu], iP, oniaMass);
    }
  }

//   hNCand[histIndex][0]->Fill(iNPix);

  //Pixel Eta, Pt, P, DR
  if(iNPix!=0){
    for(int inP=0;inP<iNPix;inP++){

//       hPixCandEta[histIndex]->Fill(ohOniaPixelEta[L3PixelCandIDForJPsi[inP]]);
//       hPixCandPt[histIndex]->Fill(ohOniaPixelPt[L3PixelCandIDForJPsi[inP]]);
//       hPixCandP[histIndex]->Fill(ohOniaPixelPt[L3PixelCandIDForJPsi[inP]] * cosh(ohOniaPixelEta[L3PixelCandIDForJPsi[inP]]));

//       if(iNPix>=2){
//         for(int jnP=inP+1;jnP<iNPix;jnP++){
//           if(inP!=jnP){
//              double dEta = fabs(ohOniaPixelEta[L3PixelCandIDForJPsi[inP]]-ohOniaPixelEta[L3PixelCandIDForJPsi[jnP]]);
//              double dPhi = fabs(ohOniaPixelPhi[L3PixelCandIDForJPsi[inP]]-ohOniaPixelPhi[L3PixelCandIDForJPsi[jnP]]);
//              if(dPhi>TMath::Pi()) dPhi = 2.0*TMath::Pi()-dPhi;
//              hPixCanddr[histIndex]->Fill(sqrt(pow(dEta,2)+pow(dPhi,2)));
//           }
//         }
//       }
    }
  }

  return iNPix;
}

int OHltTree::OpenHltMuTrackPassed_JPsi(double ptTrack, double pTrack, double etaTrack, double DxyTrack, double DzTrack, int NHitsTrack, double normChi2Track, double *massMinTrack, double *massMaxTrack, double DzMuonTrack, bool checkChargeTrack, int histIndex)
{

  double pixMatchingDeltaR = 0.01;
  const double muMass = 0.105658367;
  TLorentzVector track4Mom, mu4Mom, onia4Mom;
  int iNTrack = 0;

  //0.) check how many pixel-muon combinations there are:
  int nComb = 0;
  for(int iMu = 0; iMu < 10; iMu++)
    if(L3MuPixCandIDForJPsi[iMu] > -1)
      nComb++;

  //   printf("OpenHltMuTrackPassed_JPsi: %d incoming pixels and %d tracks\n", nComb, NohOniaTrack);
//   Int_t countCut = 0, countOniaCut = 0;
  //1.) loop over the Tracker tracks
  for(int iT = 0; iT < NohOniaTrack; iT++){

    //select those that survive the kinematical and
    //topological selection cuts
//     countCut = 0;
//     hEta[histIndex][1][countCut]->Fill(ohOniaTrackEta[iT]);
//     hPt[histIndex][1][countCut]->Fill(ohOniaTrackPt[iT]);
//     hHits[histIndex][1][countCut]->Fill(ohOniaTrackHits[iT]);
//     hNormChi2[histIndex][1][countCut]->Fill(ohOniaTrackNormChi2[iT]);
//     hDxy[histIndex][1][countCut]->Fill(ohOniaTrackDr[iT]);
//     hDz[histIndex][1][countCut]->Fill(ohOniaTrackDz[iT]);

    if(fabs(ohOniaTrackEta[iT]) > etaTrack) continue; //eta cut
    if(ohOniaTrackPt[iT] < ptTrack) continue; //pT cut
    double momThisTrack = ohOniaTrackPt[iT] * cosh(ohOniaTrackEta[iT]);
    //     printf("track[%d] has eta %f, pT %f and mom %f\n",
    //         iT, ohOniaTrackEta[iT], ohOniaTrackPt[iT], momThisTrack);
//     hP[histIndex][1][countCut]->Fill(momThisTrack);
//     countCut++;

    if(momThisTrack < pTrack) continue; //momentum cut
    if(ohOniaTrackHits[iT] <  NHitsTrack) continue; //min. nb. of hits
    if(ohOniaTrackNormChi2[iT] > normChi2Track) continue; //chi2 cut
    if(fabs(ohOniaTrackDr[iT]) > DxyTrack) continue; //Dr cut
    if(fabs(ohOniaTrackDz[iT]) > DzTrack) continue;

//     hEta[histIndex][1][countCut]->Fill(ohOniaTrackEta[iT]);
//     hPt[histIndex][1][countCut]->Fill(ohOniaTrackPt[iT]);
//     hHits[histIndex][1][countCut]->Fill(ohOniaTrackHits[iT]);
//     hNormChi2[histIndex][1][countCut]->Fill(ohOniaTrackNormChi2[iT]);
//     hDxy[histIndex][1][countCut]->Fill(ohOniaTrackDr[iT]);
//     hDz[histIndex][1][countCut]->Fill(ohOniaTrackDz[iT]);
//     hP[histIndex][1][countCut]->Fill(momThisTrack);
//     countCut++;

    //     printf("track %d surviving kinematical pre-selection\n", iT);
    //2.) loop over the pixels candidates to see whether the track
    //under investigation has a match to the pixel track
    bool trackMatched = false;
    for(int iPix = 0; iPix < nComb; iPix++){
      if(trackMatched) break; //in case the track was already matched
      if(L3PixelCandIDForJPsi[iPix] < 0) continue; //in case the pixel has been matched to a previous track

      double deltaEta = ohOniaPixelEta[L3PixelCandIDForJPsi[iPix]] - ohOniaTrackEta[iT];
      double deltaPhi = ohOniaPixelPhi[L3PixelCandIDForJPsi[iPix]] - ohOniaTrackPhi[iT];
      if(deltaPhi>TMath::Pi()) deltaPhi = 2.0*TMath::Pi()-deltaPhi;
      double deltaR = sqrt(pow(deltaEta,2) + pow(deltaPhi,2));

      //       printf("delta R = %f\n", deltaR);
      if(deltaR > pixMatchingDeltaR) continue;
      //       printf("track[%d] and pixel[%d] are compatible (deltaR %f)\n", iT, L3PixelCandIDForJPsi[iPix], deltaR);

      trackMatched = true;
      L3PixelCandIDForJPsi[iPix] = -1; //deactivate this candidate to not match it to any further track

      track4Mom.SetPtEtaPhiM(ohOniaTrackPt[iT], ohOniaTrackEta[iT], ohOniaTrackPhi[iT], muMass);
      //check if the matched tracker track combined with the
      //muon gives again an opposite sign onia:
      mu4Mom.SetPtEtaPhiM(ohMuL3Pt[L3MuPixCandIDForJPsi[iPix]],
                          ohMuL3Eta[L3MuPixCandIDForJPsi[iPix]],
                          ohMuL3Phi[L3MuPixCandIDForJPsi[iPix]], muMass);
      onia4Mom = track4Mom + mu4Mom;

      double oniaMass = onia4Mom.M();
      //       printf("mu[%d]-track[%d] inv. mass %f\n",
      //           L3MuPixCandIDForJPsi[iPix], iT, oniaMass);

//       countOniaCut = 0;
//       if(oniaMass>5.0) continue; //Only JPsi
//       hOniaEta[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta());
//       hOniaRap[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity());
//       hOniaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Pt());
//       hOniaMass[histIndex][1][countOniaCut]->Fill(onia4Mom.M());
//       hOniaP[histIndex][1][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       hOniaEtaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
//       hOniaRapP[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       countOniaCut++;

      if(oniaMass < massMinTrack[0] || oniaMass > massMaxTrack[0]) continue; //mass cut
//       hOniaEta[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta());
//       hOniaRap[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity());
//       hOniaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Pt());
//       hOniaMass[histIndex][1][countOniaCut]->Fill(onia4Mom.M());
//       hOniaP[histIndex][1][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       hOniaEtaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
//       hOniaRapP[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       countOniaCut++;

      if(checkChargeTrack)
        if(ohMuL3Chg[L3MuPixCandIDForJPsi[iPix]] == ohOniaTrackChg[iT]) continue; //charge cut

//       hOniaEta[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta());
//       hOniaRap[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity());
//       hOniaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Pt());
//       hOniaMass[histIndex][1][countOniaCut]->Fill(onia4Mom.M());
//       hOniaP[histIndex][1][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       hOniaEtaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
//       hOniaRapP[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       countOniaCut++;

      if(fabs(ohMuL3Dz[L3MuPixCandIDForJPsi[iPix]] - ohOniaTrackDz[iT]) > DzMuonTrack) continue; //deltaZ cut
//       hOniaEta[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta());
//       hOniaRap[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity());
//       hOniaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Pt());
//       hOniaMass[histIndex][1][countOniaCut]->Fill(onia4Mom.M());
//       hOniaP[histIndex][1][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       hOniaEtaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
//       hOniaRapP[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));

      //store the surviving track-muon combinations:
      if(iNTrack < 10)
        iNTrack++;
      break; //don't check further pixels... go to next track
    }
  }

//   hNCand[histIndex][1]->Fill(iNTrack);
  return iNTrack;
}


int OHltTree::OpenHltMuPixelPassed_Ups(double ptPix, double pPix, double etaPix, double DxyPix, double DzPix, int NHitsPix, double normChi2Pix, double *massMinPix, double *massMaxPix, double DzMuonPix, bool checkChargePix, int histIndex)
{

  const double muMass = 0.105658367;
  TLorentzVector pix4Mom, mu4Mom, onia4Mom;
  int iNPix = 0;
  //reset counter variables:
  for(int iMu = 0; iMu < 10; iMu++){
    L3PixelCandIDForUps[iMu] = -1;
    L3MuPixCandIDForUps[iMu] = -1;
  }

  //0.) check how many L3 muons there are:
  int nMuons = 0;
  for(int iMu = 0; iMu < 10; iMu++)
    if(L3MuCandIDForOnia[iMu] > -1)
      nMuons++;

//   Int_t countCut = 0, countOniaCut = 0;
  //1.) loop over the Pixel tracks
  for(int iP = 0; iP < NohOniaPixel; iP++){

//     countCut = 0;
//     hEta[histIndex][0][countCut]->Fill(ohOniaPixelEta[iP]);
//     hPt[histIndex][0][countCut]->Fill(ohOniaPixelPt[iP]);
//     hHits[histIndex][0][countCut]->Fill(ohOniaPixelHits[iP]);
//     hNormChi2[histIndex][0][countCut]->Fill(ohOniaPixelNormChi2[iP]);
//     hDxy[histIndex][0][countCut]->Fill(ohOniaPixelDr[iP]);
//     hDz[histIndex][0][countCut]->Fill(ohOniaPixelDz[iP]);

    //select those that survive the kinematical and
    //topological selection cuts
    if(fabs(ohOniaPixelEta[iP]) > etaPix) continue; //eta cut
    if(ohOniaPixelPt[iP] < ptPix) continue; //pT cut
    double momThisPix = ohOniaPixelPt[iP] * cosh(ohOniaPixelEta[iP]);

//     hP[histIndex][0][countCut]->Fill(momThisPix);
//     countCut++;

    if(momThisPix < pPix) continue; //momentum cut
    if(ohOniaPixelHits[iP] <  NHitsPix) continue; //min. nb. of hits
    if(ohOniaPixelNormChi2[iP] > normChi2Pix) continue; //chi2 cut
    if(fabs(ohOniaPixelDr[iP]) > DxyPix) continue; //Dr cut
    if(fabs(ohOniaPixelDz[iP]) > DzPix) continue;

//     hEta[histIndex][0][countCut]->Fill(ohOniaPixelEta[iP]);
//     hPt[histIndex][0][countCut]->Fill(ohOniaPixelPt[iP]);
//     hHits[histIndex][0][countCut]->Fill(ohOniaPixelHits[iP]);
//     hNormChi2[histIndex][0][countCut]->Fill(ohOniaPixelNormChi2[iP]);
//     hDxy[histIndex][0][countCut]->Fill(ohOniaPixelDr[iP]);
//     hDz[histIndex][0][countCut]->Fill(ohOniaPixelDz[iP]);
//     hP[histIndex][0][countCut]->Fill(momThisPix);
//     countCut++;

    pix4Mom.SetPtEtaPhiM(ohOniaPixelPt[iP], ohOniaPixelEta[iP], ohOniaPixelPhi[iP], muMass);
    //2.) loop now over all L3 muons and check if they would give a
    //Ups pair:
    for(int iMu = 0; iMu < nMuons; iMu++){
      mu4Mom.SetPtEtaPhiM(ohMuL3Pt[L3MuCandIDForOnia[iMu]],
                          ohMuL3Eta[L3MuCandIDForOnia[iMu]],
                          ohMuL3Phi[L3MuCandIDForOnia[iMu]], muMass);
      onia4Mom = pix4Mom + mu4Mom;
      double oniaMass = onia4Mom.M();

//       countOniaCut = 0;
//       if(oniaMass < 8.0) continue; //Only Ups
//       hOniaEta[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta());
//       hOniaRap[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity());
//       hOniaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Pt());
//       hOniaP[histIndex][0][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       hOniaMass[histIndex][0][countOniaCut]->Fill(onia4Mom.M());
//       hOniaEtaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
//       hOniaRapP[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       countOniaCut++;

      if(oniaMass < massMinPix[0] || oniaMass > massMaxPix[0]) continue; //mass cut

//       hOniaEta[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta());
//       hOniaRap[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity());
//       hOniaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Pt());
//       hOniaMass[histIndex][0][countOniaCut]->Fill(onia4Mom.M());
//       hOniaP[histIndex][0][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       hOniaEtaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
//       hOniaRapP[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       countOniaCut++;

      if(checkChargePix)
        if(ohMuL3Chg[iMu] == ohOniaPixelChg[iP]) continue; //charge cut

//       hOniaEta[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta());
//       hOniaRap[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity());
//       hOniaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Pt());
//       hOniaMass[histIndex][0][countOniaCut]->Fill(onia4Mom.M());
//       hOniaP[histIndex][0][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       hOniaEtaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
//       hOniaRapP[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       countOniaCut++;

      if(fabs(ohMuL3Dz[iMu] - ohOniaPixelDz[iP]) > DzMuonPix) continue;
//       hOniaEta[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta());
//       hOniaRap[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity());
//       hOniaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Pt());
//       hOniaMass[histIndex][0][countOniaCut]->Fill(onia4Mom.M());
//       hOniaP[histIndex][0][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       hOniaEtaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
//       hOniaRapP[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));

      //store the surviving pixel-muon combinations:
      if(iNPix < 10){
        L3PixelCandIDForUps[iNPix] = iP;
        L3MuPixCandIDForUps[iNPix] = iMu;
        iNPix++;
      }
      //       printf("mu[%d]-pixel[%d] inv. mass %f\n",
      //           L3MuCandIDForOnia[iMu], iP, oniaMass);
    }
  }

//   hNCand[histIndex][0]->Fill(iNPix);

  if(iNPix!=0){
    for(int inP=0;inP<iNPix;inP++){

//       hPixCandEta[histIndex]->Fill(ohOniaPixelEta[L3PixelCandIDForUps[inP]]);
//       hPixCandPt[histIndex]->Fill(ohOniaPixelPt[L3PixelCandIDForUps[inP]]);
//       hPixCandP[histIndex]->Fill(ohOniaPixelPt[L3PixelCandIDForUps[inP]] * cosh(ohOniaPixelEta[L3PixelCandIDForUps[inP]]));

//       if(iNPix>=2){
//         for(int jnP=inP+1;jnP<iNPix;jnP++){
//            if(inP!=jnP){
//              double dEta = fabs(ohOniaPixelEta[L3PixelCandIDForUps[inP]]-ohOniaPixelEta[L3PixelCandIDForUps[jnP]]);
//              double dPhi = fabs(ohOniaPixelPhi[L3PixelCandIDForUps[inP]]-ohOniaPixelPhi[L3PixelCandIDForUps[jnP]]);
//              if(dPhi>TMath::Pi()) dPhi = 2.0*TMath::Pi()-dPhi;
// //              hPixCanddr[histIndex]->Fill(sqrt(pow(dEta,2)+pow(dPhi,2)));
//            }
//         }
//       }
    }
  }

  return iNPix;
}

int OHltTree::OpenHltMuTrackPassed_Ups(double ptTrack, double pTrack, double etaTrack, double DxyTrack, double DzTrack, int NHitsTrack, double normChi2Track, double *massMinTrack, double *massMaxTrack, double DzMuonTrack, bool checkChargeTrack, int histIndex)
{

  double pixMatchingDeltaR = 0.01;
  const double muMass = 0.105658367;
  TLorentzVector track4Mom, mu4Mom, onia4Mom;
  int iNTrack = 0;

  //0.) check how many pixel-muon combinations there are:
  int nComb = 0;
  for(int iMu = 0; iMu < 10; iMu++)
    if(L3MuPixCandIDForUps[iMu] > -1)
      nComb++;
//   Int_t countCut = 0, countOniaCut = 0;
  //1.) loop over the Tracker tracks
  for(int iT = 0; iT < NohOniaTrack; iT++){

    //select those that survive the kinematical and
    //topological selection cuts
//     countCut++;
//     hEta[histIndex][1][countCut]->Fill(ohOniaTrackEta[iT]);
//     hPt[histIndex][1][countCut]->Fill(ohOniaTrackPt[iT]);
//     hHits[histIndex][1][countCut]->Fill(ohOniaTrackHits[iT]);
//     hNormChi2[histIndex][1][countCut]->Fill(ohOniaTrackNormChi2[iT]);
//     hDxy[histIndex][1][countCut]->Fill(ohOniaTrackDr[iT]);
//     hDz[histIndex][1][countCut]->Fill(ohOniaTrackDz[iT]);

    if(fabs(ohOniaTrackEta[iT]) > etaTrack) continue; //eta cut
    if(ohOniaTrackPt[iT] < ptTrack) continue; //pT cut
    double momThisTrack = ohOniaTrackPt[iT] * cosh(ohOniaTrackEta[iT]);
    //     printf("track[%d] has eta %f, pT %f and mom %f\n",
    //         iT, ohOniaTrackEta[iT], ohOniaTrackPt[iT], momThisTrack);

//     hP[histIndex][1][countCut]->Fill(momThisTrack);
//     countCut++;

    if(momThisTrack < pTrack) continue; //momentum cut
    if(ohOniaTrackHits[iT] <  NHitsTrack) continue; //min. nb. of hits
    if(ohOniaTrackNormChi2[iT] > normChi2Track) continue; //chi2 cut
    if(fabs(ohOniaTrackDr[iT]) > DxyTrack) continue; //Dr cut
    if(fabs(ohOniaTrackDz[iT]) > DzTrack) continue;

//     hEta[histIndex][1][countCut]->Fill(ohOniaTrackEta[iT]);
//     hPt[histIndex][1][countCut]->Fill(ohOniaTrackPt[iT]);
//     hHits[histIndex][1][countCut]->Fill(ohOniaTrackHits[iT]);
//     hNormChi2[histIndex][1][countCut]->Fill(ohOniaTrackNormChi2[iT]);
//     hDxy[histIndex][1][countCut]->Fill(ohOniaTrackDr[iT]);
//     hDz[histIndex][1][countCut]->Fill(ohOniaTrackDz[iT]);
//     hP[histIndex][1][countCut]->Fill(momThisTrack);

    //     printf("track %d surviving kinematical pre-selection\n", iT);
    //2.) loop over the pixels candidates to see whether the track
    //under investigation has a match to the pixel track
    bool trackMatched = false;
    for(int iPix = 0; iPix < nComb; iPix++){

      if(trackMatched) break; //in case the track was already matched
      if(L3PixelCandIDForUps[iPix] < 0) continue; //in case the pixel has been matched to a previous track

      double deltaEta = ohOniaPixelEta[L3PixelCandIDForUps[iPix]] - ohOniaTrackEta[iT];
      double deltaPhi = ohOniaPixelPhi[L3PixelCandIDForUps[iPix]] - ohOniaTrackPhi[iT];
      if(deltaPhi>TMath::Pi()) deltaPhi = 2.0*TMath::Pi()-deltaPhi;
      double deltaR = sqrt(pow(deltaEta,2) + pow(deltaPhi,2));

      //       printf("delta R = %f\n", deltaR);
      if(deltaR > pixMatchingDeltaR) continue;
      //       printf("track[%d] and pixel[%d] are compatible (deltaR %f)\n", iT, L3PixelCandIDForUps[iPix], deltaR);

      trackMatched = true;
      L3PixelCandIDForUps[iPix] = -1; //deactivate this candidate to not match it to any further track

      track4Mom.SetPtEtaPhiM(ohOniaTrackPt[iT], ohOniaTrackEta[iT], ohOniaTrackPhi[iT], muMass);
      //check if the matched tracker track combined with the
      //muon gives again an opposite sign onia:
      mu4Mom.SetPtEtaPhiM(ohMuL3Pt[L3MuPixCandIDForUps[iPix]],
                          ohMuL3Eta[L3MuPixCandIDForUps[iPix]],
                          ohMuL3Phi[L3MuPixCandIDForUps[iPix]], muMass);
      onia4Mom = track4Mom + mu4Mom;

      double oniaMass = onia4Mom.M();
      //       printf("mu[%d]-track[%d] inv. mass %f\n",
      //           L3MuPixCandIDForUps[iPix], iT, oniaMass);

//       countOniaCut = 0;
//       if(oniaMass < 8.0) continue; //Only Ups
//       hOniaEta[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta());
//       hOniaRap[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity());
//       hOniaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Pt());
//       hOniaMass[histIndex][1][countOniaCut]->Fill(onia4Mom.M());
//       hOniaP[histIndex][1][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       hOniaEtaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
//       hOniaRapP[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       countOniaCut++;

      if(oniaMass < massMinTrack[0] || oniaMass > massMaxTrack[0]) continue; //mass cut

      //       printf("surviving: mu[%d]-track[%d] inv. mass %f\n",
      //           L3MuPixCandIDForUps[iPix], iT, oniaMass);

//       hOniaEta[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta());
//       hOniaRap[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity());
//       hOniaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Pt());
//       hOniaMass[histIndex][1][countOniaCut]->Fill(onia4Mom.M());
//       hOniaP[histIndex][1][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       hOniaEtaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
//       hOniaRapP[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       countOniaCut++;

      if(checkChargeTrack)
        if(ohMuL3Chg[L3MuPixCandIDForUps[iPix]] == ohOniaTrackChg[iT]) continue; //charge cut

//       hOniaEta[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta());
//       hOniaRap[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity());
//       hOniaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Pt());
//       hOniaMass[histIndex][1][countOniaCut]->Fill(onia4Mom.M());
//       hOniaP[histIndex][1][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       hOniaEtaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
//       hOniaRapP[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       countOniaCut++;

      if(fabs(ohMuL3Dz[L3MuPixCandIDForUps[iPix]] - ohOniaTrackDz[iT]) > DzMuonTrack) continue; //deltaZ cut

//       hOniaEta[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta());
//       hOniaRap[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity());
//       hOniaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Pt());
//       hOniaMass[histIndex][1][countOniaCut]->Fill(onia4Mom.M());
//       hOniaP[histIndex][1][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
//       hOniaEtaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
//       hOniaRapP[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));

      //store the surviving track-muon combinations:
      if(iNTrack < 10)
        iNTrack++;

      break; //don't check further pixels... go to next track
    }
  }

  //   if(iNTrack > 0)
  //     printf("found %d final candidates!!!\n", iNTrack);

//   hNCand[histIndex][1]->Fill(iNTrack);
  return iNTrack;
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

int OHltTree::OpenHlt1JetPassed(double pt, double etamax)
{
  int rc = 0;
  //ccla
  // Loop over all oh jets 
  for (int i=0;i<NrecoJetCal;i++) {
    if(recoJetCalPt[i]>pt && fabs(recoJetCalEta[i])<etamax ) {  // Jet pT cut
      rc++;
    }
  }

  return rc;
}

int OHltTree::OpenHlt1JetPassed(double pt, double etamax, double emfmin, double emfmax)
{
  int rc = 0;
  //ccla
  // Loop over all oh jets 
  for (int i=0;i<NrecoJetCal;i++) {
    if(recoJetCalPt[i]>pt && 
       fabs(recoJetCalEta[i])<etamax && 
       recoJetCalEMF[i] > emfmin &&
       recoJetCalEMF[i] < emfmax
       ) {  // Jet pT cut
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
  //std::cout << "FL: NrecoJetCal = " << NrecoJetCal << std::endl;
  for (int i=0;i<NrecoJetCal;i++) { 
    for (int j=0;j<NrecoJetCal && j!=i;j++) {      
      if((recoJetCalPt[i]+recoJetCalPt[j])/2.0 > pt) {  // Jet pT cut 
	//      if((recoJetCalE[i]/cosh(recoJetCalEta[i])+recoJetCalE[j]/cosh(recoJetCalEta[j]))/2.0 > pt) {
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
      //if((recoJetCorCalPt[i]+recoJetCorCalPt[j])/2.0 > pt) {  // Jet pT cut  
      if((recoJetCorCalE[i]/cosh(recoJetCorCalEta[i])+recoJetCorCalE[j]/cosh(recoJetCorCalEta[j]))/2.0 > pt) {
        rc++;  
      } 
    }  
  }   
  return rc;  
} 

int OHltTree::OpenHltQuadCorJetPassed(double pt)
{
  int njet = 0;
  int rc = 0;
  
  // Loop over all oh jets
  for (int i=0;i<NrecoJetCorCal;i++) {
      if(recoJetCorCalPt[i] > pt && recoJetCorCalEta[i] < 5.0) {  // Jet pT cut
	//std::cout << "FL: fires the jet pt cut" << std::endl;
	njet++;
      }
  }

  if(njet >= 4)
  {
    rc = 1;
  }

  return rc;
}

int OHltTree::OpenHltQuadJetPassed(double pt)
{
  int njet = 0;
  int rc = 0;
  
  // Loop over all oh jets
  for (int i=0;i<NrecoJetCal;i++) {
      if(recoJetCalPt[i] > pt && recoJetCalEta[i] < 5.0) {  // Jet pT cut
	njet++;
    }
  }

  if(njet >= 4)
    rc = 1;

  return rc;
}


int OHltTree::OpenHltFwdCorJetPassed(double esum)
{
  int rc = 0; 
  double gap = 0.; 

  // Loop over all oh jets, count the sum of energy deposited in HF 
  for (int i=0;i<NrecoJetCorCal;i++) {   
    if(((recoJetCorCalEta[i] > 3.0 && recoJetCorCalEta[i] < 5.0) || (recoJetCorCalEta[i] < -3.0 && recoJetCorCalEta[i] > -5.0))) { 
      gap+=recoJetCorCalE[i]; 
    }   
  }    

  // Backward FWD physics logic - we want to select the events *without* large jet energy in HF 
  if(gap < esum) 
    rc = 1; 
  else 
    rc = 0; 

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

int OHltTree::OpenHltHTJetNJPassed(double HTthreshold, double jetthreshold,
				   double etamax, int nj)
{
  int rc = 0,  njets=0;   
  double sumHT = 0.;   

  // Loop over all oh jets, sum up the energy  
  for (int i=0;i<NrecoJetCal;++i) {     
    if(recoJetCalPt[i] >= jetthreshold && fabs(recoJetCalEta[i])<etamax) { 
      //sumHT+=recoJetCalPt[i];
      njets++;
      sumHT+=(recoJetCalE[i]/cosh(recoJetCalEta[i]));
    }     
  }      
   
  if(sumHT >= HTthreshold && njets>=nj) rc = 1;

  return rc;    
}

int OHltTree::OpenHltMHT(double MHTthreshold, double jetthreshold)
{
  int rc = 0; 
  double mhtx=0., mhty=0.;
  for (int i=0;i<NrecoJetCorCal;++i)
  {     
    if(recoJetCorCalPt[i] >= jetthreshold)
    { 
      mhtx-=recoJetCorCalPt[i]*cos(recoJetCorCalPhi[i]);   
      mhty-=recoJetCorCalPt[i]*sin(recoJetCorCalPhi[i]);   
    }     
  }
  if (sqrt(mhtx*mhtx+mhty*mhty)>MHTthreshold) rc = 1;
  else rc = 0; 
  //std::cout << "sqrt(mhtx*mhtx+mhty*mhty) = " << sqrt(mhtx*mhtx+mhty*mhty) << std::endl;
  return rc;  
}
int OHltTree::OpenHltMHTU(double MHTthreshold, double jetthreshold)
{
  int rc = 0; 
  double mhtx=0., mhty=0.;
  for (int i=0;i<NrecoJetCal;++i)
  {     
    if(recoJetCalPt[i] >= jetthreshold)
    { 
      mhtx-=recoJetCalPt[i]*cos(recoJetCalPhi[i]);   
      mhty-=recoJetCalPt[i]*sin(recoJetCalPhi[i]);   
    }     
  }
  if (sqrt(mhtx*mhtx+mhty*mhty)>MHTthreshold) rc = 1;
  else rc = 0; 
  //std::cout << "sqrt(mhtx*mhtx+mhty*mhty) = " << sqrt(mhtx*mhtx+mhty*mhty) << std::endl;
  return rc;  
}

int OHltTree::OpenHltPT12U(double PT12threshold, double jetthreshold)
{
  int rc = 0; 
  int njets = 0;
  double pt12tx=0., pt12ty=0.;
  for (int i=0;i<NrecoJetCal;++i)
  {     
    if((recoJetCalPt[i] >= jetthreshold) && (fabs(recoJetCalEta[i]) <3))
    { 
      njets++;
      if (njets<3){
      pt12tx-=recoJetCalPt[i]*cos(recoJetCalPhi[i]);   
      pt12ty-=recoJetCalPt[i]*sin(recoJetCalPhi[i]);
      }   
    }     
  }
  if ((njets >= 2) && (sqrt(pt12tx*pt12tx+pt12ty*pt12ty)>PT12threshold)) rc = 1;
  else rc = 0; 
  //std::cout << "sqrt(mhtx*mhtx+mhty*mhty) = " << sqrt(mhtx*mhtx+mhty*mhty) << std::endl;
  return rc;  
}

int OHltTree::OpenHltSumHTPassed(double sumHTthreshold, double jetthreshold) 
{ 
  int rc = 0;   
  double sumHT = 0.;   


  // Loop over all oh jets, sum up the energy  
  for (int i=0;i<NrecoJetCal;++i) {     
    if(recoJetCalPt[i] >= jetthreshold) { 
      //sumHT+=recoJetCorCalPt[i];

      sumHT+=(recoJetCalE[i]/cosh(recoJetCalEta[i]));
    }     
  }      
   
  if(sumHT >= sumHTthreshold) rc = 1;

  return rc;    
} 

int OHltTree::OpenHltMeffU(double Meffthreshold, double jetthreshold)
{
  int rc = 0; 
  //MHT
  double mhtx=0., mhty=0.;
  for (int i=0;i<NrecoJetCal;++i)
  {     
    if(recoJetCalPt[i] >= jetthreshold)
    { 
      mhtx-=recoJetCalPt[i]*cos(recoJetCalPhi[i]);   
      mhty-=recoJetCalPt[i]*sin(recoJetCalPhi[i]);   
    }     
  }
  //HT
  double sumHT = 0.;   
  // Loop over all oh jets, sum up the energy  
  for (int i=0;i<NrecoJetCal;++i) {     
    if(recoJetCalPt[i] >= jetthreshold) { 
      //sumHT+=recoJetCorCalPt[i];

      sumHT+=(recoJetCalE[i]/cosh(recoJetCalEta[i]));
    }     
  }      
   
  if (sqrt(mhtx*mhtx+mhty*mhty)+sumHT>Meffthreshold) rc = 1;
  else rc = 0; 
  //std::cout << "sqrt(mhtx*mhtx+mhty*mhty) = " << sqrt(mhtx*mhtx+mhty*mhty) << std::endl;
  return rc;  
}

int OHltTree::OpenHltSumHTPassed(double sumHTthreshold, double jetthreshold, double etajetthreshold) 
{ 
  int rc = 0;   
  double sumHT = 0.;   


  // Loop over all oh jets, sum up the energy  
  for (int i=0;i<NrecoJetCal;++i) {     
    if(recoJetCalPt[i] >= jetthreshold && fabs(recoJetCalEta[i])< etajetthreshold) { 
      //sumHT+=recoJetCorCalPt[i];

      sumHT+=(recoJetCalE[i]/cosh(recoJetCalEta[i]));
    }     
  }      
   
  if(sumHT >= sumHTthreshold) rc = 1;

  return rc;    
} 

int OHltTree::OpenHltSumHTPassed(double sumHTthreshold, double jetthreshold, double etajetthreshold, int Njetthreshold) 
{ 
  int rc = 0;   
  double sumHT = 0.;   
  int Njet = 0.;   


  // Loop over all oh jets, sum up the energy  
  for (int i=0;i<NrecoJetCal;++i) {     
    if(recoJetCalPt[i] >= jetthreshold && fabs(recoJetCalEta[i])< etajetthreshold) { 
      //sumHT+=recoJetCorCalPt[i];
			Njet++;

      sumHT+=(recoJetCalE[i]/cosh(recoJetCalEta[i]));
    }     
  }      
   
  if(sumHT >= sumHTthreshold && Njet >= Njetthreshold) rc = 1;

  return rc;    
} 

int OHltTree::OpenHlt1PixelTrackPassed(float minpt, float minsep, float miniso)
{
  int rc = 0;

  // Loop over all oh pixel tracks, check threshold and separation
  for(int i = 0; i < NohPixelTracksL3; i++)
    {
      for(int i = 0; i < NohPixelTracksL3; i++)
	{
	  if(ohPixelTracksL3Pt[i] > minpt)
	    {
	      float closestdr = 999.;

	      // Calculate separation from other tracks above threshold
	      for(int j = 0; j < NohPixelTracksL3 && j != i; j++)
		{
		  if(ohPixelTracksL3Pt[j] > minpt)
		    {
		      float dphi = ohPixelTracksL3Phi[i]-ohPixelTracksL3Phi[j];
		      float deta = ohPixelTracksL3Eta[i]-ohPixelTracksL3Eta[j];
		      float dr = sqrt((deta*deta) + (dphi*dphi));
		      if(dr < closestdr)
			closestdr = dr;
		    }
		}
	      if(closestdr > minsep)
		{
		  // Calculate isolation from *all* other tracks without threshold.
		  if(miniso > 0)
		    {
		      int tracksincone = 0;
		      for(int k = 0; k < NohPixelTracksL3 && k != i; k++)
			{
			  float dphi = ohPixelTracksL3Phi[i]-ohPixelTracksL3Phi[k];
			  float deta = ohPixelTracksL3Eta[i]-ohPixelTracksL3Eta[k];
			  float dr = sqrt((deta*deta) + (dphi*dphi));
			  if(dr < miniso)
			    tracksincone++;
			}
		      if(tracksincone == 0)
			rc++;
		    }
		  else
		    rc++;
		}
	    }
	}
    }

  return rc;
}

int OHltTree::OpenHlt1L3MuonPassed(double pt, double eta) 
{ 
  //for BTagMu trigger 
 
  int rcL3 = 0; 
  // Loop over all oh L3 muons and apply cuts 
  for (int i=0;i<NohMuL3;i++) {   
    if(ohMuL3Pt[i] > pt && fabs(ohMuL3Eta[i]) < eta ) { // L3 pT and eta cut  
      rcL3++; 
    } 
  } 
   
  return rcL3; 
 
} 

vector<int>  OHltTree::VectorOpenHlt1PhotonPassed(float Et, int L1iso, float Tiso, float Eiso, float HisoBR, float HisoEC, float HoverE, float R9, float ClusShapEB, float ClusShapEC) 
{ 
  vector<int> rc; 
  // Loop over all oh photons 
  for (int i=0;i<NohPhot;i++) { 
    if ( ohPhotEt[i] > Et) {  
      if( TMath::Abs(ohPhotEta[i]) < 2.65 ) {  
        if ( ohPhotL1iso[i] >= L1iso ) {  
          if( ohPhotTiso[i] < Tiso + 0.001*ohPhotEt[i] ) { 
            if( ohPhotEiso[i] < Eiso  + 0.006*ohPhotEt[i]) {  
              if( (TMath::Abs(ohPhotEta[i]) < 1.479 && ohPhotHiso[i] < HisoBR + 0.0025*ohPhotEt[i] && ohEleClusShap[i] < ClusShapEB && ohPhotR9[i] < R9)  || 
                  (1.479 < TMath::Abs(ohPhotEta[i]) && TMath::Abs(ohPhotEta[i]) < 2.65 && ohPhotHiso[i] < HisoEC + 0.0025*ohPhotEt[i] && ohEleClusShap[i] < ClusShapEC)) {  
		float EcalEnergy = ohPhotEt[i]/(sin (2*atan(exp(0-ohPhotEta[i])))); 
		if( ohPhotHforHoverE[i]/EcalEnergy < HoverE) 
		  if( ohPhotL1Dupl[i] == false) // remove double-counted L1 SCs   
		    rc.push_back(i); 
              } 
            } 
          } 
        } 
      } 
    } 
  } 
 
  return rc; 
} 

int OHltTree::OpenL1SetSingleJetBit(const float& thresh){

  // count number of L1 central, forward and tau jets above threshold

  int rc=0;

  bool CenJet=false,ForJet=false,TauJet=false;

  //size_t size = sizeof(L1CenJetEt)/sizeof(*L1CenJetEt);
  const size_t size = 4;
  // cout << thresh << "\t" << size << endl;

  int ncenjet=0;
  for (unsigned int i=0;i<size; ++i){
    if (L1CenJetEt[i] >= thresh) ++ncenjet;
  }
  CenJet=ncenjet>=1;


  int nforjet=0;
  for (unsigned int i=0;i<size; ++i){
    if (L1ForJetEt[i] >= thresh) ++nforjet;
  }
  ForJet=nforjet>=1;

  int ntaujet=0;
  for (unsigned int i=0;i<size; ++i){
    if (L1TauEt[i] >= thresh) ++ntaujet;
  }
  TauJet=ntaujet>=1;

  bool L1SingleJet=(CenJet || ForJet || TauJet );

  if (L1SingleJet) rc=1;

  return ( rc );
 
}
