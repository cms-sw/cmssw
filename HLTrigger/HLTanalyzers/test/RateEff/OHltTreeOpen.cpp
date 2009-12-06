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


  //////////////////////////////////////////////////////////////////
  // Check OpenHLT trigger

  /* Single Jet */
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1Jet6") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }   
    }    
  }    
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1Jet15") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }   
    }    
  }    
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet15U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1JetPassed(15.)>=1) {
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }   
      }    
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet30U") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1JetPassed(30.)>=1) {
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }   
      }    
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet50U") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1JetPassed(50.)>=1) {
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }   
      }    
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet30") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1CorJetPassed(30)>=1) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }   
      }    
    }    
  }    
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet50") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1CorJetPassed(50)>=1) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }    
      }     
    }     
  }     
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet80") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1CorJetPassed(80)>=1) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }     
      }      
    }      
  }      
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet110") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1CorJetPassed(110)>=1) {      
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }     
      }      
    }      
  }      
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet140") == 0) {       
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if(OpenHlt1CorJetPassed(140)>=1) {       
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }       
    }       
  }       
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Jet180") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1CorJetPassed(180)>=1) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  } 
 
  /* DiJetAve */
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DiJetAve15") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHltDiJetAvePassed(15.)>=1) {   
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }  
      }   
    }   
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DiJetAve30") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHltDiJetAvePassed(30.)>=1) {   
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }  
      }   
    }   
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DiJetAve50") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if(OpenHltDiJetAvePassed(50.)>=1) {    
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }   
      }    
    }    
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DiJetAve70") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if(OpenHltDiJetAvePassed(70.)>=1) {    
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }   
      }    
    }    
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DiJetAve130") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if(OpenHltDiJetAvePassed(130.)>=1) {    
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }   
      }    
    }    
  } 

  /* Forward & MultiJet */
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_FwdJet20U") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHltFwdJetPassed(20.)>=1) {      
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }     
      }      
    }      
  }      
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_QuadJet15U") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHltQuadJetPassed(15.)>=1) {    
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }   
      }    
    }    
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_FwdJet40") == 0) {       
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHltFwdCorJetPassed(40.)>=1) {       
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }       
    }       
  }       
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_QuadJet30") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHltQuadCorJetPassed(30.)>=1) {     
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }    
      }     
    }     
  } 

  /* MET */
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1MET20") == 0) {       
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
    }     
  }       
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET25") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(recoMetCal > 25.) {  
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }  
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET35") == 0) {         
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(recoMetCal > 35.) {         
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }        
      }         
    }         
  }         
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET45") == 0) {          
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if(recoMetCal > 45.) {          
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }         
      }          
    }          
  }          
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET50") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(recoMetCal > 50.) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET60") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if(recoMetCal > 60.) {  
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }  
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET65") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(recoMetCal > 65.) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MET100") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(recoMetCal > 100.) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  } 

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT300_MHT100") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      //if(recoHTCal > 100. && (OpenHltSumHTPassed(300., 30.) == 1)) {
      //std:: cout << "OpenHltMHT(100., 30.) = " << OpenHltMHT(100., 30.) << std::endl;
      //std:: cout << "OpenHltSumHTPassed(300., 30.) = " << OpenHltSumHTPassed(300., 30.) << std::endl;
      if(OpenHltMHT(100., 30.)==1 && (OpenHltSumHTPassed(300., 30.) == 1)) {
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }   
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT200_JT20") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {   
      if(OpenHltSumHTPassed(200., 20.) == 1) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }     
      }   
    }   
  }   
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT200") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {    
      if(OpenHltSumHTPassed(200., 30.) == 1) {  
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }    
    }    
  }    
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT200_JT40") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {    
      if(OpenHltSumHTPassed(200., 40.) == 1) {  
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }    
    }    
  }    
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT250_JT20") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {     
      if(OpenHltSumHTPassed(250., 20.) == 1) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }    
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT250") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {     
      if(OpenHltSumHTPassed(250., 30.) == 1) {  
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }     
      }   
    }   
  }   
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_HT250_JT40") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {     
      if(OpenHltSumHTPassed(250., 40.) == 1) {  
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }     
      }   
    }   
  }   

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_SumET120") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(recoMetCalSum > 120.) {
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }       
      }       
    }       
  }  

  /* Muons */
  else if (menu->GetTriggerName(it).CompareTo("OpenAlCa_RPCMuonNormalisation") == 0) {          
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      int rc = 0;
      for(int i=0;i<NL1OpenMu;i++) { 
	if(L1OpenMuEta[i] > -1.6 && L1OpenMuEta[i] < 1.6) 
	  rc++;
      }
      if(rc > 0)
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }         
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
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }        
    }         
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu") == 0) {        
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }       
    }        
  }        
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu20") == 0) {         
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }        
    }         
  }    
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu25") == 0) {            
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      int rc = 0;   
      for(int i=0;i<NL1OpenMu;i++) {   
        if(L1OpenMuPt[i] > 25.0)  
          rc++;   
      }   
      if(rc > 0)   
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }           
    }            
  }           
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu30") == 0) {           
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      int rc = 0;  
      for(int i=0;i<NL1OpenMu;i++) {  
        if(L1OpenMuPt[i] > 30.0) 
          rc++;  
      }  
      if(rc > 0)  
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }          
    }           
  }          
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu20HQ") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {   
      int rc = 0;   
      for(int i=0;i<NL1Mu;i++) {   
        if(L1MuPt[i] > 20.0)  
          if(L1MuQal[i] == 7) 
            rc++;   
      }   
      if(rc > 0)   
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }           
    }            
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu9") == 0) {          
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      int rc = 0;
      for(int i = 0; i < NohMuL2; i++) {
	if(ohMuL2Pt[i] > 9.) {
	  rc++;
	}
      }
      if(rc>0) {
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }         
      }          
    }          
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu11") == 0) {          
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      int rc = 0;
      for(int i = 0; i < NohMuL2; i++) {
	if(ohMuL2Pt[i] > 11.) {
	  rc++;
	}
      }
      if(rc>0) {
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }         
      }          
    }          
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu3") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1MuonPassed(3.,3.,3.,2.,0)>=1) {  
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }  
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1) {   
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }     
      }   
    }   
  }   
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu7") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1MuonPassed(5.,5.,7.,2.,0)>=1) {  
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }  
      }  
    }  
  }  
      
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu9") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1MuonPassed(7.,7.,9.,2.,0)>=1) {  
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }  
      }  
    }  
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu11") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1MuonPassed(7.,9.,11.,2.,0)>=1) {   
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }   
      }   
    }   
  }   
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Mu15") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1MuonPassed(10.,12.,15.,2.,0)>=1) {    
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }    
      }    
    }    
  }    
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleMu3") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt2MuonPassed(3.,3.,3.,2.,0)>=2) {   
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }   
      }   
    }   
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1DoubleMuOpen") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(1) {    // Pass through
	  if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }  
      }    
    } 
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleMu0") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
	if(OpenHlt2MuonPassed(0.,0.,0.,2.,0)>=2) {     
	  if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }   
	}
    }     
  }     
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoMu3") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1MuonPassed(3.,3.,3.,2.,1)>=1) {   
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }   
      }   
    }   
  }     
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoMu9") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1MuonPassed(7.,7.,9.,2.,1)>=1) {   
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }    
      }   
    }   
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu14") == 0) {         
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }     
    }            
  }         
  
  /* Electrons */
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1SingleEG5") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(true) { // passthrough      
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }      
    }      
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1SingleEG8") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(true) { // passthrough       
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }       
      }       
    }       
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1DoubleEG5") == 0) {       
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(true) { // passthrough        
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }        
      }        
    }        
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele10_LW_EleId_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1EleIdLWElectronPassed(10.,0,9999.,9999.)>=1) {
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }
      }
    }
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele10_LW_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1LWElectronPassed(10.,0,9999.,9999.)>=1) {
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }
      }
    }
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele15_LW_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1LWElectronPassed(15.,0,9999.,9999.)>=1) {
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele15_SW_EleId_L1R") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {   
      if(OpenHlt1ElectronEleIDPassed(15.,0,9999.,9999.)>=1) {   
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }        
      }        
    } 
  } 
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele20_LW_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1LWElectronPassed(20.,0,9999.,9999.)>=1) {
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }
      }
    }
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele25_LW_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1LWElectronPassed(25.,0,9999.,9999.)>=1) {
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }
      }
    }
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleEle5_SW_Jpsi_L1R") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) { 
      if(OpenHlt2ElectronMassWinPassed(5.0, 0, 9.0, 2.0, 4.5)>=1) {
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }
      }
    }
  }  
    else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleEle5_SW_Upsilon_L1R") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) { 
      if(OpenHlt2ElectronMassWinPassed(5.0, 0, 9.0, 8.0, 11.0)>=1) {
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }
      }
    }
  }  

  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleEle5_SW_L1R") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1ElectronPassed(5.,0,9999.,9999.)>=2) {       
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }      
    }      
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele15_SC10_LW_L1R") == 0) {         
    float Et = 15.; 
    int L1iso = 0;  
    float Tiso = 9999.;  
    float Hiso = 9999.; 
    int rc = 0; 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      for (int i=0;i<NohEleLW;i++) {  
        if ( ohEleEtLW[i] > Et) {  
	  if( TMath::Abs(ohEleEtaLW[i]) < 2.65) {
	    if ( ohEleHisoLW[i] < Hiso || ohEleHisoLW[i]/ohEleEtLW[i] < 0.05) { 
	      if (ohEleNewSCLW[i]==1) { 
		if (ohElePixelSeedsLW[i]>0) { 
		  if ( (ohEleTisoLW[i] < Tiso && ohEleTisoLW[i] != -999.) || (Tiso == 9999.) ){ 
		    if ( ohEleL1isoLW[i] >= L1iso ) {  // L1iso is 0 or 1  
		      if( ohEleLWL1Dupl[i] == false) { // remove double-counted L1 SCs
			for(int j=0;j<NohEleLW && j != i;j++) { 
			  if(ohEleEtLW[j] > 10.) { 
			    if(TMath::Abs(ohEleEtaLW[j]) < 2.65) {
			      if(ohEleEt[i] != ohEleEtLW[j])
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
    
    if(rc >= 1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }         
    }
  }  
  
  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleEle10_LW_L1R") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1LWElectronPassed(10.,0,9999.,9999.)>=2) {       
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }       
      }       
    }       
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele10_SW_L1R") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1ElectronPassed(10.,0,9999.,9999.)>=1) { 
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }      
    }      
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele15_SW_L1R") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1ElectronPassed(15.,0,9999.,9999.)>=1) { 
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }      
    }      
  }        
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele20_SW_L1R") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1ElectronPassed(20.,0,9999.,9999.)>=1) { 
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }      
    }      
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele25_SW_L1R") == 0) {       
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1ElectronPassed(25.,0,9999.,9999.)>=1) {  
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }       
      }       
    }       
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele200_LW_L1R") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1LWElectronPassed(20.,0,9999.,9999.)>=1) { 
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }      
    }      
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_LooseIsoEle15_LW_L1R") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1LWElectronPassed(15.,0,0.12,6.)>=1) {      
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
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
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }        
    }        
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoEle18_L1R") == 0) {       
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1ElectronPassed(18.,1,0.06,3.)>=1) {       
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }       
      }       
    }       
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoEle20_LW_L1R") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1LWElectronPassed(20.,0,0.06,3.)>=1) {      
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }      
    }      
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoEle15_LW_L1I") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1LWElectronPassed(15.,1,0.06,3.)>=1) {      
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }      
    }      
  }        
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoEle20_LW_L1I") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1LWElectronPassed(20.,1,0.06,3.)>=1) {      
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }      
    }      
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleEle10_LW_OnlyPixelM_L1R") == 0) {   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1LWElectronPassed(10.,1,9999.,9999.)>=2) {       
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }       
      }       
    }       
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleEle5_LW_L1R") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1LWElectronPassed(5.,0,9999.,9999.)>=2) {       
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }      
    }      
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleEle10_SW_L1R") == 0) {       
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1ElectronPassed(10.,0,9999.,9999.)>=2) {        
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }       
      }       
    }       
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele15_SC15_SW_LooseTrackIso_L1R") == 0) {        
    float Et = 15.; 
    int L1iso = 0;  
    float Tiso = 8.0;  
    float Hiso = 9999.; 
    float Tisoratio = 0.5;  
    int rc = 0; 

    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      for (int i=0;i<NohEle;i++) {  
        if ( ohEleEt[i] > Et) { 
	  if( TMath::Abs(ohEleEta[i]) < 2.65 ) {
	    if ( ohEleHiso[i] < Hiso || ohEleHiso[i]/ohEleEt[i] < 0.05) { 
	      if (ohEleNewSC[i]==1) { 
		if (ohElePixelSeeds[i]>0) { 
		  if ( (ohEleTiso[i] < Tisoratio || (ohEleTiso[i]*ohEleEt[i]) < Tiso) && ohEleTiso[i] != -999.) { 
		    if ( ohEleL1iso[i] >= L1iso ) {  // L1iso is 0 or 1  
		      for(int j=0;j<NohEle && j != i;j++) { 
			if(ohEleEt[j] > 15.) { 
			  if( TMath::Abs(ohEleEta[j]) < 2.65 ) {
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
    } 
    if(rc > 0) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }         
    }         
  }        
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele15_SC15_SW_EleId_L1R") == 0) {   
    float Et = 15.;   
    int rc = 0;   
 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      for (int i=0;i<NohEle;i++) {   
	if ( ohEleEt[i] > Et) {   
	  if( TMath::Abs(ohEleEta[i]) < 2.65 ) {
	    if (ohEleNewSC[i]==1) {  
	      if (ohElePixelSeeds[i]>0) {  
		if ( (TMath::Abs(ohEleEta[i]) < 1.475 && ohEleClusShap[i] < 0.015) ||   
		     (1.475 < TMath::Abs(ohEleEta[i]) && TMath::Abs(ohEleEta[i]) < 2.65 && ohEleClusShap[i] < 0.04) ) { 
		  if ( (ohEleDeta[i] < 0.008) && (ohEleDphi[i] < 0.1) ) { 
		    for(int j=0;j<NohEle && j != i;j++) {  
		      if(ohEleEt[j] > 15.) {  
			if(TMath::Abs(ohEleEta[j]) < 2.65) {
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
    if(rc > 0) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }          
    }    
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele20_SC15_SW_L1R") == 0) {         
    float Et = 20.;  
    int L1iso = 0;   
    float Hiso = 9999.;  
    int rc = 0;  
 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
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
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }          
    } 
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele25_SW_EleId_LooseTrackIso_L1R") == 0) {    
    float Et = 25.;    
    float Tiso = 8.0;    
    float Hiso = 9999.;  
    float Tisoratio = 0.5;  
    int L1iso = 0; 
    int rc = 0;    
  
    if ( map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) { 
      for (int i=0;i<NohEle;i++) {   
        if ( ohEleEt[i] > Et) {   
          if( TMath::Abs(ohEleEta[i]) < 2.65 ) {
          if ( ohEleHiso[i] < Hiso || ohEleHiso[i]/ohEleEt[i] < 0.05)   
            if (ohEleNewSC[i]==1)   
              if (ohElePixelSeeds[i]>0)   
                if ( (ohEleTiso[i] < Tisoratio || (ohEleTiso[i]*ohEleEt[i]) < Tiso) && ohEleTiso[i] != -999.)   
                  if ( ohEleL1iso[i] >= L1iso )   // L1iso is 0 or 1   
                    if ( (TMath::Abs(ohEleEta[i]) < 1.475 && ohEleClusShap[i] < 0.015) ||   
                         (1.475 < TMath::Abs(ohEleEta[i]) && TMath::Abs(ohEleEta[i]) < 2.65 && ohEleClusShap[i] < 0.04) )    
                      if ( (ohEleDeta[i] < 0.008) && (ohEleDphi[i] < 0.1) )   
                        if( ohEleL1Dupl[i] == false) // remove double-counted L1 SCs     
                          rc++;         
	  }
        }   
      }   
      if(rc > 0) {   
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }           
      }     
    }  
  } 
  
 else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele15_SiStrip_L1R") == 0) {   
   if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {   
      // JH: This is just a placeholder for now until strip-seeded electrons are available  
      if(1) {    
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }        
      }        
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_JPsiUpsilonEE") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      TLorentzVector e1;  
      TLorentzVector e2;  
      TLorentzVector meson; 
      int rc = 0; 
      for (int i=0;i<NohEle;i++) { 
	for (int j=0;j<NohEle && j != i;j++) {  
	  
	  if (ohElePixelSeeds[i]>0 && ohElePixelSeeds[j]>0) {
	    
	    e1.SetPtEtaPhiM(ohEleEt[i],ohEleEta[i],ohElePhi[i],0.0); 
	    e2.SetPtEtaPhiM(ohEleEt[j],ohEleEta[j],ohElePhi[j],0.0); 
	    meson = e1 + e2; 
	    
	    float mesonmass = meson.M();  
	    if(mesonmass > 2.0)
	      rc++;
	  }
	}
      }
      if(rc >= 1) {  
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }  
      }
    }
  }

   
    /* Photons */
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L1Photon5") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(true) { // passthrough     
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }     
      }     
    }     
  }      
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon15_L1R") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1PhotonPassed(15.,0,999.,999.,999.,999.)>=1) { // added track iso!
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }     
      }     
    }     
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon10_LooseEcalIso_L1R") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1PhotonLooseEcalIsoPassed(10.,0,999.,3.,999.,999.)>=1) {  
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }      
    }      
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon15_LooseEcalIso_L1R") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1PhotonLooseEcalIsoPassed(15.,0,999.,3.,999.,999.)>=1) { 
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }     
      }     
    }     
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon10_L1R") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1PhotonPassed(10.,0,999.,999.,999.,999.)>=1) {     
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }     
      }     
    }     
  }      
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon20_L1R") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1PhotonPassed(20.,0,999.,999.,999.,999.)>=1) {     
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }     
      }     
    }     
  }      
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon25_L1R") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1PhotonPassed(25.,0,999.,999.,999.,999.)>=1) {       
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }       
      }       
    }       
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon30_L1R") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1PhotonPassed(30.,0,999.,999.,999.,999.)>=1) {      
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }      
    }      
  }       
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Photon70_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if(OpenHlt1PhotonPassed(70.,0,999.,999.,999.,999.)>=1) {       
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }       
      }       
    }       
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoPhoton10_L1R") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1PhotonPassed(10.,0,1.,1.5,6.,4.)>=1) {      
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }      
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoPhoton15_L1R") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1PhotonPassed(15.,0,1.,1.5,6.,4.)>=1) {      
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }      
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_IsoPhoton20_L1R") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1PhotonPassed(20.,0,1.,1.5,6.,4.)>=1) {      
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }      
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoublePhoton5_Jpsi_L1R") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt2PhotonMassWinPassed(5.,0,999.,999.,999.,999.,2.,4.5)==1) {     
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }     
      }     
    }     
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoublePhoton5_Upsilon_L1R") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt2PhotonMassWinPassed(5.,0,999.,999.,999.,999.,8.,11.)==1) {     
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }     
      }     
    }     
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoublePhoton5_eeRes_L1R") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt2PhotonMassWinPassed(5.,0,999.,999.,999.,999.,2.,9999999.)==1) {     
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }     
      }     
    }     
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoublePhoton5_L1R") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1PhotonPassed(5.,0,999.,999.,999.,999.)>=2) {     
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }     
      }     
    }     
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoublePhoton10_L1R") == 0) {    
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1PhotonPassed(10.,0,999.,999.,999.,999.)>=2) {     
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }     
      }     
    }     
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoublePhoton15_L1R") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1PhotonPassed(15.,0,999.,999.,999.,999.)>=2) {      
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }      
    }      
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleIsoPhoton10_L1R") == 0) {      
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1PhotonPassed(10.,0,1.,1.5,6.,4.)>=2) {       
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }       
      }       
    }       
  }  
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleIsoPhoton20_L1R") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1PhotonPassed(20.,0,1.,1.5,6.,4.)>=2) {      
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }      
    }      
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon15_TrackIso_L1R") == 0) {     
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1PhotonPassed(15.,0,0.05,999.,999.,999.)>=1) { // added track iso! 
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }      
    }      
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon10_LooseEcalIso_TrackIso_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1PhotonLooseEcalIsoPassed(10.,0,0.05,3.,999.,999.)>=1) {   
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }       
      }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon20_LooseEcalIso_TrackIso_L1R") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1PhotonLooseEcalIsoPassed(15.,0,0.05,3.,999.,999.)>=1) {    
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }        
      } 
    }
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Photon25_LooseEcalIso_TrackIso_L1R") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1PhotonLooseEcalIsoPassed(25.,0,0.05,3.,999.,999.)>=1) {    
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }        
      } 
    }
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoublePhoton15_VeryLooseEcalIso_L1R") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1PhotonVeryLooseEcalIsoPassed(15.,0,999.,5.,999.,999.)>=2) {     
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }         
      }  
    }
  }

  /* Taus */
  /* Clean up tau code: remove study triggers, _L2R triggers become standard and are removed from names */
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_SingleLooseIsoTau20") == 0) {        
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHltTauL2SCPassed(20.,0.,0,0.,0,30.,70.)>=1) { 
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }        
    }        
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleLooseIsoTau15") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHltTauL2SCPassed(15.,0.,0,0.,0,40.,100.)>=2) { 
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      } 
    } 
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_SingleIsoTau20_Trk5") == 0) {        
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHltTauL2SCPassed(30.,5.,0,0.,1,30.,70.)>=1) { 
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
      }        
    }        
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_SingleIsoTau30_Trk5") == 0) {         
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {  
      if(OpenHltTauL2SCPassed(30.,5.,0,0.,1,30.,70.)>=1) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }       
      }         
    }         
  } 

  // 1 Leg L3 isolation - prototype version from Chi-Nhan
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleLooseIsoTau15_Trk5") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt2Tau1LegL3IsoPassed(15.,5.,0,0.,40.,100.)==1) { 
  	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    } 
  }
  
  // No isolation version
  // else if (menu->GetTriggerName(it).CompareTo("OpenHLT_DoubleLooseIsoTau15_Trk5") == 0) { 
  //  if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
  //    if(OpenHltTauL2SCPassed(15.,5.,0,0.,0)>=2) { 
  //	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
  //    } 
  //  } 
  //}
  /* End: Taus */

  /* BTag */
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_Jet10") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      int rc = 0; 
      int max =  (NohBJetL2Corrected > 2) ? 2 : NohBJetL2Corrected;
      for(int i = 0; i < max; i++) { 
	if(ohBJetL2CorrectedEt[i] > 10.) { // ET cut
	  if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	    if(ohBJetMuL3Tag[i] > 0.5) { // Level 3 b tag
	      rc++; 
	    } 
	  }
	}
      }
      if(rc >= 1) { 
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagMu_Jet20") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      int rc = 0; 
      int max =  (NohBJetL2Corrected > 2) ? 2 : NohBJetL2Corrected;
      for(int i = 0; i < max; i++) { 
	if(ohBJetL2CorrectedEt[i] > 20.) { // ET cut
	  if(ohBJetMuL25Tag[i] > 0.5) { // Level 2.5 b tag
	    if(ohBJetMuL3Tag[i] > 0.5) { // Level 3 b tag
	      rc++; 
	    } 
	  }
	}
      }
      if(rc >= 1) { 
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
      } 
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagIP_Jet50") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
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
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }          
      }
    }  
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagIP_Jet80") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
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
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }          
      }
    }  
  } 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_BTagIP_Jet120") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
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
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }           
      } 
    }   
  }

  /* Minbias */
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MinBiasHcal") == 0) {         
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(true) { // passthrough       
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }       
      }       
    }         
  }         
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_MinBiasEcal") == 0) {          
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(true) { // passthrough        
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }        
      }        
    }          
  }          
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_MinBiasBSC") == 0) {
    bool techTriggerBSC1 = (bool) L1Tech_BSC_minBias_threshold1_v0;
    bool techTriggerBSC2 = (bool) L1Tech_BSC_minBias_threshold2_v0;
    bool techTriggerBS3 = (bool) L1Tech_BSC_minBias_inner_threshold1_v0;
    bool techTriggerBS4 = (bool) L1Tech_BSC_minBias_inner_threshold2_v0;

    if(techTriggerBSC1 || techTriggerBSC2 || techTriggerBS3 || techTriggerBS4)
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_MinBiasPixel_SingleTrack") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if(OpenHlt1PixelTrackPassed(0.0, 1.0, 0.0)>=1){
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }
      }
    }
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_MinBiasPixel_DoubleTrack") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if(OpenHlt1PixelTrackPassed(0.0, 1.0, 0.0)>=2){
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }
      }
    }
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_MinBiasPixel_DoubleIsoTrack5") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {
      if(OpenHlt1PixelTrackPassed(5.0, 1.0, 1.0)>=0){
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }
      }
    }
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_ZeroBias") == 0) { 
    if(map_BitOfStandardHLTPath.find("OpenL1_ZeroBias")->second == 1)
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }         
  }
  
  /* AlCa */
  else if (menu->GetTriggerName(it).CompareTo("OpenAlCa_HcalPhiSym") == 0) {
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(ohHighestEnergyHFRecHit > 0 || ohHighestEnergyHBHERecHit > 0) {
	// Require one RecHit with E > 0 MeV in any HCAL subdetector
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }         
      }
    }
  }

  else if (menu->GetTriggerName(it).CompareTo("OpenAlCa_EcalPi0") == 0) { 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 

      TLorentzVector gamma1; 
      TLorentzVector gamma2; 
      TLorentzVector meson;
      TLorentzVector gammaiso;

      int rc = 0;

      for(int i = 0; i < Nalcapi0clusters; i++) {
	for(int j = i+1;j < Nalcapi0clusters && j != i; j++) {
	  gamma1.SetPtEtaPhiM(ohAlcapi0ptClusAll[i],ohAlcapi0etaClusAll[i],ohAlcapi0phiClusAll[i],0.0);
	  gamma2.SetPtEtaPhiM(ohAlcapi0ptClusAll[j],ohAlcapi0etaClusAll[j],ohAlcapi0phiClusAll[j],0.0);
	  meson = gamma1 + gamma2;

          float mesonpt = meson.Pt(); 
          float mesoneta = meson.Eta(); 
          float mesonmass = meson.M(); 

	  float iso = 0.0;
	  float dr = 0.0;
	  float deta = 0.0;
	  for(int k = 0;k < Nalcapi0clusters && k != i && k != j; k++) { 
	    gammaiso.SetPtEtaPhiM(ohAlcapi0ptClusAll[k],ohAlcapi0etaClusAll[k],ohAlcapi0phiClusAll[k],0.0);
	    dr = gammaiso.DeltaR(meson);
	    deta = TMath::Abs(ohAlcapi0etaClusAll[k] - mesoneta);

	    if((dr < 0.2) && (deta < 0.05)) {
	      iso = iso + ohAlcapi0ptClusAll[k];
	    }
	  }
	  
	  if(TMath::Abs(ohAlcapi0etaClusAll[i]) < 1.479 && TMath::Abs(ohAlcapi0etaClusAll[j]) < 1.479) {  
	    //pi0 barrel
	    if(ohAlcapi0ptClusAll[i] > 1.0 && ohAlcapi0ptClusAll[j] > 1.0 && 
	       ohAlcapi0s4s9ClusAll[i] > 0.83 && ohAlcapi0s4s9ClusAll[j] > 0.83 &&  
	       iso < 0.5 &&  
	       mesonpt > 2.0 && 
	       mesonmass > 0.06 && mesonmass < 0.22) 
	      rc++; 
	  }
	  if(TMath::Abs(ohAlcapi0etaClusAll[i]) > 1.479 && TMath::Abs(ohAlcapi0etaClusAll[j]) > 1.479)  {   
            //pi0 endcap  
            if(ohAlcapi0ptClusAll[i] > 0.8 && ohAlcapi0ptClusAll[j] > 0.8 && 
               ohAlcapi0s4s9ClusAll[i] > 0.9 && ohAlcapi0s4s9ClusAll[j] > 0.9 &&  
	       iso < 0.5 && 
               mesonpt > 3.0 && 
               mesonmass > 0.05 && mesonmass < 0.3) 
              rc++; 
	  }
	}
      }
      if(rc > 0) {
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }          
      }
    } 
  } 

  else if (menu->GetTriggerName(it).CompareTo("OpenAlCa_EcalEta") == 0) {  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
 
      TLorentzVector gamma1;  
      TLorentzVector gamma2;  
      TLorentzVector meson; 
      TLorentzVector gammaiso;

      int rc = 0; 
 
      for(int i = 0; i < Nalcapi0clusters; i++) { 
        for(int j = i+1;j < Nalcapi0clusters && j != i; j++) { 
          gamma1.SetPtEtaPhiM(ohAlcapi0ptClusAll[i],ohAlcapi0etaClusAll[i],ohAlcapi0phiClusAll[i],0.0); 
          gamma2.SetPtEtaPhiM(ohAlcapi0ptClusAll[j],ohAlcapi0etaClusAll[j],ohAlcapi0phiClusAll[j],0.0); 
          meson = gamma1 + gamma2; 
          float mesonpt = meson.Pt(); 
          float mesoneta = meson.Eta(); 
          float mesonmass = meson.M(); 

          float iso = 0.0; 
          float dr = 0.0; 
          float deta = 0.0; 
          for(int k = 0;k < Nalcapi0clusters && k != i && k != j; k++) {  
            gammaiso.SetPtEtaPhiM(ohAlcapi0ptClusAll[k],ohAlcapi0etaClusAll[k],ohAlcapi0phiClusAll[k],0.0); 
            dr = gammaiso.DeltaR(meson); 
            deta = TMath::Abs(ohAlcapi0etaClusAll[k] - mesoneta); 
                   
            if((dr < 0.3) && (deta < 0.1)) { 
              iso = iso + ohAlcapi0ptClusAll[k]; 
            } 
          } 

	  if(TMath::Abs(ohAlcapi0etaClusAll[i]) < 1.479 && TMath::Abs(ohAlcapi0etaClusAll[j]) < 1.479) { 
	    //eta barrel  
	    if(ohAlcapi0ptClusAll[i] > 1.2 && ohAlcapi0ptClusAll[j] > 1.2 &&    
               ohAlcapi0s4s9ClusAll[i] > 0.9 && ohAlcapi0s4s9ClusAll[j] > 0.9 &&     
               dr > 0.3 && deta > 0.1 &&    
               mesonpt > 4.0 &&    
	       mesonmass > 0.3 && mesonmass < 0.8)    
	      rc++;    
	  } 
	  if(TMath::Abs(ohAlcapi0etaClusAll[i]) > 1.479 && TMath::Abs(ohAlcapi0etaClusAll[j]) > 1.479) {  
	    //eta endcap   
	    if(ohAlcapi0ptClusAll[i] > 1.5 && ohAlcapi0ptClusAll[j] > 1.5 &&     
	       ohAlcapi0s4s9ClusAll[i] > 0.9 && ohAlcapi0s4s9ClusAll[j] > 0.9 &&      
	       dr > 0.3 && deta > 0.1 &&     
	       mesonpt > 5.0 &&     
	       mesonmass > 0.3 && mesonmass < 0.8)     
	      rc++;     
	  }
	} 
      }
      if(rc > 0) { 
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }           
      } 
    }  
  }
  
  /* Cross Triggers (approved in Jan 2009) */

  // SGL - lepton+jet cross-triggers. These are for 1E31, so the *corrected* 
  // jets are used at HLT. 
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu9_1JetU15") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      int rc = 0; 
      if(OpenHlt1CorJetPassed(30)>=1){ // Require 1 corrected jet above threshold
	for(int i = 0; i < NohMuL2; i++) { 
	  if(ohMuL2Pt[i] > 9.) { // Count L2 muons above threshold
	    rc++; 
	  } 
	} 
      } 
      if(rc>0) { 
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }     
      }           
    } 
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele10_SW_L1R_3Jet30_3JetL1") == 0){
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      int rc = 0;
      if(OpenHlt1CorJetPassed(30)>=3){
	if(OpenHlt1ElectronPassed(10.,0,9999.,9999.)>=1) {
	  if(rc>0) {  
	    if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }      
	  }
	}
      }
    }
  }
  // John Paul Chou - e(gamma) + mu cross-trigger. 
  // One non-isolated photon plus one non-isolated L2 muons.
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu5_Photon9_L1R") == 0){
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1L2MuonPassed(3.,5.,2.)>=1 && OpenHlt1PhotonPassed(9.,0,9999.,9999.,9999.,9999.)>=1)
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }        
    } 
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu5_Photon11_L1R") == 0){
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1L2MuonPassed(5.,5.,2.)>=1 && OpenHlt1PhotonPassed(11.,0,9999.,9999.,9999.,9999.)>=1)
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }        
    } 
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu5_Photon13_L1R") == 0){
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1L2MuonPassed(5.,5.,2.)>=1 && OpenHlt1PhotonPassed(13.,0,9999.,9999.,9999.,9999.)>=1)
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }        
    } 
  }
    

  // Exotica mu + e/gamma, mu + jet, and mu + MET L1-passthrough cross-triggers
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu14_L1SingleEG10") == 0){
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }
    }
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu14_L1SingleJet6") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
    } 
  } 
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu14_L1SingleJet15") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
    } 
  } 
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu14_L1SingleJet20") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      int rc = 0;   
      for(int i=0;i<NL1CenJet;i++) if(L1CenJetEt[i] >= 20.0) rc++;   
      for(int i=0;i<NL1ForJet;i++) if(L1ForJetEt[i] >= 20.0) rc++;   
      for(int i=0;i<NL1Tau   ;i++) if(L1TauEt   [i] >= 20.0) rc++;   
      if(rc > 0)
				if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
			
    } 
  } 
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu14_L1ETM30") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
    } 
  }
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_L1Mu14_L1ETM40") == 0){  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) {  
      if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }  
    }  
  } 
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_HT50") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {    
      if(OpenHltSumHTPassed(50., 30.) == 1) { 
        if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1)     
          if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }         
      } 
    } 
  } 
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Mu5_HT80") == 0){   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {      
      if(OpenHltSumHTPassed(80., 30.) == 1) {   
        if(OpenHlt1MuonPassed(3.,4.,5.,2.,0)>=1)       
          if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }           
      }   
    }   
  }   
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Mu8_HT50") == 0){  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {     
      if(OpenHltSumHTPassed(50., 30.) == 1) {  
        if(OpenHlt1MuonPassed(3.,4.,8.,2.,0)>=1)      
          if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }          
      }  
    }  
  }  
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu8_HT50") == 0){   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {      
      if(OpenHltSumHTPassed(50., 30.) == 1) {   
        if(ohMuL2Pt[0] > 8.0) { 
          if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }           
        }   
      }   
    }   
  } 
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele10_LW_L1R_HT150") == 0){ 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {   
      if(OpenHltSumHTPassed(150., 30.) == 1) { 
        if(OpenHlt1LWElectronPassed(10.,0,9999.,9999.)>=1) { 
          if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }        
        } 
      }          
    }          
  } 
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele10_LW_L1R_HT180") == 0){   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {     
      if(OpenHltSumHTPassed(180., 30.) == 1) {   
        if(OpenHlt1LWElectronPassed(10.,0,9999.,9999.)>=1) {   
          if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }          
        }   
      }            
    }            
  }   
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele10_LW_L1R_HT200") == 0){  
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {    
      if(OpenHltSumHTPassed(200., 30.) == 1) {  
        if(OpenHlt1LWElectronPassed(10.,0,9999.,9999.)>=1) {  
          if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }         
        }   
      }            
    }            
  }   
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_Ele15_LW_L1R_HT180") == 0){   
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second>0) {     
      if(OpenHltSumHTPassed(180., 30.) == 1) {   
        if(OpenHlt1LWElectronPassed(15.,0,9999.,9999.)>=1) {   
          if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }          
        }  
      }           
    }           
  }  
  // triple jet b-tag trigger for top
  else if(menu->GetTriggerName(it).CompareTo("OpenHLT_BTagIP_TripleJet20U") == 0){
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
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
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }  
      }  
    } 
  }
  // Lepton+jet triggers for... top? exotica? b-tagging?
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_L2Mu9_DiJet30") == 0) {
    int njetswithmu = 0;
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1L2MuonPassed(9.,9.,2.)>=1) {
	for(int i = 0; i < NrecoJetCal; i++) {
	  if(recoJetCorCalPt[i] > 30.) { // Cut on corrected jet energy
	    njetswithmu++;
	  }
	}
      }
      if(njetswithmu >= 2) // Require >= 2 jets above threshold
	if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; }
    }
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele10_SW_L1R_TripleJet30") == 0) {
    int njetswithe = 0; 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1ElectronPassed(10.,0,9999.,9999.)>=1) {
        for(int i = 0; i < NrecoJetCal; i++) { 
          if(recoJetCorCalPt[i] > 30.) { // Cut on corrected jet energy 
            njetswithe++; 
          } 
        } 
      } 
      if(njetswithe >= 3) // Require >= 3 jets above threshold 
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
    } 
  }
  else if (menu->GetTriggerName(it).CompareTo("OpenHLT_Ele15_SW_L1R_TripleJet30") == 0) {
    int njetswithe = 0; 
    if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(it))->second==1) { 
      if(OpenHlt1ElectronPassed(15.,0,9999.,9999.)>=1) {
        for(int i = 0; i < NrecoJetCal; i++) { 
          if(recoJetCorCalPt[i] > 30.) { // Cut on corrected jet energy 
            njetswithe++; 
          } 
        } 
      } 
      if(njetswithe >= 3) // Require >= 3 jets above threshold 
        if (prescaleResponse(menu,cfg,rcounter,it)) { triggerBit[it] = true; } 
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
	      if (OpenHltL1L2TauMatching(ohTauEta[i], ohTauPhi[i], L1TauThr, L1CenJetThr) == 1)
		rc++;
    }
    if (ohTauL3Tiso[i] >= 1) l3iso++;
  }
  
  if (rc>=2 && l3iso>=1)
    return 1;
  else
    return 0;
}

// e-tau
int OHltTree::OpenHltElecTauL2SCPassed(float elecEt, int elecL1iso, float elecTiso, float elecHiso,
				       float tauEt,float tauL25Tpt, int tauL25Tiso, float tauL3Tpt, int tauL3Tiso)
{
  int rc = 0;

  // Loop over all oh taus
  for (int i=0;i<NohTau;i++) {
    if (ohTauPt[i] >= tauEt) {
      if (ohTauEiso[i] < (5 + 0.025*ohTauPt[i] + 0.0015*ohTauPt[i]*ohTauPt[i])) { // sliding cut
        if (ohTauL25Tpt[i] >= tauL25Tpt) { 
          if (ohTauL25Tiso[i] >= tauL25Tiso) { 
            if (ohTauL3Tpt[i] >= tauL3Tpt) { 
              if (ohTauL3Tiso[i] >= tauL3Tiso) {	
		
		// Loop over all oh LW electrons 
		for (int j=0;j<NohEleLW;j++) { 
		  if ( ohEleEtLW[j] > elecEt) { 
		    if ( ohEleHisoLW[j] < elecHiso || ohEleHisoLW[j]/ohEleEtLW[j] < 0.05) {
		      if (ohEleNewSCLW[j]==1) {
			if (ohElePixelSeedsLW[j]>0) {
			  if ( ohEleTisoLW[j] < elecTiso && ohEleTisoLW[j] != -999.) {
			    if ( ohEleL1isoLW[j] >= elecL1iso ) {  // L1iso is 0 or 1

			      // Check non-collinearity
			      float deta = (float)TMath::Abs(ohTauEta[i]-ohEleEtaLW[j]);
			      float dphi = (float)TMath::Abs(TVector2::Phi_mpi_pi((Double_t)(ohTauPhi[i]-ohElePhiLW[j])));
			      if (deta>0.3 && dphi>0.3) { // box cut			      
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

  return rc;
}



int OHltTree::OpenHlt1ElectronPassed(float Et, int L1iso, float Tiso, float Hiso)
{
  int rc = 0;
  // Loop over all oh electrons
  for (int i=0;i<NohEle;i++) {
    if ( ohEleEt[i] > Et) {
      if( TMath::Abs(ohEleEta[i]) < 2.65 ) 
	if ( ohEleHiso[i] < Hiso || ohEleHiso[i]/ohEleEt[i] < 0.05)
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

int OHltTree::OpenHlt1LWElectronPassed(float Et, int L1iso, float Tiso, float Hiso) 
{ 
  int rc = 0; 
  // Loop over all oh LW electrons 
  for (int i=0;i<NohEleLW;i++) { 
    if ( ohEleEtLW[i] > Et) { 
      if( TMath::Abs(ohEleEtaLW[i]) < 2.65 ) 
	if ( ohEleHisoLW[i] < Hiso || ohEleHisoLW[i]/ohEleEtLW[i] < 0.05) 
	  if (ohEleNewSCLW[i]==1) 
	    if (ohElePixelSeedsLW[i]>0) 
	      if ( (ohEleTisoLW[i] < Tiso && ohEleTisoLW[i] != -999.) || (Tiso == 9999.) )
		if ( ohEleL1isoLW[i] >= L1iso )   // L1iso is 0 or 1 
		  if( ohEleLWL1Dupl[i] == false) // remove double-counted L1 SCs
		    rc++;       
    } 
  }
  
  return rc; 
} 

int OHltTree::OpenHlt1EleIdLWElectronPassed(float Et, int L1iso, float Tiso, float Hiso) 
{ 
  int rc = 0; 
  // Loop over all oh LW electrons 
  for (int i=0;i<NohEleLW;i++) { 
    if ( ohEleEtLW[i] > Et) { 
      if( TMath::Abs(ohEleEtaLW[i]) < 2.65 ) {
	if ( ohEleHisoLW[i] < Hiso || ohEleHisoLW[i]/ohEleEtLW[i] < 0.05) 
	  if (ohEleNewSCLW[i]==1) 
	    if (ohElePixelSeedsLW[i]>0) 
	      if ( (ohEleTisoLW[i] < Tiso && ohEleTisoLW[i] != -999.) || (Tiso == 9999.) ) 
	      if ( ohEleL1isoLW[i] >= L1iso )   // L1iso is 0 or 1 
		if( ohEleLWL1Dupl[i] == false) // remove double-counted L1 SCs
		  //		  if ( ohEleDetaLW[i] < 0.01 && ohEleDphiLW[i] < 0.09 ) Old definitiion
		  if ( ohEleDetaLW[i] < 0.008 && ohEleDphiLW[i] < 0.1 ) 
		    if (
			(ohEleClusShapLW[i] < 0.015 && TMath::Abs(ohEleEtaLW[i]) < 1.479) ||
			(ohEleClusShapLW[i] < 0.04 && TMath::Abs(ohEleEtaLW[i]) >= 1.479) 
			//			(ohEleClusShapLW[i] < 0.014 && ohEleEtaLW[i] < 1.475) || Old definition
			//			(ohEleClusShapLW[i] < 0.0275 && ohEleEtaLW[i] >= 1.475)
			 )
		      rc++;       
      } 
    }
  }
  
  return rc; 
} 

int OHltTree::OpenHlt1ElectronEleIDPassed(float Et,int L1iso,float Tiso,float Hiso) 
{ 
  int rc = 0;  
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
  
  return rc;  
} 


int  OHltTree::OpenHlt1PhotonLooseEcalIsoPassed(float Et, int L1iso, float Tiso, float Eiso, float HisoBR, float HisoEC)
{
  int rc = 0;
  // Loop over all oh photons
  for (int i=0;i<NohPhot;i++) {
    if ( ohPhotEt[i] > Et) { 
      if( TMath::Abs(ohPhotEta[i]) < 2.65 ) {
	if ( ohPhotL1iso[i] >= L1iso ) { 
	  if( ohPhotTiso[i]<Tiso ) { 
	    if( ohPhotEiso[i] < Eiso || (ohPhotEiso[i]/ohPhotEt[i]) < 0.2 ) { 
	      if( (TMath::Abs(ohPhotEta[i]) < 1.479 && ohPhotHiso[i] < HisoBR )  ||
		  (1.479 < TMath::Abs(ohPhotEta[i]) && TMath::Abs(ohPhotEta[i]) < 2.65 && ohPhotHiso[i] < HisoEC ) || 
		  (ohPhotHiso[i]/ohPhotEt[i] < 0.05) ) {
		if( ohPhotL1Dupl[i] == false) // remove double-counted L1 SCs 
		  rc++;
	      }
	    }
	  }
	}
      }
    }
  }

  return rc;
}

int  OHltTree::OpenHlt1PhotonVeryLooseEcalIsoPassed(float Et, int L1iso, float Tiso, float Eiso, float HisoBR, float HisoEC)  
{  
  int rc = 0;  
  // Loop over all oh photons  
  for (int i=0;i<NohPhot;i++) {  
    if ( ohPhotEt[i] > Et) {   
      if( TMath::Abs(ohPhotEta[i]) < 2.65 ) { 
	if ( ohPhotL1iso[i] >= L1iso ) {   
	  if( ohPhotTiso[i]<Tiso ) {   
	    if( ohPhotEiso[i] < Eiso || (ohPhotEiso[i]/ohPhotEt[i]) < 0.2 ) {   
	      if( (TMath::Abs(ohPhotEta[i]) < 1.479 && ohPhotHiso[i] < HisoBR )  ||  
                (1.479 < TMath::Abs(ohPhotEta[i]) && TMath::Abs(ohPhotEta[i]) < 2.65 && ohPhotHiso[i] < HisoEC ) ||   
		  (ohPhotHiso[i]/ohPhotEt[i] < 0.05) ) {  
		if( ohPhotL1Dupl[i] == false) // remove double-counted L1 SCs  
		  rc++;  
	      }  
	    }
          }  
        }  
      }  
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
      if( TMath::Abs(ohPhotEta[i]) < 2.65 ) { 
	if ( ohPhotL1iso[i] >= L1iso ) { 
	  if( (ohPhotTiso[i]/ohPhotEt[i]) < Tiso ) {
	    if( ohPhotEiso[i] < Eiso ) { 
	      if( (TMath::Abs(ohPhotEta[i]) < 1.479 && ohPhotHiso[i] < HisoBR )  ||
		  (1.479 < TMath::Abs(ohPhotEta[i]) && TMath::Abs(ohPhotEta[i]) < 2.65 && ohPhotHiso[i] < HisoEC ) || 
		  (ohPhotHiso[i]/ohPhotEt[i] < 0.05) ) {
		if( ohPhotL1Dupl[i] == false) // remove double-counted L1 SCs  
		  rc++;
	      }
	    }
	  }
	}
      }
    }
  }

  return rc;
}

int  OHltTree::OpenHlt2PhotonMassWinPassed(float Et, int L1iso, float Tiso, float Eiso, float HisoBR, float HisoEC,float massLow, float massHigh)
{
  int rc = 0;
  // Loop over all oh photons
  for (int i=0;i<NohPhot;i++) {
    if ( ohPhotEt[i] > Et) { 
      if( TMath::Abs(ohPhotEta[i]) < 2.65 ) { 
	if ( ohPhotL1iso[i] >= L1iso ) { 
	  if( ohPhotTiso[i]<Tiso ) { 
	    if( ohPhotEiso[i] < Eiso ) { 
	      if( (TMath::Abs(ohPhotEta[i]) < 1.479 && ohPhotHiso[i] < HisoBR )  ||
		  (1.479 < TMath::Abs(ohPhotEta[i]) && TMath::Abs(ohPhotEta[i]) < 2.65 && ohPhotHiso[i] < HisoEC ) || 
		  (ohPhotHiso[i]/ohPhotEt[i] < 0.05) ) {
		if( ohPhotL1Dupl[i] == false) // remove double-counted L1 SCs  
		  rc++;
	      }
	    }
	  }
	}
      }
    }
  }

  TLorentzVector e1;  
  TLorentzVector e2;  
  TLorentzVector meson;
  float mass = 0.;
  for (int i=0;i<NohEleLW;i++) { 
    for (int j=0;j<NohEleLW && j != i;j++) {  	  
      //if (ohElePixelSeedsLW[i]>0 && ohElePixelSeedsLW[j]>0) {
	e1.SetPtEtaPhiM(ohEleEtLW[i],ohEleEtaLW[i],ohElePhiLW[i],0.); 
	e2.SetPtEtaPhiM(ohEleEtLW[j],ohEleEtaLW[j],ohElePhiLW[j],0.); 
	meson = e1 + e2; 
	mass = meson.M();  
      //}
    }
  }
  
  if (rc>=2 && mass>massLow && mass<massHigh)
    return 1;
  else
    return 0;
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

int OHltTree::OpenHltSumHTPassed(double sumHTthreshold, double jetthreshold) 
{ 
  int rc = 0;   
  double sumHT = 0.;   


  // Loop over all oh jets, sum up the energy  
  for (int i=0;i<NrecoJetCorCal;++i) {     
    if(recoJetCorCalPt[i] >= jetthreshold) { 
      //sumHT+=recoJetCorCalPt[i];

      sumHT+=(recoJetCorCalE[i]/cosh(recoJetCorCalEta[i]));
    }     
  }      
   
  if(sumHT >= sumHTthreshold) rc = 1;

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
