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

void OHltTree::Loop(OHltRateCounter *rc,OHltConfig *cfg,OHltMenu *menu,int procID) 
{
  cout<<"Start looping on sample "<<procID<<endl;
  if (fChain == 0) {cerr<<"Error: no tree!"<<endl; return;}

  Long64_t nentries = (Long64_t)cfg->nEntries; 
  if (cfg->nEntries <= 0)
    nentries = fChain->GetEntries();
  cout<<"Entries to be processed: "<<nentries<<endl;

  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    fChain->GetEntry(jentry);

    if (jentry%cfg->nPrintStatusEvery == 0)
      cout<<"Processing entry "<<jentry<<"/"<<nentries<<"\r"<<flush<<endl;

    // 1. Loop to check which Bit fired
    // Triggernames are assigned to trigger cuts in unambigous way!
    // If you define a new trigger also define a new unambigous name!
    if(menu->DoL1preLoop() && menu->IsHltMenu()) {
      ApplyL1Prescales(menu);
    }
    SetMapL1BitOfStandardHLTPath(menu);
    SetL1MuonQuality();
	  
    //////////////////////////////////////////////////////////////////
    // Loop over trigger paths and do rate counting
    //////////////////////////////////////////////////////////////////
    for (int i = 0; i < nTrig; i++){
      triggerBit[i] = false;
      previousBitsFired[i] = false;
      allOtherBitsFired[i] = false;
      if ( cfg->pdomucuts[procID] && MCmu3!=0 ) continue;
      if ( cfg->pdoecuts[procID] && MCel3!=0 ) continue;

      //////////////////////////////////////////////////////////////////
      // Standard paths
      if ( (map_BitOfStandardHLTPath.find(menu->GetTriggerName(i))->second==1) ) {	
	if (map_L1BitOfStandardHLTPath.find(menu->GetTriggerName(i))->second>0) 
	  if (GetIntRandom() % menu->GetPrescale(i) == 0)  
	    triggerBit[i] = true; 
      } else {
      // Open HLT paths
 	CheckOpenHlt(cfg,menu,i);
      }
    }

    /* ******************************** */
    // 2. Loop to check overlaps
    for (int it = 0; it < nTrig; it++){
      if (triggerBit[it]) {
	rc->iCount[it]++;
	for (int it2 = 0; it2 < nTrig; it2++){
	  if ( (it2<it) && triggerBit[it2] )
	    previousBitsFired[it] = true;
	  if ( (it2!=it) && triggerBit[it2] )
	    allOtherBitsFired[it] = true;
	  if (triggerBit[it2])
	    rc->overlapCount[it][it2] += 1;
	}
	if (not previousBitsFired[it])
	  rc->sPureCount[it]++;
	if (not allOtherBitsFired[it])
	  rc->pureCount[it]++;
      }
    }
    /* ******************************** */

    
  }
  
}
