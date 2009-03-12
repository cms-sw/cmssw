#include <iostream>
#include <vector>
#include <string>
#include <map>

#include <TString.h>
#include <TTree.h>
#include <TChain.h>
#include "OHltTree.h"
#include "OHltMenu.h"
#include "OHltConfig.h"
#include "OHltRatePrinter.h"

using namespace std;


/* ********************************************** */
// Declarations
/* ********************************************** */
void fillProcesses(OHltConfig *cfg,vector<OHltTree*> &procs,vector<TChain*> &chains,OHltMenu *menu);
void calcRates(OHltConfig *cfg,OHltMenu *menu,vector<OHltTree*> &procs,
	       vector<OHltRateCounter*> &rcs,OHltRatePrinter* rprint);

/* ********************************************** */
// Print out usage and example
/* ********************************************** */
inline void ShowUsage() {
  cout << "  Usage:  ./OHltRateEff <Config File>" << endl;
  cout << "default:  ./OHltRateEff hltmenu_1E31_2008Dec04.cfg" << endl;
}

/* ********************************************** */
// Main
/* ********************************************** */
int main(int argc, char *argv[]){

  int argIndex = 0;
  if (argc > ++argIndex) {
    if (TString(argv[1])=="-h") {ShowUsage();exit(0);}
  }

  // Book menu
  OHltMenu *omenu = new OHltMenu();
  // Read & parse cfg file
  OHltConfig *ocfg = new OHltConfig(argv[1],omenu);

  // Prepare process files
  vector<TChain*> chains; chains.clear();
  vector<OHltTree*> procs; procs.clear();
  fillProcesses(ocfg,procs,chains,omenu);

  /* **** */
  // Count rates
  vector<OHltRateCounter*> rcs; rcs.clear();
  for (unsigned int i=0;i<procs.size();i++) {
    rcs.push_back(new OHltRateCounter(omenu->GetTriggerSize()));
  }
  OHltRatePrinter* rprint = new OHltRatePrinter();
  calcRates(ocfg,omenu,procs,rcs,rprint);

  rprint->printRatesASCII(ocfg,omenu);
  //rprint->printCorrelationASCII();

  if (ocfg->doPrintAll) {
    rprint->printRatesTex(ocfg,omenu);    
    rprint->printPrescalesCfg(ocfg,omenu);
    rprint->writeHistos(ocfg,omenu);    
  }
  /* **** */

  //
  return 0;
}


/* ********************************************** */
// Prepare process files
/* ********************************************** */
void fillProcesses(OHltConfig *cfg,vector<OHltTree*> &procs,vector<TChain*> &chains,OHltMenu *menu) {
  for (unsigned int i=0;i<cfg->pnames.size();i++) {
    chains.push_back(new TChain("HltTree"));
    chains.back()->Add(cfg->ppaths[i]+cfg->pfnames[i]);
    //chains[i]->Print();
    procs.push_back(new OHltTree((TTree*)chains[i],menu));
  }
}

/* ********************************************** */
// Do the actual rate count & rate conversion
/* ********************************************** */
void calcRates(OHltConfig *cfg,OHltMenu *menu,vector<OHltTree*> &procs,
	       vector<OHltRateCounter*> &rcs,OHltRatePrinter* rprint) {

  const int ntrig = (int)menu->GetTriggerSize();
  vector<float> Rate,pureRate,spureRate;
  vector<float> RateErr,pureRateErr,spureRateErr;
  vector< vector<float> >coMa;
  vector<float> coDen;
  
  vector<float> ftmp;
  for (int i=0;i<ntrig;i++) { // Init
    Rate.push_back(0.);pureRate.push_back(0.);spureRate.push_back(0.);ftmp.push_back(0.);
    RateErr.push_back(0.);pureRateErr.push_back(0.);spureRateErr.push_back(0.);
    coDen.push_back(0.);
  }
  for (int j=0;j<ntrig;j++) { coMa.push_back(ftmp); }

  for (unsigned int i=0;i<procs.size();i++) {
    procs[i]->Loop(rcs[i],cfg,menu,i);

    float deno = (float)cfg->nEntries;

    float scaleddeno = -1;
    if(cfg->isRealData && cfg->nL1AcceptsRun > 0)
      scaleddeno = ((float)cfg->nEntries/(float)cfg->nL1AcceptsRun) * (float)cfg->liveTimeRun;

    float chainEntries = (float)procs[i]->fChain->GetEntries(); 
    if (deno <= 0. || deno > chainEntries) {
      deno = chainEntries;
    }


    float mu =
      cfg->bunchCrossingTime*cfg->psigmas[i]*cfg->iLumi*(float)cfg->maxFilledBunches
      /(float)cfg->nFilledBunches;
    float collisionRate =
      ((float)cfg->nFilledBunches/(float)cfg->maxFilledBunches)/cfg->bunchCrossingTime ; // Hz

    

    for (int j=0;j<ntrig;j++) {
      // JH - cosmics!
      if(cfg->isRealData) {
	Rate[j]    += OHltRateCounter::eff((float)rcs[i]->iCount[j],scaleddeno);   
	RateErr[j] += OHltRateCounter::effErr((float)rcs[i]->iCount[j],scaleddeno); 
	spureRate[j]    += OHltRateCounter::eff((float)rcs[i]->sPureCount[j],scaleddeno);   
	spureRateErr[j] += OHltRateCounter::effErr((float)rcs[i]->sPureCount[j],scaleddeno); 
	pureRate[j]    += OHltRateCounter::eff((float)rcs[i]->pureCount[j],scaleddeno);   
	pureRateErr[j] += OHltRateCounter::effErr((float)rcs[i]->pureCount[j],scaleddeno); 
	cout << "N(passing " << menu->GetTriggerName(j) << ") = " << (float)rcs[i]->iCount[j] << endl;

        for (int k=0;k<ntrig;k++){ 
          coMa[j][k] += ((float)rcs[i]->overlapCount[j][k]);
        } 
        coDen[j] += ((float)rcs[i]->iCount[j]); // ovelap denominator 
      }

      else{
	Rate[j]    += collisionRate*(1. - exp(- mu * OHltRateCounter::eff((float)rcs[i]->iCount[j],deno)));  
	RateErr[j] += pow(collisionRate*mu * OHltRateCounter::effErr((float)rcs[i]->iCount[j],deno),2.);
	//cout<<j<<" Counts: "<<rcs[i]->iCount[j]<<endl;
	spureRate[j]    += collisionRate*(1. - exp(- mu * OHltRateCounter::eff((float)rcs[i]->sPureCount[j],deno)));  
	spureRateErr[j] += pow(collisionRate*mu * OHltRateCounter::effErr((float)rcs[i]->sPureCount[j],deno),2.);
	pureRate[j]    += collisionRate*(1. - exp(- mu * OHltRateCounter::eff((float)rcs[i]->pureCount[j],deno)));  
	pureRateErr[j] += pow(collisionRate*mu * OHltRateCounter::effErr((float)rcs[i]->pureCount[j],deno),2.);
        cout << "N(passing " << menu->GetTriggerName(j) << ") = " << (float)rcs[i]->iCount[j] << endl; 

	for (int k=0;k<ntrig;k++){
	  coMa[j][k] += ((float)rcs[i]->overlapCount[j][k]) * cfg->psigmas[i];     
	}
	coDen[j] += ((float)rcs[i]->iCount[j] * cfg->psigmas[i]); // ovelap denominator
      }
    }
  }

  for (int i=0;i<ntrig;i++) {
    RateErr[i] = sqrt(RateErr[i]);
    spureRateErr[i] = sqrt(spureRateErr[i]);
    pureRateErr[i] = sqrt(pureRateErr[i]);
    //cout<<menu->GetTriggerName(i)<<" "<<Rate[i]<<" +- "<<RateErr[i]<<endl;

    for (int j=0;j<ntrig;j++){
      coMa[i][j] = coMa[i][j]/coDen[i]; 
    }
  }
  
  rprint->SetupAll(Rate,RateErr,spureRate,spureRateErr,pureRate,pureRateErr,coMa);
  
}

////////////////////////////////////////////////////////////////////////////////////////////
// END
////////////////////////////////////////////////////////////////////////////////////////////
