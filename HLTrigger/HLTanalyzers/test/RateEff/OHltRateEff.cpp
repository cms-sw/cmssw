/////////////////////////////////////////////////////////////////////////////////////////////////
//
//        Program to calculate rates of trigger paths using variables of OHltTree class,
//
//Note: OHltTree class needs to be updated if any new variables become available 
//in OpenHLT (HLTAnalyzer).
//
//        Author:  Vladimir Rekovic,     Date: 2007/12/10
//
//
/////////////////////////////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <vector>
#include <string>
#include <map>

#include <TString.h>
#include <TTree.h>
#include <TChain.h>
#include <TError.h>
#include "OHltTree.h"
#include "OHltMenu.h"
#include "OHltConfig.h"
#include "OHltRatePrinter.h"
#include "OHltEffPrinter.h"

using namespace std;

/* ********************************************** */
// Declarations
/* ********************************************** */
void fillProcesses(
      OHltConfig *cfg,
      vector<OHltTree*> &procs,
      vector<TChain*> &chains,
      OHltMenu *menu,
      HLTDatasets &hltDatasets);

void calcRates(
      OHltConfig *cfg,
      OHltMenu *menu,
      vector<OHltTree*> &procs,
      vector<OHltRateCounter*> &rcs,
      OHltRatePrinter* rprint,
      HLTDatasets &hltDatasets);

void calcEff(
      OHltConfig *cfg,
      OHltMenu *menu,
      vector<OHltTree*> &procs,
      vector<OHltRateCounter*> &ecs,
      OHltEffPrinter* eprint,
      float &DenEff,
      HLTDatasets &hltDatasets);

/* ********************************************** */
// Print out usage and example
/* ********************************************** */
inline void ShowUsage()
{
   cout << "  Usage:  ./OHltRateEff <Config File>" << endl;
   cout << "default:  ./OHltRateEff hltmenu_1E31_2008Dec04.cfg" << endl;
}

/* ********************************************** */
// Main
/* ********************************************** */
int main(int argc, char *argv[])
{

   gErrorIgnoreLevel = 5001;

   int argIndex = 0;
   if (argc > ++argIndex)
   {
      if (TString(argv[1])=="-h")
      {
         ShowUsage();
         exit(0);
      }
   }

   // Book menu
   OHltMenu *omenu = new OHltMenu();
   // Read & parse cfg file
   OHltConfig *ocfg = new OHltConfig(argv[1],omenu);
   /**
    Create a HLTDatasets object to record primary datasets.
    Make sure to register each sample with it as well!
    */
   HLTDatasets hltDatasets(
         omenu->GetTriggerNames(),
         ocfg->dsList.Data(),
         kFALSE,
         "PrefixTest");

   // Prepare process files
   vector<TChain*> chains;
   chains.clear();
   vector<OHltTree*> procs;
   procs.clear();
   fillProcesses(ocfg, procs, chains, omenu, hltDatasets);

   /* **** */
   // Count rates
   vector<OHltRateCounter*> rcs;
   rcs.clear();
   for (unsigned int i=0; i<procs.size(); i++)
   {
      rcs.push_back(new OHltRateCounter(omenu->GetTriggerSize(), omenu->GetL1TriggerSize()));
   }
   OHltRatePrinter* rprint = new OHltRatePrinter();
   calcRates(ocfg, omenu, procs, rcs, rprint, hltDatasets);

   /* **** */
   // Get Seed prescales
   omenu->SetMapL1SeedsOfStandardHLTPath(procs[0]->GetL1SeedsOfHLTPathMap());

   rprint->printRatesASCII(ocfg, omenu);
   //rprint->printCorrelationASCII();

   if (ocfg->doPrintAll)
   {
      //    rprint->printRatesTex(ocfg,omenu);    
      rprint->printRatesTwiki(ocfg, omenu);
      //    rprint->printPrescalesCfg(ocfg,omenu);
      rprint->writeHistos(ocfg, omenu);
      if(ocfg->nonlinearPileupFit != "none")
	rprint->fitRatesForPileup(ocfg, omenu);
      char sLumi[10], sEnergy[10];
      snprintf(sEnergy, 10, "%1.0f", ocfg->cmsEnergy);
      snprintf(sLumi,   10, "%1.1e", ocfg->iLumi);
      TString hltTableFileName= TString("hlt_DS_Table_") + sEnergy + "TeV_"
            + sLumi + TString("_") + ocfg->versionTag;
      // 		printf("About to call printHLTDatasets\n"); //RR
      rprint->printHLTDatasets(ocfg,omenu,hltDatasets,hltTableFileName,3);
   }
   /* **** */
   // Calculate Efficiencies
   vector<OHltRateCounter*> ecs;
   ecs.clear();
   for (unsigned int i=0; i<procs.size(); i++)
   {
      ecs.push_back(new OHltRateCounter(omenu->GetTriggerSize(), omenu->GetL1TriggerSize()));
   }
   //RR debugging couts
   // 	printf("About to call calcEff\n");
   OHltEffPrinter* eprint = new OHltEffPrinter();
   float DenEff=0;
   calcEff(ocfg, omenu, procs, ecs, eprint, DenEff, hltDatasets);
   // 	printf("calcEff just executed. About to call printEffASCII\n");
   if (DenEff != 0)
      eprint->printEffASCII(ocfg, omenu);

   //
   return 0;
}

/* ********************************************** */
// Prepare process files
/* ********************************************** */
void fillProcesses(
      OHltConfig *cfg,
      vector<OHltTree*> &procs,
      vector<TChain*> &chains,
      OHltMenu *menu,
      HLTDatasets &hltDatasets)
{
   for (unsigned int i=0; i<cfg->pnames.size(); i++)
   {
      chains.push_back(new TChain("HltTree"));
      chains.back()->Add(cfg->ppaths[i]+cfg->pfnames[i]);
      //chains[i]->Print();
      procs.push_back(new OHltTree((TTree*)chains[i],menu));
      hltDatasets.addSample(cfg->pnames[i], (cfg->pisPhysicsSample[i]==0
            ? RATE_SAMPLE
            : PHYSICS_SAMPLE)); //SAK
   }
}

/* ********************************************** */
// Do the actual rate count & rate conversion
/* ********************************************** */
void calcRates(
      OHltConfig *cfg,
      OHltMenu *menu,
      vector<OHltTree*> &procs,
      vector<OHltRateCounter*> &rcs,
      OHltRatePrinter* rprint,
      HLTDatasets &hltDatasets)
{

   const int ntrig = (int)menu->GetTriggerSize();
   const int nL1trig = (int)menu->GetL1TriggerSize();
   vector< vector< float> > RatePerLS;
   vector< vector< int > > CountPerLS;
   vector< vector< int> > RefPrescalePerLS;
   vector< vector< int> > RefL1PrescalePerLS;
   vector<float> totalRatePerLS;
   vector<int> totalCountPerLS;
   vector<int> Count;
   vector<float> Rate, pureRate, spureRate;
   vector<float> RateErr, pureRateErr, spureRateErr;
   vector< vector<float> > coMa;
   vector<float> coDen;
   vector<int> RefPrescale, RefL1Prescale;
   vector<float> weightedPrescaleRefHLT;
   vector<float> weightedPrescaleRefL1;
   vector<double> InstLumiPerLS;
   float DenEff=0.;
   Int_t nbinpt = 50;
   Float_t ptmin = 0.0;
   Float_t ptmax = 20.0;
   Int_t nbineta = 30;
   Float_t etamin = -3.0;
   Float_t etamax = 3.0;
   TH1F *h1 = new TH1F("h1","pTnum",nbinpt,ptmin,ptmax);
   TH1F *h2 = new TH1F("h2","pTden",nbinpt,ptmin,ptmax);
   TH1F *h3 = new TH1F("h3","etanum",nbineta,etamin,etamax);
   TH1F *h4 = new TH1F("h4","etaden",nbineta,etamin,etamax);

   float fTwo=2.;

   vector<float> ftmp;
   for (int i=0; i<ntrig; i++)
   { // Init
      // per lumisection
      Rate.push_back(0.);
      pureRate.push_back(0.);
      spureRate.push_back(0.);
      ftmp.push_back(0.);
      RateErr.push_back(0.);
      pureRateErr.push_back(0.);
      spureRateErr.push_back(0.);
      RefPrescale.push_back(1);
      weightedPrescaleRefHLT.push_back(0.);
      coDen.push_back(0.);
      Count.push_back(0);
   }
   for (int i=0; i<nL1trig; i++)
   { // Init
      RefL1Prescale.push_back(1);
      weightedPrescaleRefL1.push_back(0.);
   }
   for (int j=0; j<ntrig; j++)
   {
      coMa.push_back(ftmp);
   }

   for (unsigned int i=0; i<procs.size(); i++)
   {
      procs[i]->Loop(
            rcs[i],
            cfg,
            menu,
            i,
            DenEff,
            h1,
            h2,
            h3,
            h4,
            hltDatasets[i]);

      for (unsigned int iLS=0; iLS<rcs[0]->perLumiSectionCount.size(); iLS++)
      {
         RatePerLS.push_back(Rate);
         totalRatePerLS.push_back(0.);
	 CountPerLS.push_back(Count);
	 totalCountPerLS.push_back(0);
         RefPrescalePerLS.push_back(RefPrescale);
         RefL1PrescalePerLS.push_back(RefL1Prescale);
	 InstLumiPerLS.push_back(0.);
      }

      float deno = (float)cfg->nEntries;

      float scaleddeno = -1;
      float scaleddenoPerLS = -1;
      float prescaleSum = 0.0;

      float chainEntries = (float)procs[i]->fChain->GetEntries();
      if (deno <= 0. || deno > chainEntries)
      {
         deno = chainEntries;
      }

      if (cfg->isRealData == 1 && cfg->lumiSectionLength > 0)
      {
         // Effective time = # of lumi sections * length of 1 lumi section / overall prescale factor of the PD being analyzed
         float fact=cfg->lumiSectionLength/cfg->lumiScaleFactor;
         scaleddeno = (float)((procs[i]->GetNLumiSections()) * (fact))
               / ((float)(cfg->prescaleNormalization));
         //scaleddeno = (float)(1. * (fact)) / ((float)(cfg->prescaleNormalization));
         scaleddenoPerLS = (float)((fact))
               / ((float)(cfg->prescaleNormalization));
         cout << "N(Lumi Sections) = " << (procs[i]->GetNLumiSections())
               << endl;
         hltDatasets[i].computeRate(scaleddeno); //SAK -- convert event counts into rates. FOR DATA ONLY
      }

      float mu = cfg->bunchCrossingTime*cfg->psigmas[i]*cfg->iLumi
            *(float)cfg->maxFilledBunches /(float)cfg->nFilledBunches;
      float collisionRate = ((float)cfg->nFilledBunches
            /(float)cfg->maxFilledBunches)/cfg->bunchCrossingTime; // Hz

      if (!(cfg->isRealData == 1 && cfg->lumiSectionLength > 0))
         hltDatasets[i].computeRate(collisionRate, mu); //SAK -- convert event counts into rates


      for (unsigned int iLS=0; iLS<rcs[i]->perLumiSectionCount.size(); iLS++)
      {
         totalRatePerLS[iLS] += OHltRateCounter::eff(
               (float)rcs[i]->perLumiSectionTotCount[iLS],
               scaleddenoPerLS);
	 totalCountPerLS[iLS] += rcs[i]->perLumiSectionTotCount[iLS];
	 InstLumiPerLS[iLS] = (double)rcs[i]->perLumiSectionLumi[iLS];
      }

      for (int j=0; j<nL1trig; j++)
      {
         // per lumisection
         prescaleSum = 0.0;
         for (unsigned int iLS=0; iLS<rcs[i]->perLumiSectionCount.size(); iLS++)
         {
            RefL1PrescalePerLS[iLS][j]
                  = (float)rcs[i]->perLumiSectionRefL1Prescale[iLS][j];
            prescaleSum += RefL1PrescalePerLS[iLS][j];
         }
         weightedPrescaleRefL1[j] = prescaleSum
               /(rcs[i]->perLumiSectionCount.size());
      }

      for (int j=0; j<ntrig; j++)
      {

         // per lumisection
         prescaleSum = 0.0;
         for (unsigned int iLS=0; iLS<rcs[i]->perLumiSectionCount.size(); iLS++)
         {
            RatePerLS[iLS][j] += OHltRateCounter::eff(
                  (float)rcs[i]->perLumiSectionCount[iLS][j],
                  scaleddenoPerLS);
	    CountPerLS[iLS][j] += rcs[i]->perLumiSectionCount[iLS][j];
            RefPrescalePerLS[iLS][j]
                  = (float)rcs[i]->perLumiSectionRefPrescale[iLS][j];
            prescaleSum += RefPrescalePerLS[iLS][j];
         }
         weightedPrescaleRefHLT[j] = prescaleSum
               /(rcs[i]->perLumiSectionCount.size());

         if (cfg->isRealData == 1)
         {
            Rate[j] += OHltRateCounter::eff(
                  (float)rcs[i]->iCount[j],
                  scaleddeno);
            RateErr[j] += OHltRateCounter::errRate2(
                  (float)rcs[i]->iCount[j],
                  scaleddeno);
	    Count[j] += rcs[i]->iCount[j];
            spureRate[j] += OHltRateCounter::eff(
                  (float)rcs[i]->sPureCount[j],
                  scaleddeno);
            spureRateErr[j] += OHltRateCounter::errRate2(
                  (float)rcs[i]->sPureCount[j],
                  scaleddeno);
            pureRate[j] += OHltRateCounter::eff(
                  (float)rcs[i]->pureCount[j],
                  scaleddeno);
            pureRateErr[j] += OHltRateCounter::errRate2(
                  (float)rcs[i]->pureCount[j],
                  scaleddeno);
            cout << "N(passing " << menu->GetTriggerName(j) << ") = "
                  << (float)rcs[i]->iCount[j] << endl;

            for (int k=0; k<ntrig; k++)
            {
               coMa[j][k] += ((float)rcs[i]->overlapCount[j][k]);
            }
            coDen[j] += ((float)rcs[i]->iCount[j]); // ovelap denominator 
         }

         else
         {
            Rate[j] += collisionRate*(1. - exp(-mu * OHltRateCounter::eff(
                  (float)rcs[i]->iCount[j],
                  deno)));
            RateErr[j] += pow(collisionRate*mu * OHltRateCounter::effErr(
                  (float)rcs[i]->iCount[j],
                  deno), fTwo);
	    Count[j] += rcs[i]->iCount[j];
            spureRate[j] += collisionRate*(1. - exp(-mu * OHltRateCounter::eff(
                  (float)rcs[i]->sPureCount[j],
                  deno)));
            spureRateErr[j] += pow(collisionRate*mu * OHltRateCounter::effErr(
                  (float)rcs[i]->sPureCount[j],
                  deno), fTwo);
            pureRate[j] += collisionRate*(1. - exp(-mu * OHltRateCounter::eff(
                  (float)rcs[i]->pureCount[j],
                  deno)));
            pureRateErr[j] += pow(collisionRate*mu * OHltRateCounter::effErr(
                  (float)rcs[i]->pureCount[j],
                  deno), fTwo);
            cout << "N(passing " << menu->GetTriggerName(j) << ") = "
                  << (float)rcs[i]->iCount[j] << endl;

            for (int k=0; k<ntrig; k++)
            {
               coMa[j][k] += ((float)rcs[i]->overlapCount[j][k])
                     * cfg->psigmas[i];
            }
            coDen[j] += ((float)rcs[i]->iCount[j] * cfg->psigmas[i]); // ovelap denominator
         }
      }
   }

   for (int i=0; i<ntrig; i++)
   {
      RateErr[i] = sqrt(RateErr[i]);
      spureRateErr[i] = sqrt(spureRateErr[i]);
      pureRateErr[i] = sqrt(pureRateErr[i]);
      //cout<<menu->GetTriggerName(i)<<" "<<Rate[i]<<" +- "<<RateErr[i]<<endl;

      for (int j=0; j<ntrig; j++)
      {
         coMa[i][j] = coMa[i][j]/coDen[i];
      }
   }

   rprint->SetupAll(
         Rate,
         RateErr,
         spureRate,
         spureRateErr,
         pureRate,
         pureRateErr,
         coMa,
         RatePerLS,
         rcs[0]->runID,
         rcs[0]->lumiSection,
         totalRatePerLS,
         RefPrescalePerLS,
         RefL1PrescalePerLS,
         weightedPrescaleRefHLT,
         weightedPrescaleRefL1,
	 CountPerLS,
	 totalCountPerLS,
	 InstLumiPerLS);

}
void calcEff(
      OHltConfig *cfg,
      OHltMenu *menu,
      vector<OHltTree*> &procs,
      vector<OHltRateCounter*> &rcs,
      OHltEffPrinter* eprint,
      float &DenEff,
      HLTDatasets &hltDatasets)
{

   const int ntrig = (int)menu->GetTriggerSize();
   vector<float> Rate, pureRate, spureRate;
   vector<float> Eff, pureEff, spureEff;
   vector<float> EffErr, pureEffErr, spureEffErr;
   vector<int> Count;
   vector< vector<float> > coMa;
   vector<float> coDen;
   //  float DenEff=0.;
   Int_t nbinpt = 50;
   Float_t ptmin = 0.0;
   Float_t ptmax = 20.0;
   Int_t nbineta = 30;
   Float_t etamin = -3.0;
   Float_t etamax = 3.0;

   vector<float> ftmp;
   for (int i=0; i<ntrig; i++)
   { // Init
      Eff.push_back(0.);
      pureEff.push_back(0.);
      spureEff.push_back(0.);
      ftmp.push_back(0.);
      EffErr.push_back(0.);
      pureEffErr.push_back(0.);
      spureEffErr.push_back(0.);
      coDen.push_back(0.);
   }
   
   for (int j=0; j<ntrig; j++)
   {
      coMa.push_back(ftmp);
   }

   for (unsigned int i=0; i<procs.size(); i++)
   {
      //     if(cfg->pnames[i]=="zee"||cfg->pnames[i]=="zmumu"){
      if (cfg->pisPhysicsSample[i]!=0)
      {

         char filename[256];
         snprintf(filename, 255, "MyEffHist_%d.root", i);
         TFile* theFile = new TFile(filename, "RECREATE");
         theFile->cd();
         cout<< "Efficiency root file created: "<<filename <<endl;

         TH1F *h1 = new TH1F("pTnum","pTnum",nbinpt,ptmin,ptmax);
         TH1F *h2 = new TH1F("pTden","pTden",nbinpt,ptmin,ptmax);
         TH1F *h3 = new TH1F("etanum","etanum",nbineta,etamin,etamax);
         TH1F *h4 = new TH1F("etaden","etaden",nbineta,etamin,etamax);
         TH1F *Eff_pt = new TH1F("eff_pt","eff_pt",nbinpt,ptmin,ptmax);
         TH1F *Eff_eta = new TH1F("eff_eta","eff_eta",nbineta,etamin,etamax);

         procs[i]->Loop(
               rcs[i],
               cfg,
               menu,
               i,
               DenEff,
               h1,
               h2,
               h3,
               h4,
               hltDatasets[i]);

         for (int j=0; j<ntrig; j++)
         {
            Eff[j] += OHltRateCounter::eff((float)rcs[i]->iCount[j], DenEff);
            EffErr[j] += OHltRateCounter::effErrb(
                  (float)rcs[i]->iCount[j],
                  DenEff);
            //cout<<j<<" Counts: "<<rcs[i]->iCount[j]<<endl;
            spureEff[j] += OHltRateCounter::eff(
                  (float)rcs[i]->sPureCount[j],
                  DenEff);
            spureEffErr[j] += OHltRateCounter::effErrb(
                  (float)rcs[i]->sPureCount[j],
                  DenEff);
            pureEff[j] += OHltRateCounter::eff(
                  (float)rcs[i]->pureCount[j],
                  DenEff);
            pureEffErr[j] += OHltRateCounter::effErrb(
                  (float)rcs[i]->pureCount[j],
                  DenEff);

            for (int k=0; k<ntrig; k++)
            {
               coMa[j][k] += ((float)rcs[i]->overlapCount[j][k])
                     * cfg->psigmas[i];
            }
            coDen[j] += ((float)rcs[i]->iCount[j] * cfg->psigmas[i]); // ovelap denominator
         }

         Eff_pt->Divide(h1, h2, 1., 1.);
         Eff_eta->Divide(h3, h4, 1., 1.);
         int nbins_pt=h1->GetNbinsX();
         for (int i=1; i<=nbins_pt; i++)
         {
            double a = h1->GetBinContent(i);
            double n = h2->GetBinContent(i);
            if (n != 0.)
               Eff_pt->SetBinError(i, sqrt( 1./n * a/n * ( 1. - a/n )) );
         }
         int nbins_eta=h3->GetNbinsX();
         for (int i=1; i<=nbins_eta; i++)
         {
            double a = h3->GetBinContent(i);
            double n = h4->GetBinContent(i);
            if (n != 0.)
               Eff_eta->SetBinError(i, sqrt( 1./n * a/n * ( 1. - a/n )) );
         }

         h1->Write();
         h2->Write();
         h3->Write();
         h4->Write();
         Eff_pt->Write();
         Eff_eta->Write();

         theFile->Close();

      }
   }

   for (int i=0; i<ntrig; i++)
   {
      for (int j=0; j<ntrig; j++)
      {
         coMa[i][j] = coMa[i][j]/coDen[i];
      }
   }

   if (DenEff!=0)
      eprint->SetupAll(
            Eff,
            EffErr,
            spureEff,
            spureEffErr,
            pureEff,
            pureEffErr,
            coMa,
            DenEff);
}

////////////////////////////////////////////////////////////////////////////////////////////
// END
////////////////////////////////////////////////////////////////////////////////////////////
