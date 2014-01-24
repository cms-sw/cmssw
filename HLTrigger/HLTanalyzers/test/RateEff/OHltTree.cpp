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

static const size_t UnpackBxInEvent = 5;

using namespace std;

void OHltTree::Loop(
      OHltRateCounter *rc,
      OHltConfig *cfg,
      OHltMenu *menu,
      int procID,
      double &Den,
      TH1F* &h1,
      TH1F* &h2,
      TH1F* &h3,
      TH1F* &h4,
      SampleDiagnostics& primaryDatasetsDiagnostics)
{
   cout<<"Start looping on sample "<<procID<<endl;
   if (fChain == 0)
   {
      cerr<<"Error: no tree!"<<endl;
      return;
   }

   Long64_t nentries = (Long64_t)cfg->nEntries;
   Long64_t nTotEnt=fChain->GetEntries();
   if (nTotEnt <=0)
   {
      cout << "\nTrouble! Number of entries on ntuples is " << nTotEnt
            <<". Please check your input paths and fnames."
            << "\nStopping program execution." << endl;
      exit(EXIT_FAILURE);
   }
   cout<<"Succeeded initialising OHltTree. nEntries: "<< nTotEnt <<endl;

   if (cfg->nEntries <= 0)
      nentries = nTotEnt;
   else
      nentries= cfg->nEntries;

   cout<<"Entries to be processed: "<<nentries<<endl;

   // Only for experts:
   // Select certain branches to speed up code.
   // Modify only if you know what you do!
   if (cfg->doSelectBranches)
   {
      fChain->SetBranchStatus("*", kFALSE);
      fChain->SetBranchStatus("*", kFALSE);
      fChain->SetBranchStatus("MCmu*", kTRUE); // for ppMuX
      fChain->SetBranchStatus("MCel*", kTRUE); // for ppEleX
      fChain->SetBranchStatus("Run", kTRUE);
      fChain->SetBranchStatus("Event", kTRUE);
      fChain->SetBranchStatus("LumiBlock", kTRUE);
      fChain->SetBranchStatus("Bx", kTRUE);
      fChain->SetBranchStatus("Orbit", kTRUE);
      fChain->SetBranchStatus("AvgInstDelLumi", kTRUE);

      // fChain->SetBranchStatus("L1TechnicalTriggerBits",kTRUE);
      if (cfg->selectBranchL1)
      {
         fChain->SetBranchStatus("L1_*", kTRUE);
         fChain->SetBranchStatus("L1Tech_*", kTRUE);
      }
      if (cfg->selectBranchHLT)
      {
         fChain->SetBranchStatus("HLT_*", kTRUE);
         fChain->SetBranchStatus("AlCa_*", kTRUE);
      }
      if (cfg->selectBranchOpenHLT)
      {
         fChain->SetBranchStatus("*pf*", kTRUE);
         fChain->SetBranchStatus("*oh*", kTRUE);
         fChain->SetBranchStatus("*recoJet*", kTRUE);
         fChain->SetBranchStatus("*recoMet*", kTRUE);
      }
      if (cfg->selectBranchReco)
      {
         fChain->SetBranchStatus("*reco*", kTRUE);
      }
      if (cfg->selectBranchL1extra)
      {
         fChain->SetBranchStatus("*L1*", kTRUE);
      }
      if (cfg->selectBranchMC)
      {
         fChain->SetBranchStatus("*MC*", kTRUE);
      }
   }
   else
   {
      fChain->SetBranchStatus("*", kTRUE);
   }

   //   TFile*   theHistFile = new TFile("Histograms_Quarkonia.root", "RECREATE");
   //   cout<< "Histogram root file created: Histograms_Quarkonia.root"  << endl;

   //TFile *theNPVFile = new TFile("NPVFile.root", "RECREATE");

   nEventsProcessed = 0;

   double wtPU = 1.;
   double wtMC = 1.;  
   double MyWeight = 1.;
   if (cfg->isMCPUreweight == true) 
     {

       TString mcfile = cfg->MCPUfile;
       TString datafile = cfg->DataPUfile;
       TString mchisto = cfg->MCPUhisto;
       TString datahisto = cfg->DataPUhisto;

       LumiWeights_ = reweight::LumiReWeighting(std::string(mcfile), std::string(datafile), std::string(mchisto), std::string(datahisto));
     }

   //TH1F *MCPVwithPU = new TH1F("MCPVwithPU", "MCPVwithPU", 25, 0., 50.);

   for (Long64_t jentry=0; jentry<nentries; jentry++)
   {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0)
         break;
      fChain->GetEntry(jentry);

      if (jentry%cfg->nPrintStatusEvery == 0)
         cout<<"Processing entry "<<jentry<<"/"<<nentries<<"\r"<<flush<<endl;

      // When running on real data, keep track of how many LumiSections have been 
      // used. Note: this does not require LumiSections be contiguous, but does assume
      // that the user uses complete LumiSections
      if (menu->IsRealData())
      {

         if (cfg->runLumiblockList.size()>0)
         {
            if (!isInRunLumiblockList(Run, LumiBlock, cfg->runLumiblockList))
               continue;
         }

         currentLumiSection = LumiBlock;
         if (currentLumiSection != previousLumiSection)
         {
            cout<<"Run, LS: "<<Run<<" "<<LumiBlock<<endl;
            if (rc->isNewRunLS(Run, LumiBlock))
            { // check against double counting
	       rc->addRunLS(Run, LumiBlock, AvgInstDelLumi);
               nLumiSections++;

               // JH - Track per-LS changes in prescales
               for (int it = 0; it < nTrig; it++)
               {
                  rc->updateRunLSRefPrescale(
                        Run,
                        LumiBlock,
                        it,
                        readRefPrescaleFromFile(menu->GetTriggerName(it)));
               }
               for (int it = 0; it < nL1Trig; it++)
               {
                  rc->updateRunLSRefL1Prescale(
                        Run,
                        LumiBlock,
                        it,
                        readRefPrescaleFromFile(menu->GetL1TriggerName(it)));
               }
            }
         }

         previousLumiSection = currentLumiSection;
      }

      // Do operations on L1extra quantities before calling SetopenL1Bits, 
      // so that they can be used in emulation of new L1 triggers, or later 
      // by HLT paths
      SetL1MuonQuality();
      RemoveEGOverlaps();

      // Do emulations of any new L1 bits. This is done before calling 
      // ApplyL1Prescales and looping over the HLT paths, so that 
      // L1 prescales are applied coherently to real L1 bits and 
      // "OpenL1" bits
      SetOpenL1Bits();

      // ccla example to extract timing info from L1Tech_XXX_5bx bits
      // if (L1Tech_BSC_minBias_OR_v0_5bx >0){
      //   cout << "L1Tech_BSC_minBias_OR_v0_5bx: " << L1Tech_BSC_minBias_OR_v0_5bx 
      //        << " Unpacked: " ;
      //   for (unsigned int i = 0; i<UnpackBxInEvent ; ++i){
      //     bool bitOn=L1Tech_BSC_minBias_OR_v0_5bx & (1 << i);
      //     std::cout << bitOn << " ";
      //   }
      //   cout << "\n";
      // }

      // 1. Loop to check which Bit fired
      // Triggernames are assigned to trigger cuts in unambigous way!
      // If you define a new trigger also define a new unambigous name!
      if (menu->DoL1preLoop() && menu->IsHltMenu())
      {
         ApplyL1Prescales(menu, cfg, rc);
      }

      //SetMapL1BitOfStandardHLTPath(menu);
      SetMapL1BitOfStandardHLTPathUsingLogicParser(menu, nEventsProcessed);

      // Apply prefilter based on bits
      bool passesPreFilter = passPreFilterLogicParser(
            cfg->preFilterLogicString,
            nEventsProcessed);
      if (!passesPreFilter)
      {
         //cout<<"Event rejected due to prefilter!!!"<<endl;
         nEventsProcessed++;
         continue;
      }

      if (cfg->pdomucuts[procID] && MCmu3!=0)
         continue;
      if (cfg->pdoecuts[procID] && MCel3!=0)
         continue;

      //////////////////////////////////////////////////////////////////
      // Get Denominator (normalization and acc. definition) for efficiency evaluation
      //////////////////////////////////////////////////////////////////

      //     if (cfg->pnames[procID]=="zee"||cfg->pnames[procID]=="zmumu"){
      if (cfg->pisPhysicsSample[procID]!=0)
      {
         int accMCMu=0;
         int accMCEle=0;
         int accMCPi=0;
         if (cfg->selectBranchMC)
         {
            for (int iMCpart = 0; iMCpart < NMCpart; iMCpart ++)
            {
               if ((MCpid[iMCpart]==13||MCpid[iMCpart]==-13)
                     && MCstatus[iMCpart]==1 && (MCeta[iMCpart] < 2.1
                     && MCeta[iMCpart] > -2.1) && (MCpt[iMCpart]>3))
                  accMCMu=accMCMu+1;
               if ((MCpid[iMCpart]==11||MCpid[iMCpart]==-11 )
                     && MCstatus[iMCpart]==1 && (MCeta[iMCpart] < 2.5
                     && MCeta[iMCpart] > -2.5) && (MCpt[iMCpart]>5))
                  accMCEle=accMCEle+1;
               if ((MCpid[iMCpart]==211||MCpid[iMCpart]==-211 )
                     && MCstatus[iMCpart]==1 && (MCeta[iMCpart] < 2.5
                     && MCeta[iMCpart] > -2.5) && (MCpt[iMCpart]>0))
                  accMCPi=accMCPi+1;
            }
            if ((cfg->pisPhysicsSample[procID]==1 && accMCEle>=1 ))
            {
               Den=Den+1;
            }
            else if ((cfg->pisPhysicsSample[procID]==2 && accMCMu >=1))
            {
               Den=Den+1;
            }
            else if ((cfg->pisPhysicsSample[procID]==3 && accMCEle>=1
                  && accMCMu >=1))
            {
               Den=Den+1;
            }
            else if ((cfg->pisPhysicsSample[procID]==5 && accMCPi>=1 ))
            {
               Den=Den+1;
            }
            else
            {
               continue;
            }
         }
      }

      // Get PU weight
      if (cfg->isMCPUreweight == true) 
	{
	  MyWeight = LumiWeights_.weight( recoNVrt );
	  //MCPVwithPU->Fill(recoNVrt, MyWeight);
	}
      //////////////////////////////////////////////////////////////////
      // Loop over trigger paths and do rate counting
      //////////////////////////////////////////////////////////////////
      for (int i = 0; i < nTrig; i++)
      {
         triggerBit[i] = false;
         previousBitsFired[i] = false;
         allOtherBitsFired[i] = false;

         //////////////////////////////////////////////////////////////////
         // Standard paths
         TString st = menu->GetTriggerName(i);
         if (st.BeginsWith("HLT_") || st.BeginsWith("L1_")
               || st.BeginsWith("L1Tech_") || st.BeginsWith("AlCa_")
               || st.BeginsWith("OpenL1_") )
         {
            // Prefixes reserved for Standard HLT&L1	
	   if (map_L1BitOfStandardHLTPath.find(st)->second>0)
            {
               if (prescaleResponse(menu, cfg, rc, i))
               {
                  if ( (map_BitOfStandardHLTPath.find(st)->second==1))
                  {
                     triggerBit[i] = true;
                  }
               }
            }
         }
         else
         {
            CheckOpenHlt(cfg, menu, rc, i);
         }
      }
      primaryDatasetsDiagnostics.fill(triggerBit); //SAK -- record primary datasets decisions

      /* ******************************** */
      // 2. Loop to check overlaps
      for (int it = 0; it < nTrig; it++)
      {
         if (triggerBit[it])
         {  
	    if (cfg->isMCPUreweight == true) wtPU = MyWeight;
	    if (not MCWeight == 0) wtMC = MCWeight;

            rc->iCount[it] = rc->iCount[it] + (1 * wtPU * wtMC);
            rc->incrRunLSCount(Run, LumiBlock, it); // for per LS rates!
            for (int it2 = 0; it2 < nTrig; it2++)
            {
               if (triggerBit[it2])
               {
                  rc->overlapCount[it][it2] = rc->overlapCount[it][it2] + (1 * wtPU * wtMC);
                  if (it2<it)
                     previousBitsFired[it] = true;
                  if (it2!=it)
                     allOtherBitsFired[it] = true;
               }
            }
            if (not previousBitsFired[it])
            { 
               rc->sPureCount[it] = rc->sPureCount[it] + (1 * wtPU * wtMC);
               rc->incrRunLSTotCount(Run,LumiBlock); // for per LS rates!	  
            }
            if (not allOtherBitsFired[it])
	    {   
	       rc->pureCount[it] = rc->pureCount[it] + (1 * wtPU * wtMC);
	    }   
         }
      }
      /* ******************************** */

      nEventsProcessed++;
   }

   //   theHistFile->cd();

   //   for(int ihistIdx=0;ihistIdx<10;ihistIdx++){

   //      hPixCanddr[ihistIdx]->Write();
   //      hPixCandEta[ihistIdx]->Write();
   //      hPixCandPt[ihistIdx]->Write();
   //      hPixCandP[ihistIdx]->Write();

   //      for(int iTrk=0;iTrk<2;iTrk++){

   //          hNCand[ihistIdx][iTrk]->Write();

   //          for(int i=0;i<2;i++){
   //              hEta[ihistIdx][iTrk][i]->Write();
   //              hPt[ihistIdx][iTrk][i]->Write();
   //              hHits[ihistIdx][iTrk][i]->Write();
   //              hNormChi2[ihistIdx][iTrk][i]->Write();
   //              hDxy[ihistIdx][iTrk][i]->Write();
   //              hDz[ihistIdx][iTrk][i]->Write();
   //              hP[ihistIdx][iTrk][i]->Write();
   //              hP[ihistIdx][iTrk][i]->Write();
   //          }

   //          for(int j=0;j<4;j++){

   //              //if(iTrk==0) continue;
   //              hOniaEta[ihistIdx][iTrk][j]->Write();
   //              hOniaRap[ihistIdx][iTrk][j]->Write();
   //              hOniaPt[ihistIdx][iTrk][j]->Write();
   //              hOniaP[ihistIdx][iTrk][j]->Write();
   //              hOniaMass[ihistIdx][iTrk][j]->Write();
   //              hOniaEtaPt[ihistIdx][iTrk][j]->Write();
   //              hOniaRapP[ihistIdx][iTrk][j]->Write();
   //           }
   //       }
   //   }
   //   theHistFile->Close();
   //theNPVFile->cd();
   //MCPVwithPU->Write();
   //theNPVFile->Close();

}

void OHltTree::SetLogicParser(std::string l1SeedsLogicalExpression)
{

   //if (l1SeedsLogicalExpression != "") {

   //std::cout<<"@@@ L1 condition: "<<l1SeedsLogicalExpression<<std::endl;
   // check also the logical expression - add/remove spaces if needed
   m_l1AlgoLogicParser.push_back(new L1GtLogicParser(l1SeedsLogicalExpression));

   //}
}

int OHltTree::readRefPrescaleFromFile(TString st)
{
   return map_RefPrescaleOfStandardHLTPath.find(st)->second;
}

bool OHltTree::prescaleResponse(
      OHltMenu *menu,
      OHltConfig *cfg,
      OHltRateCounter *rc,
      int i)
{
   if (cfg->doDeterministicPrescale)
   {
      (rc->prescaleCount[i])++;
      if (cfg->useNonIntegerPrescales)
      {
         float prescalemod = 1.0 - fmod((float)(menu->GetPrescale(i)), 1);
         if (prescalemod == 1.0)
            prescalemod = 0.5;
         return (fmod(
               (float)(rc->prescaleCount[i]),
               (float)(menu->GetPrescale(i))) <= prescalemod);
      }
      else
         return (fmod(
               (float)(rc->prescaleCount[i]),
               (float)(menu->GetPrescale(i))) == 0);
   }
   else
   {
      float therandom = (float)(GetFloatRandom());
      if (cfg->useNonIntegerPrescales)
      {
         float prescalemod = 1.0 - fmod((float)(menu->GetPrescale(i)), 1);
         if (prescalemod == 1.0)
            prescalemod = 0.5;
         return (fmod(therandom, (float)(menu->GetPrescale(i))) <= prescalemod);
      }
      else
         return (fmod((float)(GetIntRandom()), (float)(menu->GetPrescale(i)))
               == 0);
   }
}

bool OHltTree::prescaleResponseL1(
      OHltMenu *menu,
      OHltConfig *cfg,
      OHltRateCounter *rc,
      int i)
{
   if (cfg->doDeterministicPrescale)
   {
      (rc->prescaleCountL1[i])++;
      if (cfg->useNonIntegerPrescales)
      {
         float prescalemod = 1.0 - fmod((float)(menu->GetL1Prescale(i)), 1);
         if (prescalemod == 1.0)
            prescalemod = 0.5;
         return (fmod(
               (float)(rc->prescaleCountL1[i]),
               (float)(menu->GetL1Prescale(i))) <= prescalemod);
      }
      else
         return (fmod(
               (float)(rc->prescaleCountL1[i]),
               (float)(menu->GetL1Prescale(i))) == 0);
   }
   else
   {
      if (cfg->useNonIntegerPrescales)
      {
         float prescalemod = 1.0 - fmod((float)(menu->GetL1Prescale(i)), 1);
         if (prescalemod == 1.0)
            prescalemod = 0.5;
         return (fmod(
               (float)(GetFloatRandom()),
               (float)(menu->GetL1Prescale(i))) <= prescalemod);
      }
      else
         return (fmod((float)(GetIntRandom()), (float)(menu->GetL1Prescale(i)))
               == 0);
   }
}

bool OHltTree::isInRunLumiblockList(
      int run,
      int lumiBlock,
      vector < vector <int> > list)
{
   unsigned int nrunLumiList = list.size();
   if (nrunLumiList>0)
   {
      for (unsigned int i=0; i<nrunLumiList; i++)
      {
         if (run == list[i][0] && lumiBlock >= list[i][1] && lumiBlock
               <= list[i][2])
         {
            return true;
         }
      }
   }

   //cout << "Skip event, it is not in runLumiblockList: "<<run<<" "<<lumiBlock<<endl;
   return false;
}
