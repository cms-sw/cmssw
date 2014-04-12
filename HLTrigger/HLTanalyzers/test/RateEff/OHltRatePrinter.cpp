#include <iostream>
#include <iomanip>
#include <fstream>
#include <TMath.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <TFile.h>
#include <TString.h>
#include "OHltRatePrinter.h"
#include "OHltTree.h"
#include "OHltPileupRateFitter.h"

using namespace std;

void OHltRatePrinter::SetupAll(
      vector<float> tRate,
      vector<float> tRateErr,
      vector<float> tspureRate,
      vector<float> tspureRateErr,
      vector<float> tpureRate,
      vector<float> tpureRateErr,
      vector< vector<float> >tcoMa,
      vector < vector <float> > tRatePerLS,
      vector<int> tRunID,
      vector<int> tLumiSection,
      vector<float> tTotalRatePerLS,
      vector< vector<int> > tRefPrescalePerLS,
      vector< vector<int> > tRefL1PrescalePerLS,
      vector<float> tAverageRefPrescaleHLT,
      vector<float> tAverageRefPrescaleL1,
      vector< vector<int> > tCountPerLS,
      vector<int> tTotalCountPerLS,
      vector<double> tLumiPerLS)
{
   Rate = tRate;
   RateErr = tRateErr;
   spureRate = tspureRate;
   spureRateErr = tspureRateErr;
   pureRate = tpureRate;
   pureRateErr = tpureRateErr;
   coMa = tcoMa;
   RatePerLS = tRatePerLS;
   runID = tRunID;
   lumiSection = tLumiSection;
   totalRatePerLS = tTotalRatePerLS;
   prescaleRefPerLS = tRefPrescalePerLS;
   prescaleL1RefPerLS = tRefL1PrescalePerLS;
   averageRefPrescaleHLT = tAverageRefPrescaleHLT;
   averageRefPrescaleL1 = tAverageRefPrescaleL1;
   CountPerLS = tCountPerLS;
   totalCountPerLS = tTotalCountPerLS;
   LumiPerLS = tLumiPerLS;

   ReorderRunLS(); // reorder messed up runids/LS
}

/* ********************************************** */
// Print out rate as ascii
/* ********************************************** */
void OHltRatePrinter::printRatesASCII(OHltConfig *cfg, OHltMenu *menu)
{
   cout.setf(ios::floatfield, ios::fixed);
   cout<<setprecision(3);

   cout << "\n";
   cout << "Trigger Rates [Hz] : " << "\n";
   cout
         << "         Name                       Prescale (HLT*L1)   Indiv.          Pure   Cumulative\n";
   cout
         << "----------------------------------------------------------------------------------------------\n";

   float cumulRate = 0.;
   float cumulRateErr = 0.;
   float hltPrescaleCorrection = 1.;
   float l1PrescaleCorrection = 1.; 

   for (unsigned int i=0; i<menu->GetTriggerSize(); i++)
   {
      cumulRate += spureRate[i];
      cumulRateErr += pow(spureRateErr[i], fTwo);

      TString tempTrigSeedPrescales; 
      TString tempTrigSeeds; 
      std::map<TString, std::vector<TString> > mapL1seeds = 
	menu->GetL1SeedsOfHLTPathMap(); // mapping to all seeds  

      vector<TString> vtmp;
      vector<int> itmp;

      typedef map< TString, vector<TString> > mymap;
      for (mymap::const_iterator it = mapL1seeds.begin(); it
            != mapL1seeds.end(); ++it)
      {
         if (it->first.CompareTo(menu->GetTriggerName(i)) == 0)
         {
            vtmp = it->second;
            //cout<<it->first<<endl; 
            for (unsigned int j=0; j<it->second.size(); j++)
            {
	      itmp.push_back(menu->GetL1Prescale((it->second)[j]));
               //cout<<"\t"<<(it->second)[j]<<endl; 
            }
         }
      }

      for (unsigned int j=0; j<vtmp.size(); j++)
      {

         if (cfg->readRefPrescalesFromNtuple)
         {
            for (unsigned int k=0; k<menu->GetL1TriggerSize(); k++)
            {
               if ((menu->GetL1TriggerName(k)) == (vtmp[j]))
               {
                  if ((menu->GetL1TriggerName(k)).Contains("OpenL1_"))
                  {
                     l1PrescaleCorrection = 1.0;
                     tempTrigSeedPrescales += (itmp[j]*l1PrescaleCorrection);
                  }
                  else
                  {
		    // JH
		    // l1PrescaleCorrection = averageRefPrescaleL1[k];
		    l1PrescaleCorrection = 1.0;
		    // end JH
                     tempTrigSeedPrescales += (itmp[j]*l1PrescaleCorrection);
                  }
               }
            }
         }
         else
            tempTrigSeedPrescales += itmp[j];

         if (j<(vtmp.size()-1))
         {
            tempTrigSeedPrescales = tempTrigSeedPrescales + ", ";
         }
      }

      tempTrigSeeds = menu->GetSeedCondition(menu->GetTriggerName(i));

      if (cfg->readRefPrescalesFromNtuple)
      {
         if ((menu->GetTriggerName(i).Contains("OpenHLT_")) || (menu->GetTriggerName(i).Contains("OpenAlCa_")))
         {
            TString st = menu->GetTriggerName(i);
            TString fullhlt = st.ReplaceAll("OpenHLT", "HLT");
            fullhlt = st.ReplaceAll("OpenAlCa", "AlCa");

            // For OpenHLT emulated paths, try to "guess" the correct online prescale by matching 
            // the name of the corresponding full HLT path. If the full path doesn't exist, use 1 

            hltPrescaleCorrection = 1.0;
            for (unsigned int j=0; j<menu->GetTriggerSize(); j++)
            {
               if (menu->GetTriggerName(j) == fullhlt)
                  hltPrescaleCorrection = averageRefPrescaleHLT[j];
            }
         }
         else
            hltPrescaleCorrection = averageRefPrescaleHLT[i];
      }
      else
         hltPrescaleCorrection = menu->GetReferenceRunPrescale(i);

      //JH
      //      hltPrescaleCorrection = 1.0;
      // JH
      
      cout<<setw(50)<<menu->GetTriggerName(i)<<" (" <<setw(8)
            <<(int)(menu->GetPrescale(i) * hltPrescaleCorrection)
	  << "*" <<tempTrigSeedPrescales<<setw(8)<<")  "
	//	  << "*" << " - "<<setw(5)<<")  "
            <<setw(8)<<Rate[i]<<" +- " <<setw(7)<<RateErr[i]<<" | " <<setw(8)
            <<spureRate[i]<<" | " <<setw(8)<<cumulRate <<endl;
   }

   cumulRateErr = sqrt(cumulRateErr);
   cout << "\n";
   cout << setw(60) << "TOTAL RATE : " << setw(5) << cumulRate << " +- "
         << cumulRateErr << " Hz" << "\n";
   cout
         << "----------------------------------------------------------------------------------------------\n";

}

/* ********************************************** */
// Print out rates as twiki 
/* ********************************************** */
void OHltRatePrinter::printRatesTwiki(OHltConfig *cfg, OHltMenu *menu)
{
   if (menu->IsL1Menu())
      printL1RatesTwiki(cfg, menu);
   else
      printHltRatesTwiki(cfg, menu);

}

/* ********************************************** */
// Print out HLT rates as twiki
/* ********************************************** */
void OHltRatePrinter::printHltRatesTwiki(OHltConfig *cfg, OHltMenu *menu)
{
   TString tableFileName = GetFileName(cfg, menu);

   TString twikiFile = tableFileName + TString(".twiki");
   ofstream outFile(twikiFile.Data());
   if (!outFile)
   {
      cout<<"Error opening output file"<< endl;
   }

   outFile.setf(ios::floatfield, ios::fixed);
   outFile<<setprecision(2);

   outFile << "| *Path Name*";
   outFile << " | *L1 condition*";
   outFile << " | *L1  Prescale*";
   outFile << " | *HLT Prescale*";
   outFile << " | *HLT Rate [Hz]*";
   outFile << " | *Total Rate [Hz]*";
   outFile << " | *Avg. Size [MB]*";
   outFile << " | *Total Throughput [MB/s]* |" << endl;

   float cumulRate = 0.;
   float cumulRateErr = 0.;
   float cuThru = 0.;
   float cuThruErr = 0.;
   float physCutThru = 0.;
   float physCutThruErr = 0.;
   float cuPhysRate = 0.;
   float cuPhysRateErr = 0.;
   float hltPrescaleCorrection = 1;
   float l1PrescaleCorrection = 1.;

   for (unsigned int i=0; i<menu->GetTriggerSize(); i++)
   {
      cumulRate += spureRate[i];
      cumulRateErr += pow(spureRateErr[i], fTwo);
      cuThru += spureRate[i] * menu->GetEventsize(i);
      cuThruErr += pow(spureRateErr[i]*menu->GetEventsize(i), fTwo);

      if (!(menu->GetTriggerName(i).Contains("AlCa")))
      {
         cuPhysRate += spureRate[i];
         cuPhysRateErr += pow(spureRateErr[i], fTwo);
         physCutThru += spureRate[i]*menu->GetEventsize(i);
         physCutThruErr += pow(spureRateErr[i]*menu->GetEventsize(i), fTwo);
      }

      TString tempTrigSeedPrescales;
      TString tempTrigSeeds;
      std::map<TString, std::vector<TString> > mapL1seeds =
            menu->GetL1SeedsOfHLTPathMap(); // mapping to all seeds 

      vector<TString> vtmp;
      vector<int> itmp;

      typedef map< TString, vector<TString> > mymap;
      for (mymap::const_iterator it = mapL1seeds.begin(); it
            != mapL1seeds.end(); ++it)
      {
         if (it->first.CompareTo(menu->GetTriggerName(i)) == 0)
         {
            vtmp = it->second;
            //cout<<it->first<<endl; 
            for (unsigned int j=0; j<it->second.size(); j++)
            {
	      itmp.push_back(menu->GetL1Prescale((it->second)[j]));
               //cout<<"\t"<<(it->second)[j]<<endl; 
            }
         }
      }

      for (unsigned int j=0; j<vtmp.size(); j++)
      {

         if (cfg->readRefPrescalesFromNtuple)
         {
            for (unsigned int k=0; k<menu->GetL1TriggerSize(); k++)
            {
               if ((menu->GetL1TriggerName(k)) == (vtmp[j]))
               {
                  if ((menu->GetL1TriggerName(k)).Contains("OpenL1_"))
                  {
                     l1PrescaleCorrection = 1.0;
                     tempTrigSeedPrescales += (itmp[j]*l1PrescaleCorrection);
                  }
                  else
                  {
		    // JH
		    l1PrescaleCorrection = 1.0;
                    // l1PrescaleCorrection = averageRefPrescaleL1[k];
		    // end JH
                     tempTrigSeedPrescales += (itmp[j]*l1PrescaleCorrection);
                  }
               }
            }
         }
         else
            tempTrigSeedPrescales += itmp[j];

         if (j<(vtmp.size()-1))
         {
            tempTrigSeedPrescales = tempTrigSeedPrescales + ", ";
         }
      }

      tempTrigSeeds = menu->GetSeedCondition(menu->GetTriggerName(i));

      if (cfg->readRefPrescalesFromNtuple)
      {
         if ((menu->GetTriggerName(i).Contains("OpenHLT_")) || (menu->GetTriggerName(i).Contains("OpenAlCa_")))
         {
            TString st = menu->GetTriggerName(i);
            TString fullhlt = st.ReplaceAll("OpenHLT", "HLT");
            fullhlt = st.ReplaceAll("OpenAlCa", "AlCa");

            // For OpenHLT emulated paths, try to "guess" the correct online prescale by matching  
            // the name of the corresponding full HLT path. If the full path doesn't exist, use 1  

            hltPrescaleCorrection = 1.0;
            for (unsigned int j=0; j<menu->GetTriggerSize(); j++)
            {
               if (menu->GetTriggerName(j) == fullhlt)
                  hltPrescaleCorrection = averageRefPrescaleHLT[j];
            }
         }
         else
            hltPrescaleCorrection = averageRefPrescaleHLT[i];
      }
      else
         hltPrescaleCorrection = menu->GetReferenceRunPrescale(i);

      // JH
      //      hltPrescaleCorrection = 1.0;
      // end JH

      outFile << "| !"<< menu->GetTriggerName(i) << " | !" << tempTrigSeeds
	      << " | " << tempTrigSeedPrescales << " | "
	//	      << " | " << "-" << " | "
            << (int)(menu->GetPrescale(i) * hltPrescaleCorrection) << " | "
            << Rate[i] << "+-" << RateErr[i] << " | " << cumulRate << " | "
            << menu->GetEventsize(i) << " | " << cuThru << " | " << endl;
   }

   outFile << "| *Total* " << " | *Rate (AlCa not included) [Hz]*"
         << " | *Throughput (AlCa included) [MB/s]* |" << endl;

   outFile << "| HLT " << " | " << cuPhysRate << "+-" << sqrt(cuPhysRateErr)
         << " | " << cuThru << "+-" << sqrt(cuThruErr) << " | " << endl;

   outFile.close();

}

/* ********************************************** */
// Print out L1 rates as twiki 
/* ********************************************** */
void OHltRatePrinter::printL1RatesTwiki(OHltConfig *cfg, OHltMenu *menu)
{
   TString tableFileName = GetFileName(cfg, menu);

   TString twikiFile = tableFileName + TString(".twiki");
   ofstream outFile(twikiFile.Data());
   if (!outFile)
   {
      cout<<"Error opening output file"<< endl;
   }

   outFile.setf(ios::floatfield, ios::fixed);
   outFile<<setprecision(2);

   outFile << "| *Path Name*";
   outFile << " | *L1  Prescale*";
   outFile << " | *L1 rate [Hz]*";
   outFile << " | *Total Rate [Hz]* |" << endl;

   float cumulRate = 0.;
   float cumulRateErr = 0.;
   float hltPrescaleCorrection = 1.;
   for (unsigned int i=0; i<menu->GetTriggerSize(); i++)
   {
      cumulRate += spureRate[i];
      cumulRateErr += pow(spureRateErr[i], fTwo);

      if (cfg->readRefPrescalesFromNtuple)
         hltPrescaleCorrection = averageRefPrescaleHLT[i];
      else
         hltPrescaleCorrection = menu->GetReferenceRunPrescale(i);

      TString tempTrigName = menu->GetTriggerName(i);

      outFile << "| !" << tempTrigName << " | " << (int)(menu->GetPrescale(i)
            * hltPrescaleCorrection) << " | " << Rate[i] << "+-" << RateErr[i]
            << " | " << cumulRate << " |" << endl;
   }

   outFile << "| *Total* " << " | *Rate [Hz]* |" << endl;

   outFile << "| L1 " << " | " << cumulRate << "+-" << sqrt(cumulRateErr)
         << " | " << endl;

   outFile.close();

}

int OHltRatePrinter::ivecMax(vector<int> ivec)
{
   int max = -999999;
   for (unsigned int i=0; i<ivec.size(); i++)
   {
      if (ivec[i]>max)
         max = ivec[i];
   }
   return max;
}

int OHltRatePrinter::ivecMin(vector<int> ivec)
{
   int min = 999999999;
   for (unsigned int i=0; i<ivec.size(); i++)
   {
      if (ivec[i]<min)
         min = ivec[i];
   }
   return min;
}

/* ********************************************** */
// Fill histos
/* ********************************************** */
void OHltRatePrinter::writeHistos(OHltConfig *cfg, OHltMenu *menu)
{
   TString tableFileName = GetFileName(cfg, menu);

   TFile *fr = new TFile(tableFileName+TString(".root"),"recreate");
   fr->cd();

   int nTrig = (int)menu->GetTriggerSize();
   int nL1Trig = (int)menu->GetL1TriggerSize();
   TH1F *individual = new TH1F("individual","individual",nTrig,1,nTrig+1);
   TH1F *cumulative = new TH1F("cumulative","cumulative",nTrig,1,nTrig+1);
   TH1F *throughput = new TH1F("throughput","throughput",nTrig,1,nTrig+1);
   TH1F *eventsize = new TH1F("eventsize","eventsize",nTrig,1,nTrig+1);
   TH2F *overlap = new TH2F("overlap","overlap",nTrig,1,nTrig+1,nTrig,1,nTrig+1);
   TH1F *unique = new TH1F("unique","unique",nTrig,1,nTrig+1);

   int RunLSn = RatePerLS.size();

   int RunMin = ivecMin(runID);
   int RunMax = ivecMax(runID);
   int LSMin = ivecMin(lumiSection);
   int LSMax = ivecMax(lumiSection);

   int RunLSmin = RunMin*10000 + LSMin;
   int RunLSmax = RunMax*10000 + LSMax;

   //cout<<">>>>>>>> "<<RunLSn<<" "<<RunMin<<" "<<RunMax<<" "<<LSMin<<" "<<LSMax<<endl;

   TH2F *individualPerLS = new TH2F("individualPerLS","individualPerLS",nTrig,1,nTrig+1,
         RunLSn,RunLSmin,RunLSmax);
   TH1F *totalPerLS = new TH1F("totalPerLS","totalPerLS",RunLSn,RunLSmin,RunLSmax);
   TH2F *hltprescalePerLS = new TH2F("HLTprescalePerLS","HLTprescalePerLS",nTrig,1,nTrig+1,
         RunLSn,RunLSmin,RunLSmax);
   TH2F *l1prescalePerLS = new TH2F("L1prescalePerLS","L1prescalePerLS", nL1Trig,1,nL1Trig+1,
         RunLSn,RunLSmin,RunLSmax);
   TH2F *totalprescalePerLS = new TH2F("totalprescalePerLS","totalprescalePerLS", nTrig,1,nTrig+1,
         RunLSn,RunLSmin,RunLSmax);
   TH2F *individualCountsPerLS = new TH2F("individualCountsPerLS","individualCountsPerLS",nTrig,1,nTrig+1,
	 RunLSn,RunLSmin,RunLSmax);
   TH1F *totalCountsPerLS = new TH1F("totalCountsPerLS","totalCountsPerLS",RunLSn,RunLSmin,RunLSmax);
   TH1F *instLumiPerLS = new TH1F("instLumiPerLS","instLumiPerLS",RunLSn,RunLSmin,RunLSmax);

   float cumulRate = 0.;
   float cumulRateErr = 0.;
   float cuThru = 0.;
   float cuThruErr = 0.;
   for (unsigned int i=0; i<menu->GetTriggerSize(); i++)
   {
      cumulRate += spureRate[i];
      cumulRateErr += pow(spureRateErr[i], fTwo);
      cuThru += spureRate[i] * menu->GetEventsize(i);
      cuThruErr += pow(spureRate[i]*menu->GetEventsize(i), fTwo);

      individual->SetBinContent(i+1, Rate[i]);
      individual->GetXaxis()->SetBinLabel(i+1, menu->GetTriggerName(i));
      cumulative->SetBinContent(i+1, cumulRate);
      cumulative->GetXaxis()->SetBinLabel(i+1, menu->GetTriggerName(i));
      unique->SetBinContent(i+1, pureRate[i]); 
      unique->GetXaxis()->SetBinLabel(i+1, menu->GetTriggerName(i));

      throughput->SetBinContent(i+1, cuThru);
      throughput->GetXaxis()->SetBinLabel(i+1, menu->GetTriggerName(i));
      eventsize->SetBinContent(i+1, menu->GetEventsize(i));
      eventsize->GetXaxis()->SetBinLabel(i+1, menu->GetTriggerName(i));

      for (int j=0; j<RunLSn; j++)
      {
         individualPerLS->SetBinContent(i+1, j+1, RatePerLS[j][i]);
	 individualCountsPerLS->SetBinContent(i+1, j+1, CountPerLS[j][i]);
         TString tstr = "";
         tstr += runID[j];
         tstr = tstr + " - ";
         tstr += lumiSection[j];
         individualPerLS->GetYaxis()->SetBinLabel(j+1, tstr);
         individualPerLS->GetXaxis()->SetBinLabel(i+1, menu->GetTriggerName(i));
         individualCountsPerLS->GetYaxis()->SetBinLabel(j+1, tstr);
         individualCountsPerLS->GetXaxis()->SetBinLabel(i+1, menu->GetTriggerName(i));
      }
   }
   for (int j=0; j<RunLSn; j++)
   {
      TString tstr = "";
      tstr += runID[j];
      tstr = tstr + " - ";
      tstr += lumiSection[j];
      totalPerLS->SetBinContent(j+1, totalRatePerLS[j]);
      totalPerLS->GetXaxis()->SetBinLabel(j+1, tstr);
      totalCountsPerLS->SetBinContent(j+1, totalCountPerLS[j]);
      totalCountsPerLS->GetXaxis()->SetBinLabel(j+1, tstr);
      instLumiPerLS->SetBinContent(j+1, LumiPerLS[j]);
      instLumiPerLS->GetXaxis()->SetBinLabel(j+1, tstr);

      // L1
      for (unsigned int k=0; k<menu->GetL1TriggerSize(); k++)
      {
         l1prescalePerLS->SetBinContent(k+1, j+1, prescaleL1RefPerLS[j][k]);
         l1prescalePerLS->GetYaxis()->SetBinLabel(j+1, tstr);
         l1prescalePerLS->GetXaxis()->SetBinLabel(k+1, menu->GetL1TriggerName(k));
      }

      // HLT
      for (unsigned int i=0; i<menu->GetTriggerSize(); i++)
      {
         hltprescalePerLS->SetBinContent(i+1, j+1, prescaleRefPerLS[j][i]);
         hltprescalePerLS->GetYaxis()->SetBinLabel(j+1, tstr);
         hltprescalePerLS->GetXaxis()->SetBinLabel(i+1, menu->GetTriggerName(i));

         // HLT*L1
         std::map<TString, std::vector<TString> > mapL1seeds =
               menu->GetL1SeedsOfHLTPathMap(); // mapping to all seeds  

         vector<TString> vtmp;
         vector<int> itmp;

         typedef map< TString, vector<TString> > mymap;
         for (mymap::const_iterator it = mapL1seeds.begin(); it
               != mapL1seeds.end(); ++it)
         {
            if (it->first.CompareTo(menu->GetTriggerName(i)) == 0)
            {
               vtmp = it->second;
               if (it->second.size() > 1 || it->second.size() == 0)
               {
                  // For OR'd L1 seeds, punt
                  totalprescalePerLS->SetBinContent(i+1, j+1, -999);
                  totalprescalePerLS->GetYaxis()->SetBinLabel(j+1, tstr);
                  totalprescalePerLS->GetXaxis()->SetBinLabel(i+1, menu->GetTriggerName(i));
               }
               else
               {
                  // For a single L1 seed, loop over the map, find the online prescale, and multiply by the online HLT prescale
                  TString l1seedname = (it->second)[0];

                  for (unsigned int k=0; k<menu->GetL1TriggerSize(); k++)
                  {
                     if (l1seedname == menu->GetL1TriggerName(k))
                     {
                        totalprescalePerLS->SetBinContent(
                              i+1,
                              j+1,
                              prescaleL1RefPerLS[j][k] * prescaleRefPerLS[j][i]);
                        totalprescalePerLS->GetYaxis()->SetBinLabel(j+1, tstr);
                        totalprescalePerLS->GetXaxis()->SetBinLabel(i+1, menu->GetTriggerName(i));
                     }
                  }
               }
            }
         }
      }
   }

   for (unsigned int i=0; i<menu->GetTriggerSize(); i++)
   {
      for (unsigned int j=0; j<menu->GetTriggerSize(); j++)
      {
         overlap->SetBinContent(i+1, j+1, coMa[i][j]);
         overlap->GetXaxis()->SetBinLabel(i+1, menu->GetTriggerName(i));
         overlap->GetYaxis()->SetBinLabel(j+1, menu->GetTriggerName(j));
      }
   }

   individual->SetStats(0);
   individual->SetYTitle("Rate (Hz)");
   individual->SetTitle("Individual trigger rate");
   cumulative->SetStats(0);
   cumulative->SetYTitle("Rate (Hz)");
   cumulative->SetTitle("Cumulative trigger rate");
   overlap->SetStats(0);
   overlap->SetTitle("Overlap");
   unique->SetStats(0); 
   unique->SetYTitle("Rate (Hz)"); 
   unique->SetTitle("Unique trigger rate"); 
   individual->Write();
   cumulative->Write();
   eventsize->Write();
   throughput->Write();
   overlap->Write();
   unique->Write();
   individualPerLS->SetStats(0);
   individualPerLS->SetZTitle("Rate (Hz)");
   individualPerLS->SetTitle("Individual trigger rate vs Run/LumiSection");
   individualPerLS->Write();
   totalPerLS->SetStats(0);
   totalPerLS->SetZTitle("Rate (Hz)");
   totalPerLS->SetTitle("Total trigger rate vs Run/LumiSection");
   totalPerLS->Write();
   totalprescalePerLS->SetStats(0);
   totalprescalePerLS->SetZTitle("Prescale");
   totalprescalePerLS->SetTitle("HLT*L1 Prescale vs Run/LumiSection");
   totalprescalePerLS->Write();
   hltprescalePerLS->SetStats(0);
   hltprescalePerLS->SetZTitle("Prescale");
   hltprescalePerLS->SetTitle("HLT Prescale vs Run/LumiSection");
   hltprescalePerLS->Write();
   l1prescalePerLS->SetStats(0);
   l1prescalePerLS->SetZTitle("Prescale");
   l1prescalePerLS->SetTitle("L1 Prescale vs Run/LumiSection");
   l1prescalePerLS->Write();
   individualCountsPerLS->SetStats(0);
   individualCountsPerLS->SetZTitle("Events selected");
   individualCountsPerLS->SetTitle("Individual trigger counts vs Run/LumiSection");
   individualCountsPerLS->Write();
   totalCountsPerLS->SetStats(0);
   totalCountsPerLS->SetZTitle("Events selected");
   totalCountsPerLS->SetTitle("Total trigger counts vs Run/LumiSection");
   totalCountsPerLS->Write();
   instLumiPerLS->SetStats(0);
   instLumiPerLS->SetZTitle("Events selected");
   instLumiPerLS->SetTitle("Instantaneous lumi vs Run/LumiSection");
   instLumiPerLS->Write();


   fr->Close();
}

/* ********************************************** */
// Call pileup fitting 
/* ********************************************** */
void OHltRatePrinter::fitRatesForPileup(OHltConfig *cfg, OHltMenu *menu)
{
  TString tableFileName = GetFileName(cfg, menu);

  TFile *fr = new TFile(tableFileName+TString(".root"),"UPDATE");
  fr->cd();

  OHltPileupRateFitter* pileupfitter = new OHltPileupRateFitter();
  pileupfitter->fitForPileup(
			     cfg,
			     menu,
			     RatePerLS,
			     totalRatePerLS,
			     LumiPerLS,
			     CountPerLS,
			     totalCountPerLS,
			     fr);

  fr->Close();
}

/* ********************************************** */
// Generate basic file name
/* ********************************************** */
TString OHltRatePrinter::GetFileName(OHltConfig *cfg, OHltMenu *menu)
{
   char sLumi[10], sEnergy[10];
   snprintf(sEnergy, 10, "%1.0f", cfg->cmsEnergy);
   snprintf(sLumi,   10, "%1.1e", cfg->iLumi);

   TString menuTag;
   if (menu->IsL1Menu())
      menuTag = "l1menu_";
   else
      menuTag = "hltmenu_";

   TString tableFileName = menuTag + sEnergy + TString("TeV_") + sLumi
         + TString("_") + cfg->versionTag;
   tableFileName.ReplaceAll("+", "");

   return tableFileName;
}

/* ********************************************** */
// Print out corelation matrix
/* ********************************************** */
void OHltRatePrinter::printCorrelationASCII()
{

   for (unsigned int i=0; i<Rate.size(); i++)
   {
      for (unsigned int j=0; j<Rate.size(); j++)
      {
         cout<<"("<<i<<","<<j<<") = "<<coMa[i][j]<<endl;
      }
   }
}

/* ********************************************** */
// Print out rates as tex
/* ********************************************** */
void OHltRatePrinter::printRatesTex(OHltConfig *cfg, OHltMenu *menu)
{
   if (menu->IsL1Menu())
      printL1RatesTex(cfg, menu);
   else
      printHltRatesTex(cfg, menu);

}
/* ********************************************** */
// Print out L1 rates as tex
/* ********************************************** */
void OHltRatePrinter::printL1RatesTex(OHltConfig *cfg, OHltMenu *menu)
{
   TString tableFileName = GetFileName(cfg, menu);

   char sLumi[10], sEnergy[10];
   snprintf(sEnergy, 10, "%1.0f", cfg->cmsEnergy);
   snprintf(sLumi,   10, "%1.1e", cfg->iLumi);

   TString texFile = tableFileName + TString(".tex");
   ofstream outFile(texFile.Data());
   if (!outFile)
   {
      cout<<"Error opening output file"<< endl;
   }

   outFile <<setprecision(2);
   outFile.setf(ios::floatfield, ios::fixed);
   outFile << "\\documentclass[amsmath,amssymb]{revtex4}" << endl;
   outFile << "\\usepackage{longtable}" << endl;
   outFile << "\\usepackage{color}" << endl;
   outFile << "\\usepackage{lscape}" << endl;
   outFile << "\\begin{document}" << endl;
   outFile << "\\begin{landscape}" << endl;
   outFile
         << "\\newcommand{\\met}{\\ensuremath{E\\kern-0.6em\\lower-.1ex\\hbox{\\/}\\_T}}"
         << endl;

   outFile << "\\begin{footnotesize}" << endl;
   outFile << "\\begin{longtable}{|l|c|c|r|}" << endl;
   outFile << "\\caption[Cuts]{L1 bandwidth is 17 kHz. } \\label{CUTS} \\\\ "
         << endl;

   outFile << "\\hline \\multicolumn{4}{|c|}{\\bf \\boldmath L1 for L = "
         << sLumi << "}\\\\  \\hline" << endl;
   outFile << "{\\bf Path Name} & " << endl;
   outFile
         << "\\begin{tabular}{c} {\\bf L1} \\\\ {\\bf Prescale} \\end{tabular} & "
         << endl;
   outFile
         << "\\begin{tabular}{c} {\\bf L1 Rate} \\\\ {\\bf $[$Hz$]$} \\end{tabular} & "
         << endl;
   outFile
         << "\\begin{tabular}{c} {\\bf Total Rate} \\\\ {\\bf $[$Hz$]$} \\end{tabular} \\\\ \\hline"
         << endl;
   outFile << "\\endfirsthead " << endl;

   outFile
         << "\\multicolumn{4}{r}{\\bf \\bfseries --continued from previous page (L = "
         << sLumi << ")" << "}\\\\ \\hline " << endl;
   outFile << "{\\bf Path Name} & " << endl;
   outFile
         << "\\begin{tabular}{c} {\\bf L1} \\\\ {\\bf Prescale} \\end{tabular} & "
         << endl;
   outFile
         << "\\begin{tabular}{c} {\\bf L1 Rate} \\\\ {\\bf $[$Hz$]$} \\end{tabular} & "
         << endl;
   outFile
         << "\\begin{tabular}{c} {\\bf Total Rate} \\\\ {\\bf $[$Hz$]$} \\end{tabular} \\\\ \\hline"
         << endl;
   outFile << "\\endhead " << endl;

   outFile
         << "\\hline \\multicolumn{4}{|r|}{{Continued on next page}} \\\\ \\hline "
         << endl;
   outFile << "\\endfoot " << endl;

   outFile << "\\hline " << endl;
   outFile << "\\endlastfoot " << endl;

   float cumulRate = 0.;
   float cumulRateErr = 0.;
   float hltPrescaleCorrection = 1.;
   for (unsigned int i=0; i<menu->GetTriggerSize(); i++)
   {
      cumulRate += spureRate[i];
      cumulRateErr += pow(spureRateErr[i], fTwo);

      TString tempTrigName = menu->GetTriggerName(i);
      tempTrigName.ReplaceAll("_", "\\_");

      if (cfg->readRefPrescalesFromNtuple)
         hltPrescaleCorrection = averageRefPrescaleHLT[i];
      else
         hltPrescaleCorrection = menu->GetReferenceRunPrescale(i);

      // JH
      //      hltPrescaleCorrection = 1.0;
      // JH

      outFile << "\\color{blue}" << tempTrigName << " & "
            << (int)(menu->GetPrescale(i) * hltPrescaleCorrection) << " & "
            << Rate[i] << " {$\\pm$ " << RateErr[i] << "} & " << cumulRate
            << "\\\\" << endl;
   }

   cumulRateErr = sqrt(cumulRateErr);
   outFile
         << "\\hline \\multicolumn{2}{|l|}{\\bf \\boldmath Total L1 rate (Hz)} & \\multicolumn{2}{|r|} {\\bf "
         << cumulRate << " $\\pm$ " << cumulRateErr << "} \\\\  \\hline"
         << endl;
   outFile << "\\end{longtable}" << endl;
   outFile << "\\end{footnotesize}" << endl;
   outFile << "\\clearpage" << endl;
   outFile << "\\end{landscape}" << endl;
   outFile << "\\end{document}";
   outFile.close();
}

/* ********************************************** */
// Print out Hlt rates as tex
/* ********************************************** */
void OHltRatePrinter::printHltRatesTex(OHltConfig *cfg, OHltMenu *menu)
{
   TString tableFileName = GetFileName(cfg, menu);

   char sLumi[10], sEnergy[10];
   snprintf(sEnergy, 10, "%1.0f", cfg->cmsEnergy);
   snprintf(sLumi,   10, "%1.1e", cfg->iLumi);

   TString texFile = tableFileName + TString(".tex");
   ofstream outFile(texFile.Data());
   if (!outFile)
   {
      cout<<"Error opening output file"<< endl;
   }

   outFile <<setprecision(2);
   outFile.setf(ios::floatfield, ios::fixed);
   outFile << "\\documentclass[amsmath,amssymb]{revtex4}" << endl;
   outFile << "\\usepackage{longtable}" << endl;
   outFile << "\\usepackage{color}" << endl;
   outFile << "\\usepackage{lscape}" << endl;
   outFile << "\\begin{document}" << endl;
   outFile << "\\begin{landscape}" << endl;
   outFile
         << "\\newcommand{\\met}{\\ensuremath{E\\kern-0.6em\\lower-.1ex\\hbox{\\/}\\_T}}"
         << endl;

   outFile << "\\begin{footnotesize}" << endl;
   outFile << "\\begin{longtable}{|l|l|c|c|c|r|c|r|}" << endl;
   //  outFile << "\\caption[Cuts]{Available HLT bandwith is 150 Hz = ((1 GB/s / 3) - 100 MB/s for AlCa triggers) / 1.5 MB/event. } \\label{CUTS} \\\\ " << endl;

   outFile << "\\hline \\multicolumn{8}{|c|}{\\bf \\boldmath HLT for L = "
         << sLumi << "}\\\\  \\hline" << endl;
   outFile << "{\\bf Path Name} & " << endl;
   outFile
         << "\\begin{tabular}{c} {\\bf L1} \\\\ {\\bf Condition} \\end{tabular} & "
         << endl;
   outFile
         << "\\begin{tabular}{c} {\\bf L1} \\\\ {\\bf Prescale} \\end{tabular} & "
         << endl;
   outFile
         << "\\begin{tabular}{c} {\\bf HLT} \\\\ {\\bf Prescale} \\end{tabular} & "
         << endl;
   outFile
         << "\\begin{tabular}{c} {\\bf HLT Rate} \\\\ {\\bf $[$Hz$]$} \\end{tabular} & "
         << endl;
   outFile
         << "\\begin{tabular}{c} {\\bf Total Rate} \\\\ {\\bf $[$Hz$]$} \\end{tabular} &"
         << endl;
   outFile
         << "\\begin{tabular}{c} {\\bf Avg. Size} \\\\ {\\bf $[$MB$]$} \\end{tabular} & "
         << endl;
   outFile
         << "\\begin{tabular}{c} {\\bf Total} \\\\ {\\bf Throughput} \\\\ {\\bf $[$MB/s$]$} \\end{tabular} \\\\ \\hline"
         << endl;
   outFile << "\\endfirsthead " << endl;

   outFile
         << "\\multicolumn{8}{r}{\\bf \\bfseries --continued from previous page (L = "
         << sLumi << ")" << "}\\\\ \\hline " << endl;
   outFile << "{\\bf Path Name} & " << endl;
   outFile
         << "\\begin{tabular}{c} {\\bf L1} \\\\ {\\bf Condition} \\end{tabular} & "
         << endl;
   outFile
         << "\\begin{tabular}{c} {\\bf L1} \\\\ {\\bf Prescale} \\end{tabular} & "
         << endl;
   outFile
         << "\\begin{tabular}{c} {\\bf HLT} \\\\ {\\bf Prescale} \\end{tabular} & "
         << endl;
   outFile
         << "\\begin{tabular}{c} {\\bf HLT Rate} \\\\ {\\bf $[$Hz$]$} \\end{tabular} & "
         << endl;
   outFile
         << "\\begin{tabular}{c} {\\bf Total Rate} \\\\ {\\bf $[$Hz$]$} \\end{tabular} &"
         << endl;
   outFile
         << "\\begin{tabular}{c} {\\bf Avg. Size} \\\\ {\\bf $[$MB$]$} \\end{tabular} & "
         << endl;
   outFile
         << "\\begin{tabular}{c} {\\bf Total} \\\\ {\\bf Throughput} \\\\ {\\bf $[$MB/s$]$} \\end{tabular} \\\\ \\hline"
         << endl;
   outFile << "\\endhead " << endl;

   outFile
         << "\\hline \\multicolumn{8}{|r|}{{Continued on next page}} \\\\ \\hline "
         << endl;
   outFile << "\\endfoot " << endl;

   outFile << "\\hline " << endl;
   outFile << "\\endlastfoot " << endl;

   float cumulRate = 0.;
   float cumulRateErr = 0.;
   float cuThru = 0.;
   float cuThruErr = 0.;
   float physCutThru = 0.;
   float physCutThruErr = 0.;
   float cuPhysRate = 0.;
   float cuPhysRateErr = 0.;
   float hltPrescaleCorrection = 1.;
   vector<TString> footTrigSeedPrescales;
   vector<TString> footTrigSeeds;
   vector<TString> footTrigNames;
   for (unsigned int i=0; i<menu->GetTriggerSize(); i++)
   {
      cumulRate += spureRate[i];
      cumulRateErr += pow(spureRateErr[i], fTwo);
      cuThru += spureRate[i] * menu->GetEventsize(i);
      cuThruErr += pow(spureRateErr[i]*menu->GetEventsize(i), fTwo);

      if (!(menu->GetTriggerName(i).Contains("AlCa")))
      {
         cuPhysRate += spureRate[i];
         cuPhysRateErr += pow(spureRateErr[i], fTwo);
         physCutThru += spureRate[i]*menu->GetEventsize(i);
         physCutThruErr += pow(spureRateErr[i]*menu->GetEventsize(i), fTwo);
      }

      TString tempTrigName = menu->GetTriggerName(i);
      tempTrigName.ReplaceAll("_", "\\_");

      TString tempTrigSeedPrescales;
      TString tempTrigSeeds;
      std::map<TString, std::vector<TString> > mapL1seeds =
            menu->GetL1SeedsOfHLTPathMap(); // mapping to all seeds

      vector<TString> vtmp;
      vector<int> itmp;
      typedef map< TString, vector<TString> > mymap;
      for (mymap::const_iterator it = mapL1seeds.begin(); it
            != mapL1seeds.end(); ++it)
      {
         if (it->first.CompareTo(menu->GetTriggerName(i)) == 0)
         {
            vtmp = it->second;
            //cout<<it->first<<endl;
            for (unsigned int j=0; j<it->second.size(); j++)
            {
	      itmp.push_back(menu->GetL1Prescale((it->second)[j]));
               //cout<<"\t"<<(it->second)[j]<<endl;
            }
         }
      }
      // Faster, but crashes???:
      //vector<TString> vtmp = mapL1seeds.find(TString(menu->GetTriggerName(i)))->second;
      if (vtmp.size()>2)
      {
         for (unsigned int j=0; j<vtmp.size(); j++)
         {
            tempTrigSeeds = tempTrigSeeds + vtmp[j];
            tempTrigSeedPrescales += itmp[j];
            if (j<(vtmp.size()-1))
            {
               tempTrigSeeds = tempTrigSeeds + ", ";
               tempTrigSeedPrescales = tempTrigSeedPrescales + ", ";
            }
         }

         tempTrigSeeds.ReplaceAll("_", "\\_");
         tempTrigSeedPrescales.ReplaceAll("_", "\\_");
         footTrigSeedPrescales.push_back(tempTrigSeedPrescales);
         footTrigSeeds.push_back(tempTrigSeeds);
         TString tmpstr = menu->GetTriggerName(i);
         tmpstr.ReplaceAll("_", "\\_");
         footTrigNames.push_back(tmpstr);

         tempTrigSeeds = "List Too Long";
         tempTrigSeedPrescales = "-";
      }
      else
      {
         for (unsigned int j=0; j<vtmp.size(); j++)
         {
            tempTrigSeeds = tempTrigSeeds + vtmp[j];
            tempTrigSeedPrescales += itmp[j];
            if (j<(vtmp.size()-1))
            {
               tempTrigSeeds = tempTrigSeeds + ",";
               tempTrigSeedPrescales = tempTrigSeedPrescales + ",";
            }
         }
      }
      tempTrigSeeds.ReplaceAll("_", "\\_");
      tempTrigSeedPrescales.ReplaceAll("_", "\\_");

      if (cfg->readRefPrescalesFromNtuple)
         hltPrescaleCorrection = averageRefPrescaleHLT[i];
      else
         hltPrescaleCorrection = menu->GetReferenceRunPrescale(i);

      // JH
      //      hltPrescaleCorrection = 1.0;
      // JH

      outFile << "\\color{blue}" << tempTrigName << " & " << tempTrigSeeds
	      << " & " << tempTrigSeedPrescales << " & "
	//	      << " & " << "-" << " & "
            << (int)(menu->GetPrescale(i) * hltPrescaleCorrection) << " & "
            << Rate[i] << " {$\\pm$ " << RateErr[i] << "} & " << cumulRate
            << " & " << menu->GetEventsize(i) << " & " << cuThru << "\\\\"
            << endl;
   }

   cumulRateErr = sqrt(cumulRateErr);
   cuThruErr = sqrt(cuThruErr);
   physCutThruErr = sqrt(physCutThruErr);
   cuPhysRateErr = sqrt(cuPhysRateErr);

   outFile
         << "\\hline \\multicolumn{6}{|l|}{\\bf \\boldmath Total HLT Physics rate (Hz), AlCa triggers not included } &  \\multicolumn{2}{|r|} { \\bf "
         << cuPhysRate << " $\\pm$ " << cuPhysRateErr << "} \\\\  \\hline"
         << endl;
   outFile
         << "\\hline \\multicolumn{6}{|l|}{\\bf \\boldmath Total Physics HLT throughput (MB/s), AlCa triggers not included }  & \\multicolumn{2}{|r|} { \\bf   "
         << physCutThru<< " $\\pm$ " << physCutThruErr << "} \\\\  \\hline"
         << endl;
   outFile
         << "\\hline \\multicolumn{6}{|l|}{\\bf \\boldmath Total HLT rate (Hz) }  &  \\multicolumn{2}{|r|} { \\bf "
         << cumulRate << " $\\pm$ " << cumulRateErr << "} \\\\  \\hline"
         << endl;
   outFile
         << "\\hline \\multicolumn{6}{|l|}{\\bf \\boldmath Total HLT throughput (MB/s) } &  \\multicolumn{2}{|r|} { \\bf  "
         << cuThru << " $\\pm$ " << cuThruErr << "} \\\\  \\hline" << endl;

   // Footer for remaining L1 seeds
   for (unsigned int i=0; i<footTrigNames.size(); i++)
   {
      outFile << "\\hline  { \\begin{tabular}{p{1.6cm}} \\scriptsize "
            << footTrigNames[i]
            << "\\end{tabular} } & \\multicolumn{4}{|l|}{ \\begin{tabular}{p{7cm}} \\scriptsize "
            << footTrigSeeds[i]
            << " \\end{tabular} } & \\multicolumn{3}{|l|}{ \\begin{tabular}{p{2cm}} \\scriptsize "
            << footTrigSeedPrescales[i] << " \\end{tabular} } \\\\  \\hline"
            << endl;
   }

   outFile << "\\end{longtable}" << endl;
   outFile << "\\end{footnotesize}" << endl;
   outFile << "\\clearpage" << endl;
   outFile << "\\end{landscape}" << endl;
   outFile << "\\end{document}";
   outFile.close();
}

/* ********************************************** */
// Print out Hlt rates as text file for spreadsheet entry
/* ********************************************** */
void OHltRatePrinter::printHltRatesBocci(OHltConfig *cfg, OHltMenu *menu)
{

}

/* ********************************************** */
// Print out prescales as text file 
/* ********************************************** */
void OHltRatePrinter::printPrescalesCfg(OHltConfig *cfg, OHltMenu *menu)
{
   TString tableFileName = GetFileName(cfg, menu);

   TString txtFile = tableFileName + TString("_prescales_cffsnippet.py");
   ofstream outFile(txtFile.Data());
   if (!outFile)
   {
      cout<<"Error opening prescale output file"<< endl;
   }

   //  outFile <<setprecision(2); 
   //  outFile.setf(ios::floatfield,ios::fixed); 

   outFile << "PrescaleService = cms.Service( \"PrescaleService\"," << endl;
   outFile << "\tlvl1DefaultLabel = cms.untracked.string( \"prescale1\" ), "
         << endl;
   outFile << "\tlvl1Labels = cms.vstring( 'prescale1' )," << endl;
   outFile << "\tprescaleTable = cms.VPSet( " << endl;

   outFile << "\tcms.PSet(  pathName = cms.string( \"HLTriggerFirstPath\" ),"
         << endl;
   outFile << "\t\tprescales = cms.vuint32( 1 )" << endl;
   outFile << "\t\t)," << endl;

   float hltPrescaleCorrection = 1.;
   for (unsigned int i=0; i<menu->GetTriggerSize(); i++)
   {
      if (cfg->readRefPrescalesFromNtuple)
         hltPrescaleCorrection = averageRefPrescaleHLT[i];
      else
         hltPrescaleCorrection = menu->GetReferenceRunPrescale(i);

      // JH
      //      hltPrescaleCorrection = 1.0;
      // JH

      outFile << "\tcms.PSet(  pathName = cms.string( \""
            << menu->GetTriggerName(i) << "\" )," << endl;
      outFile << "\t\tprescales = cms.vuint32( " << (int)(menu->GetPrescale(i)
            * hltPrescaleCorrection) << " )" << endl;
      outFile << "\t\t)," << endl;
   }

   outFile << "\tcms.PSet(  pathName = cms.string( \"HLTriggerFinalPath\" ),"
         << endl;
   outFile << "\t\tprescales = cms.vuint32( 1 )" << endl;
   outFile << "\t\t)" << endl;
   outFile << "\t)" << endl;
   outFile << ")" << endl;

   outFile.close();
}

/* ********************************************** */
// Print out HLTDataset report(s)
/* ********************************************** */
void OHltRatePrinter::printHLTDatasets(
      OHltConfig *cfg,
      OHltMenu *menu,
      HLTDatasets &hltDatasets,
      TString &fullPathTableName, ///< Name for the output files. You can use this to put the output in your directory of choice (don't forget the trailing slash). Directories are automatically created as necessary.
      const Int_t significantDigits = 3 ///< Number of significant digits to report percentages in.
)
{
   //  TString tableFileName = GetFileName(cfg,menu);
   char sLumi[10];
   snprintf(sLumi, 10, "%1.1e", cfg->iLumi);
   // 	printf("OHltRatePrinter::printHLTDatasets. About to call hltDatasets.report\n"); //RR
   hltDatasets.report(sLumi, fullPathTableName+ "_PS_", significantDigits); //SAK -- prints PDF tables
   // 	printf("OHltRatePrinter::printHLTDatasets. About to call hltDatasets.write\n"); //RR
   hltDatasets.write();
   float hltPrescaleCorrection = 1.;

   printf("**************************************************************************************************************************\n");
   unsigned int HLTDSsize=hltDatasets.size();
   for (unsigned int iHLTDS=0; iHLTDS< HLTDSsize; ++iHLTDS)
   {
      unsigned int SampleDiasize=(hltDatasets.at(iHLTDS)).size();
      for (unsigned int iDataset=0; iDataset< SampleDiasize; ++iDataset)
      {
         unsigned int DSsize=hltDatasets.at(iHLTDS).at(iDataset).size();
         printf("\n");
         printf("%-60s\t%10.2lf\n", hltDatasets.at(iHLTDS).at(iDataset).name.Data(), hltDatasets.at(iHLTDS).at(iDataset).rate);
         printf("\n");
         for (unsigned int iTrigger=0; iTrigger< DSsize; ++iTrigger)
         {
            TString DStriggerName(hltDatasets.at(iHLTDS).at(iDataset).at(iTrigger).name);
            for (unsigned int i=0; i<menu->GetTriggerSize(); i++)
            {

               TString tempTrigSeedPrescales;
               TString tempTrigSeeds;
               std::map<TString, std::vector<TString> > mapL1seeds =
                     menu->GetL1SeedsOfHLTPathMap(); // mapping to all seeds 

               vector<TString> vtmp;
               vector<int> itmp;

               typedef map< TString, vector<TString> > mymap;
               for (mymap::const_iterator it = mapL1seeds.begin(); it
                     != mapL1seeds.end(); ++it)
               {
                  if (it->first.CompareTo(menu->GetTriggerName(i)) == 0)
                  {
                     vtmp = it->second;
                     //cout<<it->first<<endl; 
                     for (unsigned int j=0; j<it->second.size(); j++)
                     {
                        itmp.push_back(menu->GetL1Prescale((it->second)[j]));
                        //cout<<"\t"<<(it->second)[j]<<endl; 
                     }
                  }
               }
               for (unsigned int j=0; j<vtmp.size(); j++)
               {
                  tempTrigSeedPrescales += itmp[j];
                  if (j<(vtmp.size()-1))
                  {
                     tempTrigSeedPrescales = tempTrigSeedPrescales + ", ";
                  }
               }
               tempTrigSeeds = menu->GetSeedCondition(menu->GetTriggerName(i));

               TString iMenuTriggerName(menu->GetTriggerName(i));

               if (cfg->readRefPrescalesFromNtuple)
                  hltPrescaleCorrection = averageRefPrescaleHLT[i];
               else
                  hltPrescaleCorrection = menu->GetReferenceRunPrescale(i);

	       // JH
	       //	       hltPrescaleCorrection = 1.0;
	       // JH

               if (DStriggerName.CompareTo(iMenuTriggerName)==0)
               {
                  printf(
                        "%-40s\t%-30s\t%40s\t%10d\t%10.2lf\n",
                        (menu->GetTriggerName(i)).Data(),
                        tempTrigSeeds.Data(),
                        tempTrigSeedPrescales.Data(),
                        (int)(menu->GetPrescale(i) * hltPrescaleCorrection),
                        Rate[i]);
               }
            }
         }
      }
   }
   printf("**************************************************************************************************************************\n");

}

void OHltRatePrinter::ReorderRunLS()
{
   // apply bubblesort to reorder
   int nLS = lumiSection.size();
   //for (int i=0;i<nLS;i++) {
   //  cout<<">>>>>>>>>> "<<RunID[i]<<" "<<lumiSection[i]<<" "<<endl;
   //}
   for (int i=nLS-1; i>0; i--)
   {
      for (int j=0; j<i; j++)
      {
         if ( (runID[j] > runID[j+1]) || (runID[j] == runID[j+1]
               && lumiSection[j] > lumiSection[j+1]))
         {
            //cout<<">>>>>>> "<<runID[j]<<" "<<runID[j+1]<<" "<<endl;
            //cout<<">>>>>>> "<<lumiSection[j]<<" "<<lumiSection[j+1]<<" "<<endl;
            int swap1 = runID[j];
            runID[j] = runID[j+1];
            runID[j+1] = swap1;

            int swap2 = lumiSection[j];
            lumiSection[j] = lumiSection[j+1];
            lumiSection[j+1] = swap2;

            vector<float> swap3 = RatePerLS[j];
            RatePerLS[j] = RatePerLS[j+1];
            RatePerLS[j+1] = swap3;

            float swap4 = totalRatePerLS[j];
            totalRatePerLS[j] = totalRatePerLS[j+1];
            totalRatePerLS[j+1] = swap4;

	    int swap5 = totalCountPerLS[j];
	    totalCountPerLS[j] = totalCountPerLS[j+1];
	    totalCountPerLS[j+1] = swap5;

	    vector<int> swap6 = CountPerLS[j];
	    CountPerLS[j] = CountPerLS[j+1];
	    CountPerLS[j+1] = swap6;

	    double swap7 = LumiPerLS[j];
	    LumiPerLS[j] = LumiPerLS[j+1];
	    LumiPerLS[j+1] = swap7;
	    
            //cout<<"<<<<<< "<<runID[j]<<" "<<runID[j+1]<<" "<<endl;
            //cout<<"<<<<<< "<<lumiSection[j]<<" "<<lumiSection[j+1]<<" "<<endl;
         }
      }
   }
   //for (int i=0;i<nLS;i++) {
   //  cout<<"<<<< "<<runID[i]<<" "<<lumiSection[i]<<" "<<endl;
   //}
}
