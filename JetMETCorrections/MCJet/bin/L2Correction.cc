#include <iostream>
#include <iomanip>
#include <string.h>
#include <fstream>
#include <cmath>
#include <TFile.h>
#include <TH1F.h>
#include <TF1.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TMath.h>
#include <TKey.h>
#include <TList.h>
#include "Utilities.h"

using namespace std;

int main(int argc, char**argv)
{
  CommandLine c1;
  c1.parse(argc,argv);

  std::string HistoFilename           = c1.getValue<string>("HistoFilename");
  std::string FitterFilename          = c1.getValue<string>("FitterFilename");
  std::string L3ResponseTxtFilename   = c1.getValue<string>("L3ResponseTxtFilename");
  std::string L3CorrectionTxtFilename = c1.getValue<string>("L3CorrectionTxtFilename");
  std::string L3OutputROOTFilename    = c1.getValue<string>("L3OutputROOTFilename");
  std::string L2CorrectionTxtFilename = c1.getValue<string>("L2CorrectionTxtFilename");
  std::string L2OutputROOTFilename    = c1.getValue<string>("L2OutputROOTFilename");
  std::vector<double> pt_vec          = c1.getVector<double>("RefPtBoundaries");
  std::vector<double> eta_vec         = c1.getVector<double>("EtaBoundaries");
  if (!c1.check()) return 0; 
  c1.print();
  /////////////////////////////////////////////////////////////////////////
  const int MAX_NETA = 83;
  const int MAX_NREFPTBINS = 30; 
  const int NCaloPtValues = 18;
  int NRefPtBins = pt_vec.size()-1;
  int NETA = eta_vec.size()-1; 
  if (NETA<2) return(0);
  int i,auxi,etabin,ptbin;
  char name[100],func[1024];
  double cor,e_cor;
  double MinCaloPt[MAX_NETA],MaxCaloPt[MAX_NETA];
  double correction_x[MAX_NREFPTBINS],correction_y[MAX_NREFPTBINS],correction_ex[MAX_NREFPTBINS],correction_ey[MAX_NREFPTBINS];
  double cor_rel[MAX_NREFPTBINS],ref_pt,calo_pt,control_pt;
  double L3_resp[5],L2_cor[10];
  TKey *key;
  TFile *inf;
  TFile *outf;
  TH1F *hcorrection[MAX_NREFPTBINS];
  TH1F *h;   
  TF1 *Correction[MAX_NETA]; 
  TF1 *L2Correction[MAX_NETA];
  TF1 *L3Response;
  std::ifstream L3ResponseFile;
  std::ofstream L2File;
  TGraph *g_L2Correction[MAX_NETA];
  TGraphErrors *g_EtaCorrection[MAX_NETA];
  double aux_CaloPt[NCaloPtValues] = {10,15,20,30,40,50,75,100,150,200,300,400,500,750,1000,1500,2000,3000};
  std::vector<std::string> HistoNamesList;
  inf = new TFile(FitterFilename.c_str(),"r");
  if (inf->IsZombie()) return(0);
  TIter next(inf->GetListOfKeys());
  while ((key = (TKey*)next()))
    HistoNamesList.push_back(key->GetName());
  //////////////////// Reading the L3 Response //////////////////////////////// 
  L3ResponseFile.open(L3ResponseTxtFilename.c_str());
  L3ResponseFile>>L3_resp[0]>>L3_resp[1]>>L3_resp[2]>>L3_resp[3]>>L3_resp[4];
  L3Response = new TF1("L3Response","[0]-[1]/(pow(log10(x),[2])+[3])+[4]/x",pt_vec[0],pt_vec[NRefPtBins-1]);
  for(i=0;i<5;i++)
    L3Response->SetParameter(i,L3_resp[i]);
  L3ResponseFile.close();
  /////////////////// Calculating the L2 Correction //////////////////////////////
  for (i=0;i<NRefPtBins;i++) 
   {
     sprintf(name,"Correction_vs_Eta_RefPt%d",i);
     if (!HistoExists(HistoNamesList,name)) return(0);
     hcorrection[i] = (TH1F*)inf->Get(name);
   }
  for (etabin=0;etabin<NETA;etabin++)
   {
     sprintf(name,"MeanCaloPt_Eta%d",etabin);
     if (!HistoExists(HistoNamesList,name)) return(0);
     h = (TH1F*)inf->Get(name);
     ///////////// Absolute correction calculation for every eta bin  //////////  
     auxi = 0;
     for (ptbin=0;ptbin<NRefPtBins;ptbin++)
       { 
         cor = hcorrection[ptbin]->GetBinContent(etabin+1);
         e_cor = hcorrection[ptbin]->GetBinError(etabin+1);
         if (cor>0 && e_cor>0.0001 && e_cor<0.3)
           {
             correction_x[auxi] = h->GetBinContent(ptbin+1);//average CaloPt for the eta bin
             correction_ex[auxi] = 0.;
             correction_y[auxi] = cor;
             correction_ey[auxi] = e_cor;
             auxi++;
           }
       }
     sprintf(name,"Correction%d",etabin);
     if (auxi>1)
       { 
         MaxCaloPt[etabin]=correction_x[auxi-1];
         MinCaloPt[etabin]=correction_x[1];
         if (auxi>10)
           sprintf(func,"[0]+[1]/(pow(log10(x),[2])+[3])");
         else
           sprintf(func,"[0]+[1]*log10(x)+[2]*pow(log10(x),2)");
         Correction[etabin] = new TF1(name,func,MinCaloPt[etabin],MaxCaloPt[etabin]);      
       }
     else
       {
         std::cout<<name<<": not enough points"<<std::endl;
         sprintf(func,"[0]");
         correction_x[0] = 10;
         correction_x[1] = 100;
         correction_ex[0] = 0.;
         correction_ex[1] = 0.;
         correction_y[0] = 1.;
         correction_y[1] = 1.;
         correction_ey[0] = 0.;
         correction_ey[1] = 0.;
         auxi = 2;
         Correction[etabin] = new TF1(name,func,10,100);
         MaxCaloPt[etabin] = 0;
         MinCaloPt[etabin] = 0;
       }
     g_EtaCorrection[etabin] = new TGraphErrors(auxi,correction_x,correction_y,correction_ex,correction_ey);
     Correction[etabin]->SetParameter(0,0.);
     Correction[etabin]->SetParameter(1,0.);
     Correction[etabin]->SetParameter(2,0.);
     Correction[etabin]->SetParameter(3,0.);
     g_EtaCorrection[etabin]->Fit(name,"RQ");
     std::cout<<name<<" fitted....."<<std::endl;
     ///////////// L2 Relative correction calculation for every eta bin /////////
     auxi = 0;
     for(ptbin=0;ptbin<NCaloPtValues;ptbin++)
       {
         calo_pt = aux_CaloPt[ptbin];
         if (calo_pt>=MinCaloPt[etabin] && calo_pt<=MaxCaloPt[etabin])
           {
             ref_pt = calo_pt*Correction[etabin]->Eval(calo_pt);
	     control_pt = ref_pt*(L3Response->Eval(ref_pt));
             cor_rel[auxi] = control_pt/calo_pt;
	     correction_x[auxi] = calo_pt;
             auxi++;
           }
       }
     sprintf(name,"L2Correction%d",etabin); 
     if (auxi>=2)
       {
         sprintf(func,"[0]+[1]*log10(x)+[2]*pow(log10(x),2)");
         if (auxi==2)
           sprintf(func,"[0]+[1]*log10(x)");
       }
     else
       {
         sprintf(func,"[0]");
         correction_x[0] = 10;
         correction_x[1] = 100;
         cor_rel[0] = 1.;
         cor_rel[1] = 1.;
         auxi = 2;
       }
     g_L2Correction[etabin] = new TGraph(auxi,correction_x,cor_rel); 
     L2Correction[etabin] = new TF1(name,func,correction_x[0],correction_x[auxi-1]);
     L2Correction[etabin]->SetParameter(0,0.);
     L2Correction[etabin]->SetParameter(1,0.);
     L2Correction[etabin]->SetParameter(2,0.);
     L2Correction[etabin]->SetParameter(3,0.);
     L2Correction[etabin]->SetParameter(4,0.);
     L2Correction[etabin]->SetParameter(5,0.);
     g_L2Correction[etabin]->Fit(name,"RQ");
     std::cout<<name<<" fitted....."<<std::endl;        
   }//end of eta bin loop  
  //////////////////////// Writing //////////////////////////////
  L2File.open(L2CorrectionTxtFilename.c_str());
  L2File.setf(ios::right);
  for(etabin=0;etabin<NETA;etabin++)
   {
     for(i=0;i<6;i++)
       L2_cor[i] = L2Correction[etabin]->GetParameter(i);
        L2File << setw(11) << eta_vec[etabin]
               << setw(11) << eta_vec[etabin+1]
               << setw(11) << (int)8
               << setw(12) << MinCaloPt[etabin]
               << setw(12) << MaxCaloPt[etabin]
               << setw(13) << L2_cor[0]
               << setw(13) << L2_cor[1]
               << setw(13) << L2_cor[2]
               << setw(13) << L2_cor[3]
               << setw(13) << L2_cor[4]
               << setw(13) << L2_cor[5]
               << "\n";
   } 
  L2File.close();
  std::cout<<L2CorrectionTxtFilename<<" written...."<<std::endl;
  outf = new TFile(L2OutputROOTFilename.c_str(),"RECREATE"); 
  for(etabin=0;etabin<NETA;etabin++)
    {
      sprintf(name,"Correction_EtaBin%d",etabin);
      g_EtaCorrection[etabin]->Write(name);
      sprintf(name,"L2Correction_EtaBin%d",etabin);
      g_L2Correction[etabin]->Write(name);
    }
  outf->Close(); 
}
