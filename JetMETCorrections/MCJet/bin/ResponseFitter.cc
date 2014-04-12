#include <iostream>
#include <string.h>
#include <fstream>
#include <cmath>
#include <TFile.h>
#include <TH1F.h>
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
  bool UseRatioForResponse       = c1.getValue<bool>("UseRatioForResponse");
  std::vector<double> pt_vec          = c1.getVector<double>("RefPtBoundaries");
  std::vector<double> eta_vec         = c1.getVector<double>("EtaBoundaries");
  if (!c1.check()) return 0; 
  c1.print();
  /////////////////////////////////////////////////////////////////////////
  const int MAX_NETA = 83;
  const int MAX_NREFPTBINS = 30; 
  int NRefPtBins = pt_vec.size()-1;
  int NETA = eta_vec.size()-1; 
  char name[100];
  int j,etabin;
  double e,mR,eR,sR,seR,mRBarrel,eRBarrel,sRBarrel,seRBarrel,r,c;
  double mCaloPt,eCaloPt,sCaloPt,mRefPt,eRefPt,sRefPt;
  double mRefPtEtaBin,eRefPtEtaBin,sRefPtEtaBin,mCaloPtEtaBin,eCaloPtEtaBin,sCaloPtEtaBin;
  double EtaBoundaries[MAX_NETA],RefPtBoundaries[MAX_NREFPTBINS];
  std::vector<std::string> HistoNamesList; 
  for(j=0;j<=NRefPtBins;j++)
    RefPtBoundaries[j] = pt_vec[j];
  for(j=0;j<=NETA;j++)
    EtaBoundaries[j] = eta_vec[j]; 
  TFile *inf;//Input file containing the response histograms.
  TFile *outf;//Output file containing the fitter results. 
  TH1F *BarrelResponse;//Histogram with the barrel response in RefPt bins.
  TH1F *BarrelCorrection;//Histogram with the barrel correction in RefPt bins.
  TH1F *MeanRefPt_Barrel;//Histogram with the barrel average RefPt in RefPt bins.
  TH1F *MeanCaloPt_Barrel;//Histogram with the barrel average CaloPt in RefPt bins.
  TH1F *MeanRefPt_EtaBin[MAX_NETA];//Histograms with the average RefPt in Eta & RefPt bins. 
  TH1F *MeanCaloPt_EtaBin[MAX_NETA];//Histograms with the average CaloPt in Eta & RefPt bins. 
  TH1F *ResponseVsEta_RefPt[MAX_NREFPTBINS];//Histograms with the average response vs Eta in RefPt bins.
  TH1F *CorrectionVsEta_RefPt[MAX_NREFPTBINS];//Histograms with the average correction vs Eta in RefPt bins.   
  TH1F *h;//Auxilary histogram; 
  TKey *key;
  //////////////////////////////////////////////////////////////////////////
  inf = new TFile(HistoFilename.c_str(),"r");
  if (inf->IsZombie()) return(0);
  TIter next(inf->GetListOfKeys());
  while ((key = (TKey*)next()))
    HistoNamesList.push_back(key->GetName());
  outf = new TFile(FitterFilename.c_str(),"RECREATE");
  TDirectory *dir_Response = (TDirectory*)outf->mkdir("FittedHistograms");//Directory in output file to store the fitted histograms.
  BarrelResponse = new TH1F("Response","Response",NRefPtBins,RefPtBoundaries);
  BarrelCorrection = new TH1F("Correction","Correction",NRefPtBins,RefPtBoundaries);
  MeanRefPt_Barrel = new TH1F("MeanRefPt","MeanRefPt",NRefPtBins,RefPtBoundaries); 
  MeanCaloPt_Barrel = new TH1F("MeanCaloPt","MeanCaloPt",NRefPtBins,RefPtBoundaries);
  if (NETA>1)//multiple eta bins: used for L2+L3 correction calculation
    {
      std::cout<<"************* Fitting Response Histograms in multiple Eta bins. ************"<<std::endl;     
      for(etabin=0;etabin<NETA;etabin++)
       { 
         sprintf(name,"MeanRefPt_Eta%d",etabin);
         MeanRefPt_EtaBin[etabin] = new TH1F(name,name,NRefPtBins,RefPtBoundaries);
         sprintf(name,"MeanCaloPt_Eta%d",etabin);
         MeanCaloPt_EtaBin[etabin] = new TH1F(name,name,NRefPtBins,RefPtBoundaries);
       }
      for (j=0; j<NRefPtBins; j++)//loop over RefPt bins
        {  
          std::cout<<"RefJetPt Bin: ["<<RefPtBoundaries[j]<<","<<RefPtBoundaries[j+1]<<"] GeV"<<std::endl;
          sprintf(name,"ptRef_RefPt%d_Barrel",j);
          if (!HistoExists(HistoNamesList,name)) return(0);
          h = (TH1F*)inf->Get(name);
          GetMEAN(h,mRefPt,eRefPt,sRefPt);
          sprintf(name,"ptCalo_RefPt%d_Barrel",j);
          if (!HistoExists(HistoNamesList,name)) return(0); 
          h = (TH1F*)inf->Get(name);
          GetMEAN(h,mCaloPt,eCaloPt,sCaloPt);
          sprintf(name,"Response_RefPt%d_Barrel",j);
          if (!HistoExists(HistoNamesList,name)) return(0);
          h = (TH1F*)inf->Get(name);
          GetMPV(name,h,dir_Response,mRBarrel,eRBarrel,sRBarrel,seRBarrel);  
          ///////////////// RefPt in barrel ///////////////////////////////////
          MeanRefPt_Barrel->SetBinContent(j+1,mRefPt);
          MeanRefPt_Barrel->SetBinError(j+1,eRefPt);
          ///////////////// CaloPt in barrel ///////////////////////////////////
          MeanCaloPt_Barrel->SetBinContent(j+1,mCaloPt);
          MeanCaloPt_Barrel->SetBinError(j+1,eCaloPt);
          ////////////////// Absolute response in barrel //////////////////////////
          CalculateResponse(UseRatioForResponse,mRefPt,eRefPt,mRBarrel,eRBarrel,r,e);
          BarrelResponse->SetBinContent(j+1,r);
          BarrelResponse->SetBinError(j+1,e);
          ////////////////// Absolute correction in barrel //////////////////////////
          CalculateCorrection(UseRatioForResponse,mRefPt,eRefPt,mRBarrel,eRBarrel,c,e);
          BarrelCorrection->SetBinContent(j+1,c);
          BarrelCorrection->SetBinError(j+1,e);
          ////////////////// Eta bins /////////////////////////////////////
          sprintf(name,"Response_vs_Eta_RefPt%d",j);
          ResponseVsEta_RefPt[j] = new TH1F(name,name,NETA,EtaBoundaries);
          sprintf(name,"Correction_vs_Eta_RefPt%d",j);
          CorrectionVsEta_RefPt[j] = new TH1F(name,name,NETA,EtaBoundaries);
          for(etabin=0;etabin<NETA;etabin++)//loop over eta bins
            {
              ///////////////////////////////////////////////////////////////
              sprintf(name,"Response_RefPt%d_Eta%d",j,etabin);
              if (!HistoExists(HistoNamesList,name)) return(0);
              h = (TH1F*)inf->Get(name);
              GetMPV(name,h,dir_Response,mR,eR,sR,seR);
              sprintf(name,"ptRef_RefPt%d_Eta%d",j,etabin);
              if (!HistoExists(HistoNamesList,name)) return(0); 
              h = (TH1F*)inf->Get(name);
              GetMEAN(h,mRefPtEtaBin,eRefPtEtaBin,sRefPtEtaBin);
              sprintf(name,"ptCalo_RefPt%d_Eta%d",j,etabin);
              if (!HistoExists(HistoNamesList,name)) return(0);
              h = (TH1F*)inf->Get(name);
              GetMEAN(h,mCaloPtEtaBin,eCaloPtEtaBin,sCaloPtEtaBin);
              ///////////////// RefPt in etabin ///////////////////////////////////
              MeanRefPt_EtaBin[etabin]->SetBinContent(j+1,mRefPtEtaBin);
              MeanRefPt_EtaBin[etabin]->SetBinError(j+1,eRefPtEtaBin);
              ///////////////// CaloPt in etabin ///////////////////////////////////
              MeanCaloPt_EtaBin[etabin]->SetBinContent(j+1,mCaloPtEtaBin);
              MeanCaloPt_EtaBin[etabin]->SetBinError(j+1,eCaloPtEtaBin);
              ////////////////// Absolute response in etabin ////////////////////////// 
              CalculateResponse(UseRatioForResponse,mRefPtEtaBin,eRefPtEtaBin,mR,eR,r,e);
              ResponseVsEta_RefPt[j]->SetBinContent(etabin+1,r);
              ResponseVsEta_RefPt[j]->SetBinError(etabin+1,e);
              ////////////////// Absolute correction in etabin ////////////////////////// 
              CalculateCorrection(UseRatioForResponse,mRefPtEtaBin,eRefPtEtaBin,mR,eR,c,e);
              CorrectionVsEta_RefPt[j]->SetBinContent(etabin+1,c);
              CorrectionVsEta_RefPt[j]->SetBinError(etabin+1,e);
            }//end of EtaBin loop 
        }// end of Pt loop
    }
  else//single eta bin: used for L3 correction calculation
    {  
      std::cout<<"************* Fitting Response Histograms in single eta bin. ************"<<std::endl;
      for (j=0; j<NRefPtBins; j++)//loop over Pt bins
        {  
          std::cout<<"RefJetPt Bin: ["<<RefPtBoundaries[j]<<","<<RefPtBoundaries[j+1]<<"] GeV"<<std::endl; 
          sprintf(name,"ptRef_RefPt%d",j);
          if (!HistoExists(HistoNamesList,name)) return(0);
          h = (TH1F*)inf->Get(name);
          GetMEAN(h,mRefPt,eRefPt,sRefPt);
          sprintf(name,"ptCalo_RefPt%d",j);
          if (!HistoExists(HistoNamesList,name)) return(0);
          h = (TH1F*)inf->Get(name);
          GetMEAN(h,mCaloPt,eCaloPt,sCaloPt);
          sprintf(name,"Response_RefPt%d",j);
          if (!HistoExists(HistoNamesList,name)) return(0);
          h = (TH1F*)inf->Get(name);
          GetMPV(name,h,dir_Response,mRBarrel,eRBarrel,sRBarrel,seRBarrel);  
          ///////////////// RefPt in barrel ///////////////////////////////////
          MeanRefPt_Barrel->SetBinContent(j+1,mRefPt);
          MeanRefPt_Barrel->SetBinError(j+1,eRefPt);
          ///////////////// CaloPt in barrel ///////////////////////////////////
          MeanCaloPt_Barrel->SetBinContent(j+1,mCaloPt);
          MeanCaloPt_Barrel->SetBinError(j+1,eCaloPt);
          ////////////////// Absolute response in barrel //////////////////////////
          CalculateResponse(UseRatioForResponse,mRefPt,eRefPt,mRBarrel,eRBarrel,r,e);
          BarrelResponse->SetBinContent(j+1,r);
          BarrelResponse->SetBinError(j+1,e);
          ////////////////// Absolute correction in barrel //////////////////////////
          CalculateCorrection(UseRatioForResponse,mRefPt,eRefPt,mRBarrel,eRBarrel,c,e);
          BarrelCorrection->SetBinContent(j+1,c);
          BarrelCorrection->SetBinError(j+1,e);
        }// end of Pt loop
    }
  ////////////////////// writing ///////////////////////////////
  outf->cd();
  MeanRefPt_Barrel->Write();
  MeanCaloPt_Barrel->Write();
  BarrelResponse->Write();
  BarrelCorrection->Write();
  if (NETA>1)
    {
      for(etabin=0;etabin<NETA;etabin++)
        {
          MeanRefPt_EtaBin[etabin]->Write();
          MeanCaloPt_EtaBin[etabin]->Write();
        }
      for(j=0;j<NRefPtBins;j++)
        {
          ResponseVsEta_RefPt[j]->Write();
          CorrectionVsEta_RefPt[j]->Write();
        }
    }
  outf->Close();  
}
