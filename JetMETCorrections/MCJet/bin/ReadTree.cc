#include <iostream>
#include <string.h>
#include <fstream>
#include <cmath>
#include <TFile.h>
#include <TH1F.h>
#include <TMath.h>
#include <TTree.h>
#include "Utilities.cc"
#include "JetMETAnalysis/JetUtilities/interface/CommandLine.h"

using namespace std;

int main(int argc, char**argv)
{
  CommandLine c1;
  c1.parse(argc,argv);

  string TreeFilename      = c1.getValue<string>("TreeFilename");
  string HistoFilename     = c1.getValue<string>("HistoFilename");
  bool UseRatioForResponse = c1.getValue<bool>("UseRatioForResponse");
  double CALOPT_MIN        = c1.getValue<double>("CALOPT_MIN"); 
  double DR_MIN            = c1.getValue<double>("DR_MIN");
  int NJETS_MAX            = c1.getValue<int>("NJETS_MAX"); 
  vector<double> pt_vec    = c1.getVector<double>("RefPtBoundaries");
  vector<double> eta_vec   = c1.getVector<double>("EtaBoundaries");
  if (!c1.check()) return 0; 
  c1.print();
  /////////////////////////////////////////////////////////////////////////
  const int MAX_NETA = 83; 
  const int MAX_NREFPTBINS = 30; 
  TH1F *hPtRef[MAX_NREFPTBINS][MAX_NETA];
  TH1F *hPtCalo[MAX_NREFPTBINS][MAX_NETA];
  TH1F *hPtRefBarrel[MAX_NREFPTBINS];
  TH1F *hPtCaloBarrel[MAX_NREFPTBINS];
  TH1F *hResponseBarrel[MAX_NREFPTBINS];
  TH1F *hResponse[MAX_NREFPTBINS][MAX_NETA];
  int NRefPtBins = pt_vec.size()-1;
  int NETA = eta_vec.size()-1;   
  char name[100];
  bool cut_ptmin,cut_dR,cut_njets;
  float ptCalo,ptGen,etaCalo,etaGen,phiCalo,phiGen,dR,resp;
  int rank;
  int i,j,ind_pt,ind_eta,responseBins(2200);
  double responseLow(-1000.),responseHigh(100.);
  unsigned int entry;
  if (UseRatioForResponse)
    {
      responseBins = 200;
      responseLow = 0.;
      responseHigh = 2;
    } 
  vector<string> HistoNamesList; 
  TFile *inf = new TFile(TreeFilename.c_str(),"R");
  if (inf->IsZombie()) return(0);
  TFile *outf = new TFile(HistoFilename.c_str(),"RECREATE");
  
  TTree *tr = (TTree*)inf->Get("mcTruthTree");
  TBranch *br_ptCalo = (TBranch*)tr->GetBranch("ptCalo");
  br_ptCalo->SetAddress(&ptCalo);
  TBranch *br_ptGen = (TBranch*)tr->GetBranch("ptGen");
  br_ptGen->SetAddress(&ptGen);
  TBranch *br_etaCalo = (TBranch*)tr->GetBranch("etaCalo");
  br_etaCalo->SetAddress(&etaCalo);
  TBranch *br_etaGen = (TBranch*)tr->GetBranch("etaGen");
  br_etaGen->SetAddress(&etaGen);
  TBranch *br_phiCalo = (TBranch*)tr->GetBranch("phiCalo");
  br_phiCalo->SetAddress(&phiCalo);
  TBranch *br_phiGen = (TBranch*)tr->GetBranch("phiGen");
  br_phiGen->SetAddress(&phiGen);
  TBranch *br_dR = (TBranch*)tr->GetBranch("dR");
  br_dR->SetAddress(&dR);
  TBranch *br_rank = (TBranch*)tr->GetBranch("rank");
  br_rank->SetAddress(&rank);
  cout<<"Total entries:"<<tr->GetEntries()<<endl;
  if (NETA>1)
    {
      for (i=0;i<NRefPtBins;i++)
       {
         sprintf(name,"ptRef_RefPt%d_Barrel",i);
         hPtRefBarrel[i] = new TH1F(name,name,2000,0,2000);
         sprintf(name,"ptCalo_RefPt%d_Barrel",i);
         hPtCaloBarrel[i] = new TH1F(name,name,2000,0,2000);
         sprintf(name,"Response_RefPt%d_Barrel",i);
         hResponseBarrel[i] = new TH1F(name,name,responseBins,responseLow,responseHigh);     
         for (j=0;j<NETA;j++)
           {
	     sprintf(name,"ptRef_RefPt%d_Eta%d",i,j);
             hPtRef[i][j] = new TH1F(name,name,2000,0,2000);
             sprintf(name,"ptCalo_RefPt%d_Eta%d",i,j);
             hPtCalo[i][j] = new TH1F(name,name,2000,0,2000);
	     sprintf(name,"Response_RefPt%d_Eta%d",i,j);
             hResponse[i][j] = new TH1F(name,name,responseBins,responseLow,responseHigh);
           }
       }
    }
  else
    {
      for (i=0;i<NRefPtBins;i++)
       {
         sprintf(name,"ptRef_RefPt%d",i);
         hPtRefBarrel[i] = new TH1F(name,name,2000,0,2000);
         sprintf(name,"ptCalo_RefPt%d",i);
         hPtCaloBarrel[i] = new TH1F(name,name,2000,0,2000);
         sprintf(name,"Response_RefPt%d",i);
         hResponseBarrel[i] = new TH1F(name,name,responseBins,responseLow,responseHigh);     
       }
    }
  for(entry=0;entry<tr->GetEntries();entry++)
    {
      if (entry % (tr->GetEntries()/10) == 0)
        cout<<"Entries: "<<entry<<endl;
      tr->GetEntry(entry);
      cut_ptmin = (ptCalo>CALOPT_MIN);
      cut_dR    = (dR<DR_MIN);
      cut_njets = (rank<NJETS_MAX);
      if (cut_ptmin && cut_dR && cut_njets)
        {
	  ind_pt = getBin(ptGen,pt_vec);
	  ind_eta = getBin(etaCalo,eta_vec);   
          resp = 0.;
          if (UseRatioForResponse && ptGen>0)
            resp = ptCalo/ptGen;
          else
            resp = ptCalo - ptGen; 
          if (NETA>1)
            { 
              if (fabs(etaCalo)<1.3 && ind_pt>=0)
                {
                  hPtRefBarrel[ind_pt]->Fill(ptGen);
                  hPtCaloBarrel[ind_pt]->Fill(ptCalo);
                  hResponseBarrel[ind_pt]->Fill(resp);
                }  
              if (ind_pt>=0 && ind_eta>=0 && NETA>1)
                { 
	          hPtRef[ind_pt][ind_eta]->Fill(ptGen);
                  hPtCalo[ind_pt][ind_eta]->Fill(ptCalo);
	          hResponse[ind_pt][ind_eta]->Fill(resp);
                }
            }
          if (NETA==1)
            if (ind_pt>=0)
              {
                hPtRefBarrel[ind_pt]->Fill(ptGen);
                hPtCaloBarrel[ind_pt]->Fill(ptCalo);
                hResponseBarrel[ind_pt]->Fill(resp);
              }  
        }
    }
  outf->Write();
  outf->Close();
}
