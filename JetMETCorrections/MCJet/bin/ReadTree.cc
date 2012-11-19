#include <iostream>
#include <string.h>
#include <fstream>
#include <cmath>
#include <TFile.h>
#include <TH1F.h>
#include <TMath.h>
#include <TTree.h>
#include "Utilities.h"

using namespace std;

int main(int argc, char**argv)
{
  CommandLine c1;
  c1.parse(argc,argv);

  std::string TreeFilename      = c1.getValue<string> ("TreeFilename"             );
  std::string HistoFilename     = c1.getValue<string> ("HistoFilename"            );
  bool UseRatioForResponse = c1.getValue<bool>   ("UseRatioForResponse", true);
  bool IsPFJet             = c1.getValue<bool>   ("IsPFJet",            false);
  int NJETS_MAX            = c1.getValue<int>    ("NJETS_MAX",              2);
  double JetPT_MIN         = c1.getValue<double> ("JetPT_MIN",            1.0); 
  double DR_MIN            = c1.getValue<double> ("DR_MIN",              0.25);
  std::vector<double> pt_vec    = c1.getVector<double>("RefPtBoundaries"          );
  std::vector<double> eta_vec   = c1.getVector<double>("EtaBoundaries"            );
  if (!c1.check()) return 0; 
  c1.print();
  /////////////////////////////////////////////////////////////////////////
  const int MAX_NETA = 83; 
  const int MAX_NPT  = 30; 
  TH1F *hPtRef[MAX_NPT][MAX_NETA];
  TH1F *hPtJet[MAX_NPT][MAX_NETA];
  TH1F *hPtRefBarrel[MAX_NPT];
  TH1F *hPtJetBarrel[MAX_NPT];
  TH1F *hResponseBarrel[MAX_NPT];
  TH1F *hResponse[MAX_NPT][MAX_NETA];
  int NPT  = pt_vec.size()-1;
  int NETA = eta_vec.size()-1;
  char name[1024];
  bool cut_ptmin,cut_dR,cut_njets;
  float ptJet,ptGen,etaJet,emfJet,chfJet,etaGen,phiJet,phiGen,dR,resp;
  int rank;
  int i,j,ind_pt,ind_eta,responseBins(2200);
  double responseLow(-1000.),responseHigh(100.);
  unsigned int entry;
  if (UseRatioForResponse)
    {
      responseBins = 200;
      responseLow  = 0.;
      responseHigh = 2;
    } 
  std::vector<std::string> HistoNamesList; 
  TFile *inf = new TFile(TreeFilename.c_str(),"R");
  if (inf->IsZombie()) return 0;
  TFile *outf = new TFile(HistoFilename.c_str(),"RECREATE");
  
  TTree *tr = (TTree*)inf->Get("mcTruthTree");
  TBranch *br_ptJet = (TBranch*)tr->GetBranch("ptJet");
  br_ptJet->SetAddress(&ptJet);
  TBranch *br_ptGen = (TBranch*)tr->GetBranch("ptGen");
  br_ptGen->SetAddress(&ptGen);
  TBranch *br_emfJet,*br_chfJet;
  if (!IsPFJet)
    {
      br_emfJet = (TBranch*)tr->GetBranch("emfJet");
      br_emfJet->SetAddress(&emfJet);
    }
  else
    {
      br_chfJet = (TBranch*)tr->GetBranch("chfJet");
      br_chfJet->SetAddress(&chfJet);
    }
  TBranch *br_etaJet = (TBranch*)tr->GetBranch("etaJet");
  br_etaJet->SetAddress(&etaJet);
  TBranch *br_etaGen = (TBranch*)tr->GetBranch("etaGen");
  br_etaGen->SetAddress(&etaGen);
  TBranch *br_phiJet = (TBranch*)tr->GetBranch("phiJet");
  br_phiJet->SetAddress(&phiJet);
  TBranch *br_phiGen = (TBranch*)tr->GetBranch("phiGen");
  br_phiGen->SetAddress(&phiGen);
  TBranch *br_dR = (TBranch*)tr->GetBranch("dR");
  br_dR->SetAddress(&dR);
  TBranch *br_rank = (TBranch*)tr->GetBranch("rank");
  br_rank->SetAddress(&rank);
  std::cout<<"Total entries:"<<tr->GetEntries()<<std::endl;
  if (NETA>1)
    {
      for (i=0;i<NPT;i++)
       {
         sprintf(name,"ptRef_RefPt%d_Barrel",i);
         hPtRefBarrel[i] = new TH1F(name,name,2000,0,2000);
         sprintf(name,"ptJet_RefPt%d_Barrel",i);
         hPtJetBarrel[i] = new TH1F(name,name,2000,0,2000);
         sprintf(name,"Response_RefPt%d_Barrel",i);
         hResponseBarrel[i] = new TH1F(name,name,responseBins,responseLow,responseHigh);     
         for (j=0;j<NETA;j++)
           {
	     sprintf(name,"ptRef_RefPt%d_Eta%d",i,j);
             hPtRef[i][j] = new TH1F(name,name,2000,0,2000);
             sprintf(name,"ptCalo_RefPt%d_Eta%d",i,j);
             hPtJet[i][j] = new TH1F(name,name,2000,0,2000);
	     sprintf(name,"Response_RefPt%d_Eta%d",i,j);
             hResponse[i][j] = new TH1F(name,name,responseBins,responseLow,responseHigh);
           }
       }
    }
  else
    {
      for (i=0;i<NPT;i++)
       {
         sprintf(name,"ptRef_RefPt%d",i);
         hPtRefBarrel[i] = new TH1F(name,name,2000,0,2000);
         sprintf(name,"ptCalo_RefPt%d",i);
         hPtJetBarrel[i] = new TH1F(name,name,2000,0,2000);
         sprintf(name,"Response_RefPt%d",i);
         hResponseBarrel[i] = new TH1F(name,name,responseBins,responseLow,responseHigh);     
       }
    }
  std::cout<<"Histograms booked"<<std::endl;
  for(entry=0;entry<tr->GetEntries();entry++)
    {
      if (entry % (tr->GetEntries()/10) == 0)
        std::cout<<"Entries: "<<entry<<std::endl;
      tr->GetEntry(entry);
      cut_ptmin = (ptJet>JetPT_MIN);
      cut_dR    = (dR<DR_MIN);
      cut_njets = (rank<NJETS_MAX);
      if (cut_ptmin && cut_dR && cut_njets)
        {
	  ind_pt = getBin(ptGen,pt_vec);
	  ind_eta = getBin(etaJet,eta_vec);   
          resp = 0.;
          if (UseRatioForResponse && ptGen>0)
            resp = ptJet/ptGen;
          else
            resp = ptJet - ptGen; 
          if (NETA>1)
            { 
              if (fabs(etaJet)<1.3 && ind_pt>=0)
                {
                  hPtRefBarrel[ind_pt]->Fill(ptGen);
                  hPtJetBarrel[ind_pt]->Fill(ptJet);
                  hResponseBarrel[ind_pt]->Fill(resp);
                }  
              if (ind_pt>=0 && ind_eta>=0 && NETA>1)
                { 
	          hPtRef[ind_pt][ind_eta]->Fill(ptGen);
                  hPtJet[ind_pt][ind_eta]->Fill(ptJet);
	          hResponse[ind_pt][ind_eta]->Fill(resp);
                }
            }
          if (NETA==1)
            if (ind_pt>=0)
              {
                hPtRefBarrel[ind_pt]->Fill(ptGen);
                hPtJetBarrel[ind_pt]->Fill(ptJet);
                hResponseBarrel[ind_pt]->Fill(resp);
              }  
        }
    }
  outf->Write();
  outf->Close();
  return 0;
}
