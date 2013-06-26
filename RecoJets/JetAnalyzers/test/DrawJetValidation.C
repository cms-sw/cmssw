#include <vector.h>
void DrawJetValidation()
{  
  PrintMessage();
}
void DrawJetValidation(char s1[1024])
{  
  char filename[1][1024];
  sprintf(filename[0],"%s",s1);
  MainProgram(1,filename);
}
void DrawJetValidation(char s1[1024],char s2[1024])
{ 
  char filename[2][1024];
  sprintf(filename[0],"%s",s1);
  sprintf(filename[1],"%s",s2);
  MainProgram(2,filename);
}
void MainProgram(const int NAlg,char filename[][1024])
{
  gROOT->SetStyle("Plain");
  //gStyle->SetOptStat(0000);
  //gStyle->SetOptFit(000); 
  gStyle->SetPalette(1);
 
  char name[100];
  int alg,hCounter,pCounter,i;
  TFile *inf[NAlg]; 
  TH1F *h[NAlg][100];
  TProfile *p[NAlg][100];
  TCanvas *hCan[100];
  TCanvas *pCan[100];
  TKey *key;
  ///////////////////////////////////////////////////////////////
  for(alg=0;alg<NAlg;alg++)
    {
      inf[alg] = new TFile(filename[alg],"r");
      TIter next(inf[alg]->GetListOfKeys());
      hCounter = 0;
      pCounter = 0;
      while ((key = (TKey*)next()))
       {
         if (strcmp(key->GetClassName(),"TH1F")==0)
           {
             h[alg][hCounter] = (TH1F*)inf[alg]->Get(key->GetName());
             hCounter++;
           } 
         if (strcmp(key->GetClassName(),"TProfile")==0)
           {
             p[alg][pCounter] = (TProfile*)inf[alg]->Get(key->GetName());
             pCounter++;
           }
       }
    }
  for(i=0;i<hCounter;i++)
    {
      sprintf(name,"can_%s",h[0][i]->GetName());       
      hCan[i] = new TCanvas(name,name,900,600);
      TLegend *leg = new TLegend(0.5,0.15,0.85,0.4);
      for(alg=0;alg<NAlg;alg++)
        {
          h[alg][i]->SetLineColor(alg+1);   
          h[alg][i]->SetMarkerColor(alg+1); 
          h[alg][i]->Draw("same");
          leg->AddEntry(h[alg][i],filename[alg],"L");
        }
      leg->SetFillColor(0);
      leg->SetLineColor(0);
      leg->Draw();
    }
  for(i=0;i<pCounter;i++)
    {
      sprintf(name,"can_%s",p[0][i]->GetName());       
      pCan[i] = new TCanvas(name,name,900,600);
      TLegend *leg = new TLegend(0.5,0.15,0.85,0.4);
      for(alg=0;alg<NAlg;alg++)
        {
          p[alg][i]->SetLineColor(alg+1); 
          p[alg][i]->SetMarkerColor(alg+1);   
          p[alg][i]->Draw("same");
          leg->AddEntry(h[alg][i],filename[alg],"L");
        }
      leg->SetFillColor(0);
      leg->SetLineColor(0);
      leg->Draw();
    }
}

void PrintMessage()
{
  cout<<"This ROOT macro compares histograms from up to 2 files."<<endl;
  cout<<"Usage: .X DrawJetValidation.C(\"filename1\",\"filename2\")"<<endl;
}
