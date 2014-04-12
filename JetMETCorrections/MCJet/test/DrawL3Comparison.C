void DrawL3Comparison()
{  
  PrintMessage();
}
void DrawL3Comparison(char s1[1024])
{  
  char filename[1][1024];
  sprintf(filename[0],"%s",s1);
  MainProgram(1,filename);
}
void DrawL3Comparison(char s1[1024],char s2[1024])
{ 
  char filename[2][1024];
  sprintf(filename[0],"%s",s1);
  sprintf(filename[1],"%s",s2);
  MainProgram(2,filename);
}
void DrawL3Comparison(char s1[1024],char s2[1024],char s3[1024])
{ 
  char filename[3][1024];
  sprintf(filename[0],"%s",s1);
  sprintf(filename[1],"%s",s2);
  sprintf(filename[2],"%s",s3);
  MainProgram(3,filename);
}
void DrawL3Comparison(char s1[1024],char s2[1024],char s3[1024],char s4[1024])
{ 
  char filename[4][1024];
  sprintf(filename[0],"%s",s1);
  sprintf(filename[1],"%s",s2);
  sprintf(filename[2],"%s",s3);
  sprintf(filename[3],"%s",s4);
  MainProgram(4,filename);
}
void DrawL3Comparison(char s1[1024],char s2[1024],char s3[1024],char s4[1024],char s5[1024])
{ 
  char filename[5][1024];
  sprintf(filename[0],"%s",s1);
  sprintf(filename[1],"%s",s2);
  sprintf(filename[2],"%s",s3);
  sprintf(filename[3],"%s",s4);
  sprintf(filename[4],"%s",s5);
  MainProgram(5,filename);
}
void MainProgram(const int NAlg,char filename[][1024])
{
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0000);
  gStyle->SetOptFit(000); 
  gStyle->SetPalette(1);
 
  char name[1024];
  int alg,color;
  TFile *inf[NAlg];
  
  TGraphErrors *g_Cor[NAlg],*g_Resp[NAlg];
  TF1 *CorFit[NAlg],*RespFit[NAlg];
  ///////////////////////////////////////////////////////////////
  for(alg=0;alg<NAlg;alg++)
    {
      inf[alg] = new TFile(filename[alg],"r");
      g_Cor[alg] = (TGraphErrors*)inf[alg]->Get("Correction_vs_CaloPt");
      sprintf(name,"Correction_%s",filename[alg]);
      g_Cor[alg]->SetName(name);
      CorFit[alg] = (TF1*)g_Cor[alg]->GetFunction("CorFit");
      sprintf(name,"CorFit_%s",filename[alg]); 
      CorFit[alg]->SetName(name);
      CorFit[alg]->SetRange(5,5000);
      g_Resp[alg] = (TGraphErrors*)inf[alg]->Get("Response_vs_RefPt");
      sprintf(name,"Response_%s",filename[alg]);
      g_Resp[alg]->SetName(name);
      RespFit[alg] = (TF1*)g_Resp[alg]->GetFunction("RespFit");
      sprintf(name,"RespFit_%s",filename[alg]); 
      RespFit[alg]->SetName(name);
      RespFit[alg]->SetRange(5,5000);
    }
  ////////////////////// Correction ///////////////////////////////////////
  sprintf(name,"L3CorrectionComparison");
  TCanvas *c_Correction = new TCanvas(name,name,900,600);
  c_Correction->cd();
  gPad->SetLogx(); 
  g_Cor[0]->SetTitle("");
  g_Cor[0]->GetXaxis()->SetTitle("Uncorrected jet p_{T} (GeV)");
  g_Cor[0]->GetYaxis()->SetTitle("Correction Factor");
  g_Cor[0]->SetMaximum(3.5);
  g_Cor[0]->SetMinimum(0.8);
  g_Cor[0]->Draw("AP");
  TLegend *leg = new TLegend(0.5,0.6,0.85,0.85);
  for(alg=0;alg<NAlg;alg++)
    {
      color = alg+1;
      if (color==5)
        color = 7;
      g_Cor[alg]->SetMarkerStyle(20+alg);
      g_Cor[alg]->SetMarkerColor(color);
      g_Cor[alg]->SetLineColor(color);
      g_Cor[alg]->SetLineStyle(1);
      CorFit[alg]->SetLineColor(color);
      CorFit[alg]->SetLineStyle(1);
      CorFit[alg]->SetLineWidth(2); 
      g_Cor[alg]->Draw("sameP");
      leg->AddEntry(g_Cor[alg],filename[alg],"LP");
    }
  leg->SetFillColor(0);
  leg->SetLineColor(0);
  leg->Draw();
  ////////////////////// Response ///////////////////////////////////////
  sprintf(name,"ResponseComparison");
  TCanvas *c_Correction = new TCanvas(name,name,900,600);
  c_Correction->cd();
  gPad->SetLogx(); 
  g_Resp[0]->SetTitle("");
  g_Resp[0]->GetXaxis()->SetTitle("p_{T}^{gen} (GeV)");
  g_Resp[0]->GetYaxis()->SetTitle("Response");
  g_Resp[0]->SetMaximum(1);
  g_Resp[0]->SetMinimum(0);
  g_Resp[0]->Draw("AP");
  TLegend *leg = new TLegend(0.5,0.15,0.85,0.4);
  for(alg=0;alg<NAlg;alg++)
    {
      color = alg+1;
      if (color==5)
        color = 7;
      g_Resp[alg]->SetMarkerStyle(20+alg);
      g_Resp[alg]->SetMarkerColor(color);
      g_Resp[alg]->SetLineColor(color);
      g_Resp[alg]->SetLineStyle(1);
      RespFit[alg]->SetLineColor(color);
      RespFit[alg]->SetLineStyle(1);
      RespFit[alg]->SetLineWidth(2); 
      g_Resp[alg]->Draw("sameP");
      leg->AddEntry(g_Resp[alg],filename[alg],"LP");
    }
  leg->SetFillColor(0);
  leg->SetLineColor(0);
  leg->Draw();
}

void PrintMessage()
{
  cout<<"This ROOT macro can compare up to 5 cases."<<endl;
  cout<<"Usage: .X DrawL3Comparison.C(\"filename1\",...,\"filename5\")"<<endl;
}

