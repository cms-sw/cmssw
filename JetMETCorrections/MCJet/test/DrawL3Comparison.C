void DrawL3Comparison()
{
  gROOT->Reset();
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0000);
  gStyle->SetOptFit(000); 
  //gStyle->SetPadGridX(1);
  //gStyle->SetPadGridY(1);
  gStyle->SetPalette(1);
 
  char name[100];
  const int NAlg = 4;
  const char algorithm[NAlg][100] = {"iterativeCone5","midPointCone5","sisCone5","fastjet4"};
  //const char algorithm[NAlg][100] = {"iterativeCone7","midPointCone7","sisCone7","fastjet6"};
  int alg;
  TFile *inf[NAlg];
  
  TGraphErrors *g_Resp[NAlg];
  TGraph *g_Cor[NAlg];
  TF1 *CorFit[NAlg];
  TF1 *RespFit[NAlg];
  ///////////////////////////////////////////////////////////////
  for(alg=0;alg<NAlg;alg++)
    {
      sprintf(name,"L3Graphs_%s.root",algorithm[alg]); 
      inf[alg] = new TFile(name,"r");
      sprintf(name,"Response_vs_GenPt");      
      g_Resp[alg] = (TGraphErrors*)inf[alg]->Get(name);
      sprintf(name,"Response_%s",algorithm[alg]);
      g_Resp[alg]->SetName(name);
      sprintf(name,"Correction_vs_CaloPt");      
      g_Cor[alg] = (TGraph*)inf[alg]->Get(name);
      sprintf(name,"Correction_%s",algorithm[alg]);
      g_Cor[alg]->SetName(name);
      sprintf(name,"CorFit");
      CorFit[alg] = (TF1*)g_Cor[alg]->GetFunction(name);
      sprintf(name,"CorFit_%s",algorithm[alg]);
      CorFit[alg]->SetName(name);
      sprintf(name,"RespFit");
      RespFit[alg] = (TF1*)g_Resp[alg]->GetFunction(name);
      sprintf(name,"RespFit_%s",algorithm[alg]);
      RespFit[alg]->SetName(name);
    }
  ////////////////////// Response ///////////////////////////////////////
  sprintf(name,"Response");
  TCanvas *c_Response = new TCanvas(name,name,900,600);
  c_Response->cd();
  gPad->SetLogx(); 
  g_Resp[0]->SetTitle("Jet Response, |#eta|<1.3");
  g_Resp[0]->GetXaxis()->SetTitle("<p_{T}^{gen}> (GeV)");
  g_Resp[0]->GetYaxis()->SetTitle("1+#frac{<<#Delta p_{T}>>}{<p_{T}^{gen}>}");
  g_Resp[0]->SetMaximum(1.);
  g_Resp[0]->SetMinimum(0.);
  g_Resp[0]->Draw("AP");
  TLegend *leg = new TLegend(0.6,0.15,0.9,0.45);
  for(alg=0;alg<NAlg;alg++)
    {
      g_Resp[alg]->SetMarkerStyle(20+alg);
      g_Resp[alg]->SetMarkerColor(alg+1);
      g_Resp[alg]->SetLineColor(alg+1);
      g_Resp[alg]->SetLineStyle(1);
      RespFit[alg]->SetLineColor(alg+1);
      RespFit[alg]->SetLineStyle(alg+1);
      RespFit[alg]->SetLineWidth(2);
      g_Resp[alg]->Draw("sameP");
      sprintf(name,"%s measurement",algorithm[alg]);
      leg->AddEntry(g_Resp[alg],name,"p");
      sprintf(name,"%s fit",algorithm[alg]);
      leg->AddEntry(RespFit[alg],name,"l");
    }
  leg->SetFillColor(0);
  leg->Draw();
  ////////////////////// Correction ///////////////////////////////////////
  sprintf(name,"Correction");
  TCanvas *c_Correction = new TCanvas(name,name,900,600);
  c_Correction->cd();
  gPad->SetLogx(); 
  g_Cor[0]->SetTitle("L3 Correction");
  g_Cor[0]->GetXaxis()->SetTitle("p_{T} (GeV)");
  g_Cor[0]->GetYaxis()->SetTitle("#frac{p_{T}^{gen}}{p_{T}}");
  g_Cor[0]->SetMaximum(4.);
  g_Cor[0]->SetMinimum(1.);
  g_Cor[0]->Draw("AP");
  TLegend *leg = new TLegend(0.6,0.6,0.9,0.9);
  for(alg=0;alg<NAlg;alg++)
    {
      g_Cor[alg]->SetMarkerStyle(20+alg);
      g_Cor[alg]->SetMarkerColor(1+alg);
      g_Cor[alg]->SetLineColor(1+alg);
      g_Cor[alg]->SetLineStyle(1);
      CorFit[alg]->SetLineColor(1+alg);
      CorFit[alg]->SetLineStyle(alg+1);
      CorFit[alg]->SetLineWidth(2); 
      g_Cor[alg]->Draw("sameP");
      sprintf(name,"%s_measurement",algorithm[alg]);
      leg->AddEntry(g_Cor[alg],name,"p");
      sprintf(name,"%s_fit",algorithm[alg]);
      leg->AddEntry(CorFit[alg],name,"l");
    }
  leg->SetFillColor(0);
  leg->Draw();
}



