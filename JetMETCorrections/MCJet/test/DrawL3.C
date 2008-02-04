void DrawL3(const char algorithm[100])
{
  gROOT->Reset();
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0000);
  gStyle->SetOptFit(11); 
  //gStyle->SetPadGridX(1);
  //gStyle->SetPadGridY(1);
  gStyle->SetPalette(1);
 
  char name[100];
  sprintf(name,"L3Graphs_%s.root",algorithm);
  TFile *inf = new TFile(name,"r");
  
  TGraphErrors *g_Resp;
  TGraph *g_Cor;
  TF1 *CorFit;
  TF1 *RespFit;
  ///////////////////////////////////////////////////////////////
  sprintf(name,"Response_vs_GenPt");      
  g_Resp = (TGraphErrors*)inf->Get(name);
  sprintf(name,"Correction_vs_CaloPt");      
  g_Cor = (TGraph*)inf->Get(name);
  sprintf(name,"CorFit");
  CorFit   = (TF1*)g_Cor->GetFunction(name);
  sprintf(name,"RespFit");
  RespFit  = (TF1*)g_Resp->GetFunction(name);
  ////////////////////// Response ///////////////////////////////////////
  TCanvas *c_Response = new TCanvas("Response","Response",900,600);
  c_Response->cd();
  gPad->SetLogx(); 
  g_Resp->SetTitle("Jet Response, |#eta|<1.3");
  g_Resp->GetXaxis()->SetTitle("p_{T}^{gen} (GeV)");
  g_Resp->GetYaxis()->SetTitle("1+#frac{<<#Delta p_{T}>>}{<p_{T}^{gen}>}");
  g_Resp->SetMarkerStyle(20);
  g_Resp->SetMarkerColor(1);
  g_Resp->SetMinimum(0.);
  g_Resp->SetMaximum(1.);
  g_Resp->SetLineColor(1);
  RespFit->SetLineColor(1);
  RespFit->SetParNames("b0","b1","b2","b3","b4","b5");
  g_Resp->Draw("AP");
  TLegend *leg = new TLegend(0.6,0.15,0.9,0.35);
  leg->AddEntry(g_Resp,"Response measurement","p");
  leg->AddEntry(RespFit,"b0 - #frac{b1}{[log(p_{T})]^{b2}+b3} + #frac{b4}{p_{T}}","l");
  leg->SetFillColor(0); 
  leg->Draw();
  TPaveLabel *pave = new TPaveLabel(0.4,0.8,0.6,0.9,algorithm,"NDC");
  pave->SetFillColor(0);
  pave->Draw(); 
  ////////////////////// Correction ///////////////////////////////////////
  TCanvas *c_Correction = new TCanvas("Correction","Correction",900,600);
  c_Correction->cd(); 
  gPad->SetLogx();
  TLegend *leg = new TLegend(0.6,0.5,0.9,0.7);
  sprintf(name,"L3Correction");
  g_Cor->SetTitle(name); 
  g_Cor->SetMarkerStyle(20);
  g_Cor->SetMarkerColor(1);
  g_Cor->SetLineColor(1);
  g_Cor->GetXaxis()->SetTitle("p_{T} (GeV)");
  g_Cor->GetYaxis()->SetTitle("#frac{p_{T}^{gen}}{p_{T}}");
  g_Cor->SetMinimum(1.0);
  g_Cor->SetMaximum(4.0);
  CorFit->SetLineColor(1);
  CorFit->SetParNames("a0","a1","a2","a3","a4");
  leg->AddEntry(g_Cor,"measurement","p");
  leg->AddEntry(CorFit,"a0 + #frac{a1}{[log(p_{T})]^{a2}+a3}","l");
  g_Cor->Draw("AP");
  leg->SetFillColor(0);
  leg->Draw();
  TPaveLabel *pave = new TPaveLabel(0.4,0.8,0.6,0.9,algorithm,"NDC");
  pave->SetFillColor(0);
  pave->Draw();
  ////////////////////// MisFit ///////////////////////////////////////
}



