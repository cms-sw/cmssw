void DrawL2(const char algorithm[100],int etabin)
{
  gROOT->Reset();
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0000);
  gStyle->SetOptFit(000); 
  gStyle->SetPalette(1);
  const int NETA = 82;
  const double eta_boundaries[NETA+1]={-5.191,-4.889,-4.716,-4.538,-4.363,-4.191,-4.013,-3.839,-3.664, 
			             -3.489,-3.314,-3.139,-2.964,-2.853,-2.650,-2.500,-2.322,-2.172,
                                     -2.043,-1.930,-1.830,-1.740,-1.653,-1.566,-1.479,-1.392,-1.305,
			             -1.218,-1.131,-1.044,-0.957,-0.879,-0.783,-0.696,-0.609,-0.522,
			             -0.435,-0.348,-0.261,-0.174,-0.087,0.000,0.087,0.174,0.261,0.348,
                                      0.435,0.522,0.609,0.696,0.783,0.879,0.957,1.044,1.131,1.218,1.305,
                                      1.392,1.479,1.566,1.653,1.740,1.830,1.930,2.043,2.172,2.322,2.500,
                                      2.650,2.853,2.964,3.139,3.314,3.489,3.664,3.839,4.013,4.191,4.363,4.538,
                                      4.716,4.889,5.191};
  if (etabin<0 || etabin>=NETA)
    {
      cout<<"Eta bin must be >=0 and <"<<NETA<<endl;
      break;
    }
  TGraphErrors *g_EtaResponse;
  TGraph *g_L2Correction;
  TCanvas *c_Resp;
  TCanvas *c_L2Cor;
  TF1 *L2Fit;
  TF1 *RespFit;
  
  char filename[100],name[100];
  sprintf(name,"L2Graphs_%s.root",algorithm);
  TFile *rel_f = new TFile(name,"r");
  if (!rel_f->IsOpen()) break;
  /////////////////////////////// Response /////////////////////////
   
  sprintf(name,"Response");
  c_Resp = new TCanvas(name,name,900,700);
  sprintf(name,"Response_EtaBin%d",etabin);
  g_EtaResponse = (TGraphErrors*)rel_f->Get(name);
  sprintf(name,"Response%d",etabin);      
  RespFit = (TF1*)g_EtaResponse->GetFunction(name);
  if (RespFit->GetXmax()>200) 
    gPad->SetLogx();
  g_EtaResponse->SetMarkerStyle(21);
  g_EtaResponse->SetMarkerColor(1);
  g_EtaResponse->SetLineColor(1);
  g_EtaResponse->SetMinimum(0.);
  g_EtaResponse->SetMaximum(1.2);
  g_EtaResponse->GetXaxis()->SetTitle("p_{T}^{gen} (GeV)");
  g_EtaResponse->GetYaxis()->SetTitle("Response");  
  RespFit->SetLineColor(2);  
  g_EtaResponse->Draw("AP");
  sprintf(name,"%1.3f<#eta<%1.3f",eta_boundaries[etabin],eta_boundaries[etabin+1]);
  g_EtaResponse->SetTitle(name);
  TPaveLabel *pave = new TPaveLabel(0.15,0.15,0.35,0.25,algorithm,"NDC");
  pave->SetFillColor(0);
  pave->Draw();
    
  /////////////////////////////// L2 correction ///////////////////////// 
  sprintf(name,"L2Correction");
  c_L2Cor = new TCanvas(name,name,900,700);
  sprintf(name,"L2Correction_EtaBin%d",etabin);
  g_L2Correction = (TGraph*)rel_f->Get(name);
  sprintf(name,"L2Correction%d",etabin);      
  L2Fit = (TF1*)g_L2Correction->GetFunction(name);
  if (L2Fit->GetXmax()>200) 
    gPad->SetLogx(); 
  g_L2Correction->SetMarkerStyle(21);
  g_L2Correction->SetMarkerColor(1);
  g_L2Correction->SetLineColor(1);
  g_L2Correction->SetMinimum(0.3);
  g_L2Correction->SetMaximum(1.1);
  g_L2Correction->GetXaxis()->SetTitle("p_{T} (GeV)");
  g_L2Correction->GetYaxis()->SetTitle("L2 Correction"); 
  L2Fit->SetLineColor(2); 
  g_L2Correction->Draw("AP");
  sprintf(name,"%1.3f<#eta<%1.3f",eta_boundaries[etabin],eta_boundaries[etabin+1]);
  g_L2Correction->SetTitle(name);
  TPaveLabel *pave = new TPaveLabel(0.15,0.15,0.35,0.25,algorithm,"NDC");
  pave->SetFillColor(0);
  pave->Draw();
}


