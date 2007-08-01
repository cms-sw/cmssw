TH1D ratioEff(TH1D* hn, TH1D* hd);

// ch is Chamber name
void PlotEff(const char* ch = "") {
  gROOT->LoadMacro("macros.C");     // Load service macros
  TStyle * style = getStyle("tdr");
//  style->SetTitleYOffset(1.6);
  style->cd();                      // Apply style 

  if (ch == "" ) {
    cout << "Usage: PlotEff(\"ChamberName\") " << endl;
    cout << "Ex: PlotEff(\"Wh1_Sec10_St1\") " << endl;
    return;
  }
  TCanvas *c1 = new TCanvas("c1", "c1",10,25,900,400);

  c1->Update();
  c1->Divide(3);
  gStyle->SetOptFit(0);
  gStyle->SetOptStat(0);
  gStyle->SetPalette(1);
  TFile* file = TFile::Open("DTEffAnalyzer.root");
  //TFile* file = TFile::Open("DTEffAnalyzer_4398_118kev.root");
  //TFile* file = TFile::Open("DTEffAnalyzer_4398_100kev.root");

  TString nameDen("hEffGoodSegVsPosDen_");
  nameDen+=ch;
  TH2F* h2d = (TH2F*)(file->Get(nameDen));

  TString nameNum("hEffGoodCloseSegVsPosNum_");
  nameNum+=ch;
  TH2F* h2n = (TH2F*)(file->Get(nameNum));
  TH2F h2=(*h2n)/(*h2d);
  //h2.DrawCopy("zcol");
  TH1D* proxN=h2n->ProjectionX();
  TH1D* proxD=h2d->ProjectionX();
  TH1D* proyN=h2n->ProjectionY();
  TH1D* proyD=h2d->ProjectionY();
  int firstBinY=1;
  int lastBinY=h2.GetNbinsY();
  int midbinX=h2.GetNbinsX()/2;
  // cout << "midbinX " << midbinX << endl;
  for (int i=2; i<h2.GetNbinsY()-1; ++i) {
    //cout << "GetBinContent " << i << "=" << h2.GetBinContent(midbinX,i) << endl;
    if (firstBinY==1 && h2.GetBinContent(midbinX,i)>0) {
      // cout << "firstBinY " << i << endl;
      firstBinY=i+1;
      continue;
    }
    if (firstBinY!=1 && h2.GetBinContent(midbinX,i)==0) {
      // cout << "lastBinY " << i << endl;
      lastBinY=i-1;
      break;
    }
  }
  // cout << "firstBinY " << firstBinY << endl;
  // cout << "lastBinY " << lastBinY << endl;

  int firstBinX=1;
  int lastBinX=h2.GetNbinsX();
  int midbinY=h2.GetNbinsY()/2;
  // cout << "midbinY " << midbinY << endl;
  for (int i=2; i<h2.GetNbinsX()-1; ++i) {
    //cout << "GetBinContent " << i << "=" << h2.GetBinContent(midbinY,i) << endl;
    if (firstBinX==1 && h2.GetBinContent(i,midbinY)>0) {
      // cout << "firstBinX " << i << endl;
      firstBinX=i+1;
      continue;
    }
    if (firstBinX!=1 && h2.GetBinContent(i,midbinY)==0) {
      // cout << "lastBinX " << i << endl;
      lastBinX=i-1;
      break;
    }
  }
  // cout << "firstBinX " << firstBinX << endl;
  // cout << "lastBinX " << lastBinX << endl;
  TH1D* proxNc=h2n->ProjectionX("_pxnc",firstBinY,lastBinY);
  TH1D* proxDc=h2d->ProjectionX("_pxdc",firstBinY,lastBinY);
  TH1D* proyNc=h2n->ProjectionY("_pync",firstBinX,lastBinX);
  TH1D* proyDc=h2d->ProjectionY("_pydc",firstBinX,lastBinX);

  TLegend* leg= new TLegend(0.28,0.15,0.72,0.3,"Eff");
    leg->SetFillColor(0);
  c1->cd(1);
  TGraphAsymmErrors* hx = new TGraphAsymmErrors();
  hx->BayesDivide(proxN,proxD);
  hx->SetTitle("Efficiency along X:x (cm):#epsilon");
  (hx->GetXaxis())->SetTitle("x (cm)");
  (hx->GetYaxis())->SetTitle("#epsilon");
  hx->SetMarkerColor(4);
  hx->SetMarkerStyle(22);
  hx->SetMarkerSize(.8);
  hx->SetMinimum(0.8);
  hx->SetMaximum(1.05);
  hx->Draw("ALP");
  leg->AddEntry(hx,"Full chamber","P");

  TGraphAsymmErrors* hxc = new TGraphAsymmErrors();
  hxc->BayesDivide(proxNc,proxDc);
  hxc->SetTitle("Efficiency along X (no Y border)");
  hxc->SetMarkerColor(2);
  hxc->SetMarkerStyle(20);
  hxc->SetMarkerSize(.8);
  hxc->Draw("LP");
  leg->AddEntry(hxc,"Excluding chamber edge","P");
  leg->Draw();

  c1->cd(2);
  TGraphAsymmErrors* hy = new TGraphAsymmErrors();
  hy->BayesDivide(proyN,proyD);
  hy->SetTitle("Efficiency along y");
  (hy->GetXaxis())->SetTitle("y (cm)");
  (hy->GetYaxis())->SetTitle("#epsilon");
  hy->SetMarkerColor(4);
  hy->SetMarkerStyle(22);
  hy->SetMinimum(0.80);
  hy->SetMaximum(1.05);
  hy->Draw("ALP");
  TGraphAsymmErrors* hyc = new TGraphAsymmErrors();
  hyc->BayesDivide(proyNc,proyDc);
  hyc->SetTitle("Efficiency along Y (no X border)");
  hyc->SetMarkerColor(2);
  hyc->SetMarkerStyle(20);
  hyc->SetMarkerSize(.8);
  hyc->Draw("LP");
  leg->Draw();

  c1->cd(3);
  // TH2F* h2n = (TH2F*)(file->Get(nameNum));
  // TH2F h2=(*h2n)/(*h2d);
  // h2.SetMinimum(0.5);
  h2.SetTitle("Efficiency vs (x,y)");
  (h2.GetXaxis())->SetTitle("x (cm)");
  (h2.GetXaxis())->SetRange(firstBinX-2,lastBinX+2);
  (h2.GetYaxis())->SetTitle("y (cm)");
  (h2.GetYaxis())->SetRange(firstBinY-2,lastBinY+2);
  h2.DrawCopy("zcol");
  TString f("hSegEff");
  c1->Print(f+ch+".eps");
  c1->Print(f+ch+".root");
}
