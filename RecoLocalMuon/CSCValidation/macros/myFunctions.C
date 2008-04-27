TFile* OpenFiles(std::string path, int datatype){

  std::string mtcc  = "validationHists_mtcc.root";
  std::string mc    = "validationHists_muongun.root";

  if (datatype == 1){
    std::string file = path + mtcc;
  }
  if (datatype == 2){
    std::string file = path + mc;
  }


  TFile *f;
  f = new TFile(file.c_str(),"READ");
  return f;

}

void Compare1DPlots1(std::string histoname, TFile* f1, TFile* f2, std::string histotitle, std::string savename){

  TH1F *h1  = (TH1F*)f1->Get(histoname.c_str());
  TH1F *h2  = (TH1F*)f2->Get(histoname.c_str());

  TCanvas *c = new TCanvas("c","my canvas",1);

  if (h1 && h2){
    gStyle->SetHistFillColor(92);
    gStyle->SetFrameFillColor(4000);
    gStyle->SetTitleW(0.3);
    gStyle->SetTitleH(0.07);
    gPad->SetFillColor(4000);
    c->SetFillStyle(4000);
    gStyle->SetOptStat(10);
    h1->UseCurrentStyle();
    h2->UseCurrentStyle();
    h2->SetFillColor(52);

    h1->SetTitle(histotitle.c_str());
    h1->GetXaxis()->SetLabelSize(0.04);
    h1->GetYaxis()->SetLabelSize(0.04);
    h1->GetXaxis()->SetTitleOffset(0.7);
    h1->GetXaxis()->SetTitleSize(0.06);
    h1->GetXaxis()->SetNdivisions(208,kTRUE);

    h1->Draw();
    h2->Draw("same e");

  }

  c->Update();
  c->Print(savename.c_str());

}

void Compare1DPlots2(std::string histoname1, std::string histoname2, TFile* f1, TFile* f2, std::string t1, std::string t2, std::string savename){

  // This macro compares two sets of CSCLocalValidn histograms.
  //
  TH1F *a1  = (TH1F*)f1->Get(histoname1.c_str());
  TH1F *b1  = (TH1F*)f1->Get(histoname2.c_str());

  TH1F *a2  = (TH1F*)f2->Get(histoname1.c_str());
  TH1F *b2  = (TH1F*)f2->Get(histoname2.c_str());


  TCanvas *c = new TCanvas("c","my canvas",1);
  c->Divide(1,2);
  c->cd(1);

  if (a1 && a2){
    gStyle->SetHistFillColor(92);
    gStyle->SetFrameFillColor(4000);
    gStyle->SetTitleW(0.4);
    gStyle->SetTitleH(0.09);
    gPad->SetFillColor(4000);
    c->SetFillStyle(4000);
    gStyle->SetOptStat(10);
    a1->UseCurrentStyle();
    a2->UseCurrentStyle();
    a2->SetFillColor(52);


    a1->SetTitle(t1.c_str());
    a1->GetXaxis()->SetLabelSize(0.06);
    a1->GetYaxis()->SetLabelSize(0.06);
    a1->GetXaxis()->SetTitleOffset(0.7);
    a1->GetXaxis()->SetTitleSize(0.06);
    a1->GetXaxis()->SetNdivisions(208,kTRUE);
    a1->Draw();
    a2->Draw("same e");
  }

  if (b1 && b2){
    gStyle->SetHistFillColor(72);
    b1->UseCurrentStyle();
    b2->UseCurrentStyle();
    b2->SetFillColor(52);

    b1->SetTitle(t2.c_str());
    b1->GetXaxis()->SetLabelSize(0.06);
    b1->GetYaxis()->SetLabelSize(0.06);
    b1->GetXaxis()->SetTitleOffset(0.7);
    b1->GetXaxis()->SetTitleSize(0.06);
    b1->GetXaxis()->SetNdivisions(508,kTRUE);
    c->cd(2);
    b1->Draw();
    b2->Draw("same e1");

  }

  c->Update();
  c->Print(savename.c_str());

}


void GlobalrHPosfromTree(std::string graphname, TFile* f1, TFile* f2, int station, std::string savename){

  struct posRecord {
    int endcap;
    int station;
    int ring;
    int chamber;
    int layer;
    float localx;
    float localy;
    float globalx;
    float globaly;
  } rHpos1, rHpos2;
  TTree *t1 = (TTree*)f1->Get("recHits/rHPositions");
  TTree *t2 = (TTree*)f2->Get("recHits/rHPositions");
  TBranch *b1 = t1->GetBranch("rHpos");
  TBranch *b2 = t2->GetBranch("rHpos");
  b1->SetAddress(&rHpos1);
  b2->SetAddress(&rHpos2);

  int n1 = (int)t1->GetEntries();
  int n2 = (int)t2->GetEntries();

  const int nevents1 = n1;
  const int nevents2 = n2;

  //cout << n1 << " " << n2 << " " << nevents1 << " " << nevents2 << endl;

  float globx1[nevents1];
  float globy1[nevents1];
  float globx2[nevents2];
  float globy2[nevents2];
  int nstation1 = 0;
  int nstation2 = 0;

  for (int i=0; i<nevents1; i++){
    b1->GetEntry(i);
    if (rHpos1.station == station){
      globx1[nstation1] = rHpos1.globalx;
      globy1[nstation1] = rHpos1.globaly;
      nstation1++;
    }
  }
  for (int i=0; i<nevents2; i++){
    b2->GetEntry(i);
    if (rHpos2.station == station){
      globx2[nstation2] = rHpos2.globalx;
      globy2[nstation2] = rHpos2.globaly;
      nstation2++;
    }
  }

  std::string name1 = graphname + " (Ref)";
  std::string name2 = graphname + " (New)";
  TCanvas *c = new TCanvas("c","my canvas",1);
  c->SetCanvasSize(1300,700);
  c->Divide(2,1);
  TGraph *graph1 = new TGraph(nstation1,globx1,globy1);
  TGraph *graph2 = new TGraph(nstation2,globx2,globy2);


  std::string name1 = graphname + " (Ref)";
  std::string name2 = graphname + " (New)";

  gStyle->SetTitleW(0.6);
  gStyle->SetTitleH(0.1);

  c->cd(1);
  graph1->SetTitle(name1.c_str());
  graph1->UseCurrentStyle();
  graph1->Draw("AP");
  c->cd(2);
  graph2->SetTitle(name2.c_str());
  graph2->UseCurrentStyle();
  graph2->Draw("AP");

  //c->Update();
  c->Print(savename.c_str());

} 

void GlobalsegPosfromTree(std::string graphname, TFile* f1, TFile* f2, int station, std::string savename){

  struct posRecord {
    int endcap;
    int station;
    int ring;
    int chamber;
    int layer;
    float localx;
    float localy;
    float globalx;
    float globaly;
  } segpos1, segpos2;
  TTree *t1 = (TTree*)f1->Get("Segments/segPositions");
  TTree *t2 = (TTree*)f2->Get("Segments/segPositions");
  TBranch *b1 = t1->GetBranch("segpos");
  TBranch *b2 = t2->GetBranch("segpos");
  b1->SetAddress(&segpos1);
  b2->SetAddress(&segpos2);

  int n1 = (int)t1->GetEntries();
  int n2 = (int)t2->GetEntries();

  const int nevents1 = n1;
  const int nevents2 = n2;

  //cout << n1 << " " << n2 << " " << nevents1 << " " << nevents2 << endl;

  float globx1[nevents1];
  float globy1[nevents1];
  float globx2[nevents2];
  float globy2[nevents2];
  int nstation1 = 0;
  int nstation2 = 0;

  for (int i=0; i<nevents1; i++){
    b1->GetEntry(i);
    if (segpos1.station == station){
      globx1[nstation1] = segpos1.globalx;
      globy1[nstation1] = segpos1.globaly;
      nstation1++;
    }
  }
  for (int i=0; i<nevents2; i++){
    b2->GetEntry(i);
    if (segpos2.station == station){
      globx2[nstation2] = segpos2.globalx;
      globy2[nstation2] = segpos2.globaly;
      nstation2++;
    }
  }

  std::string name1 = graphname + " (Ref)";
  std::string name2 = graphname + " (New)";
  TCanvas *c = new TCanvas("c","my canvas",1);
  c->SetCanvasSize(1300,700);
  c->Divide(2,1);
  TGraph *graph1 = new TGraph(nstation1,globx1,globy1);
  TGraph *graph2 = new TGraph(nstation2,globx2,globy2);

  gStyle->SetTitleW(0.6);
  gStyle->SetTitleH(0.1);

  c->cd(1);
  graph1->SetTitle(name1.c_str());
  graph1->UseCurrentStyle();
  graph1->Draw("AP");
  c->cd(2);
  graph2->SetTitle(name2.c_str());
  graph2->UseCurrentStyle();
  graph2->Draw("AP");

  //c->Update();
  c->Print(savename.c_str());

}
 
