TFile* OpenFiles(std::string path){

  TFile *f;
  f = new TFile(path.c_str(),"READ");
  return f;

}

void make1DPlot(std::string histoname, TFile* f1, std::string histotitle, int statoption, std::string savename){

  TH1F *h1 = (TH1F*)f1->Get(histoname.c_str());

  TCanvas *c = new TCanvas("c","my canvas",1);

  if (h1){
    if (statoption == 0) gStyle->SetOptStat(kFALSE);
    else gStyle->SetOptStat(statoption);
    gStyle->SetHistFillColor(92);
    gStyle->SetFrameFillColor(4000);
    gStyle->SetTitleW(0.7);
    gStyle->SetTitleH(0.07);
    gPad->SetFillColor(4000);
    c->SetFillStyle(4000);
    gStyle->SetStatColor(0);
    gStyle->SetTitleFillColor(0);
    h1->UseCurrentStyle();

    h1->SetTitle(histotitle.c_str());
    h1->GetXaxis()->SetLabelSize(0.04);
    h1->GetYaxis()->SetLabelSize(0.04);
    h1->GetXaxis()->SetTitleOffset(0.7);
    h1->GetXaxis()->SetTitleSize(0.06);
    h1->GetXaxis()->SetNdivisions(208,kTRUE);

    h1->Draw();

    c->Update();
    c->Print(savename.c_str());
  }

}

void make1DPlot2(std::string histoname1, std::string histoname2, int statoption, TFile* f1, std::string t1, std::string t2, std::string savename){

  // use this if you want two plots on one canvas 
 
  TH1F *a1 = (TH1F*)f1->Get(histoname1.c_str());
  TH1F *b1 = (TH1F*)f1->Get(histoname2.c_str());

  TCanvas *c = new TCanvas("c","my canvas",1);
  c->Divide(1,2);

  if (a1){
    c->cd(1);
    if (statoption == 0) gStyle->SetOptStat(kFALSE);
    else gStyle->SetOptStat(statoption);
    gStyle->SetHistFillColor(92);
    gStyle->SetFrameFillColor(4000);
    gStyle->SetTitleW(0.4);
    gStyle->SetTitleH(0.09);
    gStyle->SetStatColor(0);
    gStyle->SetTitleFillColor(0);
    gPad->SetFillColor(4000);
    c->SetFillStyle(4000);
    a1->UseCurrentStyle();
    a1->SetTitle(t1.c_str());
    a1->GetXaxis()->SetLabelSize(0.06);
    a1->GetYaxis()->SetLabelSize(0.06);
    a1->GetXaxis()->SetTitleOffset(0.7);
    a1->GetXaxis()->SetTitleSize(0.06);
    a1->GetXaxis()->SetNdivisions(208,kTRUE);
    a1->Draw();
  }

  if (b1){
    gStyle->SetHistFillColor(72);
    b1->UseCurrentStyle();

    t2 = t2 + " (run " + run + ")";
    b1->SetTitle(t2.c_str());
    b1->GetXaxis()->SetLabelSize(0.06);
    b1->GetYaxis()->SetLabelSize(0.06);
    b1->GetXaxis()->SetTitleOffset(0.7);
    b1->GetXaxis()->SetTitleSize(0.06);
    b1->GetXaxis()->SetNdivisions(508,kTRUE);
    c->cd(2);
    b1->Draw();
  }

  c->Update();
  c->Print(savename.c_str());

}

void makeEffGif(std::string histoname, TFile* f1, std::string histotitle, std::string savename){


  TH1F *h1 = (TH1F*)f1->Get(histoname.c_str());

  TCanvas *c = new TCanvas("c","my canvas",1);


  if (h1){
    gStyle->SetOptStat(kFALSE);
    gStyle->SetHistFillColor(92);
    gStyle->SetFrameFillColor(4000);
    gStyle->SetTitleW(0.7);
    gStyle->SetTitleH(0.07);
    gPad->SetFillColor(4000);
    gStyle->SetStatColor(0);
    gStyle->SetTitleFillColor(0);
    c->SetFillStyle(4000);
    h1->UseCurrentStyle();

    h1->SetTitle(histotitle.c_str());
    h1->GetXaxis()->SetLabelSize(0.04);
    h1->GetYaxis()->SetLabelSize(0.04);
    h1->GetXaxis()->SetTitleOffset(0.7);
    h1->GetXaxis()->SetTitleSize(0.06);
    h1->GetXaxis()->SetNdivisions(208,kTRUE);
    h1->GetYaxis()->SetRangeUser(0.5,1.1);
    h1->SetMarkerStyle(6);
    h1->GetXaxis()->SetBinLabel(1,"ME +1/1b");
    h1->GetXaxis()->SetBinLabel(2,"ME +1/2");
    h1->GetXaxis()->SetBinLabel(3,"ME +1/3");
    h1->GetXaxis()->SetBinLabel(4,"ME +1/1a");
    h1->GetXaxis()->SetBinLabel(5,"ME +2/1");
    h1->GetXaxis()->SetBinLabel(6,"ME +2/2");
    h1->GetXaxis()->SetBinLabel(7,"ME +3/1");
    h1->GetXaxis()->SetBinLabel(8,"ME +3/2");
    h1->GetXaxis()->SetBinLabel(9,"ME +4/1");
    h1->GetXaxis()->SetBinLabel(10,"ME +4/2");
    h1->GetXaxis()->SetBinLabel(11,"ME -1/1b");
    h1->GetXaxis()->SetBinLabel(12,"ME -1/2");
    h1->GetXaxis()->SetBinLabel(13,"ME -1/3");
    h1->GetXaxis()->SetBinLabel(14,"ME -1/1a");
    h1->GetXaxis()->SetBinLabel(15,"ME -2/1");
    h1->GetXaxis()->SetBinLabel(16,"ME -2/2");
    h1->GetXaxis()->SetBinLabel(17,"ME -3/1");
    h1->GetXaxis()->SetBinLabel(18,"ME -3/2");
    h1->GetXaxis()->SetBinLabel(19,"ME -4/1");
    h1->GetXaxis()->SetBinLabel(20,"ME -4/2");
    h1->Draw();
    c->Update();
    c->Print(savename.c_str());
  }
}

void Draw2DTempPlot(std::string histo, TFile* f1, std::string savename){

  TCanvas *c = new TCanvas("c","my canvas",1);
  gStyle->SetPalette(1,0);
  gPad->SetFillColor(4000);
  c->SetFillStyle(4000);
  gStyle->SetStatColor(0);
  gStyle->SetTitleFillColor(0);

  TH2I *plot  = (TH2I*)f1->Get(histo.c_str());

  plot->SetStats(kFALSE);
  
  plot->GetYaxis()->SetBinLabel(1,"ME- 4/2");
  plot->GetYaxis()->SetBinLabel(2,"ME- 4/1");
  plot->GetYaxis()->SetBinLabel(3,"ME- 3/2");
  plot->GetYaxis()->SetBinLabel(4,"ME- 3/1");
  plot->GetYaxis()->SetBinLabel(5,"ME- 2/2");
  plot->GetYaxis()->SetBinLabel(6,"ME- 2/1");
  plot->GetYaxis()->SetBinLabel(10,"ME- 1/1a");
  plot->GetYaxis()->SetBinLabel(7,"ME- 1/3");
  plot->GetYaxis()->SetBinLabel(8,"ME- 1/2");
  plot->GetYaxis()->SetBinLabel(9,"ME- 1/1b");
  plot->GetYaxis()->SetBinLabel(12,"ME+ 1/1b");
  plot->GetYaxis()->SetBinLabel(13,"ME+ 1/2");
  plot->GetYaxis()->SetBinLabel(14,"ME+ 1/3");
  plot->GetYaxis()->SetBinLabel(11,"ME+ 1/1a");
  plot->GetYaxis()->SetBinLabel(15,"ME+ 2/1");
  plot->GetYaxis()->SetBinLabel(16,"ME+ 2/2");
  plot->GetYaxis()->SetBinLabel(17,"ME+ 3/1");
  plot->GetYaxis()->SetBinLabel(18,"ME+ 3/2");
  plot->GetYaxis()->SetBinLabel(19,"ME+ 4/1");
  plot->GetYaxis()->SetBinLabel(20,"ME+ 4/2");

  for (int i = 1; i < 37; i++){
    ostringstream oss1;
    oss1 << i;
    string ch = oss1.str();
    plot->GetXaxis()->SetBinLabel(i,ch.c_str());
  }

  c->SetRightMargin(0.12);

  plot->GetYaxis()->SetNdivisions(20,kFALSE);
  plot->GetXaxis()->SetNdivisions(36,kFALSE);

  plot->GetXaxis()->SetTitle("Chamber #");

  c->SetGrid();

  plot->Draw("COLZ");

  c->Update();
  c->Print(savename.c_str());


}

void GlobalPosfromTree(std::string graphname, TFile* f1, int endcap, int station, std::string type, std::string savename){

  TTree *t1;
  TBranch *b1;
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
  } points;


  if (type == "rechit"){
    t1 = (TTree*)f1->Get("recHits/rHPositions");
    b1 = t1->GetBranch("rHpos");
    b1->SetAddress(&points);
  }

  if (type == "segment"){
    t1 = (TTree*)f1->Get("Segments/segPositions");
    b1 = t1->GetBranch("segpos");
    b1->SetAddress(&points);
  }

  int n1 = (int)t1->GetEntries();
  const int nevents1 = n1;
  float globx1[nevents1];
  float globy1[nevents1];
  int nstation1 = 0;

  for (int i=0; i<nevents1; i++){
    b1->GetEntry(i);
    if (points.station == station && points.endcap == endcap){
      globx1[nstation1] = points.globalx;
      globy1[nstation1] = points.globaly;
      nstation1++;
    }
  }

  TCanvas *c = new TCanvas("c","my canvas",1);
  c->SetCanvasSize(700,700);
  TGraph *graph1 = new TGraph(nstation1,globx1,globy1);

  std::string name1 = graphname;

  gStyle->SetPalette(1,0);
  gStyle->SetTitleW(0.7);
  gStyle->SetTitleH(0.1);
  gStyle->SetStatColor(0);
  gStyle->SetTitleFillColor(0);
  gPad->SetFillColor(4000);
  c->SetFillStyle(4000);
  gStyle->SetOptStat(10);


  graph1->GetXaxis()->SetLimits(-720,720);
  graph1->GetYaxis()->SetLimits(-720,720);
  graph1->GetXaxis()->SetRangeUser(-720,720);
  graph1->GetYaxis()->SetRangeUser(-720,720);

  graph1->SetTitle(name1.c_str());
  graph1->UseCurrentStyle();
  graph1->Draw("AP");

  drawChamberLines(station);

  c->Print(savename.c_str());

}

drawChamberLines(int station){

  gStyle->SetLineWidth(2);
  float pi = 3.14159;
  TVector3 x(0,0,1);
  int linecolor = 1;
  //for alternating colors, set 2 diff colors here
  int lc1 = 1;
  int lc2 = 1;


  if (station == 1){
    TVector3 p1(101,9.361,0);
    TVector3 p2(101,-9.361,0);
    TVector3 p3(260,-22.353,0);
    TVector3 p4(260,22.353,0);

    TLine *line1; 
    TLine *line2;
    TLine *line3;
    TLine *line4;

    for (int i = 0; i < 36; i++){

      if (linecolor == lc1) linecolor = lc2;
      else linecolor = lc1;

      line1 = new TLine(p1(0),p1(1),p2(0),p2(1));
      line2 = new TLine(p2(0),p2(1),p3(0),p3(1));
      line3 = new TLine(p3(0),p3(1),p4(0),p4(1));
      line4 = new TLine(p4(0),p4(1),p1(0),p1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      p1.Rotate(pi/18,x);
      p2.Rotate(pi/18,x);
      p3.Rotate(pi/18,x);
      p4.Rotate(pi/18,x);

    }

    TVector3 q1(281.49,25.5,0);
    TVector3 q2(281.49,-25.5,0);
    TVector3 q3(455.99,-41.87,0);
    TVector3 q4(455.99,41.87,0);

    for (int i = 0; i < 36; i++){
 
      if (linecolor == lc2) linecolor = lc1;
      else linecolor = lc2;

      line1 = new TLine(q1(0),q1(1),q2(0),q2(1));
      line2 = new TLine(q2(0),q2(1),q3(0),q3(1));
      line3 = new TLine(q3(0),q3(1),q4(0),q4(1));
      line4 = new TLine(q4(0),q4(1),q1(0),q1(1));

      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      q1.Rotate(pi/18,x);
      q2.Rotate(pi/18,x);
      q3.Rotate(pi/18,x);
      q4.Rotate(pi/18,x);

    }

    TVector3 r1(511.99,31.7,0);
    TVector3 r2(511.99,-31.7,0);
    TVector3 r3(676.15,-46.05,0);
    TVector3 r4(676.15,46.05.6,0);

    for (int i = 0; i < 36; i++){

      if (linecolor == lc1) linecolor = lc2;
      else linecolor = lc1;

      line1 = new TLine(r1(0),r1(1),r2(0),r2(1));
      line2 = new TLine(r2(0),r2(1),r3(0),r3(1));
      line3 = new TLine(r3(0),r3(1),r4(0),r4(1));
      line4 = new TLine(r4(0),r4(1),r1(0),r1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      r1.Rotate(pi/18,x);
      r2.Rotate(pi/18,x);
      r3.Rotate(pi/18,x);
      r4.Rotate(pi/18,x);

    }

  }


  if (station == 2){

    TVector3 p1(146.9,27.0,0);
    TVector3 p2(146.9,-27.0,0);
    TVector3 p3(336.56,-62.855,0);
    TVector3 p4(336.56,62.855,0);

    p1.Rotate(pi/36,x);
    p2.Rotate(pi/36,x);
    p3.Rotate(pi/36,x);
    p4.Rotate(pi/36,x);

    TLine *line1;
    TLine *line2;
    TLine *line3;
    TLine *line4;

    for (int i = 0; i < 36; i++){

      if (linecolor == lc1) linecolor = lc2;
      else linecolor = lc1;

      line1 = new TLine(p1(0),p1(1),p2(0),p2(1));
      line2 = new TLine(p2(0),p2(1),p3(0),p3(1));
      line3 = new TLine(p3(0),p3(1),p4(0),p4(1));
      line4 = new TLine(p4(0),p4(1),p1(0),p1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      p1.Rotate(pi/9,x);
      p2.Rotate(pi/9,x);
      p3.Rotate(pi/9,x);
      p4.Rotate(pi/9,x);

    }

    TVector3 q1(364.02,33.23,0);
    TVector3 q2(364.02,-33.23,0);
    TVector3 q3(687.08,-63.575,0);
    TVector3 q4(687.08,63.575,0);

    for (int i = 0; i < 36; i++){

      if (linecolor == lc2) linecolor = lc1;
      else linecolor = lc2;

      line1 = new TLine(q1(0),q1(1),q2(0),q2(1));
      line2 = new TLine(q2(0),q2(1),q3(0),q3(1));
      line3 = new TLine(q3(0),q3(1),q4(0),q4(1));
      line4 = new TLine(q4(0),q4(1),q1(0),q1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      q1.Rotate(pi/18,x);
      q2.Rotate(pi/18,x);
      q3.Rotate(pi/18,x);
      q4.Rotate(pi/18,x);

    }

  }

  if (station == 3){

    TVector3 p1(166.89,30.7,0);
    TVector3 p2(166.89,-30.7,0);
    TVector3 p3(336.59,-62.855,0);
    TVector3 p4(336.59,62.855,0);

    p1.Rotate(pi/36,x);
    p2.Rotate(pi/36,x);
    p3.Rotate(pi/36,x);
    p4.Rotate(pi/36,x);

    TLine *line1;
    TLine *line2;
    TLine *line3;
    TLine *line4;

    for (int i = 0; i < 36; i++){

      if (linecolor == lc1) linecolor = lc2;
      else linecolor = lc1;

      line1 = new TLine(p1(0),p1(1),p2(0),p2(1));
      line2 = new TLine(p2(0),p2(1),p3(0),p3(1));
      line3 = new TLine(p3(0),p3(1),p4(0),p4(1));
      line4 = new TLine(p4(0),p4(1),p1(0),p1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      p1.Rotate(pi/9,x);
      p2.Rotate(pi/9,x);
      p3.Rotate(pi/9,x);
      p4.Rotate(pi/9,x);

    }

    TVector3 q1(364.02,33.23,0);
    TVector3 q2(364.02,-33.23,0);
    TVector3 q3(687.08,-63.575,0);
    TVector3 q4(687.08,63.575,0);

    for (int i = 0; i < 36; i++){

      if (linecolor == lc2) linecolor = lc1;
      else linecolor = lc2;

      line1 = new TLine(q1(0),q1(1),q2(0),q2(1));
      line2 = new TLine(q2(0),q2(1),q3(0),q3(1));
      line3 = new TLine(q3(0),q3(1),q4(0),q4(1));
      line4 = new TLine(q4(0),q4(1),q1(0),q1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      q1.Rotate(pi/18,x);
      q2.Rotate(pi/18,x);
      q3.Rotate(pi/18,x);
      q4.Rotate(pi/18,x);

    }

  }

  if (station == 4){

    TVector3 p1(186.99,34.505.15,0);
    TVector3 p2(186.99,-34.505,0);
    TVector3 p3(336.41,-62.825,0);
    TVector3 p4(336.41,62.825,0);

    p1.Rotate(pi/36,x);
    p2.Rotate(pi/36,x);
    p3.Rotate(pi/36,x);
    p4.Rotate(pi/36,x);

    TLine *line1;
    TLine *line2;
    TLine *line3;
    TLine *line4;

    for (int i = 0; i < 36; i++){

      if (linecolor == lc1) linecolor = lc2;
      else linecolor = lc1;

      line1 = new TLine(p1(0),p1(1),p2(0),p2(1));
      line2 = new TLine(p2(0),p2(1),p3(0),p3(1));
      line3 = new TLine(p3(0),p3(1),p4(0),p4(1));
      line4 = new TLine(p4(0),p4(1),p1(0),p1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      p1.Rotate(pi/9,x);
      p2.Rotate(pi/9,x);
      p3.Rotate(pi/9,x);
      p4.Rotate(pi/9,x);

    }

  }

} 

void compare1DPlot(std::string histoname, TFile* f1, TFile* f2, std::string histotitle, int statoption, std::string savename){ 

  // used to compare two of the same histogram from different releases/runs/etc

  TH1F *h2  = (TH1F*)f1->Get(histoname.c_str());
  TH1F *h1  = (TH1F*)f2->Get(histoname.c_str());

  TCanvas *c = new TCanvas("c","my canvas",1);

  if (h1 && h2){
    gStyle->SetHistFillColor(92);
    gStyle->SetFrameFillColor(4000);
    gStyle->SetTitleW(0.3);
    gStyle->SetTitleH(0.07);
    gStyle->SetStatColor(0);
    gStyle->SetTitleFillColor(0);
    gPad->SetFillColor(4000);
    c->SetFillStyle(4000);
    gStyle->SetOptStat(statoption);
    h1->UseCurrentStyle();
    h2->UseCurrentStyle();
    h2->SetFillColor(52);

    h1->SetTitle(histotitle.c_str());
    h1->GetXaxis()->SetLabelSize(0.04);
    h1->GetYaxis()->SetLabelSize(0.04);
    h1->GetXaxis()->SetTitleOffset(0.7);
    h1->GetXaxis()->SetTitleSize(0.06);
    h1->GetXaxis()->SetNdivisions(208,kTRUE);

    TLegend *leg = new TLegend(0.6,0.6,0.8,0.8);
    leg->AddEntry(h1,"ref","f");
    leg->AddEntry(h2,"new","l");

    h1->Draw();
    h2->Draw("same e");
    leg->Draw();

  }

  c->Update();
  c->Print(savename.c_str());

}

void compareEffGif(std::string histoname, TFile* f1, TFile* f2, std::string histotitle, std::string savename){


  TH1F *h1 = (TH1F*)f1->Get(histoname.c_str());
  TH1F *h2 = (TH1F*)f2->Get(histoname.c_str());

  TCanvas *c = new TCanvas("c","my canvas",1);


  if (h1 && h2){
    gStyle->SetOptStat(kFALSE);
    gStyle->SetHistFillColor(92);
    gStyle->SetFrameFillColor(4000);
    gStyle->SetTitleW(0.7);
    gStyle->SetTitleH(0.07);
    gPad->SetFillColor(4000);
    gStyle->SetStatColor(0);
    gStyle->SetTitleFillColor(0);
    c->SetFillStyle(4000);
    h1->UseCurrentStyle();

    h1->SetTitle(histotitle.c_str());
    h1->GetXaxis()->SetLabelSize(0.04);
    h1->GetYaxis()->SetLabelSize(0.04);
    h1->GetXaxis()->SetTitleOffset(0.7);
    h1->GetXaxis()->SetTitleSize(0.06);
    h1->GetXaxis()->SetNdivisions(208,kTRUE);
    h1->GetYaxis()->SetRangeUser(0.5,1.1);
    h1->SetMarkerStyle(6);
    h1->SetMarkerColor(kBlue);
    h2->SetMarkerStyle(6);
    h2->SetMarkerColor(kRed);
    h1->GetXaxis()->SetBinLabel(1,"ME +1/1b");
    h1->GetXaxis()->SetBinLabel(2,"ME +1/2");
    h1->GetXaxis()->SetBinLabel(3,"ME +1/3");
    h1->GetXaxis()->SetBinLabel(4,"ME +1/1a");
    h1->GetXaxis()->SetBinLabel(5,"ME +2/1");
    h1->GetXaxis()->SetBinLabel(6,"ME +2/2");
    h1->GetXaxis()->SetBinLabel(7,"ME +3/1");
    h1->GetXaxis()->SetBinLabel(8,"ME +3/2");
    h1->GetXaxis()->SetBinLabel(9,"ME +4/1");
    h1->GetXaxis()->SetBinLabel(10,"ME +4/2");
    h1->GetXaxis()->SetBinLabel(11,"ME -1/1b");
    h1->GetXaxis()->SetBinLabel(12,"ME -1/2");
    h1->GetXaxis()->SetBinLabel(13,"ME -1/3");
    h1->GetXaxis()->SetBinLabel(14,"ME -1/1a");
    h1->GetXaxis()->SetBinLabel(15,"ME -2/1");
    h1->GetXaxis()->SetBinLabel(16,"ME -2/2");
    h1->GetXaxis()->SetBinLabel(17,"ME -3/1");
    h1->GetXaxis()->SetBinLabel(18,"ME -3/2");
    h1->GetXaxis()->SetBinLabel(19,"ME -4/1");
    h1->GetXaxis()->SetBinLabel(20,"ME -4/2");

    TLegend *leg = new TLegend(0.6,0.7,0.89,0.89);
    leg->AddEntry(h1,"new","p");
    leg->AddEntry(h2,"ref","p");

    h1->Draw();
    h2->Draw("same");
    leg->Draw();
    c->Update();
    c->Print(savename.c_str());
  }
}


void GlobalPosfromTreeCompare(std::string graphname, TFile* f1, TFile* f2, int endcap, int station, std::string type, std::string savename){

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
  } points1, points2;

  TTree *t1;
  TTree *t2;
  TBranch *b1;
  TBranch *b2;

  if (type == "rechit"){
    t1 = (TTree*)f1->Get("recHits/rHPositions");
    t2 = (TTree*)f2->Get("recHits/rHPositions");
    b1 = t1->GetBranch("rHpos");
    b2 = t2->GetBranch("rHpos");
    b1->SetAddress(&points1);
    b2->SetAddress(&points2);
  }

  if (type == "segment"){
    t1 = (TTree*)f1->Get("Segments/segPositions");
    t2 = (TTree*)f2->Get("Segments/segPositions");
    b1 = t1->GetBranch("segpos");
    b2 = t2->GetBranch("segpos");
    b1->SetAddress(&points1);
    b2->SetAddress(&points2);
  }

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
    if (points1.station == station){
      globx1[nstation1] = points1.globalx;
      globy1[nstation1] = points1.globaly;
      nstation1++;
    }
  }
  for (int i=0; i<nevents2; i++){
    b2->GetEntry(i);
    if (points2.station == station){
      globx2[nstation2] = points2.globalx;
      globy2[nstation2] = points2.globaly;
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
  gStyle->SetStatColor(0);
  gStyle->SetTitleFillColor(0);
  gStyle->SetOptStat(10);

  c->cd(1);
  graph1->SetTitle(name1.c_str());
  graph1->UseCurrentStyle();
  drawChamberLines(station);
  graph1->Draw("AP");
  c->cd(2);
  graph2->SetTitle(name2.c_str());
  graph2->UseCurrentStyle();
  drawChamberLines(station);
  graph2->Draw("AP");

  //c->Update();
  c->Print(savename.c_str());

}


