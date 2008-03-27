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
  gStyle->SetHistFillColor(92);
  gStyle->SetFrameFillColor(4000);
  gStyle->SetTitleW(0.3);
  gStyle->SetTitleH(0.07);
  gPad->SetFillColor(4000);
  c->SetFillStyle(4000);
  gStyle->SetOptStat(10);
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

void Compare1DPlots2(std::string histoname1, std::string histoname2, TFile* f1, TFile* f2, std::string t1, std::string t2, std::string savename){

  // This macro compares two sets of CSCLocalValidn histograms.
  //
  TH1F *a1  = (TH1F*)f1->Get(histoname1.c_str());
  TH1F *b1  = (TH1F*)f1->Get(histoname2.c_str());

  TH1F *a2  = (TH1F*)f2->Get(histoname1.c_str());
  TH1F *b2  = (TH1F*)f2->Get(histoname2.c_str());


  TCanvas *c = new TCanvas("c","my canvas",1);
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


  c->Divide(1,2);
  c->cd(1);
  a1->Draw();
  c->cd(2);
  b1->Draw();

  c->Update();
  c->Print(savename.c_str());

}

void Draw2DTempPlot(std::string histo, TFile* f1, std::string savename){

  TCanvas *c = new TCanvas("c","my canvas",1);
  gStyle->SetPalette(1,0);
  TH2I *plot  = (TH2I*)f1->Get(histo.c_str());

  plot->SetStats(kFALSE);
  
  plot->GetYaxis()->SetBinLabel(1,"ME- 4/2");
  plot->GetYaxis()->SetBinLabel(2,"ME- 4/1");
  plot->GetYaxis()->SetBinLabel(3,"ME- 3/2");
  plot->GetYaxis()->SetBinLabel(4,"ME- 3/1");
  plot->GetYaxis()->SetBinLabel(5,"ME- 2/2");
  plot->GetYaxis()->SetBinLabel(6,"ME- 2/1");
  plot->GetYaxis()->SetBinLabel(7,"ME- 1/1a");
  plot->GetYaxis()->SetBinLabel(8,"ME- 1/3");
  plot->GetYaxis()->SetBinLabel(9,"ME- 1/2");
  plot->GetYaxis()->SetBinLabel(10,"ME- 1/1b");
  plot->GetYaxis()->SetBinLabel(11,"ME+ 1/1b");
  plot->GetYaxis()->SetBinLabel(12,"ME+ 1/2");
  plot->GetYaxis()->SetBinLabel(13,"ME+ 1/3");
  plot->GetYaxis()->SetBinLabel(14,"ME+ 1/1a");
  plot->GetYaxis()->SetBinLabel(15,"ME+ 2/1");
  plot->GetYaxis()->SetBinLabel(16,"ME+ 2/2");
  plot->GetYaxis()->SetBinLabel(17,"ME+ 3/1");
  plot->GetYaxis()->SetBinLabel(18,"ME+ 3/2");
  plot->GetYaxis()->SetBinLabel(19,"ME+ 4/1");
  plot->GetYaxis()->SetBinLabel(20,"ME+ 4/2");

  c->SetRightMargin(0.12);

  plot->GetXaxis()->SetBinLabel(1,"1");
  plot->GetXaxis()->SetBinLabel(2,"2");
  plot->GetXaxis()->SetBinLabel(3,"3");
  plot->GetXaxis()->SetBinLabel(4,"4");
  plot->GetXaxis()->SetBinLabel(5,"5");
  plot->GetXaxis()->SetBinLabel(6,"6");
  plot->GetXaxis()->SetBinLabel(7,"7");
  plot->GetXaxis()->SetBinLabel(8,"8");
  plot->GetXaxis()->SetBinLabel(9,"9");
  plot->GetXaxis()->SetBinLabel(10,"10");
  plot->GetXaxis()->SetBinLabel(11,"11");
  plot->GetXaxis()->SetBinLabel(12,"12");
  plot->GetXaxis()->SetBinLabel(13,"13");
  plot->GetXaxis()->SetBinLabel(14,"14");
  plot->GetXaxis()->SetBinLabel(15,"15");
  plot->GetXaxis()->SetBinLabel(16,"16");
  plot->GetXaxis()->SetBinLabel(17,"17");
  plot->GetXaxis()->SetBinLabel(18,"18");
  plot->GetXaxis()->SetBinLabel(19,"19");
  plot->GetXaxis()->SetBinLabel(20,"20");
  plot->GetXaxis()->SetBinLabel(21,"21");
  plot->GetXaxis()->SetBinLabel(22,"22");
  plot->GetXaxis()->SetBinLabel(23,"23");
  plot->GetXaxis()->SetBinLabel(24,"24");
  plot->GetXaxis()->SetBinLabel(25,"25");
  plot->GetXaxis()->SetBinLabel(26,"26");
  plot->GetXaxis()->SetBinLabel(27,"27");
  plot->GetXaxis()->SetBinLabel(28,"28");
  plot->GetXaxis()->SetBinLabel(29,"29");
  plot->GetXaxis()->SetBinLabel(30,"30");
  plot->GetXaxis()->SetBinLabel(31,"31");
  plot->GetXaxis()->SetBinLabel(32,"32");
  plot->GetXaxis()->SetBinLabel(33,"33");
  plot->GetXaxis()->SetBinLabel(34,"34");
  plot->GetXaxis()->SetBinLabel(35,"35");
  plot->GetXaxis()->SetBinLabel(36,"36");

  plot->GetYaxis()->SetNdivisions(20,kFALSE);
  plot->GetXaxis()->SetNdivisions(36,kFALSE);

  plot->GetXaxis()->SetTitle("Chamber #");

  c->SetGrid();

  plot->Draw("COLZ");

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
  c->SetCanvasSize(700,700);
  TGraph *graph1 = new TGraph(nstation1,globx1,globy1);
  TGraph *graph2 = new TGraph(nstation2,globx2,globy2);


  std::string name1 = graphname;
  std::string name2 = graphname;

  gStyle->SetTitleW(0.6);
  gStyle->SetTitleH(0.1);

  graph1->GetXaxis()->SetLimits(-720,720);
  graph1->GetYaxis()->SetLimits(-720,720);
  graph2->GetXaxis()->SetLimits(-720,720);
  graph2->GetYaxis()->SetLimits(-720,720);
  graph1->GetXaxis()->SetRangeUser(-720,720);
  graph1->GetYaxis()->SetRangeUser(-720,720);
  graph2->GetXaxis()->SetRangeUser(-720,720);
  graph2->GetYaxis()->SetRangeUser(-720,720);

  graph1->SetTitle(name1.c_str());
  graph1->UseCurrentStyle();
  graph1->Draw("AP");

  gStyle->SetLineWidth(2);
  float pi = 3.14159;
  TVector3 x(0,0,1);
  int linecolor = 1;
  //for alternating colors, set 2 diff colors here
  int lc1 = 1;
  int lc2 = 1;


  if (station == 1){
    TVector3 p1(97.5,15.55,0);
    TVector3 p2(97.5,-15.55,0);
    TVector3 p3(265.5,-30.65,0);
    TVector3 p4(265.5,30.65,0);

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

    TVector3 q1(279.7,37.0,0);
    TVector3 q2(279.7,-37.0,0);
    TVector3 q3(459.7,-53.9,0);
    TVector3 q4(459.7,53.9,0);

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

    TVector3 r1(490.15,42.95,0);
    TVector3 r2(490.15,-42.95,0);
    TVector3 r3(680.15,-59.6,0);
    TVector3 r4(680.15,59.6,0);

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
    TVector3 p1(139.45,37.55,0);
    TVector3 p2(139.45,-37.55,0);
    TVector3 p3(345.95,-76.7,0);
    TVector3 p4(345.59,76.7,0);

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

    TVector3 q1(357.25,44.75,0);
    TVector3 q2(357.25,-44.75,0);
    TVector3 q3(695.25,-76.5,0);
    TVector3 q4(695.25,76.5,0);

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
    TVector3 p1(160.45,41.75,0);
    TVector3 p2(160.45,-41.75,0);
    TVector3 p3(344.95,-76.7,0);
    TVector3 p4(344.95,76.7,0);


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

    TVector3 q1(357.25,44.75,0);
    TVector3 q2(357.25,-44.75,0);
    TVector3 q3(695.25,-76.5,0);
    TVector3 q4(695.25,76.5,0);

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
    TVector3 p1(179.4,45.15,0);
    TVector3 p2(179.4,-45.15,0);
    TVector3 p3(34.59,-76.7,0);
    TVector3 p4(34.59,76.7,0);

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

  std::string name1 = graphname;
  std::string name2 = graphname;
  TCanvas *c = new TCanvas("c","my canvas",1);
  c->SetCanvasSize(700,700);
  TGraph *graph1 = new TGraph(nstation1,globx1,globy1);
  TGraph *graph2 = new TGraph(nstation2,globx2,globy2);

  gStyle->SetTitleW(0.6);
  gStyle->SetTitleH(0.1);

  graph1->GetXaxis()->SetLimits(-720,720);
  graph1->GetYaxis()->SetLimits(-720,720);
  graph2->GetXaxis()->SetLimits(-720,720);
  graph2->GetYaxis()->SetLimits(-720,720);
  graph1->GetXaxis()->SetRangeUser(-720,720);
  graph1->GetYaxis()->SetRangeUser(-720,720);
  graph2->GetXaxis()->SetRangeUser(-720,720);
  graph2->GetYaxis()->SetRangeUser(-720,720);


  graph1->SetTitle(name1.c_str());
  graph1->UseCurrentStyle();
  graph1->Draw("AP");


  gStyle->SetLineWidth(2);
  float pi = 3.14159;
  TVector3 x(0,0,1);
  int linecolor = 1;
  int lc1 = 1;
  int lc2 = 1;

  if (station == 1){
    TVector3 p1(97.5,15.55,0);
    TVector3 p2(97.5,-15.55,0);
    TVector3 p3(265.5,-30.65,0);
    TVector3 p4(265.5,30.65,0);

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

    TVector3 q1(279.7,37.0,0);
    TVector3 q2(279.7,-37.0,0);
    TVector3 q3(459.7,-53.9,0);
    TVector3 q4(459.7,53.9,0);

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

    TVector3 r1(490.15,42.95,0);
    TVector3 r2(490.15,-42.95,0);
    TVector3 r3(680.15,-59.6,0);
    TVector3 r4(680.15,59.6,0);

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
    TVector3 p1(139.45,37.55,0);
    TVector3 p2(139.45,-37.55,0);
    TVector3 p3(345.95,-76.7,0);
    TVector3 p4(345.59,76.7,0);

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

    TVector3 q1(357.25,44.75,0);
    TVector3 q2(357.25,-44.75,0);
    TVector3 q3(695.25,-76.5,0);
    TVector3 q4(695.25,76.5,0);

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
    TVector3 p1(160.45,41.75,0);
    TVector3 p2(160.45,-41.75,0);
    TVector3 p3(344.95,-76.7,0);
    TVector3 p4(344.95,76.7,0);

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

    TVector3 q1(357.25,44.75,0);
    TVector3 q2(357.25,-44.75,0);
    TVector3 q3(695.25,-76.5,0);
    TVector3 q4(695.25,76.5,0);

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
    TVector3 p1(179.4,45.15,0);
    TVector3 p2(179.4,-45.15,0);
    TVector3 p3(34.59,-76.7,0);
    TVector3 p4(34.59,76.7,0);

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

  //c->Update();
  c->Print(savename.c_str());

}
 
