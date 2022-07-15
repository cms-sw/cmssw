TFile *OpenFiles(std::string path) {
  TFile *f;
  f = new TFile(path.c_str(), "READ");
  return f;
}

void printEmptyChambers(std::string histoname, std::string oname, TFile *f) {
  TH2I *plot = (TH2I *)f->Get(histoname.c_str());
  std::string endcap, chamber;
  int limitr, limitc;
  std::vector<string> deadchambers;

  for (int e = 0; e < 2; e++) {
    for (int s = 0; s < 4; s++) {
      if (s == 0)
        limitr = 4;
      if (s == 1 || s == 2)
        limitr = 2;
      if (s == 3)
        limitr = 1;
      for (int r = 0; r < limitr; r++) {
        if (s == 0)
          limitc = 36;
        if (s != 0 && r == 0)
          limitc = 18;
        if (s != 0 && r == 1)
          limitc = 36;
        for (int c = 0; c < limitc; c++) {
          int type = 0;
          if (s == 0 && r == 0)
            type = 2;
          else if (s == 0 && r == 1)
            type = 3;
          else if (s == 0 && r == 2)
            type = 4;
          else if (s == 0 && r == 3)
            type = 1;
          else
            type = (s + 1) * 2 + (r + 1);
          if (e == 0)
            type = type + 10;
          if (e == 1)
            type = 11 - type;
          int bin = plot->GetBin((c + 1), type);
          float content = plot->GetBinContent(bin);
          std::ostringstream oss;
          if (e == 0)
            endcap = "+";
          if (e == 1)
            endcap = "-";
          oss << "ME " << endcap << (s + 1) << "/" << (r + 1) << "/" << (c + 1);
          chamber = oss.str();
          if (content == 0) {
            if (oname == "wire digis" && (s == 0 && r == 3))
              continue;
            else
              deadchambers.push_back(chamber);
          }
        }
      }
    }
  }

  int n_dc = deadchambers.size();
  ofstream file;
  file.open("deadchamberlist.txt", ios::app);
  file << "Chambers with missing " << oname << "...\n" << endl;
  if (n_dc > 0) {
    for (int n = 0; n < n_dc; n++) {
      file << deadchambers[n] << endl;
    }
  }
  file << "\n\n\n\n";
  file.close();
}

void make1DPlot(std::string histoname, TFile *f1, std::string histotitle, int statoption, std::string savename) {
  TH1F *h1 = (TH1F *)f1->Get(histoname.c_str());

  TCanvas *c = new TCanvas("c", "my canvas", 1);

  if (h1) {
    if (statoption == 0)
      gStyle->SetOptStat(kFALSE);
    else
      gStyle->SetOptStat(statoption);
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
    h1->GetXaxis()->SetNdivisions(208, kTRUE);

    h1->Draw();

    c->Update();
    c->Print(savename.c_str(), "png");
  }
  delete c;
}

void makeCSCOccupancy(std::string histoname, TFile *f1, std::string histotitle, std::string savename) {
  TH1F *h1 = (TH1F *)f1->Get(histoname.c_str());

  TCanvas *c = new TCanvas("c", "my canvas", 1);

  if (h1) {
    gStyle->SetOptStat(kFALSE);
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
    h1->GetXaxis()->SetNdivisions(208, kTRUE);

    h1->GetXaxis()->SetBinLabel(2, "Total Events");
    h1->GetXaxis()->SetBinLabel(4, "# Events with Wires");
    h1->GetXaxis()->SetBinLabel(6, "# Events with Strips");
    h1->GetXaxis()->SetBinLabel(8, "# Events with Wires&Strips");
    h1->GetXaxis()->SetBinLabel(10, "# Events with Rechits");
    h1->GetXaxis()->SetBinLabel(12, "# Events with Segments");
    h1->GetXaxis()->SetBinLabel(14, "Events Rejected");

    h1->Draw();

    c->Update();
    c->Print(savename.c_str(), "png");
  }

  delete c;
}

void make1DPlot2(std::string histoname1,
                 std::string histoname2,
                 int statoption,
                 TFile *f1,
                 std::string t1,
                 std::string t2,
                 std::string savename) {
  // use this if you want two plots on one canvas

  TH1F *a1 = (TH1F *)f1->Get(histoname1.c_str());
  TH1F *b1 = (TH1F *)f1->Get(histoname2.c_str());

  TCanvas *c = new TCanvas("c", "my canvas", 1);
  c->Divide(1, 2);

  if (a1) {
    c->cd(1);
    if (statoption == 0)
      gStyle->SetOptStat(kFALSE);
    else
      gStyle->SetOptStat(statoption);
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
    a1->GetXaxis()->SetNdivisions(208, kTRUE);
    a1->Draw();
  }

  if (b1) {
    gStyle->SetHistFillColor(72);
    b1->UseCurrentStyle();

    t2 = t2 + " (run " + run + ")";
    b1->SetTitle(t2.c_str());
    b1->GetXaxis()->SetLabelSize(0.06);
    b1->GetYaxis()->SetLabelSize(0.06);
    b1->GetXaxis()->SetTitleOffset(0.7);
    b1->GetXaxis()->SetTitleSize(0.06);
    b1->GetXaxis()->SetNdivisions(508, kTRUE);
    c->cd(2);
    b1->Draw();
  }

  c->Update();
  c->Print(savename.c_str(), "png");
  delete c;
}

void makeEffGif(std::string histoname, TFile *f1, std::string histotitle, std::string savename) {
  TH1F *ho = (TH1F *)f1->Get(histoname.c_str());

  TCanvas *c = new TCanvas("c", "my canvas", 1);

  TH1F *hn = new TH1F("tmp", histotitle.c_str(), 20, 0.5, 20.5);

  if (ho) {
    float Num = 1;
    float Den = 1;
    for (int i = 0; i < 20; i++) {
      Num = ho->GetBinContent(i + 1);
      Den = ho->GetBinContent(i + 21);
      //getEfficiency(Num, Den, eff);
      float Eff = 0.;
      float EffE = 0.;
      if (fabs(Den) > 0.000000001) {
        Eff = Num / Den;
        if (Num < Den) {
          EffE = sqrt((1. - Eff) * Eff / Den);
        }
      }
      hn->SetBinContent(i + 1, Eff);
      hn->SetBinError(i + 1, EffE);
    }

    gStyle->SetOptStat(kFALSE);
    gStyle->SetHistFillColor(92);
    gStyle->SetFrameFillColor(4000);
    gStyle->SetTitleW(0.7);
    gStyle->SetTitleH(0.07);
    gPad->SetFillColor(4000);
    gStyle->SetStatColor(0);
    gStyle->SetTitleFillColor(0);
    c->SetFillStyle(4000);
    hn->UseCurrentStyle();

    hn->SetTitle(histotitle.c_str());
    hn->GetXaxis()->SetLabelSize(0.04);
    hn->GetYaxis()->SetLabelSize(0.04);
    hn->GetXaxis()->SetTitleOffset(0.7);
    hn->GetXaxis()->SetTitleSize(0.06);
    hn->GetXaxis()->SetNdivisions(208, kTRUE);
    hn->GetYaxis()->SetRangeUser(0.5, 1.1);
    hn->SetMarkerStyle(6);
    hn->GetXaxis()->SetBinLabel(1, "ME +1/1b");
    hn->GetXaxis()->SetBinLabel(2, "ME +1/2");
    hn->GetXaxis()->SetBinLabel(3, "ME +1/3");
    hn->GetXaxis()->SetBinLabel(4, "ME +1/1a");
    hn->GetXaxis()->SetBinLabel(5, "ME +2/1");
    hn->GetXaxis()->SetBinLabel(6, "ME +2/2");
    hn->GetXaxis()->SetBinLabel(7, "ME +3/1");
    hn->GetXaxis()->SetBinLabel(8, "ME +3/2");
    hn->GetXaxis()->SetBinLabel(9, "ME +4/1");
    hn->GetXaxis()->SetBinLabel(10, "ME +4/2");
    hn->GetXaxis()->SetBinLabel(11, "ME -1/1b");
    hn->GetXaxis()->SetBinLabel(12, "ME -1/2");
    hn->GetXaxis()->SetBinLabel(13, "ME -1/3");
    hn->GetXaxis()->SetBinLabel(14, "ME -1/1a");
    hn->GetXaxis()->SetBinLabel(15, "ME -2/1");
    hn->GetXaxis()->SetBinLabel(16, "ME -2/2");
    hn->GetXaxis()->SetBinLabel(17, "ME -3/1");
    hn->GetXaxis()->SetBinLabel(18, "ME -3/2");
    hn->GetXaxis()->SetBinLabel(19, "ME -4/1");
    hn->GetXaxis()->SetBinLabel(20, "ME -4/2");
    hn->Draw();
    c->Update();
    c->Print(savename.c_str(), "png");
  }
  delete c;
  delete hn;
}

void Draw2DProfile(std::string histoname, TFile *f1, std::string title, std::string option, std::string savename) {
  TProfile2D *test = f1->Get(histoname.c_str());
  TH2D *plot = test->ProjectionXY("test2", option.c_str());

  if (plot) {
    TCanvas *c = new TCanvas("c", "my canvas", 1);
    gStyle->SetPalette(1, 0);
    gPad->SetFillColor(4000);
    c->SetFillStyle(4000);
    gStyle->SetStatColor(0);
    gStyle->SetTitleFillColor(0);
    plot->SetStats(kFALSE);
    plot->GetYaxis()->SetBinLabel(1, "ME- 4/1");
    plot->GetYaxis()->SetBinLabel(2, "ME- 3/2");
    plot->GetYaxis()->SetBinLabel(3, "ME- 3/1");
    plot->GetYaxis()->SetBinLabel(4, "ME- 2/2");
    plot->GetYaxis()->SetBinLabel(5, "ME- 2/1");
    plot->GetYaxis()->SetBinLabel(6, "ME- 1/3");
    plot->GetYaxis()->SetBinLabel(7, "ME- 1/2");
    plot->GetYaxis()->SetBinLabel(8, "ME- 1/1b");
    plot->GetYaxis()->SetBinLabel(9, "ME- 1/1a");
    plot->GetYaxis()->SetBinLabel(10, "ME+ 1/1a");
    plot->GetYaxis()->SetBinLabel(11, "ME+ 1/1b");
    plot->GetYaxis()->SetBinLabel(12, "ME+ 1/2");
    plot->GetYaxis()->SetBinLabel(13, "ME+ 1/3");
    plot->GetYaxis()->SetBinLabel(14, "ME+ 2/1");
    plot->GetYaxis()->SetBinLabel(15, "ME+ 2/2");
    plot->GetYaxis()->SetBinLabel(16, "ME+ 3/1");
    plot->GetYaxis()->SetBinLabel(17, "ME+ 3/2");
    plot->GetYaxis()->SetBinLabel(18, "ME+ 4/1");

    plot->SetTitle(title.c_str());

    for (int i = 1; i < 37; i++) {
      ostringstream oss1;
      oss1 << i;
      string ch = oss1.str();
      plot->GetXaxis()->SetBinLabel(i, ch.c_str());
    }

    c->SetRightMargin(0.12);
    plot->GetYaxis()->SetNdivisions(20, kFALSE);
    plot->GetXaxis()->SetNdivisions(36, kFALSE);
    plot->GetXaxis()->SetTitle("Chamber #");
    c->SetGrid();

    plot->Draw("colz");
    c->Update();
    c->Print(savename.c_str(), "png");
    delete c;
  }
}

void Draw2DEfficiency(std::string histo, TFile *f1, std::string title, std::string savename) {
  TCanvas *c = new TCanvas("c", "my canvas", 1);
  gStyle->SetPalette(1, 0);
  gPad->SetFillColor(4000);
  c->SetFillStyle(4000);
  gStyle->SetStatColor(0);
  gStyle->SetTitleFillColor(0);

  TH2F *num = (TH2F *)f1->Get(histo.c_str());
  TH2F *denom = (TH2F *)f1->Get("Efficiency/hEffDenominator");

  TH2F *plot = new TH2F("plot", title.c_str(), 36, 0.5, 36.5, 18, 0.5, 18.5);

  plot->Divide(num, denom, 1., 1., "B");

  plot->SetStats(kFALSE);

  plot->GetYaxis()->SetBinLabel(1, "ME- 4/1");
  plot->GetYaxis()->SetBinLabel(2, "ME- 3/2");
  plot->GetYaxis()->SetBinLabel(3, "ME- 3/1");
  plot->GetYaxis()->SetBinLabel(4, "ME- 2/2");
  plot->GetYaxis()->SetBinLabel(5, "ME- 2/1");
  plot->GetYaxis()->SetBinLabel(9, "ME- 1/1a");
  plot->GetYaxis()->SetBinLabel(6, "ME- 1/3");
  plot->GetYaxis()->SetBinLabel(7, "ME- 1/2");
  plot->GetYaxis()->SetBinLabel(8, "ME- 1/1b");
  plot->GetYaxis()->SetBinLabel(11, "ME+ 1/1b");
  plot->GetYaxis()->SetBinLabel(12, "ME+ 1/2");
  plot->GetYaxis()->SetBinLabel(13, "ME+ 1/3");
  plot->GetYaxis()->SetBinLabel(10, "ME+ 1/1a");
  plot->GetYaxis()->SetBinLabel(14, "ME+ 2/1");
  plot->GetYaxis()->SetBinLabel(15, "ME+ 2/2");
  plot->GetYaxis()->SetBinLabel(16, "ME+ 3/1");
  plot->GetYaxis()->SetBinLabel(17, "ME+ 3/2");
  plot->GetYaxis()->SetBinLabel(18, "ME+ 4/1");

  for (int i = 1; i < 37; i++) {
    ostringstream oss1;
    oss1 << i;
    string ch = oss1.str();
    plot->GetXaxis()->SetBinLabel(i, ch.c_str());
  }

  c->SetRightMargin(0.12);

  plot->GetYaxis()->SetNdivisions(20, kFALSE);
  plot->GetXaxis()->SetNdivisions(36, kFALSE);

  plot->GetXaxis()->SetTitle("Chamber #");

  c->SetGrid();

  plot->Draw("COLZ");

  c->Update();
  c->Print(savename.c_str(), "png");
  delete c;
  delete plot;
}

void Draw2DTempPlot(std::string histo, TFile *f1, bool includeME11, std::string savename) {
  TCanvas *c = new TCanvas("c", "my canvas", 1);
  gStyle->SetPalette(1, 0);
  gPad->SetFillColor(4000);
  c->SetFillStyle(4000);
  gStyle->SetStatColor(0);
  gStyle->SetTitleFillColor(0);

  TH2I *plot = (TH2I *)f1->Get(histo.c_str());

  plot->SetStats(kFALSE);

  if (includeME11) {
    plot->GetYaxis()->SetBinLabel(1, "ME- 4/2");
    plot->GetYaxis()->SetBinLabel(2, "ME- 4/1");
    plot->GetYaxis()->SetBinLabel(3, "ME- 3/2");
    plot->GetYaxis()->SetBinLabel(4, "ME- 3/1");
    plot->GetYaxis()->SetBinLabel(5, "ME- 2/2");
    plot->GetYaxis()->SetBinLabel(6, "ME- 2/1");
    plot->GetYaxis()->SetBinLabel(10, "ME- 1/1a");
    plot->GetYaxis()->SetBinLabel(7, "ME- 1/3");
    plot->GetYaxis()->SetBinLabel(8, "ME- 1/2");
    plot->GetYaxis()->SetBinLabel(9, "ME- 1/1b");
    plot->GetYaxis()->SetBinLabel(12, "ME+ 1/1b");
    plot->GetYaxis()->SetBinLabel(13, "ME+ 1/2");
    plot->GetYaxis()->SetBinLabel(14, "ME+ 1/3");
    plot->GetYaxis()->SetBinLabel(11, "ME+ 1/1a");
    plot->GetYaxis()->SetBinLabel(15, "ME+ 2/1");
    plot->GetYaxis()->SetBinLabel(16, "ME+ 2/2");
    plot->GetYaxis()->SetBinLabel(17, "ME+ 3/1");
    plot->GetYaxis()->SetBinLabel(18, "ME+ 3/2");
    plot->GetYaxis()->SetBinLabel(19, "ME+ 4/1");
    plot->GetYaxis()->SetBinLabel(20, "ME+ 4/2");
  } else {
    plot->GetYaxis()->SetBinLabel(1, "ME- 4/1");
    plot->GetYaxis()->SetBinLabel(2, "ME- 3/2");
    plot->GetYaxis()->SetBinLabel(3, "ME- 3/1");
    plot->GetYaxis()->SetBinLabel(4, "ME- 2/2");
    plot->GetYaxis()->SetBinLabel(5, "ME- 2/1");
    plot->GetYaxis()->SetBinLabel(6, "ME- 1/3");
    plot->GetYaxis()->SetBinLabel(7, "ME- 1/2");
    plot->GetYaxis()->SetBinLabel(8, "ME- 1/1b");
    plot->GetYaxis()->SetBinLabel(9, "ME- 1/1a");
    plot->GetYaxis()->SetBinLabel(10, "ME+ 1/1a");
    plot->GetYaxis()->SetBinLabel(11, "ME+ 1/1b");
    plot->GetYaxis()->SetBinLabel(12, "ME+ 1/2");
    plot->GetYaxis()->SetBinLabel(13, "ME+ 1/3");
    plot->GetYaxis()->SetBinLabel(14, "ME+ 2/1");
    plot->GetYaxis()->SetBinLabel(15, "ME+ 2/2");
    plot->GetYaxis()->SetBinLabel(16, "ME+ 3/1");
    plot->GetYaxis()->SetBinLabel(17, "ME+ 3/2");
    plot->GetYaxis()->SetBinLabel(18, "ME+ 4/1");
  }

  for (int i = 1; i < 37; i++) {
    ostringstream oss1;
    oss1 << i;
    string ch = oss1.str();
    plot->GetXaxis()->SetBinLabel(i, ch.c_str());
  }

  c->SetRightMargin(0.12);

  plot->GetYaxis()->SetNdivisions(20, kFALSE);
  plot->GetXaxis()->SetNdivisions(36, kFALSE);

  plot->GetXaxis()->SetTitle("Chamber #");

  c->SetGrid();

  plot->Draw("COLZ");

  c->Update();
  c->Print(savename.c_str(), "png");
  delete c;
}

void GlobalPosfromTree(
    std::string graphname, TFile *f1, int endcap, int station, std::string type, std::string savename) {
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

  if (type == "rechit") {
    t1 = (TTree *)f1->Get("recHits/rHPositions");
    b1 = t1->GetBranch("rHpos");
    b1->SetAddress(&points);
  }

  if (type == "segment") {
    t1 = (TTree *)f1->Get("Segments/segPositions");
    b1 = t1->GetBranch("segpos");
    b1->SetAddress(&points);
  }

  int n1 = (int)t1->GetEntries();
  const int nevents1 = n1;
  float globx1[nevents1];
  float globy1[nevents1];
  int nstation1 = 0;
  const int num_of_rings = 4;
  const int num_of_chambers = 36;
  int nchamber1[num_of_rings][num_of_chambers];
  for (int i = 0; i < num_of_rings; i++) {
    for (int j = 0; j < num_of_chambers; j++) {
      nchamber1[i][j] = 0;
    }
  }

  for (int i = 0; i < nevents1; i++) {
    b1->GetEntry(i);
    if (points.station == station && points.endcap == endcap) {
      globx1[nstation1] = points.globalx;
      globy1[nstation1] = points.globaly;
      nstation1++;
      nchamber1[points.ring - 1][points.chamber - 1]++;
    }
  }

  TCanvas *c = new TCanvas("c", "my canvas", 1);
  c->SetCanvasSize(700, 700);
  TGraph *graph1 = new TGraph(nstation1, globx1, globy1);

  std::string name1 = graphname;

  gStyle->SetPalette(1, 0);
  gStyle->SetTitleW(0.9);
  gStyle->SetTitleH(0.1);
  gStyle->SetStatColor(0);
  gStyle->SetTitleFillColor(0);
  gPad->SetFillColor(4000);
  c->SetFillStyle(4000);
  gStyle->SetOptStat(10);

  graph1->GetXaxis()->SetLimits(-720, 720);
  graph1->GetYaxis()->SetLimits(-720, 720);
  graph1->GetXaxis()->SetRangeUser(-720, 720);
  graph1->GetYaxis()->SetRangeUser(-720, 720);

  graph1->SetTitle(name1.c_str());
  graph1->UseCurrentStyle();
  graph1->Draw("AP");

  //  drawChamberLines(station);
  drawColoredChamberLines(station, nchamber1);

  c->Print(savename.c_str(), "png");
  delete c;

}  // end GlobalPosfromTree

drawChamberLines(int station) {
  gStyle->SetLineWidth(2);
  float pi = 3.14159;
  TVector3 x(0, 0, 1);
  int linecolor = 1;
  //for alternating colors, set 2 diff colors here
  int lc1 = 1;
  int lc2 = 1;

  if (station == 1) {
    TVector3 p1(101, 9.361, 0);
    TVector3 p2(101, -9.361, 0);
    TVector3 p3(260, -22.353, 0);
    TVector3 p4(260, 22.353, 0);

    TLine *line1;
    TLine *line2;
    TLine *line3;
    TLine *line4;

    for (int i = 0; i < 36; i++) {
      if (linecolor == lc1)
        linecolor = lc2;
      else
        linecolor = lc1;

      line1 = new TLine(p1(0), p1(1), p2(0), p2(1));
      line2 = new TLine(p2(0), p2(1), p3(0), p3(1));
      line3 = new TLine(p3(0), p3(1), p4(0), p4(1));
      line4 = new TLine(p4(0), p4(1), p1(0), p1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      p1.Rotate(pi / 18, x);
      p2.Rotate(pi / 18, x);
      p3.Rotate(pi / 18, x);
      p4.Rotate(pi / 18, x);
    }

    TVector3 q1(281.49, 25.5, 0);
    TVector3 q2(281.49, -25.5, 0);
    TVector3 q3(455.99, -41.87, 0);
    TVector3 q4(455.99, 41.87, 0);

    for (int i = 0; i < 36; i++) {
      if (linecolor == lc2)
        linecolor = lc1;
      else
        linecolor = lc2;

      line1 = new TLine(q1(0), q1(1), q2(0), q2(1));
      line2 = new TLine(q2(0), q2(1), q3(0), q3(1));
      line3 = new TLine(q3(0), q3(1), q4(0), q4(1));
      line4 = new TLine(q4(0), q4(1), q1(0), q1(1));

      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      q1.Rotate(pi / 18, x);
      q2.Rotate(pi / 18, x);
      q3.Rotate(pi / 18, x);
      q4.Rotate(pi / 18, x);
    }

    TVector3 r1(511.99, 31.7, 0);
    TVector3 r2(511.99, -31.7, 0);
    TVector3 r3(676.15, -46.05, 0);
    TVector3 r4(676.15, 46.05.6, 0);

    for (int i = 0; i < 36; i++) {
      if (linecolor == lc1)
        linecolor = lc2;
      else
        linecolor = lc1;

      line1 = new TLine(r1(0), r1(1), r2(0), r2(1));
      line2 = new TLine(r2(0), r2(1), r3(0), r3(1));
      line3 = new TLine(r3(0), r3(1), r4(0), r4(1));
      line4 = new TLine(r4(0), r4(1), r1(0), r1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      r1.Rotate(pi / 18, x);
      r2.Rotate(pi / 18, x);
      r3.Rotate(pi / 18, x);
      r4.Rotate(pi / 18, x);
    }
  }

  if (station == 2) {
    TVector3 p1(146.9, 27.0, 0);
    TVector3 p2(146.9, -27.0, 0);
    TVector3 p3(336.56, -62.855, 0);
    TVector3 p4(336.56, 62.855, 0);

    p1.Rotate(pi / 36, x);
    p2.Rotate(pi / 36, x);
    p3.Rotate(pi / 36, x);
    p4.Rotate(pi / 36, x);

    TLine *line1;
    TLine *line2;
    TLine *line3;
    TLine *line4;

    for (int i = 0; i < 36; i++) {
      if (linecolor == lc1)
        linecolor = lc2;
      else
        linecolor = lc1;

      line1 = new TLine(p1(0), p1(1), p2(0), p2(1));
      line2 = new TLine(p2(0), p2(1), p3(0), p3(1));
      line3 = new TLine(p3(0), p3(1), p4(0), p4(1));
      line4 = new TLine(p4(0), p4(1), p1(0), p1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      p1.Rotate(pi / 9, x);
      p2.Rotate(pi / 9, x);
      p3.Rotate(pi / 9, x);
      p4.Rotate(pi / 9, x);
    }

    TVector3 q1(364.02, 33.23, 0);
    TVector3 q2(364.02, -33.23, 0);
    TVector3 q3(687.08, -63.575, 0);
    TVector3 q4(687.08, 63.575, 0);

    for (int i = 0; i < 36; i++) {
      if (linecolor == lc2)
        linecolor = lc1;
      else
        linecolor = lc2;

      line1 = new TLine(q1(0), q1(1), q2(0), q2(1));
      line2 = new TLine(q2(0), q2(1), q3(0), q3(1));
      line3 = new TLine(q3(0), q3(1), q4(0), q4(1));
      line4 = new TLine(q4(0), q4(1), q1(0), q1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      q1.Rotate(pi / 18, x);
      q2.Rotate(pi / 18, x);
      q3.Rotate(pi / 18, x);
      q4.Rotate(pi / 18, x);
    }
  }

  if (station == 3) {
    TVector3 p1(166.89, 30.7, 0);
    TVector3 p2(166.89, -30.7, 0);
    TVector3 p3(336.59, -62.855, 0);
    TVector3 p4(336.59, 62.855, 0);

    p1.Rotate(pi / 36, x);
    p2.Rotate(pi / 36, x);
    p3.Rotate(pi / 36, x);
    p4.Rotate(pi / 36, x);

    TLine *line1;
    TLine *line2;
    TLine *line3;
    TLine *line4;

    for (int i = 0; i < 36; i++) {
      if (linecolor == lc1)
        linecolor = lc2;
      else
        linecolor = lc1;

      line1 = new TLine(p1(0), p1(1), p2(0), p2(1));
      line2 = new TLine(p2(0), p2(1), p3(0), p3(1));
      line3 = new TLine(p3(0), p3(1), p4(0), p4(1));
      line4 = new TLine(p4(0), p4(1), p1(0), p1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      p1.Rotate(pi / 9, x);
      p2.Rotate(pi / 9, x);
      p3.Rotate(pi / 9, x);
      p4.Rotate(pi / 9, x);
    }

    TVector3 q1(364.02, 33.23, 0);
    TVector3 q2(364.02, -33.23, 0);
    TVector3 q3(687.08, -63.575, 0);
    TVector3 q4(687.08, 63.575, 0);

    for (int i = 0; i < 36; i++) {
      if (linecolor == lc2)
        linecolor = lc1;
      else
        linecolor = lc2;

      line1 = new TLine(q1(0), q1(1), q2(0), q2(1));
      line2 = new TLine(q2(0), q2(1), q3(0), q3(1));
      line3 = new TLine(q3(0), q3(1), q4(0), q4(1));
      line4 = new TLine(q4(0), q4(1), q1(0), q1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      q1.Rotate(pi / 18, x);
      q2.Rotate(pi / 18, x);
      q3.Rotate(pi / 18, x);
      q4.Rotate(pi / 18, x);
    }
  }

  if (station == 4) {
    TVector3 p1(186.99, 34.505.15, 0);
    TVector3 p2(186.99, -34.505, 0);
    TVector3 p3(336.41, -62.825, 0);
    TVector3 p4(336.41, 62.825, 0);

    p1.Rotate(pi / 36, x);
    p2.Rotate(pi / 36, x);
    p3.Rotate(pi / 36, x);
    p4.Rotate(pi / 36, x);

    TLine *line1;
    TLine *line2;
    TLine *line3;
    TLine *line4;

    for (int i = 0; i < 36; i++) {
      if (linecolor == lc1)
        linecolor = lc2;
      else
        linecolor = lc1;

      line1 = new TLine(p1(0), p1(1), p2(0), p2(1));
      line2 = new TLine(p2(0), p2(1), p3(0), p3(1));
      line3 = new TLine(p3(0), p3(1), p4(0), p4(1));
      line4 = new TLine(p4(0), p4(1), p1(0), p1(1));
      line1->SetLineColor(linecolor);
      line2->SetLineColor(linecolor);
      line3->SetLineColor(linecolor);
      line4->SetLineColor(linecolor);

      line1->Draw();
      line2->Draw();
      line3->Draw();
      line4->Draw();

      p1.Rotate(pi / 9, x);
      p2.Rotate(pi / 9, x);
      p3.Rotate(pi / 9, x);
      p4.Rotate(pi / 9, x);
    }
  }
}

void compare1DPlot(
    std::string histoname, TFile *f1, TFile *f2, std::string histotitle, int statoption, std::string savename) {
  // used to compare two of the same histogram from different releases/runs/etc

  TH1F *h2 = (TH1F *)f1->Get(histoname.c_str());
  TH1F *h1 = (TH1F *)f2->Get(histoname.c_str());

  TCanvas *c = new TCanvas("c", "my canvas", 1);

  if (h1 && h2) {
    gStyle->SetHistFillColor(92);
    gStyle->SetFrameFillColor(4000);
    gStyle->SetTitleW(0.5);
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
    h1->GetXaxis()->SetNdivisions(208, kTRUE);

    TLegend *leg = new TLegend(0.79, 0.74, 0.89, 0.84);
    leg->AddEntry(h1, "ref", "f");
    leg->AddEntry(h2, "new", "l");

    h1->Draw();
    h2->Draw("same e");
    leg->Draw();
  }

  c->Update();
  c->Print(savename.c_str(), "png");
  delete c;
}

void compareEffGif(std::string histoname, TFile *f1, TFile *f2, std::string histotitle, std::string savename) {
  TH1F *h1 = (TH1F *)f1->Get(histoname.c_str());
  TH1F *h2 = (TH1F *)f2->Get(histoname.c_str());

  TCanvas *c = new TCanvas("c", "my canvas", 1);

  TH1F *hn1 = new TH1F("tmp1", histotitle.c_str(), 20, 0.5, 20.5);
  TH1F *hn2 = new TH1F("tmp2", histotitle.c_str(), 20, 0.5, 20.5);

  if (h1 && h2) {
    float Num = 1;
    float Den = 1;
    for (int i = 0; i < 20; i++) {
      Num = h1->GetBinContent(i + 1);
      Den = h1->GetBinContent(i + 21);
      //getEfficiency(Num, Den, eff);
      float Eff = 0.;
      float EffE = 0.;
      if (fabs(Den) > 0.000000001) {
        Eff = Num / Den;
        if (Num < Den) {
          EffE = sqrt((1. - Eff) * Eff / Den);
        }
      }
      hn1->SetBinContent(i + 1, Eff);
      hn1->SetBinError(i + 1, EffE);
    }

    float Num = 1;
    float Den = 1;
    for (int i = 0; i < 20; i++) {
      Num = h2->GetBinContent(i + 1);
      Den = h2->GetBinContent(i + 21);
      //getEfficiency(Num, Den, eff);
      float Eff = 0.;
      float EffE = 0.;
      if (fabs(Den) > 0.000000001) {
        Eff = Num / Den;
        if (Num < Den) {
          EffE = sqrt((1. - Eff) * Eff / Den);
        }
      }
      hn2->SetBinContent(i + 1, Eff);
      hn2->SetBinError(i + 1, EffE);
    }

    gStyle->SetOptStat(kFALSE);
    gStyle->SetHistFillColor(92);
    gStyle->SetFrameFillColor(4000);
    gStyle->SetTitleW(0.7);
    gStyle->SetTitleH(0.07);
    gPad->SetFillColor(4000);
    gStyle->SetStatColor(0);
    gStyle->SetTitleFillColor(0);
    c->SetFillStyle(4000);
    hn1->UseCurrentStyle();

    hn1->SetTitle(histotitle.c_str());
    hn1->GetXaxis()->SetLabelSize(0.04);
    hn1->GetYaxis()->SetLabelSize(0.04);
    hn1->GetXaxis()->SetTitleOffset(0.7);
    hn1->GetXaxis()->SetTitleSize(0.06);
    hn1->GetXaxis()->SetNdivisions(208, kTRUE);
    hn1->GetYaxis()->SetRangeUser(0.5, 1.1);
    hn1->SetMarkerStyle(6);
    hn1->SetMarkerColor(kBlue);
    hn2->SetMarkerStyle(6);
    hn2->SetMarkerColor(kRed);
    hn1->GetXaxis()->SetBinLabel(1, "ME +1/1b");
    hn1->GetXaxis()->SetBinLabel(2, "ME +1/2");
    hn1->GetXaxis()->SetBinLabel(3, "ME +1/3");
    hn1->GetXaxis()->SetBinLabel(4, "ME +1/1a");
    hn1->GetXaxis()->SetBinLabel(5, "ME +2/1");
    hn1->GetXaxis()->SetBinLabel(6, "ME +2/2");
    hn1->GetXaxis()->SetBinLabel(7, "ME +3/1");
    hn1->GetXaxis()->SetBinLabel(8, "ME +3/2");
    hn1->GetXaxis()->SetBinLabel(9, "ME +4/1");
    hn1->GetXaxis()->SetBinLabel(10, "ME +4/2");
    hn1->GetXaxis()->SetBinLabel(11, "ME -1/1b");
    hn1->GetXaxis()->SetBinLabel(12, "ME -1/2");
    hn1->GetXaxis()->SetBinLabel(13, "ME -1/3");
    hn1->GetXaxis()->SetBinLabel(14, "ME -1/1a");
    hn1->GetXaxis()->SetBinLabel(15, "ME -2/1");
    hn1->GetXaxis()->SetBinLabel(16, "ME -2/2");
    hn1->GetXaxis()->SetBinLabel(17, "ME -3/1");
    hn1->GetXaxis()->SetBinLabel(18, "ME -3/2");
    hn1->GetXaxis()->SetBinLabel(19, "ME -4/1");
    hn1->GetXaxis()->SetBinLabel(20, "ME -4/2");

    TLegend *leg = new TLegend(0.79, 0.79, 0.89, 0.89);
    leg->AddEntry(hn1, "new", "p");
    leg->AddEntry(hn2, "ref", "p");

    hn1->Draw();
    hn2->Draw("same");
    leg->Draw();
    c->Update();
    c->Print(savename.c_str(), "png");
  }
  delete c;
  delete hn1;
  delete hn2;
}

void GlobalPosfromTreeCompare(
    std::string graphname, TFile *f1, TFile *f2, int endcap, int station, std::string type, std::string savename) {
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

  if (type == "rechit") {
    t1 = (TTree *)f1->Get("recHits/rHPositions");
    t2 = (TTree *)f2->Get("recHits/rHPositions");
    b1 = t1->GetBranch("rHpos");
    b2 = t2->GetBranch("rHpos");
    b1->SetAddress(&points1);
    b2->SetAddress(&points2);
  }

  if (type == "segment") {
    t1 = (TTree *)f1->Get("Segments/segPositions");
    t2 = (TTree *)f2->Get("Segments/segPositions");
    b1 = t1->GetBranch("segpos");
    b2 = t2->GetBranch("segpos");
    b1->SetAddress(&points1);
    b2->SetAddress(&points2);
  }

  int n1 = (int)t1->GetEntries();
  int n2 = (int)t2->GetEntries();

  const int nevents1 = n1;
  const int nevents2 = n2;

  float globx1[nevents1];
  float globy1[nevents1];
  float globx2[nevents2];
  float globy2[nevents2];
  int nstation1 = 0;
  int nstation2 = 0;

  for (int i = 0; i < nevents1; i++) {
    b1->GetEntry(i);
    if (points1.station == station && points1.endcap == endcap) {
      globx1[nstation1] = points1.globalx;
      globy1[nstation1] = points1.globaly;
      nstation1++;
    }
  }
  for (int i = 0; i < nevents2; i++) {
    b2->GetEntry(i);
    if (points2.station == station && points2.endcap == endcap) {
      globx2[nstation2] = points2.globalx;
      globy2[nstation2] = points2.globaly;
      nstation2++;
    }
  }

  std::string name1 = graphname + " (New)";
  std::string name2 = graphname + " (Ref)";
  TCanvas *c = new TCanvas("c", "my canvas", 1);
  c->SetCanvasSize(1300, 700);
  c->Divide(2, 1);
  TGraph *graph1 = new TGraph(nstation1, globx1, globy1);
  TGraph *graph2 = new TGraph(nstation2, globx2, globy2);

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
  c->Print(savename.c_str(), "png");
  delete c;
}

void NikolaiPlots(TFile *f_in, int flag) {
  gROOT->SetStyle("Plain");  // to get rid of gray color of pad and have it white
  gStyle->SetPalette(1, 0);  //

  std::ostringstream ss, ss1;

  if (flag == 1) {  // gas gain results
    std::string folder = "GasGain/";

    std::string input_histName = "gas_gain_rechit_adc_3_3_sum_location_ME_";
    std::string input_title_X = "Location=(layer-1)*nsegm+segm";
    std::string input_title_Y = "3X3 ADC Sum";

    std::string slice_title_X = "3X3 ADC Sum Location";

    Int_t ny = 30;
    Float_t ylow = 1.0, yhigh = 31.0;
    std::string result_histName = "mean_gas_gain_vs_location_csc_ME_";
    std::string result_histTitle = "Mean 3X3 ADC Sum";
    std::string result_title_Y = "Location=(layer-1)*nsegm+segm";

    std::string result_histNameEntries = "entries_gas_gain_vs_location_csc_ME_";
    std::string result_histTitleEntries = "Entries 3X3 ADC Sum";
  }

  if (flag == 2) {  // AFEB timing results
    std::string folder = "AFEBTiming/";

    std::string input_histName = "afeb_time_bin_vs_afeb_occupancy_ME_";
    std::string input_title_X = "AFEB";
    std::string input_title_Y = "Time Bin";

    std::string slice_title_X = "AFEB";

    Int_t ny = 42;
    Float_t ylow = 1.0, yhigh = 42.0;
    std::string result_histName = "mean_afeb_time_bin_vs_afeb_csc_ME_";
    std::string result_histTitle = "AFEB Mean Time Bin";
    std::string result_title_Y = "AFEB";

    std::string result_histNameEntries = "entries_afeb_time_bin_vs_afeb_csc_ME_";
    std::string result_histTitleEntries = "Entries AFEB Time Bin";
  }

  if (flag == 3) {  // Comparator timing results
    std::string folder = "CompTiming/";

    std::string input_histName = "comp_time_bin_vs_cfeb_occupancy_ME_";
    std::string input_title_X = "CFEB";
    std::string input_title_Y = "Time Bin";

    std::string slice_title_X = "CFEB";

    Int_t ny = 5;
    Float_t ylow = 1.0, yhigh = 6.0;
    std::string result_histName = "mean_comp_time_bin_vs_cfeb_csc_ME_";
    std::string result_histTitle = "Comparator Mean Time Bin";
    std::string result_title_Y = "CFEB";

    std::string result_histNameEntries = "entries_comp_time_bin_vs_cfeb_csc_ME_";
    std::string result_histTitleEntries = "Entries Comparator Time Bin";
  }

  if (flag == 4) {  // Strip ADC timing results
    std::string folder = "ADCTiming/";

    std::string input_histName = "adc_3_3_weight_time_bin_vs_cfeb_occupancy_ME_";
    std::string input_title_X = "CFEB";
    std::string input_title_Y = "Time Bin";

    std::string slice_title_X = "CFEB";

    Int_t ny = 5;
    Float_t ylow = 1.0, yhigh = 6.0;
    std::string result_histName = "mean_adc_time_bin_vs_cfeb_csc_ME_";
    std::string result_histTitle = "ADC 3X3 Mean Time Bin";
    std::string result_title_Y = "CFEB";

    std::string result_histNameEntries = "entries_adc_time_bin_vs_cfeb_csc_ME_";
    std::string result_histTitleEntries = "Entries ADC 3X3 Time Bin";
  }

  std::vector<std::string> xTitle;
  xTitle.push_back("ME+1/1 CSC");
  xTitle.push_back("ME+1/2 CSC");
  xTitle.push_back("ME+1/3 CSC");
  xTitle.push_back("ME+2/1 CSC");
  xTitle.push_back("ME+2/2 CSC");
  xTitle.push_back("ME+3/1 CSC");
  xTitle.push_back("ME+3/2 CSC");
  xTitle.push_back("ME+4/1 CSC");
  xTitle.push_back("ME+4/2 CSC");
  xTitle.push_back("ME-1/1 CSC");
  xTitle.push_back("ME-1/2 CSC");
  xTitle.push_back("ME-1/3 CSC");
  xTitle.push_back("ME-2/1 CSC");
  xTitle.push_back("ME-2/2 CSC");
  xTitle.push_back("ME-3/1 CSC");
  xTitle.push_back("ME-3/2 CSC");
  xTitle.push_back("ME-4/1 CSC");
  xTitle.push_back("ME-4/2 CSC");

  TH2F *h2[500];
  TH2F *h;
  Int_t esr[18] = {111, 112, 113, 121, 122, 131, 132, 141, 142, 211, 212, 213, 221, 222, 231, 232, 241, 242};
  Int_t entries[18] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Int_t k = 0;
  TCanvas *c1 = new TCanvas("c1", "canvas");
  c1->cd();

  //if(flag==2) { // adding special case for AFEB timing
  ss.str("");
  ss << "mean_afeb_time_bin_vs_csc_ME";
  ss1.str("");
  ss1 << "Mean AFEB time bin vs CSC and ME";
  gStyle->SetOptStat(0);
  TH2F *hb = new TH2F(ss.str().c_str(), ss1.str().c_str(), 36, 1.0, 37.0, 18, 1.0, 19.0);
  hb->SetStats(kFALSE);
  hb->GetXaxis()->SetTitle("CSC #");
  hb->GetZaxis()->SetLabelSize(0.03);
  hb->SetOption("COLZ");

  hb->GetYaxis()->SetBinLabel(1, "ME- 4/2");
  hb->GetYaxis()->SetBinLabel(2, "ME- 4/1");
  hb->GetYaxis()->SetBinLabel(3, "ME- 3/2");
  hb->GetYaxis()->SetBinLabel(4, "ME- 3/1");
  hb->GetYaxis()->SetBinLabel(5, "ME- 2/2");
  hb->GetYaxis()->SetBinLabel(6, "ME- 2/1");
  hb->GetYaxis()->SetBinLabel(7, "ME- 1/3");
  hb->GetYaxis()->SetBinLabel(8, "ME- 1/2");
  hb->GetYaxis()->SetBinLabel(9, "ME- 1/1");
  hb->GetYaxis()->SetBinLabel(10, "ME+ 1/1");
  hb->GetYaxis()->SetBinLabel(11, "ME+ 1/2");
  hb->GetYaxis()->SetBinLabel(12, "ME+ 1/3");
  hb->GetYaxis()->SetBinLabel(13, "ME+ 2/1");
  hb->GetYaxis()->SetBinLabel(14, "ME+ 2/2");
  hb->GetYaxis()->SetBinLabel(15, "ME+ 3/1");
  hb->GetYaxis()->SetBinLabel(16, "ME+ 3/2");
  hb->GetYaxis()->SetBinLabel(17, "ME+ 4/1");
  hb->GetYaxis()->SetBinLabel(18, "ME+ 4/2");
  //}

  for (Int_t jesr = 0; jesr < 18; jesr++) {
    ss.str("");
    ss << result_histName.c_str() << esr[jesr];
    ss1.str("");
    ss1 << result_histTitle;
    TH2F *h = new TH2F(ss.str().c_str(), ss1.str().c_str(), 40, 0.0, 40.0, ny, ylow, yhigh);
    h->SetStats(kFALSE);
    h->GetXaxis()->SetTitle(xTitle[jesr].c_str());
    h->GetYaxis()->SetTitle(result_title_Y.c_str());
    h->GetZaxis()->SetLabelSize(0.03);
    h->SetOption("COLZ");

    ss.str("");
    ss << result_histNameEntries.c_str() << esr[jesr];
    ss1.str("");
    ss1 << result_histTitleEntries;
    TH2F *hentr = new TH2F(ss.str().c_str(), ss1.str().c_str(), 40, 0.0, 40.0, ny, ylow, yhigh);
    hentr->SetStats(kFALSE);
    hentr->GetXaxis()->SetTitle(xTitle[jesr].c_str());
    hentr->GetYaxis()->SetTitle(result_title_Y.c_str());
    hentr->GetZaxis()->SetLabelSize(0.03);
    hentr->SetOption("COLZ");

    if (flag == 2) {  // adding special cases for AFEB timing
      ss.str("");
      ss << "normal_afeb_time_bin_vs_csc_ME_" << esr[jesr];
      ss1.str("");
      ss1 << "Normalized AFEB time bin, %";
      TH2F *ha = new TH2F(ss.str().c_str(), ss1.str().c_str(), 40, 0.0, 40.0, 16, 0.0, 16.0);
      ha->SetStats(kFALSE);
      ha->GetXaxis()->SetTitle(xTitle[jesr].c_str());
      ha->GetYaxis()->SetTitle("Time Bin");
      ha->GetZaxis()->SetLabelSize(0.03);
      ha->SetOption("COLZ");
    }

    for (Int_t csc = 1; csc < 37; csc++) {
      Int_t idchamber = esr[jesr] * 100 + csc;
      ss.str("");
      ss << folder.c_str() << input_histName.c_str() << idchamber;
      f_in->cd();
      TH2F *h2[1];
      h2[k] = (TH2F *)f_in->Get(ss.str().c_str());
      if (h2[k] != NULL) {
        // saving original, adding X,Y titles, color and "BOX" option
        h2[k]->GetXaxis()->SetTitle(input_title_X.c_str());
        h2[k]->GetYaxis()->SetTitle(input_title_Y.c_str());
        h2[k]->GetYaxis()->SetTitleOffset(1.2);
        h2[k]->SetFillColor(4);
        h2[k]->SetOption("BOX");
        gStyle->SetOptStat(1001111);

        // saving Y projection of the whole 2D hist for given chamber
        ss.str("");
        ss << input_histName.c_str() << idchamber << "_Y_all";
        TH1D *h1d = h2[k]->ProjectionY(ss.str().c_str(), 1, h2[k]->GetNbinsX(), "");
        h1d->GetYaxis()->SetTitle("Entries");
        h1d->GetYaxis()->SetTitleOffset(1.2);
        gStyle->SetOptStat(1001111);

        if (flag == 2 && h1d->GetEntries() > 0) {  // adding spec. case for afeb timing
          Float_t entr = h1d->GetEntries();
          for (Int_t m = 1; m < h1d->GetNbinsX(); m++) {
            Float_t w = h1d->GetBinContent(m);
            w = 100.0 * w / entr;
            ha->SetBinContent(csc + 1, m, w);
          }
          Float_t mean = h1d->GetMean();
          Int_t me;
          if (jesr < 9)
            me = 10 + jesr;
          if (jesr > 8)
            me = 18 - jesr;
          hb->SetBinContent(csc, me, mean);
        }
        delete h1d;

        // saving slices, finding MEAN in each slice, fill 2D hist
        for (Int_t j = 1; j <= h2[k]->GetNbinsX(); j++) {
          Int_t n = j;
          ss.str("");
          ss << input_histName.c_str() << idchamber << "_Y_" << n;
          TH1D *h1d = h2[k]->ProjectionY(ss.str().c_str(), j, j, "");
          if (h1d->GetEntries() > 0) {
            Float_t mean = h1d->GetMean();
            Float_t entr = h1d->GetEntries();
            entries[jesr] = entries[jesr] + 1;
            h->SetBinContent(csc + 1, j, mean);
            hentr->SetBinContent(csc + 1, j, entr);
            ss.str("");
            ss << slice_title_X << " " << n;
            h1d->GetXaxis()->SetTitle(ss.str().c_str());
            h1d->GetYaxis()->SetTitle("Entries");
            h1d->GetYaxis()->SetTitleOffset(1.2);
            gStyle->SetOptStat(1001111);
          }
          delete h1d;
        }
      }
    }
    if (entries[jesr] > 0) {
      h->SetStats(kFALSE);
      hentr->SetStats(kFALSE);
      c1->Update();

      // printing

      h->Draw();
      ss.str("");
      ss << result_histName.c_str() << esr[jesr] << ".png";
      c1->Print(ss.str().c_str(), "png");

      hentr->Draw();
      ss.str("");
      ss << result_histNameEntries.c_str() << esr[jesr] << ".png";
      c1->Print(ss.str().c_str(), "png");
    }
    delete h;
    delete hentr;
    if (flag == 2)
      delete ha;
  }
  if (flag == 2) {
    hb->Draw();
    ss.str("");
    ss << "mean_afeb_time_bin_vs_csc_ME"
       << ".png";
    c1->Print(ss.str().c_str(), "png");

    c1->Update();
    //delete hb;
  }
  delete hb;
  delete c1;
}

drawColoredChamberLines(int station, int nchamber1[4][36]) {
  // thanks to Luca Sabbatini for this coe
  const int maxRingIdxOfRelevance = 2;
  const int maxChamberIdxOfRelevance = 35;
  const int totalNumberOfChambersOfRelevance = 540;

  Int_t thisRingIdx = 0;
  Int_t numChambersInRing = 0;

  float nchamber1Avg = 0.;
  float thisRingNchamber1Avg = 0;
  float thisRingNchamber1Max = 0;
  // Note that thisRingNchamber1Min is only the minumum *non-zero*
  // occupancy among all the cmabers for the given ring
  float thisRingNchamber1Min = 0;
  Int_t thisRingNchamber1Base = 0;

  // rFloatRGB, gFloatRGB, bFloatRGB, are from the RGB color space;
  // all three range between [0,1]
  // hFloatHLS, lFloatHLS, sFloatHLS, are from the HLS color space;
  // lFloatHLS and sFloatHLS range between [0,1], while hFloatHLS
  // ranges between [0,360]
  Float_t rFloatRGB = 0.;
  Float_t gFloatRGB = 0.;
  Float_t bFloatRGB = 0.;
  Float_t hFloatHLS = 0.;
  Float_t lFloatHLS = 0.5;
  Float_t sFloatHLS = 1.0;
  TColor tempColor;

  // compute average chamber occupancy over all CSCs
  for (int i = 0; i < maxRingIdxOfRelevance + 1; i++) {
    for (int j = 0; j < maxChamberIdxOfRelevance + 1; j++) {
      nchamber1Avg += nchamber1[i][j];
    }
  }
  nchamber1Avg = nchamber1Avg / totalNumberOfChambersOfRelevance;

  Float_t myFavoriteLineWidth = 2.0;
  gStyle->SetLineWidth(myFavoriteLineWidth);
  float pi = 3.14159;
  TVector3 x(0, 0, 1);
  int linecolor = 1;

  // emptyLineColor is the color of the outline of empty chambers.
  // if emptyLineColor is negative, no outline is drawn at all for
  // that chamber.
  Int_t emptyLineColor = kYellow;

  // note that these can be greater than 360 because I have included
  // a line down below that performs a modulus operation: %360, which
  // ensures that the resulting hue will indeed be with [0,360]
  //
  // Hue:    0   30      60      120     180     240 270 300     360
  //         |   |       |       |       |       |   |   |       |
  // Color: Red Orange Yellow   Green   Cyan  Blue Vilet Magenta Red
  Float_t lineHueMin = 240;
  Float_t lineHueMax = 360;

  TLine *line1;
  TLine *line2;
  TLine *line3;
  TLine *line4;

  if (station == 1) {
    // station 1, ring 1 (inner-most ring)
    thisRingIdx = 0;
    numChambersInRing = 36;
    TVector3 p1(101, 9.361, 0);
    TVector3 p2(101, -9.361, 0);
    TVector3 p3(260, -22.353, 0);
    TVector3 p4(260, 22.353, 0);

    // compute thisRingNchamber1Min, thisRingNchamber1Max,
    //  thisRingNchamber1Avg and thisRingNchamber1Base
    thisRingNchamber1Avg = 0.;
    thisRingNchamber1Max = 0.;
    for (int j = 0; j < numChambersInRing; j++) {
      thisRingNchamber1Avg += nchamber1[thisRingIdx][j];
      if (thisRingNchamber1Max < nchamber1[thisRingIdx][j]) {
        thisRingNchamber1Max = nchamber1[thisRingIdx][j];
      }
    }
    thisRingNchamber1Min = thisRingNchamber1Max;
    for (int j = 0; j < numChambersInRing; j++) {
      if (thisRingNchamber1Min > nchamber1[thisRingIdx][j] &&  //
          nchamber1[thisRingIdx][j] != 0) {
        thisRingNchamber1Min = nchamber1[thisRingIdx][j];
      }
    }
    thisRingNchamber1Avg = thisRingNchamber1Avg / numChambersInRing;
    thisRingNchamber1Base = thisRingNchamber1Min;

    for (int i = 0; i < numChambersInRing; i++) {
      // set the line color
      if (nchamber1[thisRingIdx][i] != 0) {
        hFloatHLS = int((nchamber1[0][i] - thisRingNchamber1Base) / (thisRingNchamber1Max - thisRingNchamber1Base) *  //
                            (lineHueMax - lineHueMin) +
                        lineHueMin) %
                    360;
        tempColor.HLS2RGB(hFloatHLS, lFloatHLS, sFloatHLS, rFloatRGB, gFloatRGB, bFloatRGB);
        linecolor = tempColor.GetColor(rFloatRGB, gFloatRGB, bFloatRGB);
      } else if (emptyLineColor >= 0) {
        linecolor = emptyLineColor;
      }

      // draw the chamber outline using the line color (so long
      // as the the chamber isn't empty *and* the user did not
      // set a negative line color for empty chambers)
      if ((nchamber1[thisRingIdx][i] != 0) ||  //
          (nchamber1[thisRingIdx][i] == 0 && emptyLineColor >= 0)) {
        line1 = new TLine(p1(0), p1(1), p2(0), p2(1));
        line2 = new TLine(p2(0), p2(1), p3(0), p3(1));
        line3 = new TLine(p3(0), p3(1), p4(0), p4(1));
        line4 = new TLine(p4(0), p4(1), p1(0), p1(1));
        line1->SetLineColor(linecolor);
        line2->SetLineColor(linecolor);
        line3->SetLineColor(linecolor);
        line4->SetLineColor(linecolor);

        line1->Draw();
        line2->Draw();
        line3->Draw();
        line4->Draw();
      }

      // Rotate coordinate by 1 chamber
      p1.Rotate(2 * pi / numChambersInRing, x);
      p2.Rotate(2 * pi / numChambersInRing, x);
      p3.Rotate(2 * pi / numChambersInRing, x);
      p4.Rotate(2 * pi / numChambersInRing, x);
    }

    // station 1, ring 2 (middle-ring)
    thisRingIdx = 1;
    numChambersInRing = 36;
    TVector3 q1(281.49, 25.5, 0);
    TVector3 q2(281.49, -25.5, 0);
    TVector3 q3(455.99, -41.87, 0);
    TVector3 q4(455.99, 41.87, 0);

    thisRingNchamber1Avg = 0.;
    thisRingNchamber1Max = 0.;
    for (int j = 0; j < numChambersInRing; j++) {
      thisRingNchamber1Avg += nchamber1[thisRingIdx][j];
      if (thisRingNchamber1Max < nchamber1[thisRingIdx][j]) {
        thisRingNchamber1Max = nchamber1[thisRingIdx][j];
      }
    }
    thisRingNchamber1Min = thisRingNchamber1Max;
    for (int j = 0; j < numChambersInRing; j++) {
      if (thisRingNchamber1Min > nchamber1[thisRingIdx][j] &&  //
          nchamber1[thisRingIdx][j] != 0) {
        thisRingNchamber1Min = nchamber1[thisRingIdx][j];
      }
    }
    thisRingNchamber1Avg = thisRingNchamber1Avg / numChambersInRing;
    thisRingNchamber1Base = thisRingNchamber1Min;

    for (int i = 0; i < numChambersInRing; i++) {
      if (nchamber1[thisRingIdx][i] != 0) {
        hFloatHLS = int((nchamber1[thisRingIdx][i] - thisRingNchamber1Base) /
                            (thisRingNchamber1Max - thisRingNchamber1Base) *  //
                            (lineHueMax - lineHueMin) +
                        lineHueMin) %
                    360;
        tempColor.HLS2RGB(hFloatHLS, lFloatHLS, sFloatHLS, rFloatRGB, gFloatRGB, bFloatRGB);
        linecolor = tempColor.GetColor(rFloatRGB, gFloatRGB, bFloatRGB);
      } else if (emptyLineColor >= 0) {
        linecolor = emptyLineColor;
      }

      if ((nchamber1[thisRingIdx][i] != 0) ||  //
          (nchamber1[thisRingIdx][i] == 0 && emptyLineColor >= 0)) {
        line1 = new TLine(q1(0), q1(1), q2(0), q2(1));
        line2 = new TLine(q2(0), q2(1), q3(0), q3(1));
        line3 = new TLine(q3(0), q3(1), q4(0), q4(1));
        line4 = new TLine(q4(0), q4(1), q1(0), q1(1));
        line1->SetLineColor(linecolor);
        line2->SetLineColor(linecolor);
        line3->SetLineColor(linecolor);
        line4->SetLineColor(linecolor);

        line1->Draw();
        line2->Draw();
        line3->Draw();
        line4->Draw();
      }
      q1.Rotate(2 * pi / numChambersInRing, x);
      q2.Rotate(2 * pi / numChambersInRing, x);
      q3.Rotate(2 * pi / numChambersInRing, x);
      q4.Rotate(2 * pi / numChambersInRing, x);
    }

    // station 1, ring 3 (outer-most ring)
    thisRingIdx = 2;
    numChambersInRing = 36;
    TVector3 r1(511.99, 31.7, 0);
    TVector3 r2(511.99, -31.7, 0);
    TVector3 r3(676.15, -46.05, 0);
    TVector3 r4(676.15, 46.05.6, 0);

    thisRingNchamber1Avg = 0.;
    thisRingNchamber1Max = 0.;
    for (int j = 0; j < numChambersInRing; j++) {
      thisRingNchamber1Avg += nchamber1[thisRingIdx][j];
      if (thisRingNchamber1Max < nchamber1[thisRingIdx][j]) {
        thisRingNchamber1Max = nchamber1[thisRingIdx][j];
      }
    }
    thisRingNchamber1Min = thisRingNchamber1Max;
    for (int j = 0; j < numChambersInRing; j++) {
      if (thisRingNchamber1Min > nchamber1[thisRingIdx][j] &&  //
          nchamber1[thisRingIdx][j] != 0) {
        thisRingNchamber1Min = nchamber1[thisRingIdx][j];
      }
    }
    thisRingNchamber1Avg = thisRingNchamber1Avg / numChambersInRing;
    thisRingNchamber1Base = thisRingNchamber1Min;

    for (int i = 0; i < numChambersInRing; i++) {
      if (nchamber1[thisRingIdx][i] != 0) {
        hFloatHLS = int((nchamber1[thisRingIdx][i] - thisRingNchamber1Base) /
                            (thisRingNchamber1Max - thisRingNchamber1Base) *  //
                            (lineHueMax - lineHueMin) +
                        lineHueMin) %
                    360;
        tempColor.HLS2RGB(hFloatHLS, lFloatHLS, sFloatHLS, rFloatRGB, gFloatRGB, bFloatRGB);
        linecolor = tempColor.GetColor(rFloatRGB, gFloatRGB, bFloatRGB);
      } else if (emptyLineColor >= 0) {
        linecolor = emptyLineColor;
      }

      if ((nchamber1[thisRingIdx][i] != 0) ||  //
          (nchamber1[thisRingIdx][i] == 0 && emptyLineColor >= 0)) {
        line1 = new TLine(r1(0), r1(1), r2(0), r2(1));
        line2 = new TLine(r2(0), r2(1), r3(0), r3(1));
        line3 = new TLine(r3(0), r3(1), r4(0), r4(1));
        line4 = new TLine(r4(0), r4(1), r1(0), r1(1));
        line1->SetLineColor(linecolor);
        line2->SetLineColor(linecolor);
        line3->SetLineColor(linecolor);
        line4->SetLineColor(linecolor);

        line1->Draw();
        line2->Draw();
        line3->Draw();
        line4->Draw();
      }
      r1.Rotate(2 * pi / numChambersInRing, x);
      r2.Rotate(2 * pi / numChambersInRing, x);
      r3.Rotate(2 * pi / numChambersInRing, x);
      r4.Rotate(2 * pi / numChambersInRing, x);
    }
  }

  if (station == 2) {
    // station 2, ring 1 (inner ring)
    thisRingIdx = 0;
    numChambersInRing = 18;
    TVector3 p1(146.9, 27.0, 0);
    TVector3 p2(146.9, -27.0, 0);
    TVector3 p3(336.56, -62.855, 0);
    TVector3 p4(336.56, 62.855, 0);

    // must "pre-rotate" by one-fourth of a chamber
    p1.Rotate((1 / 4.) * 2 * pi / numChambersInRing, x);
    p2.Rotate((1 / 4.) * 2 * pi / numChambersInRing, x);
    p3.Rotate((1 / 4.) * 2 * pi / numChambersInRing, x);
    p4.Rotate((1 / 4.) * 2 * pi / numChambersInRing, x);

    thisRingNchamber1Avg = 0.;
    thisRingNchamber1Max = 0.;
    for (int j = 0; j < numChambersInRing; j++) {
      thisRingNchamber1Avg += nchamber1[thisRingIdx][j];
      if (thisRingNchamber1Max < nchamber1[thisRingIdx][j]) {
        thisRingNchamber1Max = nchamber1[thisRingIdx][j];
      }
    }
    thisRingNchamber1Min = thisRingNchamber1Max;
    for (int j = 0; j < numChambersInRing; j++) {
      if (thisRingNchamber1Min > nchamber1[thisRingIdx][j] &&  //
          nchamber1[thisRingIdx][j] != 0) {
        thisRingNchamber1Min = nchamber1[thisRingIdx][j];
      }
    }
    thisRingNchamber1Avg = thisRingNchamber1Avg / numChambersInRing;
    thisRingNchamber1Base = thisRingNchamber1Min;

    for (int i = 0; i < numChambersInRing; i++) {
      if (nchamber1[thisRingIdx][i] != 0) {
        hFloatHLS = int((nchamber1[0][i] - thisRingNchamber1Base) / (thisRingNchamber1Max - thisRingNchamber1Base) *  //
                            (lineHueMax - lineHueMin) +
                        lineHueMin) %
                    360;
        tempColor.HLS2RGB(hFloatHLS, lFloatHLS, sFloatHLS, rFloatRGB, gFloatRGB, bFloatRGB);
        linecolor = tempColor.GetColor(rFloatRGB, gFloatRGB, bFloatRGB);
      } else if (emptyLineColor >= 0) {
        linecolor = emptyLineColor;
      }

      if ((nchamber1[thisRingIdx][i] != 0) ||  //
          (nchamber1[thisRingIdx][i] == 0 && emptyLineColor >= 0)) {
        line1 = new TLine(p1(0), p1(1), p2(0), p2(1));
        line2 = new TLine(p2(0), p2(1), p3(0), p3(1));
        line3 = new TLine(p3(0), p3(1), p4(0), p4(1));
        line4 = new TLine(p4(0), p4(1), p1(0), p1(1));
        line1->SetLineColor(linecolor);
        line2->SetLineColor(linecolor);
        line3->SetLineColor(linecolor);
        line4->SetLineColor(linecolor);

        line1->Draw();
        line2->Draw();
        line3->Draw();
        line4->Draw();
      }
      p1.Rotate(2 * pi / numChambersInRing, x);
      p2.Rotate(2 * pi / numChambersInRing, x);
      p3.Rotate(2 * pi / numChambersInRing, x);
      p4.Rotate(2 * pi / numChambersInRing, x);
    }

    // station 2, ring 2 (outer ring)
    thisRingIdx = 1;
    numChambersInRing = 36;
    TVector3 q1(364.02, 33.23, 0);
    TVector3 q2(364.02, -33.23, 0);
    TVector3 q3(687.08, -63.575, 0);
    TVector3 q4(687.08, 63.575, 0);

    thisRingNchamber1Avg = 0.;
    thisRingNchamber1Max = 0.;
    for (int j = 0; j < numChambersInRing; j++) {
      thisRingNchamber1Avg += nchamber1[thisRingIdx][j];
      if (thisRingNchamber1Max < nchamber1[thisRingIdx][j]) {
        thisRingNchamber1Max = nchamber1[thisRingIdx][j];
      }
    }
    thisRingNchamber1Min = thisRingNchamber1Max;
    for (int j = 0; j < numChambersInRing; j++) {
      if (thisRingNchamber1Min > nchamber1[thisRingIdx][j] &&  //
          nchamber1[thisRingIdx][j] != 0) {
        thisRingNchamber1Min = nchamber1[thisRingIdx][j];
      }
    }
    thisRingNchamber1Avg = thisRingNchamber1Avg / numChambersInRing;
    thisRingNchamber1Base = thisRingNchamber1Min;

    for (int i = 0; i < numChambersInRing; i++) {
      if (nchamber1[thisRingIdx][i] != 0) {
        hFloatHLS = int((nchamber1[thisRingIdx][i] - thisRingNchamber1Base) /
                            (thisRingNchamber1Max - thisRingNchamber1Base) *  //
                            (lineHueMax - lineHueMin) +
                        lineHueMin) %
                    360;
        tempColor.HLS2RGB(hFloatHLS, lFloatHLS, sFloatHLS, rFloatRGB, gFloatRGB, bFloatRGB);
        linecolor = tempColor.GetColor(rFloatRGB, gFloatRGB, bFloatRGB);
      } else if (emptyLineColor >= 0) {
        linecolor = emptyLineColor;
      }

      if ((nchamber1[thisRingIdx][i] != 0) ||  //
          (nchamber1[thisRingIdx][i] == 0 && emptyLineColor >= 0)) {
        line1 = new TLine(q1(0), q1(1), q2(0), q2(1));
        line2 = new TLine(q2(0), q2(1), q3(0), q3(1));
        line3 = new TLine(q3(0), q3(1), q4(0), q4(1));
        line4 = new TLine(q4(0), q4(1), q1(0), q1(1));
        line1->SetLineColor(linecolor);
        line2->SetLineColor(linecolor);
        line3->SetLineColor(linecolor);
        line4->SetLineColor(linecolor);

        line1->Draw();
        line2->Draw();
        line3->Draw();
        line4->Draw();
      }
      q1.Rotate(2 * pi / numChambersInRing, x);
      q2.Rotate(2 * pi / numChambersInRing, x);
      q3.Rotate(2 * pi / numChambersInRing, x);
      q4.Rotate(2 * pi / numChambersInRing, x);
    }
  }

  if (station == 3) {
    // station 3, ring 1 (inner ring)
    thisRingIdx = 0;
    numChambersInRing = 18;
    TVector3 p1(166.89, 30.7, 0);
    TVector3 p2(166.89, -30.7, 0);
    TVector3 p3(336.59, -62.855, 0);
    TVector3 p4(336.59, 62.855, 0);

    // must "pre-rotate" by one-fourth of a chamber
    p1.Rotate((1 / 4.) * 2 * pi / numChambersInRing, x);
    p2.Rotate((1 / 4.) * 2 * pi / numChambersInRing, x);
    p3.Rotate((1 / 4.) * 2 * pi / numChambersInRing, x);
    p4.Rotate((1 / 4.) * 2 * pi / numChambersInRing, x);

    TLine *line1;
    TLine *line2;
    TLine *line3;
    TLine *line4;

    thisRingNchamber1Avg = 0.;
    thisRingNchamber1Max = 0.;
    for (int j = 0; j < numChambersInRing; j++) {
      thisRingNchamber1Avg += nchamber1[thisRingIdx][j];
      if (thisRingNchamber1Max < nchamber1[thisRingIdx][j]) {
        thisRingNchamber1Max = nchamber1[thisRingIdx][j];
      }
    }
    thisRingNchamber1Min = thisRingNchamber1Max;
    for (int j = 0; j < numChambersInRing; j++) {
      if (thisRingNchamber1Min > nchamber1[thisRingIdx][j] &&  //
          nchamber1[thisRingIdx][j] != 0) {
        thisRingNchamber1Min = nchamber1[thisRingIdx][j];
      }
    }
    thisRingNchamber1Avg = thisRingNchamber1Avg / numChambersInRing;
    thisRingNchamber1Base = thisRingNchamber1Min;

    for (int i = 0; i < numChambersInRing; i++) {
      if (nchamber1[thisRingIdx][i] != 0) {
        hFloatHLS = int((nchamber1[0][i] - thisRingNchamber1Base) / (thisRingNchamber1Max - thisRingNchamber1Base) *  //
                            (lineHueMax - lineHueMin) +
                        lineHueMin) %
                    360;
        tempColor.HLS2RGB(hFloatHLS, lFloatHLS, sFloatHLS, rFloatRGB, gFloatRGB, bFloatRGB);
        linecolor = tempColor.GetColor(rFloatRGB, gFloatRGB, bFloatRGB);
      } else if (emptyLineColor >= 0) {
        linecolor = emptyLineColor;
      }

      if ((nchamber1[thisRingIdx][i] != 0) ||  //
          (nchamber1[thisRingIdx][i] == 0 && emptyLineColor >= 0)) {
        line1 = new TLine(p1(0), p1(1), p2(0), p2(1));
        line2 = new TLine(p2(0), p2(1), p3(0), p3(1));
        line3 = new TLine(p3(0), p3(1), p4(0), p4(1));
        line4 = new TLine(p4(0), p4(1), p1(0), p1(1));
        line1->SetLineColor(linecolor);
        line2->SetLineColor(linecolor);
        line3->SetLineColor(linecolor);
        line4->SetLineColor(linecolor);

        line1->Draw();
        line2->Draw();
        line3->Draw();
        line4->Draw();
      }
      p1.Rotate(2 * pi / numChambersInRing, x);
      p2.Rotate(2 * pi / numChambersInRing, x);
      p3.Rotate(2 * pi / numChambersInRing, x);
      p4.Rotate(2 * pi / numChambersInRing, x);
    }

    // station 3, ring 2 (outer ring)
    thisRingIdx = 1;
    numChambersInRing = 36;
    TVector3 q1(364.02, 33.23, 0);
    TVector3 q2(364.02, -33.23, 0);
    TVector3 q3(687.08, -63.575, 0);
    TVector3 q4(687.08, 63.575, 0);

    thisRingNchamber1Avg = 0.;
    thisRingNchamber1Max = 0.;
    for (int j = 0; j < numChambersInRing; j++) {
      thisRingNchamber1Avg += nchamber1[thisRingIdx][j];
      if (thisRingNchamber1Max < nchamber1[thisRingIdx][j]) {
        thisRingNchamber1Max = nchamber1[thisRingIdx][j];
      }
    }
    thisRingNchamber1Min = thisRingNchamber1Max;
    for (int j = 0; j < numChambersInRing; j++) {
      if (thisRingNchamber1Min > nchamber1[thisRingIdx][j] &&  //
          nchamber1[thisRingIdx][j] != 0) {
        thisRingNchamber1Min = nchamber1[thisRingIdx][j];
      }
    }
    thisRingNchamber1Avg = thisRingNchamber1Avg / numChambersInRing;
    thisRingNchamber1Base = thisRingNchamber1Min;

    for (int i = 0; i < numChambersInRing; i++) {
      if (nchamber1[thisRingIdx][i] != 0) {
        hFloatHLS = int((nchamber1[thisRingIdx][i] - thisRingNchamber1Base) /
                            (thisRingNchamber1Max - thisRingNchamber1Base) *  //
                            (lineHueMax - lineHueMin) +
                        lineHueMin) %
                    360;
        tempColor.HLS2RGB(hFloatHLS, lFloatHLS, sFloatHLS, rFloatRGB, gFloatRGB, bFloatRGB);
        linecolor = tempColor.GetColor(rFloatRGB, gFloatRGB, bFloatRGB);
      } else if (emptyLineColor >= 0) {
        linecolor = emptyLineColor;
      }

      if ((nchamber1[thisRingIdx][i] != 0) ||  //
          (nchamber1[thisRingIdx][i] == 0 && emptyLineColor >= 0)) {
        line1 = new TLine(q1(0), q1(1), q2(0), q2(1));
        line2 = new TLine(q2(0), q2(1), q3(0), q3(1));
        line3 = new TLine(q3(0), q3(1), q4(0), q4(1));
        line4 = new TLine(q4(0), q4(1), q1(0), q1(1));
        line1->SetLineColor(linecolor);
        line2->SetLineColor(linecolor);
        line3->SetLineColor(linecolor);
        line4->SetLineColor(linecolor);

        line1->Draw();
        line2->Draw();
        line3->Draw();
        line4->Draw();
      }
      q1.Rotate(2 * pi / numChambersInRing, x);
      q2.Rotate(2 * pi / numChambersInRing, x);
      q3.Rotate(2 * pi / numChambersInRing, x);
      q4.Rotate(2 * pi / numChambersInRing, x);
    }
  }

  if (station == 4) {
    // station 4, ring 1 (the only ring on station 4... so far)
    thisRingIdx = 0;
    numChambersInRing = 18;
    TVector3 p1(186.99, 34.505.15, 0);
    TVector3 p2(186.99, -34.505, 0);
    TVector3 p3(336.41, -62.825, 0);
    TVector3 p4(336.41, 62.825, 0);

    // must "pre-rotate" by one-fourth of a chamber
    p1.Rotate((1 / 4.) * 2 * pi / numChambersInRing, x);
    p2.Rotate((1 / 4.) * 2 * pi / numChambersInRing, x);
    p3.Rotate((1 / 4.) * 2 * pi / numChambersInRing, x);
    p4.Rotate((1 / 4.) * 2 * pi / numChambersInRing, x);

    TLine *line1;
    TLine *line2;
    TLine *line3;
    TLine *line4;

    thisRingNchamber1Avg = 0.;
    thisRingNchamber1Max = 0.;
    for (int j = 0; j < numChambersInRing; j++) {
      thisRingNchamber1Avg += nchamber1[thisRingIdx][j];
      if (thisRingNchamber1Max < nchamber1[thisRingIdx][j]) {
        thisRingNchamber1Max = nchamber1[thisRingIdx][j];
      }
    }
    thisRingNchamber1Min = thisRingNchamber1Max;
    for (int j = 0; j < numChambersInRing; j++) {
      if (thisRingNchamber1Min > nchamber1[thisRingIdx][j] &&  //
          nchamber1[thisRingIdx][j] != 0) {
        thisRingNchamber1Min = nchamber1[thisRingIdx][j];
      }
    }
    thisRingNchamber1Avg = thisRingNchamber1Avg / numChambersInRing;
    thisRingNchamber1Base = thisRingNchamber1Min;

    for (int i = 0; i < numChambersInRing; i++) {
      if (nchamber1[thisRingIdx][i] != 0) {
        hFloatHLS = int((nchamber1[0][i] - thisRingNchamber1Base) / (thisRingNchamber1Max - thisRingNchamber1Base) *  //
                            (lineHueMax - lineHueMin) +
                        lineHueMin) %
                    360;
        tempColor.HLS2RGB(hFloatHLS, lFloatHLS, sFloatHLS, rFloatRGB, gFloatRGB, bFloatRGB);
        linecolor = tempColor.GetColor(rFloatRGB, gFloatRGB, bFloatRGB);
      } else if (emptyLineColor >= 0) {
        linecolor = emptyLineColor;
      }

      if ((nchamber1[thisRingIdx][i] != 0) ||  //
          (nchamber1[thisRingIdx][i] == 0 && emptyLineColor >= 0)) {
        line1 = new TLine(p1(0), p1(1), p2(0), p2(1));
        line2 = new TLine(p2(0), p2(1), p3(0), p3(1));
        line3 = new TLine(p3(0), p3(1), p4(0), p4(1));
        line4 = new TLine(p4(0), p4(1), p1(0), p1(1));
        line1->SetLineColor(linecolor);
        line2->SetLineColor(linecolor);
        line3->SetLineColor(linecolor);
        line4->SetLineColor(linecolor);

        line1->Draw();
        line2->Draw();
        line3->Draw();
        line4->Draw();
      }
      p1.Rotate(2 * pi / numChambersInRing, x);
      p2.Rotate(2 * pi / numChambersInRing, x);
      p3.Rotate(2 * pi / numChambersInRing, x);
      p4.Rotate(2 * pi / numChambersInRing, x);
    }
  }
}
