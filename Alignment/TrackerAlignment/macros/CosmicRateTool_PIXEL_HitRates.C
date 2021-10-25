void CosmicRateTool_PIXEL_HitRates(const char *fileName, unsigned int runLow = 0, unsigned int runUp = 0) {
  TString InputFile = Form("%s", fileName);
  TFile *file = new TFile(InputFile);

  bool IsFileExist;
  IsFileExist = file->IsZombie();
  if (IsFileExist) {
    cout << endl
         << "====================================================================================================="
         << endl;
    cout << fileName << " not found. Check the file!" << endl;
    cout << "====================================================================================================="
         << endl
         << endl;
    exit(EXIT_FAILURE);
  }

  TTree *tree = (TTree *)file->Get("cosmicRateAnalyzer/Run");

  double run_time;
  unsigned int runnum;
  int number_of_events;
  int number_of_tracks;
  int number_of_hits_Total;
  int number_of_hits_PIX;
  int number_of_hits_BPIX;
  int number_of_hits_BPIX_layer1;
  int number_of_hits_BPIX_layer2;
  int number_of_hits_BPIX_layer3;
  int number_of_hits_BPIX_layer4;
  int number_of_hits_FPIX;
  int number_of_hits_FPIX_disk1;
  int number_of_hits_FPIX_disk2;
  int number_of_hits_FPIX_disk3;
  int number_of_hits_FPIX_disk1_plus;
  int number_of_hits_FPIX_disk1_minus;
  int number_of_hits_FPIX_disk2_plus;
  int number_of_hits_FPIX_disk2_minus;
  int number_of_hits_FPIX_disk3_plus;
  int number_of_hits_FPIX_disk3_minus;

  tree->SetBranchAddress("run_time", &run_time);
  tree->SetBranchAddress("runnum", &runnum);
  tree->SetBranchAddress("number_of_events", &number_of_events);
  tree->SetBranchAddress("number_of_tracks", &number_of_tracks);
  tree->SetBranchAddress("number_of_hits_Total", &number_of_hits_Total);
  tree->SetBranchAddress("number_of_hits_PIX", &number_of_hits_PIX);
  tree->SetBranchAddress("number_of_hits_BPIX", &number_of_hits_BPIX);
  tree->SetBranchAddress("number_of_hits_BPIX_layer1", &number_of_hits_BPIX_layer1);
  tree->SetBranchAddress("number_of_hits_BPIX_layer2", &number_of_hits_BPIX_layer2);
  tree->SetBranchAddress("number_of_hits_BPIX_layer3", &number_of_hits_BPIX_layer3);
  tree->SetBranchAddress("number_of_hits_BPIX_layer4", &number_of_hits_BPIX_layer4);
  tree->SetBranchAddress("number_of_hits_FPIX", &number_of_hits_FPIX);
  tree->SetBranchAddress("number_of_hits_FPIX_disk1", &number_of_hits_FPIX_disk1);
  tree->SetBranchAddress("number_of_hits_FPIX_disk2", &number_of_hits_FPIX_disk2);
  tree->SetBranchAddress("number_of_hits_FPIX_disk3", &number_of_hits_FPIX_disk3);
  tree->SetBranchAddress("number_of_hits_FPIX_disk1_plus", &number_of_hits_FPIX_disk1_plus);
  tree->SetBranchAddress("number_of_hits_FPIX_disk1_minus", &number_of_hits_FPIX_disk1_minus);
  tree->SetBranchAddress("number_of_hits_FPIX_disk2_plus", &number_of_hits_FPIX_disk2_plus);
  tree->SetBranchAddress("number_of_hits_FPIX_disk2_minus", &number_of_hits_FPIX_disk2_minus);
  tree->SetBranchAddress("number_of_hits_FPIX_disk3_plus", &number_of_hits_FPIX_disk3_plus);
  tree->SetBranchAddress("number_of_hits_FPIX_disk3_minus", &number_of_hits_FPIX_disk3_minus);

  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  //		Various Rates Declerations
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  vector<double> events;
  vector<double> event_rate;
  vector<double> event_rate_err;
  vector<double> track_rate;
  vector<double> track_rate_err;
  vector<double> runNumber;
  vector<double> runNumber_err;
  vector<double> tracks;
  vector<double> tracks_err;
  vector<double> weight;

  vector<double> hits;
  vector<double> hit_rate_Total;
  vector<double> hit_rate_Total_err;
  vector<double> hit_rate_PIX;
  vector<double> hit_rate_PIX_err;
  vector<double> hit_rate_BPIX;
  vector<double> hit_rate_BPIX_err;
  vector<double> hit_rate_BPIX_layer1;
  vector<double> hit_rate_BPIX_layer1_err;
  vector<double> hit_rate_BPIX_layer2;
  vector<double> hit_rate_BPIX_layer2_err;
  vector<double> hit_rate_BPIX_layer3;
  vector<double> hit_rate_BPIX_layer3_err;
  vector<double> hit_rate_BPIX_layer4;
  vector<double> hit_rate_BPIX_layer4_err;
  vector<double> hit_rate_FPIX;
  vector<double> hit_rate_FPIX_err;
  vector<double> hit_rate_FPIX_disk1;
  vector<double> hit_rate_FPIX_disk1_err;
  vector<double> hit_rate_FPIX_disk2;
  vector<double> hit_rate_FPIX_disk2_err;
  vector<double> hit_rate_FPIX_disk3;
  vector<double> hit_rate_FPIX_disk3_err;
  vector<double> hit_rate_FPIX_disk1_plus;
  vector<double> hit_rate_FPIX_disk1_plus_err;
  vector<double> hit_rate_FPIX_disk1_minus;
  vector<double> hit_rate_FPIX_disk1_minus_err;
  vector<double> hit_rate_FPIX_disk2_plus;
  vector<double> hit_rate_FPIX_disk2_plus_err;
  vector<double> hit_rate_FPIX_disk2_minus;
  vector<double> hit_rate_FPIX_disk2_minus_err;
  vector<double> hit_rate_FPIX_disk3_plus;
  vector<double> hit_rate_FPIX_disk3_plus_err;
  vector<double> hit_rate_FPIX_disk3_minus;
  vector<double> hit_rate_FPIX_disk3_minus_err;

  string Bar_Xtitle_BPIX[6] = {"PIXEL", "BPIX", "BPIX_1", "BPIX_2", "BPIX_3", "BPIX_4"};
  string Bar_Xtitle_FPIX[11] = {
      "PIXEL", "FPIX", "FPIX_1", "FPIX_1+", "FPIX_1-", "FPIX_2", "FPIX_2+", "FPIX_2-", "FPIX_3", "FPIX_3+", "FPIX_3-"};
  string Bar_Xtitle_PIXEL[10] = {
      "PIXEL", "BPIX", "BPIX_1", "BPIX_2", "BPIX_3", "BPIX_4", "FPIX", "FPIX_1", "FPIX_2", "FPIX_3"};
  double Bar_Ytitle_BPIX[6] = {0};
  double Bar_Ytitle_FPIX[11] = {0};
  double Bar_Ytitle_PIXEL[10] = {0};

  int j = 0;
  double total_tracks = 0, nTotalEvents = 0, nTotalTracks = 0, nTotalHits = 0, nZeroRunTimeRuns = 0;

  Long64_t n = tree->GetEntriesFast();
  for (Long64_t jentry = 0; jentry < n; jentry++) {
    tree->GetEntry(jentry);
    if (run_time == 0 || run_time < 0) {
      nZeroRunTimeRuns++;
      continue;
    }

    if (runLow != 0 && runUp != 0) {
      if (runnum < runLow)
        continue;
      if (runnum > runUp)
        break;
    }

    events.push_back(number_of_events);
    event_rate.push_back(number_of_events / run_time);
    runNumber.push_back(runnum);
    track_rate.push_back(number_of_tracks / run_time);
    tracks.push_back(number_of_tracks);

    nTotalEvents += number_of_events;
    nTotalTracks += number_of_tracks;
    nTotalHits += number_of_hits_Total;

    hits.push_back(number_of_hits_Total);
    hit_rate_Total.push_back(number_of_hits_Total / run_time);
    hit_rate_Total_err.push_back(sqrt(float(number_of_hits_Total)) / run_time);
    hit_rate_PIX.push_back(number_of_hits_PIX / run_time);
    hit_rate_PIX_err.push_back(sqrt(float(number_of_hits_PIX)) / run_time);
    hit_rate_BPIX.push_back(number_of_hits_BPIX / run_time);
    hit_rate_BPIX_err.push_back(sqrt(float(number_of_hits_BPIX)) / run_time);
    hit_rate_BPIX_layer1.push_back(number_of_hits_BPIX_layer1 / run_time);
    hit_rate_BPIX_layer1_err.push_back(sqrt(float(number_of_hits_BPIX_layer1)) / run_time);
    hit_rate_BPIX_layer2.push_back(number_of_hits_BPIX_layer2 / run_time);
    hit_rate_BPIX_layer2_err.push_back(sqrt(float(number_of_hits_BPIX_layer2)) / run_time);
    hit_rate_BPIX_layer3.push_back(number_of_hits_BPIX_layer3 / run_time);
    hit_rate_BPIX_layer3_err.push_back(sqrt(float(number_of_hits_BPIX_layer3)) / run_time);
    hit_rate_BPIX_layer4.push_back(number_of_hits_BPIX_layer4 / run_time);
    hit_rate_BPIX_layer4_err.push_back(sqrt(float(number_of_hits_BPIX_layer4)) / run_time);
    hit_rate_FPIX.push_back(number_of_hits_FPIX / run_time);
    hit_rate_FPIX_err.push_back(sqrt(float(number_of_hits_FPIX)) / run_time);
    hit_rate_FPIX_disk1.push_back(number_of_hits_FPIX_disk1 / run_time);
    hit_rate_FPIX_disk1_err.push_back(sqrt(float(number_of_hits_FPIX_disk1)) / run_time);
    hit_rate_FPIX_disk2.push_back(number_of_hits_FPIX_disk2 / run_time);
    hit_rate_FPIX_disk2_err.push_back(sqrt(float(number_of_hits_FPIX_disk2)) / run_time);
    hit_rate_FPIX_disk3.push_back(number_of_hits_FPIX_disk3 / run_time);
    hit_rate_FPIX_disk3_err.push_back(sqrt(float(number_of_hits_FPIX_disk3)) / run_time);
    hit_rate_FPIX_disk1_plus.push_back(number_of_hits_FPIX_disk1_plus / run_time);
    hit_rate_FPIX_disk1_plus_err.push_back(sqrt(float(number_of_hits_FPIX_disk1_plus)) / run_time);
    hit_rate_FPIX_disk1_minus.push_back(number_of_hits_FPIX_disk1_minus / run_time);
    hit_rate_FPIX_disk1_minus_err.push_back(sqrt(float(number_of_hits_FPIX_disk1_minus)) / run_time);
    hit_rate_FPIX_disk2_plus.push_back(number_of_hits_FPIX_disk2_plus / run_time);
    hit_rate_FPIX_disk2_plus_err.push_back(sqrt(float(number_of_hits_FPIX_disk2_plus)) / run_time);
    hit_rate_FPIX_disk2_minus.push_back(number_of_hits_FPIX_disk2_minus / run_time);
    hit_rate_FPIX_disk2_minus_err.push_back(sqrt(float(number_of_hits_FPIX_disk2_minus)) / run_time);
    hit_rate_FPIX_disk3_plus.push_back(number_of_hits_FPIX_disk3_plus / run_time);
    hit_rate_FPIX_disk3_plus_err.push_back(sqrt(float(number_of_hits_FPIX_disk3_plus)) / run_time);
    hit_rate_FPIX_disk3_minus.push_back(number_of_hits_FPIX_disk3_minus / run_time);
    hit_rate_FPIX_disk3_minus_err.push_back(sqrt(float(number_of_hits_FPIX_disk3_minus)) / run_time);

    track_rate_err.push_back(sqrt(float(number_of_tracks)) / run_time);
    event_rate_err.push_back(sqrt(float(number_of_events)) / run_time);
    runNumber_err.push_back(0);

    j++;
  }
  std::cout << "Total Runs in this files: " << n << endl;
  std::cout << "Runs with negative or 0 runtime: " << nZeroRunTimeRuns << endl;
  std::cout << "Total Events: " << nTotalEvents << std::endl;
  std::cout << "Total Tracks: " << nTotalTracks << std::endl;
  std::cout << "Total Hits: " << nTotalHits << std::endl;
  std::cout << "Runs without 0 or negative runtime actually used in plotting & evaluation(j value): " << j << std::endl;

  //+++++++++++++++++++++++++++++       Make Directory     +++++++++++++++++++++++++++++++++++++
  gSystem->Exec("mkdir -p Hit_Rate_Plots");

  //-----------------------------------    PLOTTING  -------------------------------------------
  TCanvas c("c1", "c1", 604, 82, 856, 836);  // Declare canvas
  gStyle->SetOptStat(0);
  c.Range(298434.4, -0.2989256, 299381.3, 2.010954);
  c.SetFillColor(0);
  c.SetBorderMode(0);
  c.SetBorderSize(2);
  c.SetTickx(1);
  c.SetTicky(1);
  c.SetGrid();
  c.SetLeftMargin(0.1883886);
  c.SetRightMargin(0.03909953);
  c.SetTopMargin(0.0875817);
  c.SetBottomMargin(0.1294118);
  c.SetFrameLineWidth(3);
  c.SetFrameBorderMode(0);

  //============  Text  =============//
  TLatex top_right_Title = TLatex();
  top_right_Title.SetTextFont(42);
  top_right_Title.SetTextSize(0.03717);

  TLatex detector = TLatex();
  detector.SetTextFont(62);
  detector.SetTextSize(0.047);

  //-------- TVectors to be taken as input in TGraphs for plotting ---------//
  TVectorD event_rate_VecD;
  TVectorD event_rate_err_VecD;
  TVectorD track_rate_VecD;
  TVectorD track_rate_err_VecD;
  TVectorD runNumber_VecD;
  TVectorD runNumber_err_VecD;
  TVectorD hit_rate_VecD;
  TVectorD hit_rate_err_VecD;
  TVectorD hit_rate_PIX_VecD;
  TVectorD hit_rate_PIX_err_VecD;
  TVectorD hit_rate_BPIX_VecD;
  TVectorD hit_rate_BPIX_err_VecD;
  TVectorD hit_rate_BPIX_layer1_VecD;
  TVectorD hit_rate_BPIX_layer1_err_VecD;
  TVectorD hit_rate_BPIX_layer2_VecD;
  TVectorD hit_rate_BPIX_layer2_err_VecD;
  TVectorD hit_rate_BPIX_layer3_VecD;
  TVectorD hit_rate_BPIX_layer3_err_VecD;
  TVectorD hit_rate_BPIX_layer4_VecD;
  TVectorD hit_rate_BPIX_layer4_err_VecD;
  TVectorD hit_rate_FPIX_VecD;
  TVectorD hit_rate_FPIX_err_VecD;
  TVectorD hit_rate_FPIX_disk1_VecD;
  TVectorD hit_rate_FPIX_disk1_err_VecD;
  TVectorD hit_rate_FPIX_disk2_VecD;
  TVectorD hit_rate_FPIX_disk2_err_VecD;
  TVectorD hit_rate_FPIX_disk3_VecD;
  TVectorD hit_rate_FPIX_disk3_err_VecD;
  TVectorD hit_rate_FPIX_disk1_plus_VecD;
  TVectorD hit_rate_FPIX_disk1_plus_err_VecD;
  TVectorD hit_rate_FPIX_disk1_minus_VecD;
  TVectorD hit_rate_FPIX_disk1_minus_err_VecD;
  TVectorD hit_rate_FPIX_disk2_plus_VecD;
  TVectorD hit_rate_FPIX_disk2_plus_err_VecD;
  TVectorD hit_rate_FPIX_disk2_minus_VecD;
  TVectorD hit_rate_FPIX_disk2_minus_err_VecD;
  TVectorD hit_rate_FPIX_disk3_plus_VecD;
  TVectorD hit_rate_FPIX_disk3_plus_err_VecD;
  TVectorD hit_rate_FPIX_disk3_minus_VecD;
  TVectorD hit_rate_FPIX_disk3_minus_err_VecD;

  runNumber_VecD.Use(runNumber.size(), &(runNumber[0]));
  runNumber_err_VecD.Use(runNumber_err.size(), &(runNumber_err[0]));
  event_rate_VecD.Use(event_rate.size(), &(event_rate[0]));
  event_rate_err_VecD.Use(event_rate_err.size(), &(event_rate_err[0]));
  track_rate_VecD.Use(track_rate.size(), &(track_rate[0]));
  track_rate_err_VecD.Use(track_rate_err.size(), &(track_rate_err[0]));
  hit_rate_VecD.Use(hit_rate_Total.size(), &(hit_rate_Total[0]));
  hit_rate_err_VecD.Use(hit_rate_Total_err.size(), &(hit_rate_Total_err[0]));
  hit_rate_PIX_VecD.Use(hit_rate_PIX.size(), &(hit_rate_PIX[0]));
  hit_rate_PIX_err_VecD.Use(hit_rate_PIX_err.size(), &(hit_rate_PIX_err[0]));
  hit_rate_BPIX_VecD.Use(hit_rate_BPIX.size(), &(hit_rate_BPIX[0]));
  hit_rate_BPIX_err_VecD.Use(hit_rate_BPIX_err.size(), &(hit_rate_BPIX_err[0]));
  hit_rate_BPIX_layer1_VecD.Use(hit_rate_BPIX_layer1.size(), &(hit_rate_BPIX_layer1[0]));
  hit_rate_BPIX_layer1_err_VecD.Use(hit_rate_BPIX_layer1_err.size(), &(hit_rate_BPIX_layer1_err[0]));
  hit_rate_BPIX_layer2_VecD.Use(hit_rate_BPIX_layer2.size(), &(hit_rate_BPIX_layer2[0]));
  hit_rate_BPIX_layer2_err_VecD.Use(hit_rate_BPIX_layer2_err.size(), &(hit_rate_BPIX_layer2_err[0]));
  hit_rate_BPIX_layer3_VecD.Use(hit_rate_BPIX_layer3.size(), &(hit_rate_BPIX_layer3[0]));
  hit_rate_BPIX_layer3_err_VecD.Use(hit_rate_BPIX_layer3_err.size(), &(hit_rate_BPIX_layer3_err[0]));
  hit_rate_BPIX_layer4_VecD.Use(hit_rate_BPIX_layer4.size(), &(hit_rate_BPIX_layer4[0]));
  hit_rate_BPIX_layer4_err_VecD.Use(hit_rate_BPIX_layer4_err.size(), &(hit_rate_BPIX_layer4_err[0]));
  hit_rate_FPIX_VecD.Use(hit_rate_FPIX.size(), &(hit_rate_FPIX[0]));
  hit_rate_FPIX_err_VecD.Use(hit_rate_FPIX_err.size(), &(hit_rate_FPIX_err[0]));
  hit_rate_FPIX_disk1_VecD.Use(hit_rate_FPIX_disk1.size(), &(hit_rate_FPIX_disk1[0]));
  hit_rate_FPIX_disk1_err_VecD.Use(hit_rate_FPIX_disk1_err.size(), &(hit_rate_FPIX_disk1_err[0]));
  hit_rate_FPIX_disk2_VecD.Use(hit_rate_FPIX_disk2.size(), &(hit_rate_FPIX_disk2[0]));
  hit_rate_FPIX_disk2_err_VecD.Use(hit_rate_FPIX_disk2_err.size(), &(hit_rate_FPIX_disk2_err[0]));
  hit_rate_FPIX_disk3_VecD.Use(hit_rate_FPIX_disk3.size(), &(hit_rate_FPIX_disk3[0]));
  hit_rate_FPIX_disk3_err_VecD.Use(hit_rate_FPIX_disk3_err.size(), &(hit_rate_FPIX_disk3_err[0]));
  hit_rate_FPIX_disk1_plus_VecD.Use(hit_rate_FPIX_disk1_plus.size(), &(hit_rate_FPIX_disk1_plus[0]));
  hit_rate_FPIX_disk1_plus_err_VecD.Use(hit_rate_FPIX_disk1_plus_err.size(), &(hit_rate_FPIX_disk1_plus_err[0]));
  hit_rate_FPIX_disk1_minus_VecD.Use(hit_rate_FPIX_disk1_minus.size(), &(hit_rate_FPIX_disk1_minus[0]));
  hit_rate_FPIX_disk1_minus_err_VecD.Use(hit_rate_FPIX_disk1_minus_err.size(), &(hit_rate_FPIX_disk1_minus_err[0]));
  hit_rate_FPIX_disk2_plus_VecD.Use(hit_rate_FPIX_disk2_plus.size(), &(hit_rate_FPIX_disk2_plus[0]));
  hit_rate_FPIX_disk2_plus_err_VecD.Use(hit_rate_FPIX_disk2_plus_err.size(), &(hit_rate_FPIX_disk2_plus_err[0]));
  hit_rate_FPIX_disk2_minus_VecD.Use(hit_rate_FPIX_disk2_minus.size(), &(hit_rate_FPIX_disk2_minus[0]));
  hit_rate_FPIX_disk2_minus_err_VecD.Use(hit_rate_FPIX_disk2_minus_err.size(), &(hit_rate_FPIX_disk2_minus_err[0]));
  hit_rate_FPIX_disk3_plus_VecD.Use(hit_rate_FPIX_disk3_plus.size(), &(hit_rate_FPIX_disk3_plus[0]));
  hit_rate_FPIX_disk3_plus_err_VecD.Use(hit_rate_FPIX_disk3_plus_err.size(), &(hit_rate_FPIX_disk3_plus_err[0]));
  hit_rate_FPIX_disk3_minus_VecD.Use(hit_rate_FPIX_disk3_minus.size(), &(hit_rate_FPIX_disk3_minus[0]));
  hit_rate_FPIX_disk3_minus_err_VecD.Use(hit_rate_FPIX_disk3_minus_err.size(), &(hit_rate_FPIX_disk3_minus_err[0]));

  //+++++++++++++++++++++++++++++  Overall event rate  +++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_event_rate(runNumber_VecD, event_rate_VecD, runNumber_err_VecD, event_rate_err_VecD);
  gr_event_rate.GetXaxis()->SetTitle("Run Number");
  gr_event_rate.GetXaxis()->SetLabelSize(0.04);
  gr_event_rate.GetXaxis()->SetNoExponent();
  gr_event_rate.GetXaxis()->SetNdivisions(5);
  gr_event_rate.GetYaxis()->SetTitle("Event Rate (Hz)");
  gr_event_rate.GetXaxis()->SetTitleSize(0.05);
  gr_event_rate.GetYaxis()->SetLabelSize(0.05);
  gr_event_rate.GetYaxis()->SetTitleSize(0.05);
  gr_event_rate.SetMarkerStyle(20);
  gr_event_rate.SetMarkerSize(1.4);
  gr_event_rate.SetMarkerColor(kRed);
  gr_event_rate.SetTitle("");
  gr_event_rate.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "Event Rate");
  c.SetGrid();
  c.SaveAs("event_rate.png");
  c.SaveAs("event_rate.pdf");
  c.SaveAs("event_rate.C");
  c.Clear();
  gSystem->Exec("mv event_rate.png Hit_Rate_Plots");
  gSystem->Exec("mv event_rate.pdf Hit_Rate_Plots");
  gSystem->Exec("mv event_rate.C Hit_Rate_Plots");
  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++  Overall track rate  +++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_track_rate(runNumber_VecD, track_rate_VecD, runNumber_err_VecD, track_rate_err_VecD);
  gr_track_rate.GetXaxis()->SetTitle("Run Number");
  gr_track_rate.GetXaxis()->SetLabelSize(0.04);
  gr_track_rate.GetXaxis()->SetNoExponent();
  gr_track_rate.GetXaxis()->SetNdivisions(5);
  gr_track_rate.GetYaxis()->SetTitle("Track Rate (Hz)");
  gr_track_rate.GetXaxis()->SetTitleSize(0.05);
  gr_track_rate.GetYaxis()->SetLabelSize(0.05);
  gr_track_rate.GetYaxis()->SetTitleSize(0.05);
  gr_track_rate.SetMarkerStyle(20);
  gr_track_rate.SetMarkerSize(1.4);
  gr_track_rate.SetMarkerColor(kRed);
  gr_track_rate.SetTitle("");
  gr_track_rate.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "Track Rate");
  c.SetGrid();
  c.SaveAs("track_rate.png");
  c.SaveAs("track_rate.pdf");
  c.SaveAs("track_rate.C");
  c.Clear();
  gSystem->Exec("mv track_rate.png Hit_Rate_Plots");
  gSystem->Exec("mv track_rate.pdf Hit_Rate_Plots");
  gSystem->Exec("mv track_rate.C Hit_Rate_Plots");

  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++  Overall hit rate  +++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_hit_rate(runNumber_VecD, hit_rate_VecD, runNumber_err_VecD, hit_rate_err_VecD);
  gr_hit_rate.GetXaxis()->SetTitle("Run Number");
  gr_hit_rate.GetXaxis()->SetLabelSize(0.04);
  gr_hit_rate.GetXaxis()->SetNoExponent();
  gr_hit_rate.GetXaxis()->SetNdivisions(5);
  gr_hit_rate.GetYaxis()->SetTitle("Hit Rate (Hz)");
  gr_hit_rate.GetXaxis()->SetTitleSize(0.05);
  gr_hit_rate.GetYaxis()->SetLabelSize(0.05);
  gr_hit_rate.GetYaxis()->SetTitleSize(0.05);
  gr_hit_rate.SetMarkerStyle(20);
  gr_hit_rate.SetMarkerSize(1.4);
  gr_hit_rate.SetMarkerColor(kRed);
  gr_hit_rate.SetTitle("");
  gr_hit_rate.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "Hit Rate");
  c.SetGrid();
  c.SaveAs("hit_rate.png");
  c.SaveAs("hit_rate.pdf");
  c.SaveAs("hit_rate.C");
  c.Clear();
  gSystem->Exec("mv hit_rate.png Hit_Rate_Plots");
  gSystem->Exec("mv hit_rate.pdf Hit_Rate_Plots");
  gSystem->Exec("mv hit_rate.C Hit_Rate_Plots");

  //-----------------------------------------------------------------------------------------------

  //+++++++++++++++++++++++++++++++  Total Pixel hit rate +++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_hit_rate_PIX(runNumber_VecD, hit_rate_PIX_VecD, runNumber_err_VecD, hit_rate_PIX_err_VecD);
  gr_hit_rate_PIX.GetXaxis()->SetTitle("Run Number");
  gr_hit_rate_PIX.GetXaxis()->SetLabelSize(0.04);
  gr_hit_rate_PIX.GetXaxis()->SetNoExponent();
  gr_hit_rate_PIX.GetXaxis()->SetNdivisions(5);
  gr_hit_rate_PIX.GetYaxis()->SetTitle("Hit Rate (Hz)");
  gr_hit_rate_PIX.GetXaxis()->SetTitleSize(0.05);
  gr_hit_rate_PIX.GetYaxis()->SetLabelSize(0.05);
  gr_hit_rate_PIX.GetYaxis()->SetTitleSize(0.05);
  gr_hit_rate_PIX.SetMarkerStyle(20);
  gr_hit_rate_PIX.SetMarkerSize(1.4);
  gr_hit_rate_PIX.SetMarkerColor(2);
  gr_hit_rate_PIX.SetTitle("");
  gr_hit_rate_PIX.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "PIXEL");
  c.SetGrid();
  c.SaveAs("pixel_hit_rate.png");
  c.SaveAs("pixel_hit_rate.pdf");
  c.SaveAs("pixel_hit_rate.C");
  c.Clear();
  gSystem->Exec("mv pixel_hit_rate.png Hit_Rate_Plots");
  gSystem->Exec("mv pixel_hit_rate.pdf Hit_Rate_Plots");
  gSystem->Exec("mv pixel_hit_rate.C Hit_Rate_Plots");

  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  FPIX hit rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_hit_rate_FPIX(runNumber_VecD, hit_rate_FPIX_VecD, runNumber_err_VecD, hit_rate_FPIX_err_VecD);
  gr_hit_rate_FPIX.GetXaxis()->SetTitle("Run Number");
  gr_hit_rate_FPIX.GetXaxis()->SetLabelSize(0.04);
  gr_hit_rate_FPIX.GetXaxis()->SetNoExponent();
  gr_hit_rate_FPIX.GetXaxis()->SetNdivisions(5);
  gr_hit_rate_FPIX.GetYaxis()->SetTitle("Hit Rate (Hz)");
  gr_hit_rate_FPIX.GetXaxis()->SetTitleSize(0.05);
  gr_hit_rate_FPIX.GetYaxis()->SetLabelSize(0.05);
  gr_hit_rate_FPIX.GetYaxis()->SetTitleSize(0.05);
  gr_hit_rate_FPIX.SetMarkerStyle(20);
  gr_hit_rate_FPIX.SetMarkerSize(1.4);
  gr_hit_rate_FPIX.SetMarkerColor(kRed);
  gr_hit_rate_FPIX.SetTitle("");
  gr_hit_rate_FPIX.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "FPIX");
  c.SetGrid();
  c.SaveAs("fpix_hit_rate.png");
  c.SaveAs("fpix_hit_rate.pdf");
  c.SaveAs("fpix_hit_rate.C");
  c.Clear();
  gSystem->Exec("mv fpix_hit_rate.png Hit_Rate_Plots");
  gSystem->Exec("mv fpix_hit_rate.pdf Hit_Rate_Plots");
  gSystem->Exec("mv fpix_hit_rate.C Hit_Rate_Plots");
  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  FPIX Disk1 hit rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_hit_rate_FPIX_disk1(
      runNumber_VecD, hit_rate_FPIX_disk1_VecD, runNumber_err_VecD, hit_rate_FPIX_disk1_err_VecD);
  gr_hit_rate_FPIX_disk1.GetXaxis()->SetTitle("Run Number");
  gr_hit_rate_FPIX_disk1.GetXaxis()->SetLabelSize(0.04);
  gr_hit_rate_FPIX_disk1.GetXaxis()->SetNoExponent();
  gr_hit_rate_FPIX_disk1.GetXaxis()->SetNdivisions(5);
  gr_hit_rate_FPIX_disk1.GetYaxis()->SetTitle("Hit Rate (Hz)");
  gr_hit_rate_FPIX_disk1.GetXaxis()->SetTitleSize(0.05);
  gr_hit_rate_FPIX_disk1.GetYaxis()->SetLabelSize(0.05);
  gr_hit_rate_FPIX_disk1.GetYaxis()->SetTitleSize(0.05);
  gr_hit_rate_FPIX_disk1.SetMarkerStyle(20);
  gr_hit_rate_FPIX_disk1.SetMarkerSize(1.4);
  gr_hit_rate_FPIX_disk1.SetMarkerColor(kRed);
  gr_hit_rate_FPIX_disk1.SetTitle("");
  gr_hit_rate_FPIX_disk1.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "FPIX disk 1");
  c.SetGrid();
  c.SaveAs("fpix_hit_rate_disk1.png");
  c.SaveAs("fpix_hit_rate_disk1.pdf");
  c.SaveAs("fpix_hit_rate_disk1.C");
  c.Clear();
  gSystem->Exec("mv fpix_hit_rate_disk1.png Hit_Rate_Plots");
  gSystem->Exec("mv fpix_hit_rate_disk1.pdf Hit_Rate_Plots");
  gSystem->Exec("mv fpix_hit_rate_disk1.C Hit_Rate_Plots");
  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  FPIX Disk2 hit rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_hit_rate_FPIX_disk2(
      runNumber_VecD, hit_rate_FPIX_disk2_VecD, runNumber_err_VecD, hit_rate_FPIX_disk2_err_VecD);
  gr_hit_rate_FPIX_disk2.GetXaxis()->SetTitle("Run Number");
  gr_hit_rate_FPIX_disk2.GetXaxis()->SetLabelSize(0.04);
  gr_hit_rate_FPIX_disk2.GetXaxis()->SetNoExponent();
  gr_hit_rate_FPIX_disk2.GetXaxis()->SetNdivisions(5);
  gr_hit_rate_FPIX_disk2.GetYaxis()->SetTitle("Hit Rate (Hz)");
  gr_hit_rate_FPIX_disk2.GetXaxis()->SetTitleSize(0.05);
  gr_hit_rate_FPIX_disk2.GetYaxis()->SetLabelSize(0.05);
  gr_hit_rate_FPIX_disk2.GetYaxis()->SetTitleSize(0.05);
  gr_hit_rate_FPIX_disk2.SetMarkerStyle(20);
  gr_hit_rate_FPIX_disk2.SetMarkerSize(1.4);
  gr_hit_rate_FPIX_disk2.SetMarkerColor(kRed);
  gr_hit_rate_FPIX_disk2.SetTitle("");
  gr_hit_rate_FPIX_disk2.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "FPIX disk 2");
  c.SetGrid();
  c.SaveAs("fpix_hit_rate_disk2.png");
  c.SaveAs("fpix_hit_rate_disk2.pdf");
  c.SaveAs("fpix_hit_rate_disk2.C");
  c.Clear();
  gSystem->Exec("mv fpix_hit_rate_disk2.png Hit_Rate_Plots");
  gSystem->Exec("mv fpix_hit_rate_disk2.pdf Hit_Rate_Plots");
  gSystem->Exec("mv fpix_hit_rate_disk2.C Hit_Rate_Plots");
  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  FPIX Disk3 hit rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_hit_rate_FPIX_disk3(
      runNumber_VecD, hit_rate_FPIX_disk3_VecD, runNumber_err_VecD, hit_rate_FPIX_disk3_err_VecD);
  gr_hit_rate_FPIX_disk3.GetXaxis()->SetTitle("Run Number");
  gr_hit_rate_FPIX_disk3.GetXaxis()->SetLabelSize(0.04);
  gr_hit_rate_FPIX_disk3.GetXaxis()->SetNoExponent();
  gr_hit_rate_FPIX_disk3.GetXaxis()->SetNdivisions(5);
  gr_hit_rate_FPIX_disk3.GetYaxis()->SetTitle("Hit Rate (Hz)");
  gr_hit_rate_FPIX_disk3.GetXaxis()->SetTitleSize(0.05);
  gr_hit_rate_FPIX_disk3.GetYaxis()->SetLabelSize(0.05);
  gr_hit_rate_FPIX_disk3.GetYaxis()->SetTitleSize(0.05);
  gr_hit_rate_FPIX_disk3.SetMarkerStyle(20);
  gr_hit_rate_FPIX_disk3.SetMarkerSize(1.4);
  gr_hit_rate_FPIX_disk3.SetMarkerColor(kRed);
  gr_hit_rate_FPIX_disk3.SetTitle("");
  gr_hit_rate_FPIX_disk3.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "FPIX disk 3");
  c.SetGrid();
  c.SaveAs("fpix_hit_rate_disk3.png");
  c.SaveAs("fpix_hit_rate_disk3.pdf");
  c.SaveAs("fpix_hit_rate_disk3.C");
  c.Clear();
  gSystem->Exec("mv fpix_hit_rate_disk3.png Hit_Rate_Plots");
  gSystem->Exec("mv fpix_hit_rate_disk3.pdf Hit_Rate_Plots");
  gSystem->Exec("mv fpix_hit_rate_disk3.C Hit_Rate_Plots");
  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  FPIX Disk1+ hit rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_hit_rate_FPIX_disk1_plus(
      runNumber_VecD, hit_rate_FPIX_disk1_plus_VecD, runNumber_err_VecD, hit_rate_FPIX_disk1_plus_err_VecD);
  gr_hit_rate_FPIX_disk1_plus.GetXaxis()->SetTitle("Run Number");
  gr_hit_rate_FPIX_disk1_plus.GetXaxis()->SetLabelSize(0.04);
  gr_hit_rate_FPIX_disk1_plus.GetXaxis()->SetNoExponent();
  gr_hit_rate_FPIX_disk1_plus.GetXaxis()->SetNdivisions(5);
  gr_hit_rate_FPIX_disk1_plus.GetYaxis()->SetTitle("Hit Rate (Hz)");
  gr_hit_rate_FPIX_disk1_plus.GetXaxis()->SetTitleSize(0.05);
  gr_hit_rate_FPIX_disk1_plus.GetYaxis()->SetLabelSize(0.05);
  gr_hit_rate_FPIX_disk1_plus.GetYaxis()->SetTitleSize(0.05);
  gr_hit_rate_FPIX_disk1_plus.SetMarkerStyle(20);
  gr_hit_rate_FPIX_disk1_plus.SetMarkerSize(1.4);
  gr_hit_rate_FPIX_disk1_plus.SetMarkerColor(kRed);
  gr_hit_rate_FPIX_disk1_plus.SetTitle("");
  gr_hit_rate_FPIX_disk1_plus.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "FPIX disk 1+");
  c.SetGrid();
  c.SaveAs("fpix_hit_rate_disk1_plus.png");
  c.SaveAs("fpix_hit_rate_disk1_plus.C");
  c.SaveAs("fpix_hit_rate_disk1_plus.pdf");
  c.Clear();
  gSystem->Exec("mv fpix_hit_rate_disk1_plus.png Hit_Rate_Plots");
  gSystem->Exec("mv fpix_hit_rate_disk1_plus.pdf Hit_Rate_Plots");
  gSystem->Exec("mv fpix_hit_rate_disk1_plus.C Hit_Rate_Plots");
  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  FPIX Disk1- hit rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_hit_rate_FPIX_disk1_minus(
      runNumber_VecD, hit_rate_FPIX_disk1_minus_VecD, runNumber_err_VecD, hit_rate_FPIX_disk1_minus_err_VecD);
  gr_hit_rate_FPIX_disk1_minus.GetXaxis()->SetTitle("Run Number");
  gr_hit_rate_FPIX_disk1_minus.GetXaxis()->SetLabelSize(0.04);
  gr_hit_rate_FPIX_disk1_minus.GetXaxis()->SetNoExponent();
  gr_hit_rate_FPIX_disk1_minus.GetXaxis()->SetNdivisions(5);
  gr_hit_rate_FPIX_disk1_minus.GetYaxis()->SetTitle("Hit Rate (Hz)");
  gr_hit_rate_FPIX_disk1_minus.GetXaxis()->SetTitleSize(0.05);
  gr_hit_rate_FPIX_disk1_minus.GetYaxis()->SetLabelSize(0.05);
  gr_hit_rate_FPIX_disk1_minus.GetYaxis()->SetTitleSize(0.05);
  gr_hit_rate_FPIX_disk1_minus.SetMarkerStyle(20);
  gr_hit_rate_FPIX_disk1_minus.SetMarkerSize(1.4);
  gr_hit_rate_FPIX_disk1_minus.SetMarkerColor(kRed);
  gr_hit_rate_FPIX_disk1_minus.SetTitle("");
  gr_hit_rate_FPIX_disk1_minus.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "FPIX disk 1-");
  c.SetGrid();
  c.SaveAs("fpix_hit_rate_disk1_minus.png");
  c.SaveAs("fpix_hit_rate_disk1_minus.pdf");
  c.SaveAs("fpix_hit_rate_disk1_minus.C");
  c.Clear();
  gSystem->Exec("mv fpix_hit_rate_disk1_minus.png Hit_Rate_Plots");
  gSystem->Exec("mv fpix_hit_rate_disk1_minus.pdf Hit_Rate_Plots");
  gSystem->Exec("mv fpix_hit_rate_disk1_minus.C Hit_Rate_Plots");
  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  FPIX Disk2+ hit rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_hit_rate_FPIX_disk2_plus(
      runNumber_VecD, hit_rate_FPIX_disk2_plus_VecD, runNumber_err_VecD, hit_rate_FPIX_disk2_plus_err_VecD);
  gr_hit_rate_FPIX_disk2_plus.GetXaxis()->SetTitle("Run Number");
  gr_hit_rate_FPIX_disk2_plus.GetXaxis()->SetLabelSize(0.04);
  gr_hit_rate_FPIX_disk2_plus.GetXaxis()->SetNoExponent();
  gr_hit_rate_FPIX_disk2_plus.GetXaxis()->SetNdivisions(5);
  gr_hit_rate_FPIX_disk2_plus.GetYaxis()->SetTitle("Hit Rate (Hz)");
  gr_hit_rate_FPIX_disk2_plus.GetXaxis()->SetTitleSize(0.05);
  gr_hit_rate_FPIX_disk2_plus.GetYaxis()->SetLabelSize(0.05);
  gr_hit_rate_FPIX_disk2_plus.GetYaxis()->SetTitleSize(0.05);
  gr_hit_rate_FPIX_disk2_plus.SetMarkerStyle(20);
  gr_hit_rate_FPIX_disk2_plus.SetMarkerSize(1.4);
  gr_hit_rate_FPIX_disk2_plus.SetMarkerColor(kRed);
  gr_hit_rate_FPIX_disk2_plus.SetTitle("");
  gr_hit_rate_FPIX_disk2_plus.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "FPIX disk 2+");
  c.SetGrid();
  c.SaveAs("fpix_hit_rate_disk2_plus.png");
  c.SaveAs("fpix_hit_rate_disk2_plus.pdf");
  c.SaveAs("fpix_hit_rate_disk2_plus.C");
  c.Clear();
  gSystem->Exec("mv fpix_hit_rate_disk2_plus.png Hit_Rate_Plots");
  gSystem->Exec("mv fpix_hit_rate_disk2_plus.pdf Hit_Rate_Plots");
  gSystem->Exec("mv fpix_hit_rate_disk2_plus.C Hit_Rate_Plots");
  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  FPIX Disk2- hit rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_hit_rate_FPIX_disk2_minus(
      runNumber_VecD, hit_rate_FPIX_disk2_minus_VecD, runNumber_err_VecD, hit_rate_FPIX_disk2_minus_err_VecD);
  gr_hit_rate_FPIX_disk2_minus.GetXaxis()->SetTitle("Run Number");
  gr_hit_rate_FPIX_disk2_minus.GetXaxis()->SetLabelSize(0.04);
  gr_hit_rate_FPIX_disk2_minus.GetXaxis()->SetNoExponent();
  gr_hit_rate_FPIX_disk2_minus.GetXaxis()->SetNdivisions(5);
  gr_hit_rate_FPIX_disk2_minus.GetYaxis()->SetTitle("Hit Rate (Hz)");
  gr_hit_rate_FPIX_disk2_minus.GetXaxis()->SetTitleSize(0.05);
  gr_hit_rate_FPIX_disk2_minus.GetYaxis()->SetLabelSize(0.05);
  gr_hit_rate_FPIX_disk2_minus.GetYaxis()->SetTitleSize(0.05);
  gr_hit_rate_FPIX_disk2_minus.SetMarkerStyle(20);
  gr_hit_rate_FPIX_disk2_minus.SetMarkerSize(1.4);
  gr_hit_rate_FPIX_disk2_minus.SetMarkerColor(2);
  gr_hit_rate_FPIX_disk2_minus.SetTitle("");
  gr_hit_rate_FPIX_disk2_minus.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "FPIX disk 2-");
  c.SetGrid();
  c.SaveAs("fpix_hit_rate_disk2_minus.png");
  c.SaveAs("fpix_hit_rate_disk2_minus.pdf");
  c.SaveAs("fpix_hit_rate_disk2_minus.C");
  c.Clear();
  gSystem->Exec("mv fpix_hit_rate_disk2_minus.png Hit_Rate_Plots");
  gSystem->Exec("mv fpix_hit_rate_disk2_minus.pdf Hit_Rate_Plots");
  gSystem->Exec("mv fpix_hit_rate_disk2_minus.C Hit_Rate_Plots");
  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  FPIX Disk3+ hit rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_hit_rate_FPIX_disk3_plus(
      runNumber_VecD, hit_rate_FPIX_disk3_plus_VecD, runNumber_err_VecD, hit_rate_FPIX_disk3_plus_err_VecD);
  gr_hit_rate_FPIX_disk3_plus.GetXaxis()->SetTitle("Run Number");
  gr_hit_rate_FPIX_disk3_plus.GetXaxis()->SetLabelSize(0.04);
  gr_hit_rate_FPIX_disk3_plus.GetXaxis()->SetNoExponent();
  gr_hit_rate_FPIX_disk3_plus.GetXaxis()->SetNdivisions(5);
  gr_hit_rate_FPIX_disk3_plus.GetYaxis()->SetTitle("Hit Rate (Hz)");
  gr_hit_rate_FPIX_disk3_plus.GetXaxis()->SetTitleSize(0.05);
  gr_hit_rate_FPIX_disk3_plus.GetYaxis()->SetLabelSize(0.05);
  gr_hit_rate_FPIX_disk3_plus.GetYaxis()->SetTitleSize(0.05);
  gr_hit_rate_FPIX_disk3_plus.SetMarkerStyle(20);
  gr_hit_rate_FPIX_disk3_plus.SetMarkerSize(1.4);
  gr_hit_rate_FPIX_disk3_plus.SetMarkerColor(kRed);
  gr_hit_rate_FPIX_disk3_plus.SetTitle("");
  gr_hit_rate_FPIX_disk3_plus.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "FPIX disk 3+");
  c.SetGrid();
  c.SaveAs("fpix_hit_rate_disk3_plus.png");
  c.SaveAs("fpix_hit_rate_disk3_plus.pdf");
  c.SaveAs("fpix_hit_rate_disk3_plus.C");
  c.Clear();
  gSystem->Exec("mv fpix_hit_rate_disk3_plus.png Hit_Rate_Plots");
  gSystem->Exec("mv fpix_hit_rate_disk3_plus.pdf Hit_Rate_Plots");
  gSystem->Exec("mv fpix_hit_rate_disk3_plus.C Hit_Rate_Plots");
  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  FPIX Disk3- hit rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_hit_rate_FPIX_disk3_minus(
      runNumber_VecD, hit_rate_FPIX_disk3_minus_VecD, runNumber_err_VecD, hit_rate_FPIX_disk3_minus_err_VecD);
  gr_hit_rate_FPIX_disk3_minus.GetXaxis()->SetTitle("Run Number");
  gr_hit_rate_FPIX_disk3_minus.GetXaxis()->SetLabelSize(0.04);
  gr_hit_rate_FPIX_disk3_minus.GetXaxis()->SetNoExponent();
  gr_hit_rate_FPIX_disk3_minus.GetXaxis()->SetNdivisions(5);
  gr_hit_rate_FPIX_disk3_minus.GetYaxis()->SetTitle("Hit Rate (Hz)");
  gr_hit_rate_FPIX_disk3_minus.GetXaxis()->SetTitleSize(0.05);
  gr_hit_rate_FPIX_disk3_minus.GetYaxis()->SetLabelSize(0.05);
  gr_hit_rate_FPIX_disk3_minus.GetYaxis()->SetTitleSize(0.05);
  gr_hit_rate_FPIX_disk3_minus.SetMarkerStyle(20);
  gr_hit_rate_FPIX_disk3_minus.SetMarkerSize(1.4);
  gr_hit_rate_FPIX_disk3_minus.SetMarkerColor(2);
  gr_hit_rate_FPIX_disk3_minus.SetTitle("");
  gr_hit_rate_FPIX_disk3_minus.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "FPIX disk 3-");
  c.SetGrid();
  c.SaveAs("fpix_hit_rate_disk3_minus.png");
  c.SaveAs("fpix_hit_rate_disk3_minus.pdf");
  c.SaveAs("fpix_hit_rate_disk3_minus.C");
  c.Clear();
  gSystem->Exec("mv fpix_hit_rate_disk3_minus.png Hit_Rate_Plots");
  gSystem->Exec("mv fpix_hit_rate_disk3_minus.pdf Hit_Rate_Plots");
  gSystem->Exec("mv fpix_hit_rate_disk3_minus.C Hit_Rate_Plots");
  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  BPIX hit rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_hit_rate_BPIX(runNumber_VecD, hit_rate_BPIX_VecD, runNumber_err_VecD, hit_rate_BPIX_err_VecD);
  gr_hit_rate_BPIX.GetXaxis()->SetTitle("Run Number");
  gr_hit_rate_BPIX.GetXaxis()->SetLabelSize(0.04);
  gr_hit_rate_BPIX.GetXaxis()->SetNoExponent();
  gr_hit_rate_BPIX.GetXaxis()->SetNdivisions(5);
  gr_hit_rate_BPIX.GetYaxis()->SetTitle("Hit Rate (Hz)");
  gr_hit_rate_BPIX.GetXaxis()->SetTitleSize(0.05);
  gr_hit_rate_BPIX.GetYaxis()->SetLabelSize(0.05);
  gr_hit_rate_BPIX.GetYaxis()->SetTitleSize(0.05);
  gr_hit_rate_BPIX.SetMarkerStyle(20);
  gr_hit_rate_BPIX.SetMarkerSize(1.4);
  gr_hit_rate_BPIX.SetMarkerColor(2);
  gr_hit_rate_BPIX.SetTitle("");
  gr_hit_rate_BPIX.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "BPIX");
  c.SetGrid();
  c.SaveAs("bpix_hit_rate.png");
  c.SaveAs("bpix_hit_rate.pdf");
  c.SaveAs("bpix_hit_rate.C");
  c.Clear();
  gSystem->Exec("mv bpix_hit_rate.png Hit_Rate_Plots");
  gSystem->Exec("mv bpix_hit_rate.pdf Hit_Rate_Plots");
  gSystem->Exec("mv bpix_hit_rate.C Hit_Rate_Plots");

  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  BPIX layer 1 hit rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_hit_rate_BPIX_layer1(
      runNumber_VecD, hit_rate_BPIX_layer1_VecD, runNumber_err_VecD, hit_rate_BPIX_layer1_err_VecD);
  gr_hit_rate_BPIX_layer1.GetXaxis()->SetTitle("Run Number");
  gr_hit_rate_BPIX_layer1.GetXaxis()->SetLabelSize(0.04);
  gr_hit_rate_BPIX_layer1.GetXaxis()->SetNoExponent();
  gr_hit_rate_BPIX_layer1.GetXaxis()->SetNdivisions(5);
  gr_hit_rate_BPIX_layer1.GetYaxis()->SetTitle("Hit Rate (Hz)");
  gr_hit_rate_BPIX_layer1.GetXaxis()->SetTitleSize(0.05);
  gr_hit_rate_BPIX_layer1.GetYaxis()->SetLabelSize(0.05);
  gr_hit_rate_BPIX_layer1.GetYaxis()->SetTitleSize(0.05);
  gr_hit_rate_BPIX_layer1.SetMarkerStyle(20);
  gr_hit_rate_BPIX_layer1.SetMarkerSize(1.4);
  gr_hit_rate_BPIX_layer1.SetMarkerColor(2);
  gr_hit_rate_BPIX_layer1.SetTitle("");
  gr_hit_rate_BPIX_layer1.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "BPIX layer 1");
  c.SetGrid();
  c.SaveAs("bpix_layer1_hit_rate.png");
  c.SaveAs("bpix_layer1_hit_rate.pdf");
  c.SaveAs("bpix_layer1_hit_rate.C");
  c.Clear();
  gSystem->Exec("mv bpix_layer1_hit_rate.png Hit_Rate_Plots");
  gSystem->Exec("mv bpix_layer1_hit_rate.pdf Hit_Rate_Plots");
  gSystem->Exec("mv bpix_layer1_hit_rate.C Hit_Rate_Plots");

  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  BPIX layer 2 hit rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_hit_rate_BPIX_layer2(
      runNumber_VecD, hit_rate_BPIX_layer2_VecD, runNumber_err_VecD, hit_rate_BPIX_layer2_err_VecD);
  gr_hit_rate_BPIX_layer2.GetXaxis()->SetTitle("Run Number");
  gr_hit_rate_BPIX_layer2.GetXaxis()->SetLabelSize(0.04);
  gr_hit_rate_BPIX_layer2.GetXaxis()->SetNoExponent();
  gr_hit_rate_BPIX_layer2.GetXaxis()->SetNdivisions(5);
  gr_hit_rate_BPIX_layer2.GetYaxis()->SetTitle("Hit Rate (Hz)");
  gr_hit_rate_BPIX_layer2.GetXaxis()->SetTitleSize(0.05);
  gr_hit_rate_BPIX_layer2.GetYaxis()->SetLabelSize(0.05);
  gr_hit_rate_BPIX_layer2.GetYaxis()->SetTitleSize(0.05);
  gr_hit_rate_BPIX_layer2.SetMarkerStyle(20);
  gr_hit_rate_BPIX_layer2.SetMarkerSize(1.4);
  gr_hit_rate_BPIX_layer2.SetMarkerColor(2);
  gr_hit_rate_BPIX_layer2.SetTitle("");
  gr_hit_rate_BPIX_layer2.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "BPIX layer 2");
  c.SetGrid();
  c.SaveAs("bpix_layer2_hit_rate.png");
  c.SaveAs("bpix_layer2_hit_rate.pdf");
  c.SaveAs("bpix_layer2_hit_rate.C");
  c.Clear();
  gSystem->Exec("mv bpix_layer2_hit_rate.png Hit_Rate_Plots");
  gSystem->Exec("mv bpix_layer2_hit_rate.pdf Hit_Rate_Plots");
  gSystem->Exec("mv bpix_layer2_hit_rate.C Hit_Rate_Plots");

  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  BPIX layer 3 hit rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_hit_rate_BPIX_layer3(
      runNumber_VecD, hit_rate_BPIX_layer3_VecD, runNumber_err_VecD, hit_rate_BPIX_layer3_err_VecD);
  gr_hit_rate_BPIX_layer3.GetXaxis()->SetTitle("Run Number");
  gr_hit_rate_BPIX_layer3.GetXaxis()->SetLabelSize(0.04);
  gr_hit_rate_BPIX_layer3.GetXaxis()->SetNoExponent();
  gr_hit_rate_BPIX_layer3.GetXaxis()->SetNdivisions(5);
  gr_hit_rate_BPIX_layer3.GetYaxis()->SetTitle("Hit Rate (Hz)");
  gr_hit_rate_BPIX_layer3.GetXaxis()->SetTitleSize(0.05);
  gr_hit_rate_BPIX_layer3.GetYaxis()->SetLabelSize(0.05);
  gr_hit_rate_BPIX_layer3.GetYaxis()->SetTitleSize(0.05);
  gr_hit_rate_BPIX_layer3.SetMarkerStyle(20);
  gr_hit_rate_BPIX_layer3.SetMarkerSize(1.4);
  gr_hit_rate_BPIX_layer3.SetMarkerColor(2);
  gr_hit_rate_BPIX_layer3.SetTitle("");
  gr_hit_rate_BPIX_layer3.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "BPIX layer 3");
  c.SetGrid();
  c.SaveAs("bpix_layer3_hit_rate.png");
  c.SaveAs("bpix_layer3_hit_rate.pdf");
  c.SaveAs("bpix_layer3_hit_rate.C");
  c.Clear();
  gSystem->Exec("mv bpix_layer3_hit_rate.png Hit_Rate_Plots");
  gSystem->Exec("mv bpix_layer3_hit_rate.pdf Hit_Rate_Plots");
  gSystem->Exec("mv bpix_layer3_hit_rate.C Hit_Rate_Plots");

  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  BPIX layer 4 hit rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_hit_rate_BPIX_layer4(
      runNumber_VecD, hit_rate_BPIX_layer4_VecD, runNumber_err_VecD, hit_rate_BPIX_layer4_err_VecD);
  gr_hit_rate_BPIX_layer4.GetXaxis()->SetTitle("Run Number");
  gr_hit_rate_BPIX_layer4.GetXaxis()->SetLabelSize(0.04);
  gr_hit_rate_BPIX_layer4.GetXaxis()->SetNoExponent();
  gr_hit_rate_BPIX_layer4.GetXaxis()->SetNdivisions(5);
  gr_hit_rate_BPIX_layer4.GetYaxis()->SetTitle("Hit Rate (Hz)");
  gr_hit_rate_BPIX_layer4.GetXaxis()->SetTitleSize(0.05);
  gr_hit_rate_BPIX_layer4.GetYaxis()->SetLabelSize(0.05);
  gr_hit_rate_BPIX_layer4.GetYaxis()->SetTitleSize(0.05);
  gr_hit_rate_BPIX_layer4.SetMarkerStyle(20);
  gr_hit_rate_BPIX_layer4.SetMarkerSize(1.4);
  gr_hit_rate_BPIX_layer4.SetMarkerColor(2);
  gr_hit_rate_BPIX_layer4.SetTitle("");
  gr_hit_rate_BPIX_layer4.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "BPIX layer 4");
  c.SetGrid();
  c.SaveAs("bpix_layer4_hit_rate.png");
  c.SaveAs("bpix_layer4_hit_rate.pdf");
  c.SaveAs("bpix_layer4_hit_rate.C");
  c.Clear();
  gSystem->Exec("mv bpix_layer4_hit_rate.png Hit_Rate_Plots");
  gSystem->Exec("mv bpix_layer4_hit_rate.pdf Hit_Rate_Plots");
  gSystem->Exec("mv bpix_layer4_hit_rate.C Hit_Rate_Plots");

  //-----------------------------------------------------------------------------------------------

  c.Close();

  //-----------------------------------------------------------------------------------------------
  //					Weighted Mean calculation - Hit Rate
  //-----------------------------------------------------------------------------------------------

  double total_weight = 0;
  double weighted_mean_hit_rate;
  double weighted_mean_hit_rate_PIX;
  double weighted_mean_hit_rate_BPIX;
  double weighted_mean_hit_rate_BPIX_layer1;
  double weighted_mean_hit_rate_BPIX_layer2;
  double weighted_mean_hit_rate_BPIX_layer3;
  double weighted_mean_hit_rate_BPIX_layer4;
  double weighted_mean_hit_rate_FPIX;
  double weighted_mean_hit_rate_FPIX_disk1;
  double weighted_mean_hit_rate_FPIX_disk2;
  double weighted_mean_hit_rate_FPIX_disk3;
  double weighted_mean_hit_rate_FPIX_disk1_plus;
  double weighted_mean_hit_rate_FPIX_disk2_plus;
  double weighted_mean_hit_rate_FPIX_disk3_plus;
  double weighted_mean_hit_rate_FPIX_disk1_minus;
  double weighted_mean_hit_rate_FPIX_disk2_minus;
  double weighted_mean_hit_rate_FPIX_disk3_minus;

  // ---------- Weighted Hit Rate Calculation ------------//
  for (int k = 0; k < j; k++)  // Loop over all runs(without 0 or negative runtime) to get weight per run
  {
    weight.push_back(hits.at(k) / nTotalHits);
  }

  for (int a = 0; a < j;
       a++)  // Loop over runs(without 0 or negative runtime) to calculate the weighted mean hit rate per PIXEL Layer
  {
    weighted_mean_hit_rate_PIX += hit_rate_PIX.at(a) * weight.at(a);
    weighted_mean_hit_rate_BPIX += hit_rate_BPIX.at(a) * weight.at(a);
    weighted_mean_hit_rate_BPIX_layer1 += hit_rate_BPIX_layer1.at(a) * weight.at(a);
    weighted_mean_hit_rate_BPIX_layer2 += hit_rate_BPIX_layer2.at(a) * weight.at(a);
    weighted_mean_hit_rate_BPIX_layer3 += hit_rate_BPIX_layer3.at(a) * weight.at(a);
    weighted_mean_hit_rate_BPIX_layer4 += hit_rate_BPIX_layer4.at(a) * weight.at(a);
    weighted_mean_hit_rate_FPIX += hit_rate_FPIX.at(a) * weight.at(a);
    weighted_mean_hit_rate_FPIX_disk1 += hit_rate_FPIX_disk1.at(a) * weight.at(a);
    weighted_mean_hit_rate_FPIX_disk2 += hit_rate_FPIX_disk2.at(a) * weight.at(a);
    weighted_mean_hit_rate_FPIX_disk3 += hit_rate_FPIX_disk3.at(a) * weight.at(a);
    weighted_mean_hit_rate_FPIX_disk1_plus += hit_rate_FPIX_disk1_plus.at(a) * weight.at(a);
    weighted_mean_hit_rate_FPIX_disk2_plus += hit_rate_FPIX_disk2_plus.at(a) * weight.at(a);
    weighted_mean_hit_rate_FPIX_disk3_plus += hit_rate_FPIX_disk3_plus.at(a) * weight.at(a);
    weighted_mean_hit_rate_FPIX_disk1_minus += hit_rate_FPIX_disk1_minus.at(a) * weight.at(a);
    weighted_mean_hit_rate_FPIX_disk2_minus += hit_rate_FPIX_disk2_minus.at(a) * weight.at(a);
    weighted_mean_hit_rate_FPIX_disk3_minus += hit_rate_FPIX_disk3_minus.at(a) * weight.at(a);
  }

  //-----------------------------------------------------------------------------------------------
  //			Summary Plot for hit rate in BPIX
  //-----------------------------------------------------------------------------------------------
  TCanvas *canvas = new TCanvas("canvas", "canvas", 324, 57, 953, 866);
  canvas->SetFillColor(0);
  canvas->SetBorderMode(0);
  canvas->SetBorderSize(2);
  canvas->SetGridx();
  canvas->SetGridy();
  canvas->SetTickx(1);
  canvas->SetTicky(1);
  canvas->SetLeftMargin(0.1608833);
  canvas->SetRightMargin(0.05152471);

  canvas->SetFrameLineWidth(3);
  canvas->SetFrameBorderMode(0);
  canvas->SetFrameLineWidth(3);
  canvas->SetFrameBorderMode(0);

  TLatex tex;
  tex.DrawLatexNDC(0.4, 0.8, "Rate Summary");
  tex.SetLineWidth(2);
  tex.SetTextFont(62);
  tex.Draw();

  TH1F hb_BPIX("hb_BPIX", "Rate Summary BPIX", 6, 0, 6);
  hb_BPIX.SetFillColor(6);
  hb_BPIX.SetBarWidth(0.6);
  hb_BPIX.SetBarOffset(0.25);
  hb_BPIX.SetStats(0);
  hb_BPIX.GetXaxis()->SetLabelFont(42);
  hb_BPIX.GetXaxis()->SetLabelOffset(0.012);
  hb_BPIX.GetXaxis()->SetLabelSize(0.06);
  hb_BPIX.GetXaxis()->SetTitleSize(0.05);
  hb_BPIX.GetXaxis()->SetTitleFont(42);
  hb_BPIX.GetYaxis()->SetTitle("Average Hit Rate (Hz)");
  hb_BPIX.GetYaxis()->SetLabelFont(42);
  hb_BPIX.GetYaxis()->SetLabelSize(0.05);
  hb_BPIX.GetYaxis()->SetTitleSize(0.05);
  hb_BPIX.GetYaxis()->SetTitleOffset(0);

  gStyle->SetPaintTextFormat("1.3f");

  tex.DrawLatexNDC(0.4, 0.8, "Rate Summary BPIX");
  tex.SetLineWidth(2);
  tex.SetTextFont(62);
  tex.Draw();

  Bar_Ytitle_BPIX[0] = weighted_mean_hit_rate_PIX;
  Bar_Ytitle_BPIX[1] = weighted_mean_hit_rate_BPIX;
  Bar_Ytitle_BPIX[2] = weighted_mean_hit_rate_BPIX_layer1;
  Bar_Ytitle_BPIX[3] = weighted_mean_hit_rate_BPIX_layer2;
  Bar_Ytitle_BPIX[4] = weighted_mean_hit_rate_BPIX_layer3;
  Bar_Ytitle_BPIX[5] = weighted_mean_hit_rate_BPIX_layer4;

  for (int i = 1; i <= 6; i++) {
    hb_BPIX.SetBinContent(i, Bar_Ytitle_BPIX[i - 1]);
    hb_BPIX.GetXaxis()->SetBinLabel(i, Bar_Xtitle_BPIX[i - 1].c_str());
  }

  TString summary_chart_title;
  TString Format[3] = {"png", "pdf", "C"};

  hb_BPIX.Draw("bTEXT");
  summary_chart_title = "SummaryChart_HitRate_BPIX";

  for (int i = 0; i < 3; i++) {
    TString filename = summary_chart_title + "." + Format[i];
    canvas->SaveAs(filename.Data());
    TString mv_cmd = "mv " + filename + " Hit_Rate_Plots";
    gSystem->Exec(mv_cmd.Data());
  }
  canvas->Clear();

  //-----------------------------------------------------------------------------------------------
  //			Summary Plot for hit rate in FPIX
  //-----------------------------------------------------------------------------------------------

  TH1F hb_FPIX("hb_FPIX", "Rate Summary FPIX", 11, 0, 11);
  hb_FPIX.SetFillColor(6);
  hb_FPIX.SetBarWidth(0.6);
  hb_FPIX.SetBarOffset(0.25);
  hb_FPIX.SetStats(0);
  hb_FPIX.GetXaxis()->SetLabelFont(42);
  hb_FPIX.GetXaxis()->SetLabelOffset(0.012);
  hb_FPIX.GetXaxis()->SetLabelSize(0.04);
  hb_FPIX.GetXaxis()->SetTitleSize(0.05);
  hb_FPIX.GetXaxis()->SetTitleFont(42);
  hb_FPIX.GetYaxis()->SetTitle("Average Hit Rate (Hz)");
  hb_FPIX.GetYaxis()->SetLabelFont(42);
  hb_FPIX.GetYaxis()->SetLabelSize(0.05);
  hb_FPIX.GetYaxis()->SetTitleSize(0.05);
  hb_FPIX.GetYaxis()->SetTitleOffset(0);

  gStyle->SetPaintTextFormat("1.3f");

  Bar_Ytitle_FPIX[0] = weighted_mean_hit_rate_PIX;
  Bar_Ytitle_FPIX[1] = weighted_mean_hit_rate_FPIX;
  Bar_Ytitle_FPIX[2] = weighted_mean_hit_rate_FPIX_disk1;
  Bar_Ytitle_FPIX[3] = weighted_mean_hit_rate_FPIX_disk1_plus;
  Bar_Ytitle_FPIX[4] = weighted_mean_hit_rate_FPIX_disk1_minus;
  Bar_Ytitle_FPIX[5] = weighted_mean_hit_rate_FPIX_disk2;
  Bar_Ytitle_FPIX[6] = weighted_mean_hit_rate_FPIX_disk2_plus;
  Bar_Ytitle_FPIX[7] = weighted_mean_hit_rate_FPIX_disk2_minus;
  Bar_Ytitle_FPIX[8] = weighted_mean_hit_rate_FPIX_disk3;
  Bar_Ytitle_FPIX[9] = weighted_mean_hit_rate_FPIX_disk3_plus;
  Bar_Ytitle_FPIX[10] = weighted_mean_hit_rate_FPIX_disk3_minus;

  for (int i = 1; i <= 11; i++) {
    hb_FPIX.SetBinContent(i, Bar_Ytitle_FPIX[i - 1]);
    hb_FPIX.GetXaxis()->SetBinLabel(i, Bar_Xtitle_FPIX[i - 1].c_str());
  }

  hb_FPIX.Draw("bTEXT");
  summary_chart_title = "SummaryChart_HitRate_FPIX";

  for (int i = 0; i < 3; i++) {
    TString filename = summary_chart_title + "." + Format[i];
    canvas->SaveAs(filename.Data());
    TString mv_cmd = "mv " + filename + " Hit_Rate_Plots";
    gSystem->Exec(mv_cmd.Data());
  }
  canvas->Clear();

  //-----------------------------------------------------------------------------------------------
  //			Summary Plot for hit rate in PIXEL
  //-----------------------------------------------------------------------------------------------

  TH1F hb_PIXEL("hb_PIXEL", "Rate Summary PIXEL", 10, 0, 10);
  hb_PIXEL.SetFillColor(6);
  hb_PIXEL.SetBarWidth(0.6);
  hb_PIXEL.SetBarOffset(0.25);
  hb_PIXEL.SetStats(0);
  hb_PIXEL.GetXaxis()->SetLabelFont(42);
  hb_PIXEL.GetXaxis()->SetLabelOffset(0.012);
  hb_PIXEL.GetXaxis()->SetLabelSize(0.05);
  hb_PIXEL.GetXaxis()->SetTitleSize(0.05);
  hb_PIXEL.GetXaxis()->SetTitleFont(42);
  hb_PIXEL.GetYaxis()->SetTitle("Average Hit Rate (Hz)");
  hb_PIXEL.GetYaxis()->SetLabelFont(42);
  hb_PIXEL.GetYaxis()->SetLabelSize(0.05);
  hb_PIXEL.GetYaxis()->SetTitleSize(0.05);
  hb_PIXEL.GetYaxis()->SetTitleOffset(0);

  gStyle->SetPaintTextFormat("1.3f");

  Bar_Ytitle_PIXEL[0] = weighted_mean_hit_rate_PIX;
  Bar_Ytitle_PIXEL[1] = weighted_mean_hit_rate_BPIX;
  Bar_Ytitle_PIXEL[2] = weighted_mean_hit_rate_BPIX_layer1;
  Bar_Ytitle_PIXEL[3] = weighted_mean_hit_rate_BPIX_layer2;
  Bar_Ytitle_PIXEL[4] = weighted_mean_hit_rate_BPIX_layer3;
  Bar_Ytitle_PIXEL[5] = weighted_mean_hit_rate_BPIX_layer4;
  Bar_Ytitle_PIXEL[6] = weighted_mean_hit_rate_FPIX;
  Bar_Ytitle_PIXEL[7] = weighted_mean_hit_rate_FPIX_disk1;
  Bar_Ytitle_PIXEL[8] = weighted_mean_hit_rate_FPIX_disk2;
  Bar_Ytitle_PIXEL[9] = weighted_mean_hit_rate_FPIX_disk3;

  for (int i = 1; i <= 10; i++) {
    hb_PIXEL.SetBinContent(i, Bar_Ytitle_PIXEL[i - 1]);
    hb_PIXEL.GetXaxis()->SetBinLabel(i, Bar_Xtitle_PIXEL[i - 1].c_str());
  }

  hb_PIXEL.Draw("bTEXT");
  summary_chart_title = "SummaryChart_HitRate_PIXEL";

  for (int i = 0; i < 3; i++) {
    TString filename = summary_chart_title + "." + Format[i];
    canvas->SaveAs(filename.Data());
    TString mv_cmd = "mv " + filename + " Hit_Rate_Plots";
    gSystem->Exec(mv_cmd.Data());
  }
  canvas->Clear();
  canvas->Close();

  std::cout << " Weighted mean PIX: " << weighted_mean_hit_rate_PIX << std::endl;
  std::cout << " Weighted mean BPIX: " << weighted_mean_hit_rate_BPIX << std::endl;
  std::cout << " Weighted mean BPIX layer 1: " << weighted_mean_hit_rate_BPIX_layer1 << std::endl;
  std::cout << " Weighted mean BPIX layer 2 : " << weighted_mean_hit_rate_BPIX_layer2 << std::endl;
  std::cout << " Weighted mean BPIX layer 3: " << weighted_mean_hit_rate_BPIX_layer3 << std::endl;
  std::cout << " Weighted mean BPIX layer 4: " << weighted_mean_hit_rate_BPIX_layer4 << std::endl;
  std::cout << " Weighted mean FPIX : " << weighted_mean_hit_rate_FPIX << std::endl;
  std::cout << " Weighted mean FPIX disk 1: " << weighted_mean_hit_rate_FPIX_disk1 << std::endl;
  std::cout << " Weighted mean FPIX disk 2: " << weighted_mean_hit_rate_FPIX_disk2 << std::endl;
  std::cout << " Weighted mean FPIX disk 3: " << weighted_mean_hit_rate_FPIX_disk3 << std::endl;
  std::cout << " Weighted mean FPIX disk 1+: " << weighted_mean_hit_rate_FPIX_disk1_plus << std::endl;
  std::cout << " Weighted mean FPIX disk 2+: " << weighted_mean_hit_rate_FPIX_disk2_plus << std::endl;
  std::cout << " Weighted mean FPIX disk 3+: " << weighted_mean_hit_rate_FPIX_disk3_plus << std::endl;
  std::cout << " Weighted mean FPIX disk 1-: " << weighted_mean_hit_rate_FPIX_disk1_minus << std::endl;
  std::cout << " Weighted mean FPIX disk 2-: " << weighted_mean_hit_rate_FPIX_disk2_minus << std::endl;
  std::cout << " Weighted mean FPIX disk 3-: " << weighted_mean_hit_rate_FPIX_disk3_minus << std::endl;
}