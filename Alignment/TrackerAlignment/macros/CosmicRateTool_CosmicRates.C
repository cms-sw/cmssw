#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TMath.h>
#include <cmath>
#include <TString.h>

void CosmicRateTool_CosmicRates(const char *fileName, unsigned int runLow = 0, unsigned int runUp = 0) {
  TString InputFile = Form("%s", fileName);
  TFile *file = new TFile(InputFile);

  bool IsFileExist;
  IsFileExist = file->IsZombie();
  if (IsFileExist) {
    cout << endl
         << "====================================================================================================="
         << endl;
    cout << fileName << " is not found. Check the file!" << endl;
    cout << "====================================================================================================="
         << endl
         << endl;
    exit(EXIT_FAILURE);
  }

  TTree *tree;
  tree = (TTree *)file->Get("cosmicRateAnalyzer/Run");

  FILE *pFile;
  pFile = fopen("tracksInfo.txt", "w");

  double run_time;
  unsigned int runnum;
  int number_of_events;
  int number_of_tracks;
  int number_of_tracks_PIX;
  int number_of_tracks_FPIX;
  int number_of_tracks_BPIX;
  int number_of_tracks_TID;
  int number_of_tracks_TIDM;
  int number_of_tracks_TIDP;
  int number_of_tracks_TIB;
  int number_of_tracks_TEC;
  int number_of_tracks_TECP;
  int number_of_tracks_TECM;
  int number_of_tracks_TOB;

  tree->SetBranchAddress("run_time", &run_time);
  tree->SetBranchAddress("runnum", &runnum);
  tree->SetBranchAddress("number_of_events", &number_of_events);
  tree->SetBranchAddress("number_of_tracks", &number_of_tracks);
  tree->SetBranchAddress("number_of_tracks_PIX", &number_of_tracks_PIX);
  tree->SetBranchAddress("number_of_tracks_FPIX", &number_of_tracks_FPIX);
  tree->SetBranchAddress("number_of_tracks_BPIX", &number_of_tracks_BPIX);
  tree->SetBranchAddress("number_of_tracks_TID", &number_of_tracks_TID);
  tree->SetBranchAddress("number_of_tracks_TIDM", &number_of_tracks_TIDM);
  tree->SetBranchAddress("number_of_tracks_TIDP", &number_of_tracks_TIDP);
  tree->SetBranchAddress("number_of_tracks_TIB", &number_of_tracks_TIB);
  tree->SetBranchAddress("number_of_tracks_TEC", &number_of_tracks_TEC);
  tree->SetBranchAddress("number_of_tracks_TECP", &number_of_tracks_TECP);
  tree->SetBranchAddress("number_of_tracks_TECM", &number_of_tracks_TECM);
  tree->SetBranchAddress("number_of_tracks_TOB", &number_of_tracks_TOB);

  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  //		Various Rates Declerations
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  vector<double> event_rate;
  vector<double> event_rate_err;
  vector<double> track_rate;
  vector<double> track_rate_err;
  vector<double> runNumber;
  vector<double> runNumber_err;
  vector<double> track_rate_PIX;
  vector<double> track_rate_PIX_err;
  vector<double> track_rate_FPIX;
  vector<double> track_rate_FPIX_err;
  vector<double> track_rate_BPIX;
  vector<double> track_rate_BPIX_err;
  vector<double> track_rate_TOB;
  vector<double> track_rate_TOB_err;
  vector<double> track_rate_TIB;
  vector<double> track_rate_TIB_err;
  vector<double> track_rate_TID;
  vector<double> track_rate_TID_err;
  vector<double> track_rate_TEC;
  vector<double> track_rate_TEC_err;
  vector<double> track_rate_TECP;
  vector<double> track_rate_TECP_err;
  vector<double> track_rate_TECM;
  vector<double> track_rate_TECM_err;
  vector<double> tracks;
  vector<double> tracks_err;
  vector<double> tracks_bpix;
  vector<double> tracks_fpix;
  vector<double> tracks_pix;
  vector<double> tracks_tec;
  vector<double> weight;

  string Bar_Xtitle[8] = {"Event", "Track", "FPIX", "BPIX", "TIB", "TID", "TOB", "TEC"};
  double Bar_Ytitle[8] = {0};

  int j = 0;
  double total_tracks = 0;
  double bpix_tracks = 0;
  double fpix_tracks = 0;
  double pix_tracks = 0;
  double tracks_TECoff = 0;
  int nTotalEvents = 0, nTotalTracks = 0;

  fprintf(pFile, "##################################################\n");
  fprintf(pFile, "         Track rate for each run number           \n");
  fprintf(pFile, "##################################################\n");

  Long64_t n = tree->GetEntriesFast();
  std::cout << "Total Runs in this file: " << n << endl;
  for (Long64_t jentry = 0; jentry < n; jentry++)  // Loop over Runs
  {
    tree->GetEntry(jentry);
    if (run_time == 0 || run_time < 0)
      continue;

    if (runLow != 0 && runUp != 0) {
      if (runnum < runLow)
        continue;
      if (runnum > runUp)
        break;
    }

    event_rate.push_back(number_of_events / run_time);
    runNumber.push_back(runnum);
    track_rate.push_back(number_of_tracks / run_time);
    track_rate_PIX.push_back(number_of_tracks_PIX / run_time);
    track_rate_FPIX.push_back(number_of_tracks_FPIX / run_time);
    track_rate_BPIX.push_back(number_of_tracks_BPIX / run_time);
    track_rate_TOB.push_back(number_of_tracks_TOB / run_time);
    track_rate_TIB.push_back(number_of_tracks_TIB / run_time);
    track_rate_TID.push_back(number_of_tracks_TID / run_time);
    track_rate_TEC.push_back(number_of_tracks_TEC / run_time);
    track_rate_TECP.push_back(number_of_tracks_TECP / run_time);
    track_rate_TECM.push_back(number_of_tracks_TECM / run_time);
    tracks.push_back(number_of_tracks);
    tracks_bpix.push_back(number_of_tracks_BPIX);
    tracks_fpix.push_back(number_of_tracks_FPIX);
    tracks_pix.push_back(number_of_tracks_PIX);
    tracks_tec.push_back(number_of_tracks_TECM);
    total_tracks += tracks[j];
    bpix_tracks += tracks_bpix[j];
    fpix_tracks += tracks_fpix[j];
    pix_tracks += tracks_pix[j];
    nTotalEvents += number_of_events;
    nTotalTracks += number_of_tracks;

    fprintf(pFile,
            "runnum :%-7.0lf, # of tracks :%-10.0lf, track rates :%-10.2lf\n",
            runNumber.at(j),
            tracks.at(j),
            track_rate.at(j));
    track_rate_err.push_back(sqrt(float(number_of_tracks)) / run_time);
    event_rate_err.push_back(sqrt(float(number_of_events)) / run_time);
    track_rate_PIX_err.push_back(sqrt(float(number_of_tracks_PIX)) / run_time);
    track_rate_FPIX_err.push_back(sqrt(float(number_of_tracks_FPIX)) / run_time);
    track_rate_BPIX_err.push_back(sqrt(float(number_of_tracks_BPIX)) / run_time);
    track_rate_TOB_err.push_back(sqrt(float(number_of_tracks_TOB)) / run_time);
    track_rate_TIB_err.push_back(sqrt(float(number_of_tracks_TIB)) / run_time);
    track_rate_TID_err.push_back(sqrt(float(number_of_tracks_TID)) / run_time);
    track_rate_TEC_err.push_back(sqrt(float(number_of_tracks_TEC)) / run_time);
    track_rate_TECP_err.push_back(sqrt(float(number_of_tracks_TECP)) / run_time);
    track_rate_TECM_err.push_back(sqrt(float(number_of_tracks_TECM)) / run_time);

    runNumber_err.push_back(0);
    if (number_of_tracks_TECM == 0) {
      tracks_TECoff += tracks.at(j);
    }

    j++;
  }  //Loop over runs closed
  std::cout << "Total Events: " << nTotalEvents << std::endl;
  std::cout << "Total Tracks: " << nTotalTracks << std::endl;

  fprintf(pFile, "\n\n");
  fprintf(pFile, "##################################################\n");
  fprintf(pFile, "    Some information on total number of tracks    \n");
  fprintf(pFile, "##################################################\n");
  fprintf(pFile, "Total # of tracks   : %-10.0lf\n", total_tracks);
  fprintf(pFile, "# of tracks in BPIX : %-10.0lf\n", bpix_tracks);
  fprintf(pFile, "# of tracks in FPIX : %-10.0lf\n", fpix_tracks);
  fprintf(pFile, "# of tracks in PIX  : %-10.0lf\n", pix_tracks);
  fprintf(pFile, "\n\n");

  fclose(pFile);

  //+++++++++++++++++++++++++++++       Make Directories     +++++++++++++++++++++++++++++++++++++

  gSystem->Exec("mkdir -p Rate_Plots");

  //----------------------------------------------------------------------------------------------

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

  //============  Declaring TVectors for TGraphs  =============//
  TVectorD event_rate_VecD;
  TVectorD event_rate_err_VecD;
  TVectorD track_rate_VecD;
  TVectorD track_rate_err_VecD;
  TVectorD runNumber_VecD;
  TVectorD runNumber_err_VecD;
  TVectorD track_rate_PIX_VecD;
  TVectorD track_rate_PIX_err_VecD;
  TVectorD track_rate_FPIX_VecD;
  TVectorD track_rate_FPIX_err_VecD;
  TVectorD track_rate_BPIX_VecD;
  TVectorD track_rate_BPIX_err_VecD;
  TVectorD track_rate_TOB_VecD;
  TVectorD track_rate_TOB_err_VecD;
  TVectorD track_rate_TIB_VecD;
  TVectorD track_rate_TIB_err_VecD;
  TVectorD track_rate_TID_VecD;
  TVectorD track_rate_TID_err_VecD;
  TVectorD track_rate_TEC_VecD;
  TVectorD track_rate_TEC_err_VecD;
  TVectorD track_rate_TECP_VecD;
  TVectorD track_rate_TECP_err_VecD;
  TVectorD track_rate_TECM_VecD;
  TVectorD track_rate_TECM_err_VecD;

  runNumber_VecD.Use(runNumber.size(), &(runNumber[0]));
  runNumber_err_VecD.Use(runNumber_err.size(), &(runNumber_err[0]));
  event_rate_VecD.Use(event_rate.size(), &(event_rate[0]));
  event_rate_err_VecD.Use(event_rate_err.size(), &(event_rate_err[0]));

  track_rate_VecD.Use(track_rate.size(), &(track_rate[0]));
  track_rate_err_VecD.Use(track_rate_err.size(), &(track_rate_err[0]));

  track_rate_PIX_VecD.Use(track_rate_PIX.size(), &(track_rate_PIX[0]));
  track_rate_PIX_err_VecD.Use(track_rate_PIX_err.size(), &(track_rate_PIX_err[0]));
  track_rate_FPIX_VecD.Use(track_rate_FPIX.size(), &(track_rate_FPIX[0]));
  track_rate_FPIX_err_VecD.Use(track_rate_FPIX_err.size(), &(track_rate_FPIX_err[0]));
  track_rate_BPIX_VecD.Use(track_rate_BPIX.size(), &(track_rate_BPIX[0]));
  track_rate_BPIX_err_VecD.Use(track_rate_BPIX_err.size(), &(track_rate_BPIX_err[0]));
  track_rate_TOB_VecD.Use(track_rate_TOB.size(), &(track_rate_TOB[0]));
  track_rate_TOB_err_VecD.Use(track_rate_TOB_err.size(), &(track_rate_TOB_err[0]));
  track_rate_TIB_VecD.Use(track_rate_TIB.size(), &(track_rate_TIB[0]));
  track_rate_TIB_err_VecD.Use(track_rate_TIB_err.size(), &(track_rate_TIB_err[0]));
  track_rate_TID_VecD.Use(track_rate_TID.size(), &(track_rate_TID[0]));
  track_rate_TID_err_VecD.Use(track_rate_TID_err.size(), &(track_rate_TID_err[0]));
  track_rate_TEC_VecD.Use(track_rate_TEC.size(), &(track_rate_TEC[0]));
  track_rate_TEC_err_VecD.Use(track_rate_TEC_err.size(), &(track_rate_TEC_err[0]));
  track_rate_TECP_VecD.Use(track_rate_TECP.size(), &(track_rate_TECP[0]));
  track_rate_TECP_err_VecD.Use(track_rate_TECP_err.size(), &(track_rate_TECP_err[0]));
  track_rate_TECM_VecD.Use(track_rate_TECM.size(), &(track_rate_TECM[0]));
  track_rate_TECM_err_VecD.Use(track_rate_TECM_err.size(), &(track_rate_TECM_err[0]));

  //+++++++++++++++++++++++++++++  Overall event event rate  +++++++++++++++++++++++++++++++++++++

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
  //gr_event_rate.GetYaxis()->SetRangeUser(0,7);
  gr_event_rate.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "Event Rate");
  c.SetGrid();
  c.SaveAs("event_rate.png");
  c.SaveAs("event_rate.pdf");
  c.SaveAs("event_rate.C");
  c.Clear();
  gSystem->Exec("mv event_rate.png Rate_Plots");
  gSystem->Exec("mv event_rate.pdf Rate_Plots");
  gSystem->Exec("mv event_rate.C Rate_Plots");

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
  gSystem->Exec("mv track_rate.png Rate_Plots");
  gSystem->Exec("mv track_rate.pdf Rate_Plots");
  gSystem->Exec("mv track_rate.C Rate_Plots");

  //-----------------------------------------------------------------------------------------------

  //+++++++++++++++++++++++++++++++  Total Pixel track rate +++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_track_rate_PIX(runNumber_VecD, track_rate_PIX_VecD, runNumber_err_VecD, track_rate_PIX_err_VecD);
  gr_track_rate_PIX.GetXaxis()->SetTitle("Run Number");
  gr_track_rate_PIX.GetXaxis()->SetLabelSize(0.04);
  gr_track_rate_PIX.GetXaxis()->SetNoExponent();
  gr_track_rate_PIX.GetXaxis()->SetNdivisions(5);
  gr_track_rate_PIX.GetYaxis()->SetTitle("Track Rate (Hz)");
  gr_track_rate_PIX.GetXaxis()->SetTitleSize(0.05);
  gr_track_rate_PIX.GetYaxis()->SetLabelSize(0.05);
  gr_track_rate_PIX.GetYaxis()->SetTitleSize(0.05);
  gr_track_rate_PIX.SetMarkerStyle(20);
  gr_track_rate_PIX.SetMarkerSize(1.4);
  gr_track_rate_PIX.SetMarkerColor(2);
  gr_track_rate_PIX.SetTitle("");
  gr_track_rate_PIX.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "PIXEL");
  c.SetGrid();
  c.SaveAs("pixel_track_rate.png");
  c.SaveAs("pixel_track_rate.pdf");
  c.SaveAs("pixel_track_rate.C");
  c.Clear();
  gSystem->Exec("mv pixel_track_rate.png Rate_Plots");
  gSystem->Exec("mv pixel_track_rate.pdf Rate_Plots");
  gSystem->Exec("mv pixel_track_rate.C Rate_Plots");

  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  FPIX track rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_track_rate_FPIX(runNumber_VecD, track_rate_FPIX_VecD, runNumber_err_VecD, track_rate_FPIX_err_VecD);
  gr_track_rate_FPIX.GetXaxis()->SetTitle("Run Number");
  gr_track_rate_FPIX.GetXaxis()->SetLabelSize(0.04);
  gr_track_rate_FPIX.GetXaxis()->SetNoExponent();
  gr_track_rate_FPIX.GetXaxis()->SetNdivisions(5);
  gr_track_rate_FPIX.GetYaxis()->SetTitle("Track Rate (Hz)");
  gr_track_rate_FPIX.GetXaxis()->SetTitleSize(0.05);
  gr_track_rate_FPIX.GetYaxis()->SetLabelSize(0.05);
  gr_track_rate_FPIX.GetYaxis()->SetTitleSize(0.05);
  gr_track_rate_FPIX.SetMarkerStyle(20);
  gr_track_rate_FPIX.SetMarkerSize(1.4);
  gr_track_rate_FPIX.SetMarkerColor(kRed);
  gr_track_rate_FPIX.SetTitle("");
  gr_track_rate_FPIX.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "FPIX");
  c.SetGrid();
  c.SaveAs("fpix_track_rate.png");
  c.SaveAs("fpix_track_rate.pdf");
  c.SaveAs("fpix_track_rate.C");
  c.Clear();
  gSystem->Exec("mv fpix_track_rate.png Rate_Plots");
  gSystem->Exec("mv fpix_track_rate.pdf Rate_Plots");
  gSystem->Exec("mv fpix_track_rate.C Rate_Plots");

  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  BPIX track rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_track_rate_BPIX(runNumber_VecD, track_rate_BPIX_VecD, runNumber_err_VecD, track_rate_BPIX_err_VecD);
  gr_track_rate_BPIX.GetXaxis()->SetTitle("Run Number");
  gr_track_rate_BPIX.GetXaxis()->SetLabelSize(0.04);
  gr_track_rate_BPIX.GetXaxis()->SetNoExponent();
  gr_track_rate_BPIX.GetXaxis()->SetNdivisions(5);
  gr_track_rate_BPIX.GetYaxis()->SetTitle("Track Rate (Hz)");
  gr_track_rate_BPIX.GetXaxis()->SetTitleSize(0.05);
  gr_track_rate_BPIX.GetYaxis()->SetLabelSize(0.05);
  gr_track_rate_BPIX.GetYaxis()->SetTitleSize(0.05);
  gr_track_rate_BPIX.SetMarkerStyle(20);
  gr_track_rate_BPIX.SetMarkerSize(1.4);
  gr_track_rate_BPIX.SetMarkerColor(2);
  gr_track_rate_BPIX.SetTitle("");
  gr_track_rate_BPIX.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "BPIX");
  c.SetGrid();
  c.SaveAs("bpix_track_rate.png");
  c.SaveAs("bpix_track_rate.pdf");
  c.SaveAs("bpix_track_rate.C");
  c.Clear();
  gSystem->Exec("mv bpix_track_rate.png Rate_Plots");
  gSystem->Exec("mv bpix_track_rate.pdf Rate_Plots");
  gSystem->Exec("mv bpix_track_rate.C Rate_Plots");

  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  TOB track rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_track_rate_TOB(runNumber_VecD, track_rate_TOB_VecD, runNumber_err_VecD, track_rate_TOB_err_VecD);
  gr_track_rate_TOB.GetXaxis()->SetTitle("Run Number");
  gr_track_rate_TOB.GetXaxis()->SetLabelSize(0.04);
  gr_track_rate_TOB.GetXaxis()->SetNoExponent();
  gr_track_rate_TOB.GetXaxis()->SetNdivisions(5);
  gr_track_rate_TOB.GetYaxis()->SetTitle("Track Rate (Hz)");
  gr_track_rate_TOB.GetXaxis()->SetTitleSize(0.05);
  gr_track_rate_TOB.GetYaxis()->SetLabelSize(0.05);
  gr_track_rate_TOB.GetYaxis()->SetTitleSize(0.05);
  gr_track_rate_TOB.SetMarkerStyle(20);
  gr_track_rate_TOB.SetMarkerSize(1.4);
  gr_track_rate_TOB.SetMarkerColor(2);
  gr_track_rate_TOB.SetTitle("");
  gr_track_rate_TOB.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "TOB");
  c.SetGrid();
  c.SaveAs("tob_track_rate.png");
  c.SaveAs("tob_track_rate.pdf");
  c.SaveAs("tob_track_rate.C");
  c.Clear();
  gSystem->Exec("mv tob_track_rate.png Rate_Plots");
  gSystem->Exec("mv tob_track_rate.C Rate_Plots");
  gSystem->Exec("mv tob_track_rate.pdf Rate_Plots");

  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  TIB track rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_track_rate_TIB(runNumber_VecD, track_rate_TIB_VecD, runNumber_err_VecD, track_rate_TIB_err_VecD);
  gr_track_rate_TIB.GetXaxis()->SetTitle("Run Number");
  gr_track_rate_TIB.GetXaxis()->SetLabelSize(0.04);
  gr_track_rate_TIB.GetXaxis()->SetNoExponent();
  gr_track_rate_TIB.GetXaxis()->SetNdivisions(5);
  gr_track_rate_TIB.GetYaxis()->SetTitle("Track Rate (Hz)");
  gr_track_rate_TIB.GetXaxis()->SetTitleSize(0.05);
  gr_track_rate_TIB.GetYaxis()->SetLabelSize(0.05);
  gr_track_rate_TIB.GetYaxis()->SetTitleSize(0.05);
  gr_track_rate_TIB.SetMarkerStyle(20);
  gr_track_rate_TIB.SetMarkerSize(1.4);
  gr_track_rate_TIB.SetMarkerColor(2);
  gr_track_rate_TIB.SetTitle("");
  gr_track_rate_TIB.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "TIB");
  c.SetGrid();
  c.SaveAs("tib_track_rate.png");
  c.SaveAs("tib_track_rate.pdf");
  c.SaveAs("tib_track_rate.C");
  c.Clear();
  gSystem->Exec("mv tib_track_rate.png Rate_Plots");
  gSystem->Exec("mv tib_track_rate.pdf Rate_Plots");
  gSystem->Exec("mv tib_track_rate.C Rate_Plots");

  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  TID track rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_track_rate_TID(runNumber_VecD, track_rate_TID_VecD, runNumber_err_VecD, track_rate_TID_err_VecD);
  gr_track_rate_TID.GetXaxis()->SetTitle("Run Number");
  gr_track_rate_TID.GetXaxis()->SetLabelSize(0.04);
  gr_track_rate_TID.GetXaxis()->SetNoExponent();
  gr_track_rate_TID.GetXaxis()->SetNdivisions(5);
  gr_track_rate_TID.GetYaxis()->SetTitle("Track Rate (Hz)");
  gr_track_rate_TID.GetXaxis()->SetTitleSize(0.05);
  gr_track_rate_TID.GetYaxis()->SetLabelSize(0.05);
  gr_track_rate_TID.GetYaxis()->SetTitleSize(0.05);
  gr_track_rate_TID.SetMarkerStyle(20);
  gr_track_rate_TID.SetMarkerSize(1.4);
  gr_track_rate_TID.SetMarkerColor(2);
  gr_track_rate_TID.SetTitle("");
  gr_track_rate_TID.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "TID");
  c.SetGrid();
  c.SaveAs("tid_track_rate.png");
  c.SaveAs("tid_track_rate.pdf");
  c.SaveAs("tid_track_rate.C");
  c.Clear();
  gSystem->Exec("mv tid_track_rate.png Rate_Plots");
  gSystem->Exec("mv tid_track_rate.pdf Rate_Plots");
  gSystem->Exec("mv tid_track_rate.C Rate_Plots");

  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  Total TEC track rate  ++++++++++++++++++++++++++++++++++++++++++++

  TGraphErrors gr_track_rate_TEC(runNumber_VecD, track_rate_TEC_VecD, runNumber_err_VecD, track_rate_TEC_err_VecD);
  gr_track_rate_TEC.GetXaxis()->SetTitle("Run Number");
  gr_track_rate_TEC.GetXaxis()->SetLabelSize(0.04);
  gr_track_rate_TEC.GetXaxis()->SetNoExponent();
  gr_track_rate_TEC.GetXaxis()->SetNdivisions(5);
  gr_track_rate_TEC.GetYaxis()->SetTitle("Track Rate (Hz)");
  gr_track_rate_TEC.GetXaxis()->SetTitleSize(0.05);
  gr_track_rate_TEC.GetYaxis()->SetLabelSize(0.05);
  gr_track_rate_TEC.GetYaxis()->SetTitleSize(0.05);
  gr_track_rate_TEC.SetMarkerStyle(20);
  gr_track_rate_TEC.SetMarkerSize(1.4);
  gr_track_rate_TEC.SetMarkerColor(kRed);
  gr_track_rate_TEC.SetTitle("");
  gr_track_rate_TEC.Draw("AP");
  top_right_Title.DrawLatexNDC(0.79, 0.94, "cosmic rays");
  detector.DrawLatexNDC(0.23, 0.83, "TEC");
  c.SetGrid();
  c.SaveAs("tec_track_rate.png");
  c.SaveAs("tec_track_rate.pdf");
  c.SaveAs("tec_track_rate.C");
  c.Clear();
  gSystem->Exec("mv tec_track_rate.png Rate_Plots");
  gSystem->Exec("mv tec_track_rate.pdf Rate_Plots");
  gSystem->Exec("mv tec_track_rate.C Rate_Plots");

  //-----------------------------------------------------------------------------------------------

  //++++++++++++++++++++++++++++++++  TEC+/- track rate  ++++++++++++++++++++++++++++++++++++++++++++
  TMultiGraph mg("track rate", "Track Rate TEC+/-");  // Multigraph decleration

  TGraphErrors *gr_track_rate_TECP =
      new TGraphErrors(runNumber_VecD, track_rate_TECP_VecD, runNumber_err_VecD, track_rate_TECP_err_VecD);
  gr_track_rate_TECP->SetMarkerStyle(20);
  gr_track_rate_TECP->SetMarkerSize(1.4);
  gr_track_rate_TECP->SetMarkerColor(kBlack);

  TGraphErrors *gr_track_rate_TECM =
      new TGraphErrors(runNumber_VecD, track_rate_TECM_VecD, runNumber_err_VecD, track_rate_TECM_err_VecD);
  gr_track_rate_TECM->SetMarkerStyle(20);
  gr_track_rate_TECM->SetMarkerSize(1.4);
  gr_track_rate_TECM->SetMarkerColor(kGreen);

  mg.Add(gr_track_rate_TECP);
  mg.Add(gr_track_rate_TECM);
  mg.Draw("AP");
  mg.GetXaxis()->SetTitle("Run Number");
  mg.GetXaxis()->SetNoExponent();
  mg.GetXaxis()->SetNdivisions(5);
  mg.GetXaxis()->SetLabelSize(0.04);
  mg.GetXaxis()->SetTitleSize(0.05);
  mg.GetYaxis()->SetLabelSize(0.05);
  mg.GetYaxis()->SetTitleSize(0.05);
  mg.GetYaxis()->SetTitle("Track Rate (Hz)");

  TLegend leg(0.76, 0.76, 0.92, 0.90);  // Legend for TEC+/-
  leg.AddEntry(gr_track_rate_TECP, "TEC+", "p");
  leg.AddEntry(gr_track_rate_TECM, "TEC-", "p");
  leg.SetBorderSize(1);
  leg.SetShadowColor(0);
  leg.SetFillColor(0);
  leg.Draw();
  c.SetGrid();
  c.SaveAs("tec_track_ratePM.png");
  c.Clear();
  gSystem->Exec("mv tec_track_ratePM.png Rate_Plots");

  //-----------------------------------------------------------------------------------------------

  c.Close();

  //-----------------------------------------------------------------------------------------------
  //					Weighted Mean calculation
  //-----------------------------------------------------------------------------------------------

  double total_weight = 0;
  double weighted_mean_track_rate;
  double weighted_mean_track_rate_TEC;
  double weighted_mean_track_rate_TOB;
  double weighted_mean_track_rate_TIB;
  double weighted_mean_track_rate_TID;
  double weighted_mean_track_rate_FPIX;
  double weighted_mean_track_rate_BPIX;
  double weighted_mean_event_rate;

  for (int k = 0; k < j; k++)  // Loop over all runs used(j) to allot weight to each run number
  {
    weight.push_back(tracks.at(k) / total_tracks);
  }

  for (int a = 0; a < j; a++)  // Loop over all runs used(j) to evaluate weighted mean for each subdetector
  {
    weighted_mean_track_rate += track_rate.at(a) * weight.at(a);
    weighted_mean_track_rate_TEC += track_rate_TEC.at(a) * weight.at(a);
    weighted_mean_track_rate_TOB += track_rate_TOB.at(a) * weight.at(a);
    weighted_mean_track_rate_TIB += track_rate_TIB.at(a) * weight.at(a);
    weighted_mean_track_rate_TID += track_rate_TID.at(a) * weight.at(a);
    weighted_mean_track_rate_FPIX += track_rate_FPIX.at(a) * weight.at(a);
    weighted_mean_track_rate_BPIX += track_rate_BPIX.at(a) * weight.at(a);
    weighted_mean_event_rate += event_rate.at(a) * weight.at(a);
    total_weight += weight.at(a);
  }

  std::cout << " Weighted mean Event Rate : " << weighted_mean_event_rate << std::endl;
  std::cout << " Weighted mean Total Track Rate : " << weighted_mean_track_rate << std::endl;
  std::cout << " Weighted mean Track Rate BPIX: " << weighted_mean_track_rate_BPIX << std::endl;
  std::cout << " Weighted mean Track Rate FPIX : " << weighted_mean_track_rate_FPIX << std::endl;
  std::cout << " Weighted mean Track Rate TIB: " << weighted_mean_track_rate_TIB << std::endl;
  std::cout << " Weighted mean Track Rate TOB: " << weighted_mean_track_rate_TOB << std::endl;
  std::cout << " Weighted mean Track Rate TID: " << weighted_mean_track_rate_TID << std::endl;
  std::cout << " Weighted mean Track Rate TEC: " << weighted_mean_track_rate_TEC << std::endl;

  //-----------------------------------------------------------------------------------------------
  //			Summary Plot for track rate in each Subdetector
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

  TH1F h1b("h1b", "Track Rate Summary", 8, 0, 8);
  h1b.SetFillColor(kRed);
  h1b.SetBarWidth(0.6);
  h1b.SetBarOffset(0.25);
  h1b.SetStats(0);
  h1b.GetXaxis()->SetLabelFont(42);
  h1b.GetXaxis()->SetLabelOffset(0.012);
  h1b.GetXaxis()->SetLabelSize(0.06);
  h1b.GetXaxis()->SetTitleSize(0.05);
  h1b.GetXaxis()->SetTitleFont(42);
  h1b.GetYaxis()->SetTitle("Average Track Rate (Hz)");
  h1b.GetYaxis()->SetLabelFont(42);
  h1b.GetYaxis()->SetLabelSize(0.05);
  h1b.GetYaxis()->SetTitleSize(0.05);
  h1b.GetYaxis()->SetTitleOffset(0);

  Bar_Ytitle[0] = weighted_mean_event_rate;
  Bar_Ytitle[1] = weighted_mean_track_rate;
  Bar_Ytitle[2] = weighted_mean_track_rate_FPIX;
  Bar_Ytitle[3] = weighted_mean_track_rate_BPIX;
  Bar_Ytitle[4] = weighted_mean_track_rate_TIB;
  Bar_Ytitle[5] = weighted_mean_track_rate_TID;
  Bar_Ytitle[6] = weighted_mean_track_rate_TOB;
  Bar_Ytitle[7] = weighted_mean_track_rate_TEC;

  for (int i = 1; i <= 8; i++) {
    h1b.SetBinContent(i, Bar_Ytitle[i - 1]);
    h1b.GetXaxis()->SetBinLabel(i, Bar_Xtitle[i - 1].c_str());
  }

  gStyle->SetPaintTextFormat("1.3f");
  h1b.LabelsOption("d");
  h1b.Draw("bTEXT");

  // --------- Saving Summary Chart in pdf,png & C formats ------- //
  TString summary_chart_title = "SummaryChart";
  TString Format[3] = {"png", "pdf", "C"};

  for (int i = 0; i < 3; i++) {
    TString filename = summary_chart_title + "." + Format[i];
    canvas->SaveAs(filename.Data());
    TString mv_cmd = "mv " + filename + " Rate_Plots";
    gSystem->Exec(mv_cmd.Data());
  }
  canvas->Clear();
  canvas->Close();
}
