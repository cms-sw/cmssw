#include "PlotsFromDump.hh"

PlotsFromDump::PlotsFromDump(const TString& sample, const TString& build, const TString& suite, const int useARCH)
    : sample(sample), build(build), suite(suite), useARCH(useARCH) {
  // setup style for plotting
  setupStyle();

  // setup suite enum
  setupSUITEEnum(suite);

  // setup build options : true for isBenchmark-type plots, false for no CMSSW
  setupBuilds(true, false);

  // get the right build label
  label =
      std::find_if(builds.begin(), builds.end(), [&](const auto& ibuild) { return build.EqualTo(ibuild.name); })->label;
  if (label == "") {
    std::cerr << build.Data() << " build routine not specified in list of builds! Exiting..." << std::endl;
    exit(1);
  }

  // Setup test opts
  setupTests(useARCH);

  // Setup plot opts
  setupPlots();
}

PlotsFromDump::~PlotsFromDump() {}

void PlotsFromDump::RunPlotsFromDump() {
  // Open ROOT files first
  std::vector<TFile*> files(ntests);
  for (auto t = 0U; t < ntests; t++) {
    const auto& test = tests[t];
    auto& file = files[t];

    file = TFile::Open("test_" + test.arch + "_" + sample + "_" + build + "_" + test.suffix + ".root");
  }

  // Outer loop over all overplots
  for (auto p = 0U; p < nplots; p++) {
    const auto& plot = plots[p];

    // declare standard stuff
    const Bool_t isLogy =
        !(plot.name.Contains("MXPHI", TString::kExact) || plot.name.Contains("MXETA", TString::kExact));
    auto canv = new TCanvas();
    canv->cd();
    canv->SetLogy(isLogy);

    auto leg = new TLegend(0.7, 0.68, 0.98, 0.92);

    Double_t min = 1e9;
    Double_t max = -1e9;

    std::vector<TH1F*> hists(ntests);
    for (auto t = 0U; t < ntests; t++) {
      const auto& test = tests[t];
      auto& file = files[t];
      auto& hist = hists[t];

      hist = (TH1F*)file->Get(plot.name + "_" + test.suffix);
      const TString title = hist->GetTitle();
      hist->SetTitle(title + " [" + label + " - " + sample + "]");
      hist->GetXaxis()->SetTitle(plot.xtitle.Data());
      hist->GetYaxis()->SetTitle(plot.ytitle.Data());

      hist->SetLineColor(test.color);
      hist->SetMarkerColor(test.color);
      hist->SetMarkerStyle(test.marker);

      hist->Scale(1.f / hist->Integral());
      GetMinMaxHist(hist, min, max);
    }

    for (auto t = 0U; t < ntests; t++) {
      const auto& test = tests[t];
      auto& hist = hists[t];

      SetMinMaxHist(hist, min, max, isLogy);
      hist->Draw(t > 0 ? "P SAME" : "P");

      const TString mean = Form("%4.1f", hist->GetMean());
      leg->AddEntry(hist, test.arch + " " + test.suffix + " [#mu = " + mean + "]", "p");
    }

    // draw legend and save plot
    leg->Draw("SAME");
    canv->SaveAs(sample + "_" + build + "_" + plot.outname + ".png");

    // delete temps
    for (auto& hist : hists)
      delete hist;
    delete leg;
    delete canv;
  }

  // delete files
  for (auto& file : files)
    delete file;
}
