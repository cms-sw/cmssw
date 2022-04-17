#include "PlotMEIFBenchmarks.hh"

PlotMEIFBenchmarks::PlotMEIFBenchmarks(const TString& arch, const TString& sample, const TString& build)
    : arch(arch), sample(sample), build(build) {
  // setup style for plotting
  setupStyle();

  // get file
  file = TFile::Open("benchmarkMEIF_" + arch + "_" + sample + "_" + build + ".root");

  // setup enum
  setupARCHEnum(arch);

  // setup arch options
  setupArch();

  // setup events
  setupEvents();
}

PlotMEIFBenchmarks::~PlotMEIFBenchmarks() { delete file; }

void PlotMEIFBenchmarks::RunMEIFBenchmarkPlots() {
  // title options
  const TString nvu = Form("%iint", arch_opt.vumax);

  // x-axis title
  const TString xtitleth = "Number of Threads";

  // y-axis title
  const TString ytitletime = "Averarge Time per Event [s]";
  const TString ytitlespeedup = "Average Speedup per Event";

  // Do the overlaying!
  PlotMEIFBenchmarks::MakeOverlay(
      "time",
      build + " " + sample + " Multiple Events in Flight Benchmark on " + arch + " [nVU=" + nvu + "]",
      xtitleth,
      ytitletime,
      arch_opt.thmin,
      arch_opt.thmax,
      arch_opt.thmeiftimemin,
      arch_opt.thmeiftimemax);

  PlotMEIFBenchmarks::MakeOverlay(
      "speedup",
      build + " " + sample + " Multiple Events in Flight Speedup on " + arch + " [nVU=" + nvu + "]",
      xtitleth,
      ytitlespeedup,
      arch_opt.thmin,
      arch_opt.thmax,
      arch_opt.thmeifspeedupmin,
      arch_opt.thmeifspeedupmax);
}

void PlotMEIFBenchmarks::MakeOverlay(const TString& text,
                                     const TString& title,
                                     const TString& xtitle,
                                     const TString& ytitle,
                                     const Double_t xmin,
                                     const Double_t xmax,
                                     const Double_t ymin,
                                     const Double_t ymax) {
  // special setups
  const Bool_t isSpeedup = text.Contains("speedup", TString::kExact);

  // canvas
  auto canv = new TCanvas();
  canv->cd();
  canv->SetGridy();
  if (!isSpeedup)
    canv->SetLogy();
  canv->DrawFrame(xmin, ymin, xmax, ymax, "");

  // legend
  const Double_t x1 = (isSpeedup ? 0.20 : 0.60);
  const Double_t y1 = 0.65;
  auto leg = new TLegend(x1, y1, x1 + 0.25, y1 + 0.2);
  leg->SetBorderSize(0);

  // get tgraphs for meif and draw
  TGVec graphs(nevents);
  for (auto i = 0U; i < nevents; i++) {
    const auto& event = events[i];
    auto& graph = graphs[i];

    const TString nEV = Form("%i", event.nev);
    graph = (TGraph*)file->Get("g_" + build + "_MEIF_nEV" + nEV + "_" + text);

    if (graph) {
      // restyle a bit
      graph->SetTitle(title + ";" + xtitle + ";" + ytitle);

      graph->SetLineWidth(2);
      graph->SetLineColor(event.color);
      graph->SetMarkerStyle(kFullCircle);
      graph->SetMarkerColor(event.color);
      graph->GetXaxis()->SetRangeUser(xmin, xmax);
      graph->GetYaxis()->SetRangeUser(ymin, ymax);

      // draw and add to legend
      graph->Draw(i > 0 ? "LP SAME" : "ALP");
      leg->AddEntry(graph, Form("%i Events", event.nev), "LP");
    }
  }

  // Draw ideal scaling line
  TF1* scaling = NULL;
  if (isSpeedup) {
    scaling = new TF1("ideal_scaling", "x", arch_opt.thmin, arch_opt.thmeifspeedupmax);
    scaling->SetLineColor(kBlack);
    scaling->SetLineStyle(kDashed);
    scaling->SetLineWidth(2);
    scaling->Draw("SAME");
    leg->AddEntry(scaling, "Ideal Scaling", "l");
  }

  // draw legend last
  leg->Draw("SAME");

  // Save the png
  const TString outname = arch + "_" + sample + "_" + build + "_MEIF_" + text;
  canv->SaveAs(outname + ".png");

  // Save log-x version
  canv->SetLogx();
  for (auto i = 0U; i < nevents; i++) {
    auto& graph = graphs[i];

    // reset axes for logx
    if (graph) {
      graph->GetXaxis()->SetRangeUser(xmin, xmax);
      graph->GetYaxis()->SetRangeUser(ymin, ymax);
    }
  }
  canv->Update();
  canv->SaveAs(outname + "_logx.png");

  // delete everything
  for (auto& graph : graphs)
    delete graph;
  if (isSpeedup)
    delete scaling;
  delete leg;
  delete canv;
}
