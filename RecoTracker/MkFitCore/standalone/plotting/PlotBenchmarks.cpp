#include "PlotBenchmarks.hh"

#include <iostream>

PlotBenchmarks::PlotBenchmarks(const TString& arch, const TString& sample, const TString& suite)
    : arch(arch), sample(sample), suite(suite) {
  // setup style for plotting
  setupStyle();

  // get file
  file = TFile::Open("benchmark_" + arch + "_" + sample + ".root");

  // setup arch enum
  setupARCHEnum(arch);

  // setup arch options
  setupArch();

  // setup suite enum
  setupSUITEEnum(suite);

  // setup build options : true for isBenchmark-type plots, false for no CMSSW
  setupBuilds(true, false);
}

PlotBenchmarks::~PlotBenchmarks() { delete file; }

void PlotBenchmarks::RunBenchmarkPlots() {
  // title options
  const TString nth = "1";
  const TString nvu = Form("%iint", arch_opt.vumax);

  // x-axis titles
  const TString xtitlevu = "Matriplex Vector Width [floats]";
  const TString xtitleth = "Number of Threads";

  // y-axis titles
  const TString ytitletime = "Average Build Time per Event [s]";
  const TString ytitlespeedup = "Average Build Speedup per Event";

  // Do the overlaying!
  PlotBenchmarks::MakeOverlay("VU_time",
                              sample + " Vectorization Benchmark on " + arch + " [nTH=" + nth + "]",
                              xtitlevu,
                              ytitletime,
                              arch_opt.vumin,
                              arch_opt.vumax,
                              arch_opt.vutimemin,
                              arch_opt.vutimemax);

  PlotBenchmarks::MakeOverlay("TH_time",
                              sample + " Parallelization Benchmark on " + arch + " [nVU=" + nvu + "]",
                              xtitleth,
                              ytitletime,
                              arch_opt.thmin,
                              arch_opt.thmax,
                              arch_opt.thtimemin,
                              arch_opt.thtimemax);

  PlotBenchmarks::MakeOverlay("VU_speedup",
                              sample + " Vectorization Speedup on " + arch + " [nTH=" + nth + "]",
                              xtitlevu,
                              ytitlespeedup,
                              arch_opt.vumin,
                              arch_opt.vumax,
                              arch_opt.vuspeedupmin,
                              arch_opt.vuspeedupmax);

  PlotBenchmarks::MakeOverlay("TH_speedup",
                              sample + " Parallelization Speedup on " + arch + " [nVU=" + nvu + "]",
                              xtitleth,
                              ytitlespeedup,
                              arch_opt.thmin,
                              arch_opt.thmax,
                              arch_opt.thspeedupmin,
                              arch_opt.thspeedupmax);
}

void PlotBenchmarks::MakeOverlay(const TString& text,
                                 const TString& title,
                                 const TString& xtitle,
                                 const TString& ytitle,
                                 const Double_t xmin,
                                 const Double_t xmax,
                                 const Double_t ymin,
                                 const Double_t ymax) {
  // special setups
  const Bool_t isVU = text.Contains("VU", TString::kExact);
  const Bool_t isSpeedup = text.Contains("speedup", TString::kExact);

  // canvas
  auto canv = new TCanvas();
  canv->cd();
  canv->SetGridy();
  if (!isVU && !isSpeedup)
    canv->SetLogy();

  // legend
  const Double_t x1 = (isSpeedup ? 0.20 : 0.60);  // draw legend on left for speedup plots as this part is empty
  const Double_t y1 = 0.65;
  const Double_t ylength = builds.size() * 0.05;  // adjust size of legend for how many build routines we are plotting
  auto leg = new TLegend(x1, y1, x1 + 0.25, y1 + ylength);
  leg->SetBorderSize(0);

  // setup tgraphs
  TGEVec graphs(nbuilds);
  PlotBenchmarks::GetGraphs(graphs, text, title, xtitle, ytitle);

  // get tgraphs for intrinsic plot
  TGEVec graphs_int(nbuilds);
  if (isVU)
    PlotBenchmarks::GetGraphs(graphs_int, text + "_int", title, xtitle, ytitle);

  // Draw graphs
  for (auto i = 0U; i < nbuilds; i++) {
    auto& graph = graphs[i];
    auto& graph_int = graphs_int[i];
    auto& build = builds[i];

    // draph if graph exists
    if (graph) {
      graph->GetXaxis()->SetRangeUser(xmin, xmax);
      graph->GetYaxis()->SetRangeUser(ymin, ymax);
      graph->Draw(i > 0 ? "LP SAME" : "ALP");

      // add point for VU with intrinsics
      if (isVU && graph_int) {
        graph_int->GetXaxis()->SetRangeUser(xmin, xmax);
        graph_int->GetYaxis()->SetRangeUser(ymin, ymax);
        graph_int->Draw("P SAME");
      }

      // add to legend
      leg->AddEntry(graph, build.label.Data(), "LP");
    }
  }

  // Draw speedup line
  TF1* scaling = NULL;
  if (isSpeedup) {
    scaling = new TF1("ideal_scaling",
                      "x",
                      (isVU ? arch_opt.vumin : arch_opt.thmin),
                      (isVU ? arch_opt.vuspeedupmax : arch_opt.thspeedupmax));
    scaling->SetLineColor(kBlack);
    scaling->SetLineStyle(kDashed);
    scaling->SetLineWidth(2);
    scaling->Draw("SAME");
    leg->AddEntry(scaling, "Ideal Scaling", "l");
  }

  // Draw legend last
  leg->Draw("SAME");

  // Save the png
  const TString outname = arch + "_" + sample + "_" + text;
  canv->SaveAs(outname + ".png");

  // Save log-x version
  canv->SetLogx();
  for (auto i = 0U; i < nbuilds; i++) {
    auto& graph = graphs[i];
    auto& graph_int = graphs_int[i];

    // need to reset range with logx
    if (graph) {
      graph->GetXaxis()->SetRangeUser(xmin, xmax);
      graph->GetYaxis()->SetRangeUser(ymin, ymax);
      if (isVU && graph_int) {
        graph_int->GetXaxis()->SetRangeUser(xmin, xmax);
        graph_int->GetYaxis()->SetRangeUser(ymin, ymax);
      }
    }
  }
  canv->Update();
  canv->SaveAs(outname + "_logx.png");

  // delete everything
  for (auto i = 0U; i < nbuilds; i++) {
    delete graphs[i];
    if (isVU)
      delete graphs_int[i];
  }
  if (isSpeedup)
    delete scaling;
  delete leg;
  delete canv;
}

void PlotBenchmarks::GetGraphs(
    TGEVec& graphs, const TString& text, const TString& title, const TString& xtitle, const TString& ytitle) {
  // special setup for intrinsic only plot
  const Bool_t isInt = text.Contains("_int", TString::kExact);

  for (auto i = 0U; i < nbuilds; i++) {
    const auto& build = builds[i];
    auto& graph = graphs[i];

    // get graph
    graph = (TGraphErrors*)file->Get("g_" + build.name + "_" + text);

    // restyle if graph exists
    if (graph) {
      graph->SetTitle(title + ";" + xtitle + ";" + ytitle);

      graph->SetLineWidth(2);
      graph->SetLineColor(build.color);
      graph->SetMarkerStyle(isInt ? kOpenCircle : kFullCircle);
      graph->SetMarkerColor(build.color);
    }
  }
}
