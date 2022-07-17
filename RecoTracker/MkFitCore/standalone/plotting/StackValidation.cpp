#include "StackValidation.hh"

StackValidation::StackValidation(const TString& label,
                                 const TString& extra,
                                 const Bool_t cmsswComp,
                                 const TString& suite)
    : label(label), extra(extra), cmsswComp(cmsswComp), suite(suite) {
  // setup style for plotting
  setupStyle();

  // setup suite enum
  setupSUITEEnum(suite);

  // setup build options : true for isBenchmark-type plots, and include cmssw in plots if simval
  setupBuilds(false, !cmsswComp);

  // set legend y height
  y1 = 0.80;
  y2 = y1 + nbuilds * 0.04;  // full + CMSSW = 0.04*5 + 0.8 = 1.0

  // open the files
  files.resize(nbuilds);
  for (auto b = 0U; b < nbuilds; b++) {
    const auto& build = builds[b];
    auto& file = files[b];

    file = TFile::Open("validation_" + label + "_" + build.name + extra + "/plots.root");
  }

  // setup ref
  setupRef(cmsswComp);

  // setup rates
  setupRates(cmsswComp);

  // setup ptcuts
  setupPtCuts();
}

StackValidation::~StackValidation() {
  for (auto& file : files)
    delete file;
}

void StackValidation::MakeValidationStacks() {
  StackValidation::MakeRatioStacks("build");
  StackValidation::MakeKinematicDiffStacks("build");
  StackValidation::MakeQualityStacks("build");

  if (cmsswComp) {
    StackValidation::MakeRatioStacks("fit");
    StackValidation::MakeKinematicDiffStacks("fit");
    StackValidation::MakeQualityStacks("fit");
  }
}

void StackValidation::MakeRatioStacks(const TString& trk) {
  // kinematic variables to plot
  std::vector<TString> vars = {"pt", "eta", "phi", "nLayers"};
  const UInt_t nvars = vars.size();

  // indices for loops match PlotValidation.cpp
  for (auto l = 0U; l < nrates; l++) {
    const auto& rate = rates[l];

    for (auto k = 0U; k < nptcuts; k++) {
      const auto& ptcut = ptcuts[k];

      for (auto i = 0U; i < nvars; i++) {
        const auto& var = vars[i];

        auto canv = new TCanvas();
        canv->cd();

        auto leg = new TLegend(0.85, y1, 1.0, y2);

        // tmp axis titles, not sure why ROOT is deleting them
        TString xtitle = "";
        TString ytitle = "";

        std::vector<TGraphAsymmErrors*> graphs(nbuilds);

        for (auto b = 0U; b < nbuilds; b++) {
          const auto& build = builds[b];
          auto& file = files[b];
          auto& graph = graphs[b];

          graph = ((TEfficiency*)file->Get(rate.dir + refdir + "/" + rate.rate + "_" + rate.sORr + "_" + var + "_" +
                                           trk + "_pt" + ptcut))
                      ->CreateGraph();
          graph->SetLineColor(build.color);
          graph->SetMarkerColor(build.color);

          // store tmp titles
          if (b == 0) {
            xtitle = graph->GetXaxis()->GetTitle();
            ytitle = graph->GetYaxis()->GetTitle();
          }

          graph->Draw(b > 0 ? "PZ SAME" : "APZ");

          if (!rate.rate.Contains("ineff", TString::kExact) && !rate.rate.Contains("dr", TString::kExact))
            graph->GetYaxis()->SetRangeUser(0.0, 1.05);
          else
            graph->GetYaxis()->SetRangeUser(0.0, 0.25);

          leg->AddEntry(graph, build.label.Data(), "LEP");
        }

        // print standard plot for every rate/variable
        leg->Draw("SAME");
        canv->SaveAs(label + "_" + rate.rate + "_" + var + "_" + trk + "_pt" + ptcut + extra + ".png");

        // zoom in on pt range
        if (i == 0) {
          std::vector<TGraphAsymmErrors*> zoomgraphs(nbuilds);
          for (auto b = 0U; b < nbuilds; b++) {
            auto& graph = graphs[b];
            auto& zoomgraph = zoomgraphs[b];

            zoomgraph = (TGraphAsymmErrors*)graph->Clone(Form("%s_zoom", graph->GetName()));
            zoomgraph->GetXaxis()->SetRangeUser(0, 10);
            zoomgraph->Draw(b > 0 ? "PZ SAME" : "APZ");
          }

          leg->Draw("SAME");
          canv->SaveAs(label + "_" + rate.rate + "_" + var + "_zoom_" + trk + "_pt" + ptcut + extra + ".png");

          for (auto& zoomgraph : zoomgraphs)
            delete zoomgraph;
        }

        // make logx plots for pt: causes LOTS of weird effects... workarounds for now
        if (i == 0) {
          canv->SetLogx(1);

          // apparently logx removes titles and ranges???
          for (auto b = 0U; b < nbuilds; b++) {
            auto& graph = graphs[b];
            graph->GetXaxis()->SetRangeUser(0.01, graph->GetXaxis()->GetBinUpEdge(graph->GetXaxis()->GetNbins()));

            if (!rate.rate.Contains("ineff", TString::kExact) && !rate.rate.Contains("dr", TString::kExact))
              graph->GetYaxis()->SetRangeUser(0.0, 1.05);
            else
              graph->GetYaxis()->SetRangeUser(0.0, 0.25);

            graph->GetXaxis()->SetTitle(xtitle);
            graph->GetYaxis()->SetTitle(ytitle);

            graph->Draw(b > 0 ? "PZ SAME" : "APZ");
          }

          leg->Draw("SAME");
          canv->SaveAs(label + "_" + rate.rate + "_" + var + "_logx_" + trk + "_pt" + ptcut + extra + ".png");
        }

        delete leg;
        for (auto& graph : graphs)
          delete graph;
        delete canv;
      }
    }
  }
}

void StackValidation::MakeKinematicDiffStacks(const TString& trk) {
  // variables to plot
  std::vector<TString> diffs = {"nHits", "invpt", "eta", "phi"};
  const UInt_t ndiffs = diffs.size();

  // diffferent reco collections
  std::vector<TString> colls = {"allmatch", "bestmatch"};
  const UInt_t ncolls = colls.size();

  // indices for loops match PlotValidation.cpp
  for (auto o = 0U; o < ncolls; o++) {
    const auto& coll = colls[o];

    for (auto p = 0U; p < ndiffs; p++) {
      const auto& diff = diffs[p];

      for (auto k = 0U; k < nptcuts; k++) {
        const auto& ptcut = ptcuts[k];

        const Bool_t isLogy = true;
        auto canv = new TCanvas();
        canv->cd();
        canv->SetLogy(isLogy);

        auto leg = new TLegend(0.85, y1, 1.0, y2);

        // tmp min/max
        Double_t min = 1e9;
        Double_t max = -1e9;

        std::vector<TH1F*> hists(nbuilds);
        for (auto b = 0U; b < nbuilds; b++) {
          const auto& build = builds[b];
          auto& file = files[b];
          auto& hist = hists[b];

          hist = (TH1F*)file->Get("kindiffs" + refdir + "/h_d" + diff + "_" + coll + "_" + trk + "_pt" + ptcut);
          hist->SetLineColor(build.color);
          hist->SetMarkerColor(build.color);

          hist->Scale(1.f / hist->Integral());
          hist->GetYaxis()->SetTitle("Fraction of Tracks");

          GetMinMaxHist(hist, min, max);
        }

        for (auto b = 0U; b < nbuilds; b++) {
          const auto& build = builds[b];
          auto& hist = hists[b];

          SetMinMaxHist(hist, min, max, isLogy);
          hist->Draw(b > 0 ? "EP SAME" : "EP");

          const TString mean = Form("%4.1f", hist->GetMean());
          leg->AddEntry(hist, build.label + " " + " [#mu = " + mean + "]", "LEP");
        }

        leg->Draw("SAME");
        canv->SaveAs(label + "_" + coll + "_d" + diff + "_" + trk + "_pt" + ptcut + extra + ".png");

        delete leg;
        for (auto& hist : hists)
          delete hist;
        delete canv;
      }  // end pt cut loop
    }    // end var loop
  }      // end coll loop
}

void StackValidation::MakeQualityStacks(const TString& trk) {
  // diffferent reco collections
  std::vector<TString> colls = {"allreco", "fake", "allmatch", "bestmatch"};
  const UInt_t ncolls = colls.size();

  // quality plots to use: nHits/track and track score
  std::vector<TString> quals = {"nHits", "score"};
  const UInt_t nquals = quals.size();

  // indices for loops match PlotValidation.cpp
  for (auto o = 0U; o < ncolls; o++) {
    const auto& coll = colls[o];

    for (auto k = 0U; k < nptcuts; k++) {
      const auto& ptcut = ptcuts[k];

      for (auto n = 0U; n < nquals; n++) {
        const auto& qual = quals[n];

        const Bool_t isLogy = true;
        auto canv = new TCanvas();
        canv->cd();
        canv->SetLogy(isLogy);

        auto leg = new TLegend(0.85, y1, 1.0, y2);

        // tmp min/max
        Double_t min = 1e9;
        Double_t max = -1e9;

        std::vector<TH1F*> hists(nbuilds);
        for (auto b = 0U; b < nbuilds; b++) {
          const auto& build = builds[b];
          auto& file = files[b];
          auto& hist = hists[b];

          hist = (TH1F*)file->Get("quality" + refdir + "/h_" + qual + "_" + coll + "_" + trk + "_pt" + ptcut);
          hist->SetLineColor(build.color);
          hist->SetMarkerColor(build.color);

          hist->Scale(1.f / hist->Integral());
          hist->GetYaxis()->SetTitle("Fraction of Tracks");

          GetMinMaxHist(hist, min, max);
        }

        for (auto b = 0U; b < nbuilds; b++) {
          const auto& build = builds[b];
          auto& hist = hists[b];

          SetMinMaxHist(hist, min, max, isLogy);
          hist->Draw(b > 0 ? "EP SAME" : "EP");

          const TString mean = Form("%4.1f", hist->GetMean());
          leg->AddEntry(hist, build.label + " " + " [#mu = " + mean + "]", "LEP");
        }

        leg->Draw("SAME");
        canv->SaveAs(label + "_" + coll + "_" + qual + "_" + trk + "_pt" + ptcut + extra + ".png");

        delete leg;
        for (auto& hist : hists)
          delete hist;
        delete canv;

      }  // end loop over quality variable
    }    // end pt cut loop
  }      // end coll loop
}
