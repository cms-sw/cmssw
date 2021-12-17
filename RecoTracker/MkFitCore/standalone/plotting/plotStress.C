#include "TString.h"
#include "TColor.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TCanvas.h"

#include <iostream>
#include <vector>

///////////////////////
// Structs for macro //
///////////////////////

struct setStruct {
  setStruct() {}
  setStruct(const TString& label, const Double_t x) : label(label), x(x) {}

  TString label;
  UInt_t x;
};

struct xyStruct {
  xyStruct() {}
  xyStruct(const Double_t x, const Double_t y) : x(x), y(y) {}

  Double_t x;
  Double_t y;
};

struct testStruct {
  testStruct() {}
  testStruct(const TString& label, const Color_t color) : label(label), color(color) {}

  TString label;
  Color_t color;

  std::vector<xyStruct> xyPoints;
  TGraph* graph;
};

////////////////
// Main Macro //
////////////////

void plotStress(const TString& infile_name, const TString& graph_label, const TString& outfile_name) {
  // no stats boxes
  gStyle->SetOptStat(0);

  // which tests to plot
  std::vector<testStruct> tests = {{"nTH1_nEV1", kBlue},
                                   {"nTH16_nEV16", kRed + 1},
                                   {"nTH32_nEV16", kGreen + 1},
                                   {"nTH32_nEV32", kMagenta},
                                   {"nTH64_nEV32", kOrange + 1},
                                   {"nTH64_nEV64", kBlack},
                                   {"nJOB32", kViolet - 1},
                                   {"nJOB64", kAzure + 10}};

  // which instruction sets (nVU) to use
  std::vector<setStruct> sets = {{"SSE3", 4}, {"AVX2", 8}, {"AVX512", 16}};

  // make label for x-axis
  const auto nset = sets.size();
  TString set_label;
  for (auto iset = 0U; iset < nset; iset++) {
    const auto& set = sets[iset];
    set_label += Form(" %s (x=%i)%s", set.label.Data(), set.x, (iset + 1 != nset ? "," : ""));
  }

  // read input file, fill testStruct vector
  std::ifstream input(infile_name.Data(), std::ios::in);
  TString test_set_label;
  Double_t y;

  // hacky read-in, but sufficient for small number of tests
  while (input >> test_set_label >> y) {
    for (auto& test : tests) {
      if (test_set_label.Contains(test.label)) {
        for (const auto& set : sets) {
          if (test_set_label.Contains(set.label)) {
            test.xyPoints.emplace_back(set.x, y);
            break;
          }  // end check over input label contains given instruction set label
        }    // end loop over instruction set labels
      }      // end check over input label contains given test label
    }        // end loop over instruction test labels
  }          // end loop over reading input file

  // setup canvas
  auto canv = new TCanvas();
  canv->cd();
  canv->SetTickx(1);
  canv->SetTicky(1);
  canv->SetGridy(1);

  // setup legend
  auto leg = new TLegend(0.77, 0.8, 0.99, 0.99);
  leg->SetNColumns(2);

  // loop tests, fill graphs, add to canvas + legend
  for (auto itest = 0U; itest < tests.size(); itest++) {
    // get test result
    auto& test = tests[itest];

    // get test info (points, label, color, graph)
    const auto& xyPoints = test.xyPoints;
    const auto& label = test.label;
    const auto color = test.color;
    auto& graph = test.graph;

    // make new graph, set style
    graph = new TGraph(test.xyPoints.size());
    graph->SetTitle("Time vs ISA Ext " + graph_label);
    graph->SetLineColor(color);
    graph->SetMarkerColor(color);
    graph->SetMarkerStyle(kFullCircle);
    graph->SetMarkerSize(1);

    // add graph points
    for (auto ixyPoint = 0U; ixyPoint < xyPoints.size(); ixyPoint++) {
      const auto& xyPoint = xyPoints[ixyPoint];
      graph->SetPoint(ixyPoint, xyPoint.x, xyPoint.y);
    }

    // draw graph
    graph->Draw(itest > 0 ? "CP SAME" : "ACP");

    // graphs can only set x-y axis info after being drawn
    graph->GetXaxis()->SetRangeUser(0, 20);
    graph->GetYaxis()->SetRangeUser(0, 0.2);
    graph->GetXaxis()->SetTitle("Floats in 1 vector [ISA Extensions: " + set_label + "]");
    graph->GetYaxis()->SetTitle("Time / evt / physical core [s]");

    // add graph to leg
    leg->AddEntry(graph, label.Data(), "lp");
  }

  // draw leg
  leg->Draw("same");

  // save it
  canv->SaveAs(outfile_name.Data());

  // delete it all
  for (auto& test : tests)
    delete test.graph;
  delete leg;
  delete canv;
}
