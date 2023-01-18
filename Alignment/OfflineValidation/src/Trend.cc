#include <cassert>  // you may want to replace all assertions by clear error messages
#include <cstdlib>

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "TLine.h"
#include "TLatex.h"
#include "TString.h"
#include "TGraphErrors.h"
#include "TH1.h"

#include "Alignment/OfflineValidation/plugins/ColorParser.C"
#include "Alignment/OfflineValidation/plugins/StyleParser.C"
#include "Alignment/OfflineValidation/interface/Trend.h"

using namespace std;
namespace fs = boost::filesystem;
namespace pt = boost::property_tree;

Run2Lumi::Run2Lumi(fs::path file, int first, int last, float convertUnit)
    : firstRun(first), lastRun(last), convertUnit(convertUnit) {
  assert(first < last);

  assert(fs::exists(file));

  ifstream f(file);
  int run;
  while (f >> run >> runs[run])
    ;
  f.close();
}

float Run2Lumi::operator()(int run1, int run2) const

{
  float sum = 0.;
  for (auto& run : runs) {
    if (run.first < run1)
      continue;
    if (run.first >= run2)
      break;
    sum += run.second;
  }
  return sum / convertUnit;  // conversion from e.g. /pb to /fb
}

float Run2Lumi::operator()(int run) const { return operator()(firstRun, run); }

float Run2Lumi::operator()() const { return operator()(firstRun, lastRun); }

template <typename T>
void CopyStyle(T* objIn, T* objOut) {
  objOut->SetLineColor(objIn->GetLineColor());
  objOut->SetMarkerColor(objIn->GetMarkerColor());
  objOut->SetFillColorAlpha(objIn->GetFillColor(), 0.2);  // TODO??

  objOut->SetLineStyle(objIn->GetLineStyle());
  objOut->SetMarkerStyle(objIn->GetMarkerStyle());
  objOut->SetFillStyle(objIn->GetFillStyle());

  objOut->SetLineWidth(objIn->GetLineWidth());
  objOut->SetMarkerSize(objIn->GetMarkerSize());
}

TGraph* Run2Lumi::operator()(TGraph* gIn) const {
  assert(gIn != nullptr);

  vector<float> x, y, ex, ey;
  int n = gIn->GetN();
  for (int i = 0; i < n - 1; ++i) {
    int currentRun = gIn->GetPointX(i);
    if (currentRun < firstRun)
      continue;
    if (currentRun >= lastRun)
      break;

    int nextRun = gIn->GetPointX(i + 1);

    auto lumi_edge = operator()(firstRun, currentRun), lumi_width = operator()(currentRun, nextRun);
    x.push_back(lumi_edge + lumi_width / 2);
    ex.push_back(lumi_width / 2);

    auto point = gIn->GetPointY(i), error = gIn->GetErrorY(i);
    y.push_back(point);
    ey.push_back(error);
  }

  auto N = x.size();
  assert(N == y.size() && N == ex.size() && N == ey.size());
  TGraph* gOut = new TGraphErrors(N, x.data(), y.data(), ex.data(), ey.data());
  gOut->SetTitle(gIn->GetTitle());
  CopyStyle(gIn, gOut);
  return gOut;
}

TH1* Run2Lumi::operator()(TH1* hIn) const {
  vector<float> edges, contents, errors;
  edges.push_back(0);
  int N = hIn->GetNbinsX();
  for (int i = 1; i <= N; ++i) {
    auto nextRun = hIn->GetBinLowEdge(i + 1);
    if (nextRun < firstRun)
      continue;
    if (nextRun >= lastRun)
      break;

    edges.push_back(operator()(nextRun));

    auto content = hIn->GetBinContent(i), error = hIn->GetBinError(i);
    contents.push_back(content);
    errors.push_back(error);
  }

  N = edges.size() - 1;
  TString name = hIn->GetName();
  name += "_byLumi";
  TH1* hOut = new TH1F(name, hIn->GetTitle(), N, edges.data());
  for (int i = 1; i <= N; ++i) {
    hOut->SetBinContent(i, contents[i - 1]);
    hOut->SetBinError(i, errors[i - 1]);
  }
  CopyStyle(hIn, hOut);
  return hOut;
}

Trend::Trend(const char* name,
             const char* dir,
             const char* title,
             const char* ytitle,
             float ymin,
             float ymax,
             pt::ptree& json,
             const Run2Lumi& GetLumiFunctor,
             const char* lumiAxisType)
    : c(name, title, 2000, 800),
      outputDir(Form("%s", dir)),
      lgd(0.7, 0.65, 0.97, 0.89, "", "NDC"),
      JSON(json),
      GetLumi(GetLumiFunctor),
      lumiType(lumiAxisType) {
  if (JSON.count("CMSlabel"))
    CMS = Form("#scale[1.1]{#bf{CMS}} #it{%s}", JSON.get<string>("CMSlabel").data());

  if (JSON.get_child("trends").count("TitleCanvas"))
    lumi = Form("#scale[0.8]{%s}", JSON.get_child("trends").get<string>("TitleCanvas").data());

  assert(ymin < ymax);
  float xmax = GetLumi(GetLumi.firstRun, GetLumi.lastRun);
  if (JSON.get_child("trends").count("plotUnit"))
    plotUnit = JSON.get_child("trends").get<string>("plotUnit");
  const char* axistitles = Form(";%s luminosity  [%s^{-1} ];%s", lumiType, plotUnit.c_str(), ytitle);
  auto frame = c.DrawFrame(0., ymin, xmax, ymax, axistitles);
  frame->GetYaxis()->SetTitleOffset(0.8);
  frame->GetYaxis()->SetTickLength(0.01);
  frame->GetXaxis()->SetLabelSize(fontsize);
  frame->GetXaxis()->SetTitleSize(fontsize);
  frame->GetYaxis()->SetLabelSize(fontsize);
  frame->GetYaxis()->SetTitleSize(fontsize);
  lgd.SetTextSize(fontsize);

  if (ymax > 0 && ymin < 0) {
    TLine l;
    l.SetLineColor(kBlack);
    l.SetLineStyle(kDashed);
    l.DrawLine(0., 0., xmax, 0.);
  }

  c.SetTicks(1, 1);
  c.SetRightMargin(0.015);
  c.SetLeftMargin(0.07);
  c.SetTopMargin(0.07);

  // plot vertical lines (typically pixel template transitions)
  pt::ptree lines = JSON.get_child("trends.lines");
  for (auto& type : lines) {
    auto line = type.second.get_child_optional("line");
    auto runs = type.second.get_child_optional("runs");
    if (!line || !runs)
      continue;

    auto v = new TLine;

    auto style = line->get_optional<string>("style");
    if (style)
      v->SetLineStyle(StyleParser(*style));

    auto color = line->get_optional<string>("color");
    if (color)
      v->SetLineColor(ColorParser(*color));

    auto width = line->get_optional<int>("width");
    if (width)
      v->SetLineWidth(*width);

    auto title = line->get_optional<string>("title");
    if (title)
      lgd.AddEntry(v, title->c_str(), "l");

    for (auto& run : *runs) {
      auto currentRun = run.second.get_value<int>();

      auto lumi = GetLumi(GetLumi.firstRun, currentRun);

      if (lumi > 0)
        v->DrawLine(lumi, ymin, lumi, ymax);
    }
  }
}

void Trend::operator()(TObject* obj, TString drawOpt, TString lgdOpt, bool fullRange) {
  c.cd();

  TString classname = obj->ClassName();
  if (classname.Contains("TGraph")) {
    auto g = dynamic_cast<TGraph*>(obj);
    int n = g->GetN();

    if (fullRange) {
      g->Set(n);
      g->SetPoint(n, GetLumi.lastRun, 0);
    }
    g = GetLumi(g);
    g->Draw("same" + drawOpt);
  } else if (classname.Contains("TH1")) {
    auto h = dynamic_cast<TH1*>(obj);
    // TODO: full range?
    h = GetLumi(h);
    h->Draw("same" + drawOpt);
  } else {
    cerr << "No implementation for `" << classname << "`\n";
    exit(EXIT_FAILURE);
  }

  TString name = c.GetName();
  name.ReplaceAll("vs_run", "vs_lumi");
  c.SetName(name);

  TString title = obj->GetTitle();
  if (title == "")
    return;
  lgd.AddEntry(obj, "", lgdOpt);
}

Trend::~Trend() {
  c.cd();
  lgd.Draw();

  float l = c.GetLeftMargin(), t = c.GetTopMargin(), r = c.GetRightMargin(), lumiTextOffset = 0.2;

  TLatex latex;
  latex.SetNDC();
  latex.SetTextFont(42);

  latex.SetTextAlign(11);
  latex.DrawLatex(l, 1 - t + lumiTextOffset * t, CMS);

  latex.SetTextAlign(31);
  latex.DrawLatex(1 - r, 1 - t + lumiTextOffset * t, lumi);

  // plot labels
  latex.SetTextAlign(13);
  auto totLumi = GetLumi();
  assert(totLumi > 0);
  auto posY = 0.88;
  pt::ptree lines = JSON.get_child("trends.lines");
  for (auto& type : lines) {
    auto labels = type.second.get_child_optional("labels");
    auto runs = type.second.get_child_optional("runs");
    if (!labels || !runs)
      continue;

    auto runIt = runs->begin();
    auto labelIt = labels->begin();
    while (runIt != runs->end() && labelIt != labels->end()) {
      auto currentRun = runIt->second.get_value<int>();
      auto label = labelIt->second.get_value<string>();

      auto lumi = max(GetLumi(currentRun), (float)0.01);
      auto posX = l + (lumi / totLumi) / (l + 1 + r) + 0.02;

      label = "#scale[0.8]{" + label + "}";
      latex.DrawLatex(posX, posY, label.c_str());

      ++runIt;
      ++labelIt;
    }
    posY -= 0.06;
  }

  c.RedrawAxis();
  c.SaveAs(Form("%s/%s.pdf", outputDir, c.GetName()), Form("Title:%s", c.GetTitle()));
}
