#ifndef PhysicsTools_Utilities_rootPlot_h
#define PhysicsTools_Utilities_rootPlot_h
#include "PhysicsTools/Utilities/interface/rootTf1.h"
#include "TF1.h"
#include "TH1.h"
#include "TCanvas.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"

namespace root {
  namespace helper {
    struct PlotNoArg {};
  }  // namespace helper

  inline void plotTF1(const char* name,
                      TF1& fun0,
                      TF1& fun1,
                      TH1& histo,
                      double min,
                      double max,
                      Color_t lineColor0 = kRed,
                      Width_t lineWidth0 = 1,
                      Style_t lineStyle0 = kDashed,
                      Int_t npx0 = 1000,
                      Color_t lineColor1 = kGreen,
                      Width_t lineWidth1 = 1,
                      Style_t lineStyle1 = kDashed,
                      Int_t npx1 = 1000,
                      const char* title = "Histo Title",
                      const char* xTitle = "X Title",
                      const char* yTitle = "Y Title") {
    fun0.SetLineColor(lineColor0);
    fun0.SetLineWidth(lineWidth0);
    fun0.SetLineStyle(lineStyle0);
    fun0.SetNpx(npx0);
    fun1.SetLineColor(lineColor1);
    fun1.SetLineWidth(lineWidth1);
    fun1.SetLineStyle(lineStyle1);
    fun1.SetNpx(npx1);
    TCanvas* canvas = new TCanvas("canvas");
    histo.SetTitle(title);
    histo.SetXTitle(xTitle);
    histo.SetYTitle(yTitle);
    histo.Draw("e");
    fun0.Draw("same");
    fun1.Draw("same");
    std::string plotName = name;
    canvas->SaveAs(plotName.c_str());
    canvas->SetLogy();
    std::string logPlotName = "log_" + plotName;
    canvas->SaveAs(logPlotName.c_str());
  }

  inline void plotTF1(const char* name,
                      TF1& fun,
                      TH1& histo,
                      double min,
                      double max,
                      Color_t lineColor = kRed,
                      Width_t lineWidth = 1,
                      Style_t lineStyle = kDashed,
                      Int_t npx = 1000,
                      const char* title = "Histo Title",
                      const char* xTitle = "X Title",
                      const char* yTitle = "Y Title") {
    fun.SetLineColor(lineColor);
    fun.SetLineWidth(lineWidth);
    fun.SetLineStyle(lineStyle);
    fun.SetNpx(npx);
    TCanvas* canvas = new TCanvas("canvas");
    histo.SetTitle(title);
    histo.SetXTitle(xTitle);
    histo.SetYTitle(yTitle);
    histo.Draw("e");
    fun.Draw("same");
    std::string plotName = name;
    canvas->SaveAs(plotName.c_str());
    canvas->SetLogy();
    std::string logPlotName = "log_" + plotName;
    canvas->SaveAs(logPlotName.c_str());
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1("fun", f, min, max);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1("fun", f, min, max, p0);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3, p4);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4,
            const funct::Parameter& p5,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3, p4, p5);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4,
            const funct::Parameter& p5,
            const funct::Parameter& p6,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3, p4, p5, p6);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4,
            const funct::Parameter& p5,
            const funct::Parameter& p6,
            const funct::Parameter& p7,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3, p4, p5, p6, p7);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4,
            const funct::Parameter& p5,
            const funct::Parameter& p6,
            const funct::Parameter& p7,
            const funct::Parameter& p8,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4,
            const funct::Parameter& p5,
            const funct::Parameter& p6,
            const funct::Parameter& p7,
            const funct::Parameter& p8,
            const funct::Parameter& p9,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4,
            const funct::Parameter& p5,
            const funct::Parameter& p6,
            const funct::Parameter& p7,
            const funct::Parameter& p8,
            const funct::Parameter& p9,
            const funct::Parameter& p10,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4,
            const funct::Parameter& p5,
            const funct::Parameter& p6,
            const funct::Parameter& p7,
            const funct::Parameter& p8,
            const funct::Parameter& p9,
            const funct::Parameter& p10,
            const funct::Parameter& p11,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4,
            const funct::Parameter& p5,
            const funct::Parameter& p6,
            const funct::Parameter& p7,
            const funct::Parameter& p8,
            const funct::Parameter& p9,
            const funct::Parameter& p10,
            const funct::Parameter& p11,
            const funct::Parameter& p12,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4,
            const funct::Parameter& p5,
            const funct::Parameter& p6,
            const funct::Parameter& p7,
            const funct::Parameter& p8,
            const funct::Parameter& p9,
            const funct::Parameter& p10,
            const funct::Parameter& p11,
            const funct::Parameter& p12,
            const funct::Parameter& p13,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4,
            const funct::Parameter& p5,
            const funct::Parameter& p6,
            const funct::Parameter& p7,
            const funct::Parameter& p8,
            const funct::Parameter& p9,
            const funct::Parameter& p10,
            const funct::Parameter& p11,
            const funct::Parameter& p12,
            const funct::Parameter& p13,
            const funct::Parameter& p14,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4,
            const funct::Parameter& p5,
            const funct::Parameter& p6,
            const funct::Parameter& p7,
            const funct::Parameter& p8,
            const funct::Parameter& p9,
            const funct::Parameter& p10,
            const funct::Parameter& p11,
            const funct::Parameter& p12,
            const funct::Parameter& p13,
            const funct::Parameter& p14,
            const funct::Parameter& p15,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4,
            const funct::Parameter& p5,
            const funct::Parameter& p6,
            const funct::Parameter& p7,
            const funct::Parameter& p8,
            const funct::Parameter& p9,
            const funct::Parameter& p10,
            const funct::Parameter& p11,
            const funct::Parameter& p12,
            const funct::Parameter& p13,
            const funct::Parameter& p14,
            const funct::Parameter& p15,
            const funct::Parameter& p16,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4,
            const funct::Parameter& p5,
            const funct::Parameter& p6,
            const funct::Parameter& p7,
            const funct::Parameter& p8,
            const funct::Parameter& p9,
            const funct::Parameter& p10,
            const funct::Parameter& p11,
            const funct::Parameter& p12,
            const funct::Parameter& p13,
            const funct::Parameter& p14,
            const funct::Parameter& p15,
            const funct::Parameter& p16,
            const funct::Parameter& p17,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun =
        root::tf1("fun", f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4,
            const funct::Parameter& p5,
            const funct::Parameter& p6,
            const funct::Parameter& p7,
            const funct::Parameter& p8,
            const funct::Parameter& p9,
            const funct::Parameter& p10,
            const funct::Parameter& p11,
            const funct::Parameter& p12,
            const funct::Parameter& p13,
            const funct::Parameter& p14,
            const funct::Parameter& p15,
            const funct::Parameter& p16,
            const funct::Parameter& p17,
            const funct::Parameter& p18,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1(
        "fun", f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4,
            const funct::Parameter& p5,
            const funct::Parameter& p6,
            const funct::Parameter& p7,
            const funct::Parameter& p8,
            const funct::Parameter& p9,
            const funct::Parameter& p10,
            const funct::Parameter& p11,
            const funct::Parameter& p12,
            const funct::Parameter& p13,
            const funct::Parameter& p14,
            const funct::Parameter& p15,
            const funct::Parameter& p16,
            const funct::Parameter& p17,
            const funct::Parameter& p18,
            const funct::Parameter& p19,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1(
        "fun", f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const std::vector<funct::Parameter>& p,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1("fun", f, min, max, p);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }

  template <typename F>
  void plot(const char* name,
            TH1& histo,
            F& f,
            double min,
            double max,
            const std::vector<std::shared_ptr<double> >& p,
            Color_t lineColor = kRed,
            Width_t lineWidth = 1,
            Style_t lineStyle = kDashed,
            Int_t npx = 1000,
            const char* title = "Histo Title",
            const char* xTitle = "X Title",
            const char* yTitle = "Y Title") {
    TF1 fun = root::tf1("fun", f, min, max, p);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx, title, xTitle, yTitle);
  }
}  // namespace root

#endif
