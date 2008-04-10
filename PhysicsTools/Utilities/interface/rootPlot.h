#ifndef PhysicsTools_Utilities_rootPlot_h
#define PhysicsTools_Utilities_rootPlot_h
#include "PhysicsTools/Utilities/interface/rootTf1.h"
#include "TF1.h"
#include "TCanvas.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"

namespace root {

  inline void plotTF1(const char * name, TF1 & fun, TH1 & histo, 
		      double min, double max,
		      Color_t lineColor = kRed, Width_t lineWidth = 1,
		      Style_t lineStyle = kDashed) {
    fun.SetLineColor(lineColor);
    fun.SetLineWidth(lineWidth);
    fun.SetLineStyle(lineStyle);
    TCanvas *canvas = new TCanvas("canvas");
    histo.Draw("e");
    fun.Draw("same");	
    std::string plotName = name;
    canvas->SaveAs(plotName.c_str());
    canvas->SetLogy();
    std::string logPlotName = "log_" + plotName;
    canvas->SaveAs(logPlotName.c_str());
  }

  template<typename F>
  void plot(const char * name, TH1 & histo, F& f, double min, double max,
	    Color_t lineColor = kRed, Width_t lineWidth = 1,
	    Style_t lineStyle = kDashed) {
    TF1 fun = root::tf1("fun", f, min, max);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle);
  }

  template<typename F>
  void plot(const char * name, TH1 & histo, F& f, double min, double max,
	    const funct::Parameter & p0,
	    Color_t lineColor = kRed, Width_t lineWidth = 1,
	    Style_t lineStyle = kDashed) {
    TF1 fun = root::tf1("fun", f, min, max, p0);
    fun.SetParNames(p0.name().c_str());
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle);
  }

  template<typename F>
  void plot(const char * name, TH1 & histo, F& f, double min, double max,
	    const funct::Parameter & p0, 
	    const funct::Parameter & p1,
	    Color_t lineColor = kRed, Width_t lineWidth = 1,
	    Style_t lineStyle = kDashed) {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1);
    fun.SetParNames(p0.name().c_str(), p1.name().c_str());
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle);
  }

  template<typename F>
  void plot(const char * name, TH1 & histo, F& f, double min, double max,
	    const funct::Parameter & p0, 
	    const funct::Parameter & p1,
	    const funct::Parameter & p2,
	    Color_t lineColor = kRed, Width_t lineWidth = 1,
	    Style_t lineStyle = kDashed) {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2);
    fun.SetParNames(p0.name().c_str(), p1.name().c_str(), p2.name().c_str());
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle);
  }


}

#endif
