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
		      Style_t lineStyle = kDashed, Int_t npx = 1000) {
    fun.SetLineColor(lineColor);
    fun.SetLineWidth(lineWidth);
    fun.SetLineStyle(lineStyle);
    fun.SetNpx(npx);
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
	    Style_t lineStyle = kDashed, Int_t npx = 1000) {
    TF1 fun = root::tf1("fun", f, min, max);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx);
  }

  template<typename F>
  void plot(const char * name, TH1 & histo, F& f, double min, double max,
	    const funct::Parameter & p0,
	    Color_t lineColor = kRed, Width_t lineWidth = 1,
	    Style_t lineStyle = kDashed, Int_t npx = 1000) {
    TF1 fun = root::tf1("fun", f, min, max, p0);
    fun.SetParNames(p0.name().c_str());
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx);
  }

  template<typename F>
  void plot(const char * name, TH1 & histo, F& f, double min, double max,
	    const funct::Parameter & p0, 
	    const funct::Parameter & p1,
	    Color_t lineColor = kRed, Width_t lineWidth = 1,
	    Style_t lineStyle = kDashed, Int_t npx = 1000) {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1);
    fun.SetParNames(p0.name().c_str(), p1.name().c_str());
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx);
  }

  template<typename F>
  void plot(const char * name, TH1 & histo, F& f, double min, double max,
	    const funct::Parameter & p0, 
	    const funct::Parameter & p1,
	    const funct::Parameter & p2,
	    Color_t lineColor = kRed, Width_t lineWidth = 1,
	    Style_t lineStyle = kDashed, Int_t npx = 1000) {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2);
    fun.SetParNames(p0.name().c_str(), p1.name().c_str(), p2.name().c_str());
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx);
  }
  
  template<typename F>
  void plot(const char * name, TH1 & histo, F& f, double min, double max,
	    const funct::Parameter & p0, 
	    const funct::Parameter & p1,
	    const funct::Parameter & p2,
	    const funct::Parameter & p3,
	    Color_t lineColor = kRed, Width_t lineWidth = 1,
	    Style_t lineStyle = kDashed, Int_t npx = 1000) {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3);
    fun.SetParNames(p0.name().c_str(), p1.name().c_str(), p2.name().c_str(), p3.name().c_str());
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx);
  }
  
  template<typename F>
  void plot(const char * name, TH1 & histo, F& f, double min, double max,
	    const funct::Parameter & p0, 
	    const funct::Parameter & p1,
	    const funct::Parameter & p2,
	    const funct::Parameter & p3,
	    const funct::Parameter & p4,
	    Color_t lineColor = kRed, Width_t lineWidth = 1,
	    Style_t lineStyle = kDashed, Int_t npx = 1000) {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3, p4);
    fun.SetParNames(p0.name().c_str(), p1.name().c_str(), p2.name().c_str(), p3.name().c_str(), p4.name().c_str());
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx);
  }
  
  template<typename F>
  void plot(const char * name, TH1 & histo, F& f, double min, double max,
	    const funct::Parameter & p0, 
	    const funct::Parameter & p1,
	    const funct::Parameter & p2,
	    const funct::Parameter & p3,
	    const funct::Parameter & p4,
	    const funct::Parameter & p5,
	    Color_t lineColor = kRed, Width_t lineWidth = 1,
	    Style_t lineStyle = kDashed, Int_t npx = 1000) {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3, p4, p5);
    fun.SetParNames(p0.name().c_str(), p1.name().c_str(), p2.name().c_str(), p3.name().c_str(), p4.name().c_str(), 
		    p5.name().c_str());
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx);
  }
  
  template<typename F>
  void plot(const char * name, TH1 & histo, F& f, double min, double max,
	    const funct::Parameter & p0, 
	    const funct::Parameter & p1,
	    const funct::Parameter & p2,
	    const funct::Parameter & p3,
	    const funct::Parameter & p4,
	    const funct::Parameter & p5,
	    const funct::Parameter & p6,
	    Color_t lineColor = kRed, Width_t lineWidth = 1,
	    Style_t lineStyle = kDashed, Int_t npx = 1000) {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3, p4, p5, p6);
    fun.SetParNames(p0.name().c_str(), p1.name().c_str(), p2.name().c_str(), p3.name().c_str(), p4.name().c_str(), 
		    p5.name().c_str(), p6.name().c_str());
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx);
  }
  
  template<typename F>
  void plot(const char * name, TH1 & histo, F& f, double min, double max,
	    const funct::Parameter & p0, 
	    const funct::Parameter & p1,
	    const funct::Parameter & p2,
	    const funct::Parameter & p3,
	    const funct::Parameter & p4,
	    const funct::Parameter & p5,
	    const funct::Parameter & p6,
	    const funct::Parameter & p7,
	    Color_t lineColor = kRed, Width_t lineWidth = 1,
	    Style_t lineStyle = kDashed, Int_t npx = 1000) {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3, p4, p5, p6, p7);
    fun.SetParNames(p0.name().c_str(), p1.name().c_str(), p2.name().c_str(), p3.name().c_str(), p4.name().c_str(), 
		    p5.name().c_str(), p6.name().c_str(), p7.name().c_str());
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx);
  }
  
  template<typename F>
  void plot(const char * name, TH1 & histo, F& f, double min, double max,
	    const funct::Parameter & p0, 
	    const funct::Parameter & p1,
	    const funct::Parameter & p2,
	    const funct::Parameter & p3,
	    const funct::Parameter & p4,
	    const funct::Parameter & p5,
	    const funct::Parameter & p6,
	    const funct::Parameter & p7,
	    const funct::Parameter & p8,
	    Color_t lineColor = kRed, Width_t lineWidth = 1,
	    Style_t lineStyle = kDashed, Int_t npx = 1000) {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8);
    fun.SetParNames(p0.name().c_str(), p1.name().c_str(), p2.name().c_str(), p3.name().c_str(), p4.name().c_str(), 
		    p5.name().c_str(), p6.name().c_str(), p7.name().c_str(), p8.name().c_str());
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx);
  }
  
  template<typename F>
  void plot(const char * name, TH1 & histo, F& f, double min, double max,
	    const funct::Parameter & p0, 
	    const funct::Parameter & p1,
	    const funct::Parameter & p2,
	    const funct::Parameter & p3,
	    const funct::Parameter & p4,
	    const funct::Parameter & p5,
	    const funct::Parameter & p6,
	    const funct::Parameter & p7,
	    const funct::Parameter & p8,
	    const funct::Parameter & p9,
	    Color_t lineColor = kRed, Width_t lineWidth = 1,
	    Style_t lineStyle = kDashed, Int_t npx = 1000) {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
    fun.SetParNames(p0.name().c_str(), p1.name().c_str(), p2.name().c_str(), p3.name().c_str(), p4.name().c_str(), 
		    p5.name().c_str(), p6.name().c_str(), p7.name().c_str(), p8.name().c_str(), p9.name().c_str());
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx);
  }
  
  template<typename F>
  void plot(const char * name, TH1 & histo, F& f, double min, double max,
	    const std::vector<funct::Parameter> & p,
	    Color_t lineColor = kRed, Width_t lineWidth = 1,
	    Style_t lineStyle = kDashed, Int_t npx = 1000) {
    TF1 fun = root::tf1("fun", f, min, max, p);
    for(size_t i = 0; i < p.size(); ++i)
      fun.SetParName(i, p[i].name().c_str());  
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx);
  }
  
  template<typename F>
  void plot(const char * name, TH1 & histo, F& f, double min, double max,
	    const std::vector<boost::shared_ptr<double> > & p,
	    Color_t lineColor = kRed, Width_t lineWidth = 1,
	    Style_t lineStyle = kDashed, Int_t npx = 1000) {
    TF1 fun = root::tf1("fun", f, min, max, p);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle, npx);
  }
}

#endif
