#ifndef SUSYCAF_GENERIC_PLOTTER_H
#define SUSYCAF_GENERIC_PLOTTER_H

#include <vector>
#include <string>
#include <TLegend.h>
namespace edm { class ParameterSet;}
class Book;
class TH1;
class TCanvas;

class GenericPlotter {
 public:
  GenericPlotter(const edm::ParameterSet&);
  const std::string printSuffix_, plotDocument_, drawOption2D_;
  const bool normalize_,legend_;
  const unsigned canvasWidth_,canvasHeight_;
  const std::vector<std::string> replace_text_;
  const double maxRatioUncertainty_;
  const double fixRatioYAxis_;

  void plot_all(std::vector<Book*>&, int reference = -1) const;
  void plot1D(std::string, std::vector<std::string>&, std::vector<TH1*>&, int reference = -1) const;
  void plot2D(std::string, std::vector<std::string>&, std::vector<TH1*>&) const;
  void plotRatio(bool, std::vector<std::string>&, std::vector<TH1*>&, int reference, TCanvas& c) const;

  TLegend* make_legend2D(const std::string, TH1*) const;
  TLegend make_legend(const std::vector<std::string>&, const std::vector<TH1*>&) const;
  void normalize(std::vector<TH1*>&) const;
  void setLabels(std::vector<TH1*>&) const;
  void setBounds(std::vector<TH1*>&, bool, bool) const;
  double hist_maximum(TH1*) const;
  double hist_minimum(TH1*) const;

  void plotDocumentOpen() const;
  void plotDocumentClose() const;
  void plotDocumentAdd(const TCanvas&) const;
  void printFile(const std::string&, const TCanvas&) const;

  // User must clean up the created ratio histograms.
  static void make_rebinned_ratios(std::vector<TH1*>& ratios, const std::vector<TH1*>& hist, int reference, double maxUncertainty, const std::string& refName, const std::string postfix = "_ratio");
  static double ratioError2(double numerator, double numeratorError2, double denominator, double denominatorError2);

};

#endif
