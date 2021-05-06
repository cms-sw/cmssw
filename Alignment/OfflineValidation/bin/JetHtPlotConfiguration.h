// Class for histograms needed in the dijet analysis

#ifndef JETHTPLOTCONFIGURATION_H
#define JETHTPLOTCONFIGURATION_H

// Boost libraries for reading JSON files
#define BOOST_BIND_GLOBAL_PLACEHOLDERS  // This suppresses some warning message. It is annoying so just not seeing it makes me happy.
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string/replace.hpp>

// C++ includes
#include <iostream>
#include <vector>
#include <stdlib.h>

// Root includes
#include <TString.h>

class JetHtPlotConfiguration{

public:

  // Enumeration for histogram types
  enum enumHistogramType{kDz, kDzError, kDxy, kDxyError, knHistogramTypes};
  
  // Enumeration for profile types
  enum enumProfileType{kDzErrorVsPt, kDzErrorVsPhi, kDzErrorVsEta, kDxyErrorVsPt, kDxyErrorVsPhi, kDxyErrorVsEta, kDzErrorVsPtWide, kDxyErrorVsPtWide, knProfileTypes};
  
   // Enumeration for trend types
  enum enumTrendType{kDzErrorTrend, kDxyErrorTrend, knTrendTypes};
  
  // Constructors and destructor
  JetHtPlotConfiguration(); // Default constructor
  JetHtPlotConfiguration(const JetHtPlotConfiguration& in); // Copy constructor
  virtual ~JetHtPlotConfiguration(); // Destructor
  JetHtPlotConfiguration& operator=(const JetHtPlotConfiguration& obj); // Equal sign operator
  
  // Read a json configuration file
  void ReadJsonFile(const std::string fileName);
  
  // Getters for the configuration
  bool GetDrawTrackQA() const;
  bool GetDrawHistogram(const int iHistogram) const;
  bool GetDrawProfile(const int iProfile) const;
  double GetProfileZoomLow(const int iProfile) const;
  double GetProfileZoomHigh(const int iProfile) const;
  bool GetDrawTrend(const int iTrend) const;
  double GetTrendZoomLow(const int iTrend) const;
  double GetTrendZoomHigh(const int iTrend) const;
  
  int GetNInputFiles() const;
  std::string GetInputFile(const int iFile) const;
  std::vector<std::string> GetInputFiles() const;
  
  std::string GetLegendComment(const int iComment) const;
  int GetMarkerColor(const int iFile) const;
  int GetMarkerStyle(const int iFile) const;
  
  const char* GetLumiPerIovFile() const;
  bool GetDrawYearLines() const;
  std::vector<int> GetRunsForLines() const;

  std::vector<double> GetWidePtBinBorders() const;
  
  bool GetDrawPlotsForEachIOV() const;
  int GetNIovInOnePlot() const;
  bool GetUseLuminosityForTrends() const;
  bool GetSkipRunsWithNoData() const;
  bool GetNormalizeQAplots() const;
  
  const char* GetSaveComment() const;
  
  bool GetMakeIovListForSlides() const;
  const char* GetIovListForSlides() const;
  
  // Print the current configuration to console
  void PrintConfiguration() const;

private:
  
  // Strings correcponding to configuration in the JSON file
  std::string fJsonTrackQAname = "drawTrackQA";
  std::string fJsonCategoryNameHistogram = "drawHistograms";
  std::string fJsonNameHistogram[knHistogramTypes] = {"drawDz", "drawDzError", "drawDxy", "drawDxyError"};
  std::string fJsonCategoryNameProfile = "drawProfiles";
  std::string fJsonNameProfile[knProfileTypes] = {"drawDzErrorVsPt", "drawDzErrorVsPhi", "drawDzErrorVsEta", "drawDxyErrorVsPt", "drawDxyErrorVsPhi", "drawDxyErrorVsEta", "drawDzErrorVsPtWide", "drawDxyErrorVsPtWide"};
  std::string fJsonCategoryNameProfileZoom = "profileZoom";
  std::string fJsonNameProfileZoom[knProfileTypes] = {"ZoomPtProfileDz", "ZoomPhiProfileDz", "ZoomEtaProfileDz", "ZoomPtProfileDxy", "ZoomPhiProfileDxy", "ZoomEtaProfileDxy", "ZoomPtWideProfileDz", "ZoomPtWideProfileDxy"};
  std::string fJsonCategoryNameTrend = "drawTrends";
  std::string fJsonNameTrend[knHistogramTypes] = {"drawDzError", "drawDxyError"};
  std::string fJsonCategoryNameTrendZoom = "trendZoom";
  std::string fJsonNameTrendZoom[knHistogramTypes] = {"ZoomDzTrend", "ZoomDxyTrend"};
 
  std::string fJsonInputFileName = "inputFiles";
  std::string fJsonLegendComment = "legendText";
  std::string fJsonDrawPlotsForEachIOV = "drawPlotsForEachIOV";
  std::string fJsonNIovInOnePlot = "nIovInOnePlot";
  std::string fJsonUseLuminosityForTrends = "useLuminosityForTrends";
  std::string fJsonSkipRunsWithNoData = "skipRunsWithNoData";
  std::string fJsonNormalizeQAplots = "normalizeQAplots";
  std::string fJsonSaveComment = "saveComment";
  std::string fJsonLumiPerIovFile = "lumiPerIovFile";
  std::string fJsonDrawYearLines = "drawYearLines";
  std::string fJsonRunsForLines = "runsForLines";
  std::string fJsonWidePtBinBorders = "widePtBinBorders";
  std::string fJsonMakeIovListForSlides = "makeIovlistForSlides";
  std::string fJsonIovListForSlides = "iovListForSlides";
  
  int fDebugLevel;
  
  bool fDrawTrackQA;
  bool fDrawHistogram[knHistogramTypes];
  bool fDrawProfile[knProfileTypes];
  bool fDrawTrend[knTrendTypes];
  
  double fProfileZoomLow[knProfileTypes];
  double fProfileZoomHigh[knProfileTypes];
  double fTrendZoomLow[knTrendTypes];
  double fTrendZoomHigh[knTrendTypes];
  
  std::vector<std::string> fInputFileNames;
  std::vector<std::string> fLegendComments;
  std::vector<int> fMarkerColor;
  std::vector<int> fMarkerStyle;
  bool fDrawPlotsForEachIOV;
  int fNIovInOnePlot;
  bool fUseLuminosityForTrends;
  bool fSkipRunsWithNoData;
  bool fNormalizeQAplots;
  std::string fSaveComment;
  
  std::string fLumiPerIovFile;
  bool fDrawYearLines;
  std::vector<int> fRunsForLines;
  std::vector<double> fWidePtBinBorders;
  
  bool fMakeIovListForSlides;
  std::string fIovListForSlides;
  
  // Default values for colors and style
  int fDefaultColors[11] = {kBlue, kRed, kGreen+2, kMagenta, kBlack, kCyan, kViolet+3, kOrange, kPink-7, kSpring+3, kAzure-7};
  int fDefaultStyle = 20; // Full circle

};

#endif
