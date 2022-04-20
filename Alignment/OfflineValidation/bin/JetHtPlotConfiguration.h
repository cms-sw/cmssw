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
#include <string>
#include <regex>

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
  
  // Other constants
  static const int kMaxLegendColumns = 3;       // Maximum allowed number of colums in a legend
  
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
  bool GetDrawReferenceProfile() const;
  bool GetDrawCentralEtaSummaryProfile() const;
  double GetProfileZoomLow(const int iProfile) const;
  double GetProfileZoomHigh(const int iProfile) const;
  bool GetDrawTrend(const int iTrend) const;
  double GetTrendZoomLow(const int iTrend) const;
  double GetTrendZoomHigh(const int iTrend) const;
  
  double GetProfileLegendShiftTotalX() const;
  double GetProfileLegendShiftTotalY() const;
  double GetProfileLegendShiftColumnX(const int iColumn) const;
  double GetProfileLegendShiftColumnY(const int iColumn) const;
  double GetProfileLegendTextSize() const;
  int GetProfileLegendTextFont() const;
  
  double GetTrendLegendShiftTotalX() const;
  double GetTrendLegendShiftTotalY() const;
  double GetTrendLegendTextSize() const;
  int GetTrendLegendTextFont() const;
  
  bool GetDrawTrendTag() const;
  std::vector<std::string> GetTrendTagText() const;
  std::vector<double> GetTrendTagPositionX() const;
  std::vector<double> GetTrendTagPositionY() const;
  double GetTrendTagTextSize() const;
  int GetTrendTagTextFont() const;
  
  int GetTrendCanvasHeight() const;
  int GetTrendCanvasWidth() const;
  double GetTrendMarginLeft() const;
  double GetTrendMarginRight() const;
  double GetTrendMarginTop() const;
  double GetTrendMarginBottom() const;
  double GetTrendTitleOffsetX() const;
  double GetTrendTitleOffsetY() const;
  double GetTrendTitleSizeX() const;
  double GetTrendTitleSizeY() const;
  double GetTrendLabelOffsetX() const;
  double GetTrendLabelOffsetY() const;
  double GetTrendLabelSizeX() const;
  double GetTrendLabelSizeY() const;
  
  int GetNInputFiles() const;
  std::string GetInputFile(const int iFile) const;
  std::vector<std::string> GetInputFiles() const;
  
  std::string GetLegendComment(const int iComment) const;
  int GetMarkerColor(const int iFile) const;
  int GetMarkerStyle(const int iFile) const;
  int GetMarkerSize(const int iFile) const;
  bool GetCopyErrorColor(const int iFile) const;
  
  const char* GetLumiPerIovFile() const;
  const char* GetIovListMode() const;
  bool GetDrawYearLines() const;
  int GetYearLineColor() const;
  int GetYearLineWidth() const;
  int GetYearLineStyle() const;
  std::vector<int> GetRunsForLines() const;

  std::vector<double> GetWidePtBinBorders() const;
  
  std::string GetLegendTextForAllRuns() const;
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
  
  // Methods to expand environtental variables in the configuration
  void AutoExpandEnvironmentVariables(std::string &text) const;
  std::string ExpandEnvironmentVariables(const std::string &input) const;

  // Strings correcponding to configuration in the JSON file
  std::string fJsonTrackQAname = "drawTrackQA";
  std::string fJsonCategoryNameHistogram = "drawHistograms";
  std::string fJsonNameHistogram[knHistogramTypes] = {"drawDz", "drawDzError", "drawDxy", "drawDxyError"};
  std::string fJsonCategoryNameProfile = "drawProfiles";
  std::string fJsonNameProfile[knProfileTypes] = {"drawDzErrorVsPt", "drawDzErrorVsPhi", "drawDzErrorVsEta", "drawDxyErrorVsPt", "drawDxyErrorVsPhi", "drawDxyErrorVsEta", "drawDzErrorVsPtWide", "drawDxyErrorVsPtWide"};
  std::string fJsonNameReferenceProfile = "drawReferenceProfile";
  std::string fJsonNameCentralEtaSummaryProfile = "drawCentralEtaSummaryProfile";
  std::string fJsonCategoryNameProfileZoom = "profileZoom";
  std::string fJsonNameProfileZoom[knProfileTypes] = {"ZoomPtProfileDz", "ZoomPhiProfileDz", "ZoomEtaProfileDz", "ZoomPtProfileDxy", "ZoomPhiProfileDxy", "ZoomEtaProfileDxy", "ZoomPtWideProfileDz", "ZoomPtWideProfileDxy"};
  std::string fJsonCategoryNameTrend = "drawTrends";
  std::string fJsonNameTrend[knHistogramTypes] = {"drawDzError", "drawDxyError"};
  std::string fJsonCategoryNameTrendZoom = "trendZoom";
  std::string fJsonNameTrendZoom[knHistogramTypes] = {"ZoomDzTrend", "ZoomDxyTrend"};
  
  std::string fJsonNameLegendShiftTotalX = "legendShiftTotalX";
  std::string fJsonNameLegendShiftTotalY = "legendShiftTotalY";
  std::string fJsonNameLegendShiftColumnX = "legendShiftX";
  std::string fJsonNameLegendShiftColumnY = "legendShiftY";
  std::string fJsonNameLegendTextSize = "legendTextSize";
  std::string fJsonNameLegendTextFont = "legendTextFont";
  std::string fJsonNameLegendTextForAllRuns = "legendTextForAllRuns";
  
  std::string fJsonNameCanvasHeight = "canvasHeight";
  std::string fJsonNameCanvasWidth = "canvasWidth";
  std::string fJsonNameMarginLeft = "marginLeft";
  std::string fJsonNameMarginRight = "marginRight";
  std::string fJsonNameMarginTop = "marginTop";
  std::string fJsonNameMarginBottom = "marginBottom";
  std::string fJsonNameTitleOffsetX = "titleOffsetX";
  std::string fJsonNameTitleOffsetY = "titleOffsetY";
  std::string fJsonNameTitleSizeX = "titleSizeX";
  std::string fJsonNameTitleSizeY = "titleSizeY";
  std::string fJsonNameLabelOffsetX = "labelOffsetX";
  std::string fJsonNameLabelOffsetY = "labelOffsetY";
  std::string fJsonNameLabelSizeX = "labelSizeX";
  std::string fJsonNameLabelSizeY = "labelSizeY";
  
  std::string fJsonInputFileName = "inputFiles";
  std::string fJsonLegendComment = "legendText";
  std::string fJsonDrawPlotsForEachIOV = "drawPlotsForEachIOV";
  std::string fJsonNIovInOnePlot = "nIovInOnePlot";
  std::string fJsonUseLuminosityForTrends = "useLuminosityAxis";
  std::string fJsonSkipRunsWithNoData = "skipRunsWithNoData";
  std::string fJsonNormalizeQAplots = "normalizeQAplots";
  std::string fJsonSaveComment = "saveComment";
  std::string fJsonLumiPerIovFile = "lumiPerIovFile";
  std::string fJsonIovListMode = "lumiFileMode";
  std::string fJsonDrawYearLines = "drawYearLines";
  std::string fJsonYearLineColor = "yearLineColor";
  std::string fJsonYearLineWidth = "yearLineWidth";
  std::string fJsonYearLineStyle = "yearLineStyle";
  std::string fJsonRunsForLines = "runsForLines";
  std::string fJsonNameDrawTag = "drawTags";
  std::string fJsonNameTagInfo = "tagLabels";
  std::string fJsonNameTagTextSize = "tagTextSize";
  std::string fJsonNameTagTextFont = "tagTextFont";
  std::string fJsonWidePtBinBorders = "widePtBinBorders";
  std::string fJsonMakeIovListForSlides = "makeIovlistForSlides";
  std::string fJsonIovListForSlides = "iovListForSlides";
  
  int fDebugLevel;
  
  bool fDrawTrackQA;
  bool fDrawHistogram[knHistogramTypes];
  bool fDrawProfile[knProfileTypes];
  bool fDrawReferenceProfile;
  bool fDrawCentralEtaSummaryProfile;
  double fProfileLegendShiftTotalX;
  double fProfileLegendShiftTotalY;
  double fProfileLegendShiftColumnX[kMaxLegendColumns];
  double fProfileLegendShiftColumnY[kMaxLegendColumns];
  double fProfileLegendTextSize;
  int fProfileLegendTextFont;
  bool fDrawTrend[knTrendTypes];
  double fTrendLegendShiftTotalX;
  double fTrendLegendShiftTotalY;
  double fTrendLegendTextSize;
  int fTrendLegendTextFont;
  
  bool fDrawTrendTag;
  std::vector<std::string> fTrendTagText;
  std::vector<double> fTrendTagPositionX;
  std::vector<double> fTrendTagPositionY;
  double fTrendTagTextSize;
  int fTrendTagTextFont;
  
  int fTrendCanvasHeight;
  int fTrendCanvasWidth;
  double fTrendMarginLeft;
  double fTrendMarginRight;
  double fTrendMarginTop;
  double fTrendMarginBottom;
  double fTrendTitleOffsetX;
  double fTrendTitleOffsetY;
  double fTrendTitleSizeX;
  double fTrendTitleSizeY;
  double fTrendLabelOffsetX;
  double fTrendLabelOffsetY;
  double fTrendLabelSizeX;
  double fTrendLabelSizeY;
  
  double fProfileZoomLow[knProfileTypes];
  double fProfileZoomHigh[knProfileTypes];
  double fTrendZoomLow[knTrendTypes];
  double fTrendZoomHigh[knTrendTypes];
  
  std::vector<std::string> fInputFileNames;
  std::vector<std::string> fLegendComments;
  std::vector<int> fMarkerColor;
  std::vector<int> fMarkerStyle;
  std::vector<int> fMarkerSize;
  std::vector<bool> fCopyErrorColor;
  std::string fLegendTextForAllRuns;
  bool fDrawPlotsForEachIOV;
  int fNIovInOnePlot;
  bool fUseLuminosityForTrends;
  bool fSkipRunsWithNoData;
  bool fNormalizeQAplots;
  std::string fSaveComment;
  
  std::string fLumiPerIovFile;
  std::string fIovListMode;
  bool fDrawYearLines;
  int fYearLineColor;
  int fYearLineWidth;
  int fYearLineStyle;
  std::vector<int> fRunsForLines;
  std::vector<double> fWidePtBinBorders;
  
  bool fMakeIovListForSlides;
  std::string fIovListForSlides;
  
  // Default marker styles that are used if not configured in json
  int fDefaultColors[11] = {kBlue, kRed, kGreen+2, kMagenta, kBlack, kCyan, kViolet+3, kOrange, kPink-7, kSpring+3, kAzure-7};
  int fDefaultStyle = 20; // Full circle
  int fDefaultMarkerSize = 1;

};

#endif
