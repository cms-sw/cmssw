// Class for histograms needed in the dijet analysis

#ifndef ALIGNMENT_OFFLINEVALIDATION_JETHTPLOTCONFIGURATION_H
#define ALIGNMENT_OFFLINEVALIDATION_JETHTPLOTCONFIGURATION_H

// Boost libraries for reading JSON files
#define BOOST_BIND_GLOBAL_PLACEHOLDERS  // This suppresses some warning message. It is annoying so just not seeing it makes me happy.
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/algorithm/string/replace.hpp"

// C++ includes
#include <cstdlib>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

// Root includes
#include "TString.h"

class JetHtPlotConfiguration {
public:
  // Enumeration for histogram types
  enum enumHistogramType { kDz, kDzError, kDxy, kDxyError, knHistogramTypes };

  // Enumeration for profile types
  enum enumProfileType {
    kDzErrorVsPt,
    kDzErrorVsPhi,
    kDzErrorVsEta,
    kDxyErrorVsPt,
    kDxyErrorVsPhi,
    kDxyErrorVsEta,
    kDzErrorVsPtWide,
    kDxyErrorVsPtWide,
    knProfileTypes
  };

  // Enumeration for trend types
  enum enumTrendType { kDzErrorTrend, kDxyErrorTrend, knTrendTypes };

  // Other constants
  static const int kMaxLegendColumns = 3;  // Maximum allowed number of colums in a legend

  // Constructors and destructor
  JetHtPlotConfiguration();                                              // Default constructor
  JetHtPlotConfiguration(const JetHtPlotConfiguration& in);              // Copy constructor
  virtual ~JetHtPlotConfiguration();                                     // Destructor
  JetHtPlotConfiguration& operator=(const JetHtPlotConfiguration& obj);  // Equal sign operator

  // Read a json configuration file
  void readJsonFile(const std::string fileName);

  // Getters for the configuration
  bool drawTrackQA() const;
  bool drawHistogram(const int iHistogram) const;
  bool drawProfile(const int iProfile) const;
  bool drawReferenceProfile() const;
  bool drawCentralEtaSummaryProfile() const;
  double profileZoomLow(const int iProfile) const;
  double profileZoomHigh(const int iProfile) const;
  bool drawTrend(const int iTrend) const;
  double trendZoomLow(const int iTrend) const;
  double trendZoomHigh(const int iTrend) const;

  double profileLegendShiftTotalX() const;
  double profileLegendShiftTotalY() const;
  double profileLegendShiftColumnX(const int iColumn) const;
  double profileLegendShiftColumnY(const int iColumn) const;
  double profileLegendTextSize() const;
  int profileLegendTextFont() const;

  double trendLegendShiftTotalX() const;
  double trendLegendShiftTotalY() const;
  double trendLegendTextSize() const;
  int trendLegendTextFont() const;

  bool drawTrendTag() const;
  std::vector<std::string> trendTagText() const;
  std::vector<double> trendTagPositionX() const;
  std::vector<double> trendTagPositionY() const;
  double trendTagTextSize() const;
  int trendTagTextFont() const;

  int trendCanvasHeight() const;
  int trendCanvasWidth() const;
  double trendMarginLeft() const;
  double trendMarginRight() const;
  double trendMarginTop() const;
  double trendMarginBottom() const;
  double trendTitleOffsetX() const;
  double trendTitleOffsetY() const;
  double trendTitleSizeX() const;
  double trendTitleSizeY() const;
  double trendLabelOffsetX() const;
  double trendLabelOffsetY() const;
  double trendLabelSizeX() const;
  double trendLabelSizeY() const;

  int nInputFiles() const;
  std::string inputFile(const int iFile) const;
  std::vector<std::string> inputFiles() const;

  std::string legendComment(const int iComment) const;
  int markerColor(const int iFile) const;
  int markerStyle(const int iFile) const;
  int markerSize(const int iFile) const;
  bool copyErrorColor(const int iFile) const;

  const char* lumiPerIovFile() const;
  const char* iovListMode() const;
  bool drawYearLines() const;
  int yearLineColor() const;
  int yearLineWidth() const;
  int yearLineStyle() const;
  std::vector<int> runsForLines() const;

  std::vector<double> widePtBinBorders() const;

  std::string legendTextForAllRuns() const;
  bool drawPlotsForEachIOV() const;
  int nIovInOnePlot() const;
  bool useLuminosityForTrends() const;
  bool skipRunsWithNoData() const;
  bool normalizeQAplots() const;

  const char* saveComment() const;

  bool makeIovListForSlides() const;
  const char* iovListForSlides() const;

  // Print the current configuration to console
  void printConfiguration() const;

private:
  // Methods to expand environtental variables in the configuration
  void autoExpandEnvironmentVariables(std::string& text) const;
  std::string expandEnvironmentVariables(const std::string& input) const;

  // Strings correcponding to configuration in the JSON file
  std::string fJsonTrackQAname = "drawTrackQA";
  std::string fJsonCategoryNameHistogram = "drawHistograms";
  std::string fJsonNameHistogram[knHistogramTypes] = {"drawDz", "drawDzError", "drawDxy", "drawDxyError"};
  std::string fJsonCategoryNameProfile = "drawProfiles";
  std::string fJsonNameProfile[knProfileTypes] = {"drawDzErrorVsPt",
                                                  "drawDzErrorVsPhi",
                                                  "drawDzErrorVsEta",
                                                  "drawDxyErrorVsPt",
                                                  "drawDxyErrorVsPhi",
                                                  "drawDxyErrorVsEta",
                                                  "drawDzErrorVsPtWide",
                                                  "drawDxyErrorVsPtWide"};
  std::string fJsonNameReferenceProfile = "drawReferenceProfile";
  std::string fJsonNameCentralEtaSummaryProfile = "drawCentralEtaSummaryProfile";
  std::string fJsonCategoryNameProfileZoom = "profileZoom";
  std::string fJsonNameProfileZoom[knProfileTypes] = {"ZoomPtProfileDz",
                                                      "ZoomPhiProfileDz",
                                                      "ZoomEtaProfileDz",
                                                      "ZoomPtProfileDxy",
                                                      "ZoomPhiProfileDxy",
                                                      "ZoomEtaProfileDxy",
                                                      "ZoomPtWideProfileDz",
                                                      "ZoomPtWideProfileDxy"};
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
  int fDefaultColors[11] = {
      kBlue, kRed, kGreen + 2, kMagenta, kBlack, kCyan, kViolet + 3, kOrange, kPink - 7, kSpring + 3, kAzure - 7};
  int fDefaultStyle = 20;  // Full circle
  int fDefaultMarkerSize = 1;
};

#endif
