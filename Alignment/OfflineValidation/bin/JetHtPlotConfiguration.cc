#include "JetHtPlotConfiguration.h"

/*
 * Default constructor
 */
JetHtPlotConfiguration::JetHtPlotConfiguration()
    : fDebugLevel(0),
      fDrawTrackQA(false),
      fDrawReferenceProfile(false),
      fDrawCentralEtaSummaryProfile(true),
      fProfileLegendShiftTotalX(0),
      fProfileLegendShiftTotalY(0),
      fProfileLegendTextSize(0.05),
      fProfileLegendTextFont(62),
      fTrendLegendShiftTotalX(0),
      fTrendLegendShiftTotalY(0),
      fTrendLegendTextSize(0.05),
      fTrendLegendTextFont(62),
      fDrawTrendTag(false),
      fTrendTagText(0),
      fTrendTagPositionX(0),
      fTrendTagPositionY(0),
      fTrendTagTextSize(0.05),
      fTrendTagTextFont(42),
      fTrendCanvasHeight(400),
      fTrendCanvasWidth(1000),
      fTrendMarginLeft(0.08),
      fTrendMarginRight(0.03),
      fTrendMarginTop(0.06),
      fTrendMarginBottom(0.15),
      fTrendTitleOffsetX(1.1),
      fTrendTitleOffsetY(0.55),
      fTrendTitleSizeX(0.06),
      fTrendTitleSizeY(0.06),
      fTrendLabelOffsetX(0.01),
      fTrendLabelOffsetY(0.007),
      fTrendLabelSizeX(0.05),
      fTrendLabelSizeY(0.05),
      fInputFileNames(0),
      fLegendComments(0),
      fMarkerColor(0),
      fMarkerStyle(0),
      fMarkerSize(0),
      fCopyErrorColor(0),
      fLegendTextForAllRuns("All"),
      fDrawPlotsForEachIOV(false),
      fNIovInOnePlot(1),
      fUseLuminosityForTrends(true),
      fSkipRunsWithNoData(false),
      fNormalizeQAplots(true),
      fSaveComment(""),
      fLumiPerIovFile("lumiPerRun_Run2.txt"),
      fIovListMode("run"),
      fDrawYearLines(false),
      fYearLineColor(kBlack),
      fYearLineWidth(1),
      fYearLineStyle(1),
      fRunsForLines(0),
      fWidePtBinBorders(0),
      fMakeIovListForSlides(false),
      fIovListForSlides("iovListForSlides.txt") {
  // Default zoom values
  double defaultProfileZoomLow[knProfileTypes] = {28, 45, 30, 7, 40, 20, 25, 20};
  double defaultProfileZoomHigh[knProfileTypes] = {60, 80, 95, 40, 70, 90, 90, 80};
  double defaultTrendZoomLow[knTrendTypes] = {20, 10};
  double defaultTrendZoomHigh[knTrendTypes] = {95, 90};

  // Initialize arrays
  for (int iHistogram = 0; iHistogram < knHistogramTypes; iHistogram++) {
    fDrawHistogram[iHistogram] = false;
  }
  for (int iProfile = 0; iProfile < knProfileTypes; iProfile++) {
    fDrawProfile[iProfile] = false;
    fProfileZoomLow[iProfile] = defaultProfileZoomLow[iProfile];
    fProfileZoomHigh[iProfile] = defaultProfileZoomHigh[iProfile];
  }
  for (int iTrend = 0; iTrend < knTrendTypes; iTrend++) {
    fDrawTrend[iTrend] = false;
    fTrendZoomLow[iTrend] = defaultTrendZoomLow[iTrend];
    fTrendZoomHigh[iTrend] = defaultTrendZoomHigh[iTrend];
  }
  for (int iColumn = 0; iColumn < kMaxLegendColumns; iColumn++) {
    fProfileLegendShiftColumnX[iColumn] = 0;
    fProfileLegendShiftColumnY[iColumn] = 0;
  }
}

/*
 * Copy constructor
 */
JetHtPlotConfiguration::JetHtPlotConfiguration(const JetHtPlotConfiguration& in)
    : fDebugLevel(in.fDebugLevel),
      fDrawTrackQA(in.fDrawTrackQA),
      fDrawReferenceProfile(in.fDrawReferenceProfile),
      fDrawCentralEtaSummaryProfile(in.fDrawCentralEtaSummaryProfile),
      fProfileLegendShiftTotalX(in.fProfileLegendShiftTotalX),
      fProfileLegendShiftTotalY(in.fProfileLegendShiftTotalY),
      fProfileLegendTextSize(in.fProfileLegendTextSize),
      fProfileLegendTextFont(in.fProfileLegendTextFont),
      fTrendLegendShiftTotalX(in.fTrendLegendShiftTotalX),
      fTrendLegendShiftTotalY(in.fTrendLegendShiftTotalY),
      fTrendLegendTextSize(in.fTrendLegendTextSize),
      fTrendLegendTextFont(in.fTrendLegendTextFont),
      fDrawTrendTag(in.fDrawTrendTag),
      fTrendTagText(in.fTrendTagText),
      fTrendTagPositionX(in.fTrendTagPositionX),
      fTrendTagPositionY(in.fTrendTagPositionY),
      fTrendTagTextSize(in.fTrendTagTextSize),
      fTrendTagTextFont(in.fTrendTagTextFont),
      fTrendCanvasHeight(in.fTrendCanvasHeight),
      fTrendCanvasWidth(in.fTrendCanvasWidth),
      fTrendMarginLeft(in.fTrendMarginLeft),
      fTrendMarginRight(in.fTrendMarginRight),
      fTrendMarginTop(in.fTrendMarginTop),
      fTrendMarginBottom(in.fTrendMarginBottom),
      fTrendTitleOffsetX(in.fTrendTitleOffsetX),
      fTrendTitleOffsetY(in.fTrendTitleOffsetY),
      fTrendTitleSizeX(in.fTrendTitleSizeX),
      fTrendTitleSizeY(in.fTrendTitleSizeY),
      fTrendLabelOffsetX(in.fTrendLabelOffsetX),
      fTrendLabelOffsetY(in.fTrendLabelOffsetY),
      fTrendLabelSizeX(in.fTrendLabelSizeX),
      fTrendLabelSizeY(in.fTrendLabelSizeY),
      fInputFileNames(in.fInputFileNames),
      fLegendComments(in.fLegendComments),
      fMarkerColor(in.fMarkerColor),
      fMarkerStyle(in.fMarkerStyle),
      fMarkerSize(in.fMarkerSize),
      fCopyErrorColor(in.fCopyErrorColor),
      fLegendTextForAllRuns(in.fLegendTextForAllRuns),
      fDrawPlotsForEachIOV(in.fDrawPlotsForEachIOV),
      fNIovInOnePlot(in.fNIovInOnePlot),
      fUseLuminosityForTrends(in.fUseLuminosityForTrends),
      fSkipRunsWithNoData(in.fSkipRunsWithNoData),
      fNormalizeQAplots(in.fNormalizeQAplots),
      fSaveComment(in.fSaveComment),
      fLumiPerIovFile(in.fLumiPerIovFile),
      fIovListMode(in.fIovListMode),
      fDrawYearLines(in.fDrawYearLines),
      fYearLineColor(in.fYearLineColor),
      fYearLineWidth(in.fYearLineWidth),
      fYearLineStyle(in.fYearLineStyle),
      fRunsForLines(in.fRunsForLines),
      fWidePtBinBorders(in.fWidePtBinBorders),
      fMakeIovListForSlides(in.fMakeIovListForSlides),
      fIovListForSlides(in.fIovListForSlides) {
  // Copy arrays
  for (int iHistogram = 0; iHistogram < knHistogramTypes; iHistogram++) {
    fDrawHistogram[iHistogram] = in.fDrawHistogram[iHistogram];
  }
  for (int iProfile = 0; iProfile < knProfileTypes; iProfile++) {
    fDrawProfile[iProfile] = in.fDrawProfile[iProfile];
    fProfileZoomLow[iProfile] = in.fProfileZoomLow[iProfile];
    fProfileZoomHigh[iProfile] = in.fProfileZoomHigh[iProfile];
  }
  for (int iTrend = 0; iTrend < knTrendTypes; iTrend++) {
    fDrawTrend[iTrend] = in.fDrawTrend[iTrend];
    fTrendZoomLow[iTrend] = in.fTrendZoomLow[iTrend];
    fTrendZoomHigh[iTrend] = in.fTrendZoomHigh[iTrend];
  }
  for (int iColumn = 0; iColumn < kMaxLegendColumns; iColumn++) {
    fProfileLegendShiftColumnX[iColumn] = in.fProfileLegendShiftColumnX[iColumn];
    fProfileLegendShiftColumnY[iColumn] = in.fProfileLegendShiftColumnY[iColumn];
  }
}

/*
 * Assingment operator
 */
JetHtPlotConfiguration& JetHtPlotConfiguration::operator=(const JetHtPlotConfiguration& in) {
  if (&in == this)
    return *this;

  fDebugLevel = in.fDebugLevel;
  fDrawTrackQA = in.fDrawTrackQA;
  fDrawReferenceProfile = in.fDrawReferenceProfile;
  fDrawCentralEtaSummaryProfile = in.fDrawCentralEtaSummaryProfile;
  fProfileLegendShiftTotalX = in.fProfileLegendShiftTotalX;
  fProfileLegendShiftTotalY = in.fProfileLegendShiftTotalY;
  fProfileLegendTextSize = in.fProfileLegendTextSize;
  fProfileLegendTextFont = in.fProfileLegendTextFont;
  fTrendLegendShiftTotalX = in.fTrendLegendShiftTotalX;
  fTrendLegendShiftTotalY = in.fTrendLegendShiftTotalY;
  fTrendLegendTextSize = in.fTrendLegendTextSize;
  fTrendLegendTextFont = in.fTrendLegendTextFont;
  fDrawTrendTag = in.fDrawTrendTag;
  fTrendTagText = in.fTrendTagText;
  fTrendTagPositionX = in.fTrendTagPositionX;
  fTrendTagPositionY = in.fTrendTagPositionY;
  fTrendTagTextSize = in.fTrendTagTextSize;
  fTrendTagTextFont = in.fTrendTagTextFont;
  fTrendCanvasHeight = in.fTrendCanvasHeight;
  fTrendCanvasWidth = in.fTrendCanvasWidth;
  fTrendMarginLeft = in.fTrendMarginLeft;
  fTrendMarginRight = in.fTrendMarginRight;
  fTrendMarginTop = in.fTrendMarginTop;
  fTrendMarginBottom = in.fTrendMarginBottom;
  fTrendTitleOffsetX = in.fTrendTitleOffsetX;
  fTrendTitleOffsetY = in.fTrendTitleOffsetY;
  fTrendTitleSizeX = in.fTrendTitleSizeX;
  fTrendTitleSizeY = in.fTrendTitleSizeY;
  fTrendLabelOffsetX = in.fTrendLabelOffsetX;
  fTrendLabelOffsetY = in.fTrendLabelOffsetY;
  fTrendLabelSizeX = in.fTrendLabelSizeX;
  fTrendLabelSizeY = in.fTrendLabelSizeY;
  fInputFileNames = in.fInputFileNames;
  fLegendComments = in.fLegendComments;
  fMarkerColor = in.fMarkerColor;
  fMarkerStyle = in.fMarkerStyle;
  fMarkerSize = in.fMarkerSize;
  fCopyErrorColor = in.fCopyErrorColor;
  fLegendTextForAllRuns = in.fLegendTextForAllRuns;
  fDrawPlotsForEachIOV = in.fDrawPlotsForEachIOV;
  fNIovInOnePlot = in.fNIovInOnePlot;
  fUseLuminosityForTrends = in.fUseLuminosityForTrends;
  fSkipRunsWithNoData = in.fSkipRunsWithNoData;
  fNormalizeQAplots = in.fNormalizeQAplots;
  fSaveComment = in.fSaveComment;
  fLumiPerIovFile = in.fLumiPerIovFile;
  fIovListMode = in.fIovListMode;
  fDrawYearLines = in.fDrawYearLines;
  fYearLineColor = in.fYearLineColor;
  fYearLineWidth = in.fYearLineWidth;
  fYearLineStyle = in.fYearLineStyle;
  fRunsForLines = in.fRunsForLines;
  fWidePtBinBorders = in.fWidePtBinBorders;
  fMakeIovListForSlides = in.fMakeIovListForSlides;
  fIovListForSlides = in.fIovListForSlides;

  // Copy arrays
  for (int iHistogram = 0; iHistogram < knHistogramTypes; iHistogram++) {
    fDrawHistogram[iHistogram] = in.fDrawHistogram[iHistogram];
  }
  for (int iProfile = 0; iProfile < knProfileTypes; iProfile++) {
    fDrawProfile[iProfile] = in.fDrawProfile[iProfile];
    fProfileZoomLow[iProfile] = in.fProfileZoomLow[iProfile];
    fProfileZoomHigh[iProfile] = in.fProfileZoomHigh[iProfile];
  }
  for (int iTrend = 0; iTrend < knTrendTypes; iTrend++) {
    fDrawTrend[iTrend] = in.fDrawTrend[iTrend];
    fTrendZoomLow[iTrend] = in.fTrendZoomLow[iTrend];
    fTrendZoomHigh[iTrend] = in.fTrendZoomHigh[iTrend];
  }
  for (int iColumn = 0; iColumn < kMaxLegendColumns; iColumn++) {
    fProfileLegendShiftColumnX[iColumn] = in.fProfileLegendShiftColumnX[iColumn];
    fProfileLegendShiftColumnY[iColumn] = in.fProfileLegendShiftColumnY[iColumn];
  }

  return *this;
}

/*
 * Destructor
 */
JetHtPlotConfiguration::~JetHtPlotConfiguration() {}

/*
 * Read the configuration from json file
 */
void JetHtPlotConfiguration::readJsonFile(const std::string fileName) {
  // Read the file to property tree
  namespace pt = boost::property_tree;
  pt::ptree configuration;
  pt::read_json(fileName, configuration);

  // Read the alignments and styles for these
  int indexForDefault = 0;
  std::string alignmentName;
  std::string thisValue;
  int thisNumber;
  bool thisBool;
  for (pt::ptree::value_type& result : configuration.get_child("jethtplot.alignments")) {
    alignmentName = result.first;

    // Input file for the alignment
    try {
      thisValue = configuration.get_child(Form("jethtplot.alignments.%s.inputFile", alignmentName.c_str()))
                      .get_value<std::string>();

      // From the file name, expand possible environment variables
      autoExpandEnvironmentVariables(thisValue);

      // Expand CMSSW_BASE event without preceding $-sign
      boost::replace_all(thisValue, "CMSSW_BASE", getenv("CMSSW_BASE"));

      fInputFileNames.push_back(thisValue);
    } catch (const std::exception& e) {
      if (fDebugLevel > 0) {
        std::cout << "No input file given for alignment " << alignmentName
                  << "! All the configuration for this alignment is skipped." << std::endl;
      }
      continue;
    }

    // Legend text for the alignment
    try {
      thisValue = configuration.get_child(Form("jethtplot.alignments.%s.legendText", alignmentName.c_str()))
                      .get_value<std::string>();
      fLegendComments.push_back(thisValue);
    } catch (const std::exception& e) {
      if (fDebugLevel > 0) {
        std::cout << "No legend text given for alignment " << alignmentName << "! Using default value "
                  << Form("Alignment%d", indexForDefault) << "." << std::endl;
        fLegendComments.push_back(Form("Alignment%d", indexForDefault));
      }
    }

    // Marker color related to the alignment
    try {
      thisNumber =
          configuration.get_child(Form("jethtplot.alignments.%s.color", alignmentName.c_str())).get_value<int>();
      fMarkerColor.push_back(thisNumber);
    } catch (const std::exception& e) {
      if (fDebugLevel > 0) {
        std::cout << "No marker color given for alignment " << alignmentName << "! Using default value "
                  << fDefaultColors[indexForDefault] << "." << std::endl;
        fMarkerColor.push_back(fDefaultColors[indexForDefault]);
      }
    }

    // Marker style related to the alignment
    try {
      thisNumber =
          configuration.get_child(Form("jethtplot.alignments.%s.style", alignmentName.c_str())).get_value<int>();
      fMarkerStyle.push_back(thisNumber);
    } catch (const std::exception& e) {
      if (fDebugLevel > 0) {
        std::cout << "No marker style given for alignment " << alignmentName << "! Using default value "
                  << fDefaultStyle << "." << std::endl;
        fMarkerStyle.push_back(fDefaultStyle);
      }
    }

    // Marker size related to the alignment
    try {
      thisNumber =
          configuration.get_child(Form("jethtplot.alignments.%s.markerSize", alignmentName.c_str())).get_value<int>();
      fMarkerSize.push_back(thisNumber);
    } catch (const std::exception& e) {
      if (fDebugLevel > 0) {
        std::cout << "No marker size given for alignment " << alignmentName << "! Using default value "
                  << fDefaultMarkerSize << "." << std::endl;
        fMarkerSize.push_back(fDefaultMarkerSize);
      }
    }

    // Copy color for statistical error bar related to the alignment
    try {
      thisBool = configuration.get_child(Form("jethtplot.alignments.%s.copyErrorColor", alignmentName.c_str()))
                     .get_value<bool>();
      fCopyErrorColor.push_back(thisBool);
    } catch (const std::exception& e) {
      if (fDebugLevel > 0) {
        std::cout << "Not defined is marker color should be copied for statistical error bars for alignment "
                  << alignmentName << "! Using default value false." << std::endl;
        fCopyErrorColor.push_back(false);
      }
    }

    indexForDefault++;
  }

  // Read the drawing configuration for track QA
  try {
    fDrawTrackQA = configuration.get_child(Form("jethtplot.%s", fJsonTrackQAname.c_str())).get_value<bool>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s", fJsonTrackQAname.c_str())
                << " in configuration. Using default value " << fDrawTrackQA << "." << std::endl;
    }
  }

  // Read the drawing configuration for dz and dxy value and error histograms
  for (int iHistogram = 0; iHistogram < knHistogramTypes; iHistogram++) {
    try {
      fDrawHistogram[iHistogram] =
          configuration
              .get_child(
                  Form("jethtplot.%s.%s", fJsonCategoryNameHistogram.c_str(), fJsonNameHistogram[iHistogram].c_str()))
              .get_value<bool>();
    } catch (const std::exception& e) {
      if (fDebugLevel > 0) {
        std::cout << "No value "
                  << Form("jethtplot.%s.%s", fJsonCategoryNameHistogram.c_str(), fJsonNameHistogram[iHistogram].c_str())
                  << " in configuration. Using default value " << fDrawHistogram[iHistogram] << "." << std::endl;
      }
    }
  }

  // Read the drawing configuration for profile plots
  for (int iProfile = 0; iProfile < knProfileTypes; iProfile++) {
    try {
      fDrawProfile[iProfile] =
          configuration
              .get_child(Form("jethtplot.%s.%s", fJsonCategoryNameProfile.c_str(), fJsonNameProfile[iProfile].c_str()))
              .get_value<bool>();
    } catch (const std::exception& e) {
      if (fDebugLevel > 0) {
        std::cout << "No value "
                  << Form("jethtplot.%s.%s", fJsonCategoryNameProfile.c_str(), fJsonNameProfile[iProfile].c_str())
                  << " in configuration. Using default value " << fDrawProfile[iProfile] << "." << std::endl;
      }
    }
  }

  // Draw the drawing of reference profile
  try {
    fDrawReferenceProfile =
        configuration
            .get_child(Form("jethtplot.%s.%s", fJsonCategoryNameProfile.c_str(), fJsonNameReferenceProfile.c_str()))
            .get_value<bool>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value "
                << Form("jethtplot.%s.%s", fJsonCategoryNameProfile.c_str(), fJsonNameReferenceProfile.c_str())
                << " in configuration. Using default value " << fDrawReferenceProfile << "." << std::endl;
    }
  }

  // Draw the central eta summary profile to the all runs summary profile plots
  try {
    fDrawCentralEtaSummaryProfile =
        configuration
            .get_child(
                Form("jethtplot.%s.%s", fJsonCategoryNameProfile.c_str(), fJsonNameCentralEtaSummaryProfile.c_str()))
            .get_value<bool>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value "
                << Form("jethtplot.%s.%s", fJsonCategoryNameProfile.c_str(), fJsonNameCentralEtaSummaryProfile.c_str())
                << " in configuration. Using default value " << fDrawCentralEtaSummaryProfile << "." << std::endl;
    }
  }

  // Read the total legend shift in x-direction for legends in profile plots
  try {
    fProfileLegendShiftTotalX =
        configuration
            .get_child(Form("jethtplot.%s.%s", fJsonCategoryNameProfile.c_str(), fJsonNameLegendShiftTotalX.c_str()))
            .get_value<double>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value "
                << Form("jethtplot.%s.%s", fJsonCategoryNameProfile.c_str(), fJsonNameLegendShiftTotalX.c_str())
                << " in configuration. Using default value " << fProfileLegendShiftTotalX << "." << std::endl;
    }
  }

  // Read the total legend shift in y-direction for legends in profile plots
  try {
    fProfileLegendShiftTotalY =
        configuration
            .get_child(Form("jethtplot.%s.%s", fJsonCategoryNameProfile.c_str(), fJsonNameLegendShiftTotalY.c_str()))
            .get_value<double>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value "
                << Form("jethtplot.%s.%s", fJsonCategoryNameProfile.c_str(), fJsonNameLegendShiftTotalY.c_str())
                << " in configuration. Using default value " << fProfileLegendShiftTotalY << "." << std::endl;
    }
  }

  // Read the columnwise legend shift in x-direction for legends in profile plots
  for (int iColumn = 0; iColumn < kMaxLegendColumns; iColumn++) {
    try {
      fProfileLegendShiftColumnX[iColumn] =
          configuration
              .get_child(Form(
                  "jethtplot.%s.%s%d", fJsonCategoryNameProfile.c_str(), fJsonNameLegendShiftColumnX.c_str(), iColumn))
              .get_value<double>();
    } catch (const std::exception& e) {
      if (fDebugLevel > 0) {
        std::cout
            << "No value "
            << Form("jethtplot.%s.%s%d", fJsonCategoryNameProfile.c_str(), fJsonNameLegendShiftColumnX.c_str(), iColumn)
            << " in configuration. Using default value " << fProfileLegendShiftColumnX[iColumn] << "." << std::endl;
      }
    }
  }

  // Read the columnwise legend shift in y-direction for legends in profile plots
  for (int iColumn = 0; iColumn < kMaxLegendColumns; iColumn++) {
    try {
      fProfileLegendShiftColumnY[iColumn] =
          configuration
              .get_child(Form(
                  "jethtplot.%s.%s%d", fJsonCategoryNameProfile.c_str(), fJsonNameLegendShiftColumnY.c_str(), iColumn))
              .get_value<double>();
    } catch (const std::exception& e) {
      if (fDebugLevel > 0) {
        std::cout
            << "No value "
            << Form("jethtplot.%s.%s%d", fJsonCategoryNameProfile.c_str(), fJsonNameLegendShiftColumnY.c_str(), iColumn)
            << " in configuration. Using default value " << fProfileLegendShiftColumnY[iColumn] << "." << std::endl;
      }
    }
  }

  // Read the legend text size for profile plots
  try {
    fProfileLegendTextSize =
        configuration
            .get_child(Form("jethtplot.%s.%s", fJsonCategoryNameProfile.c_str(), fJsonNameLegendTextSize.c_str()))
            .get_value<double>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value "
                << Form("jethtplot.%s.%s", fJsonCategoryNameProfile.c_str(), fJsonNameLegendTextSize.c_str())
                << " in configuration. Using default value " << fProfileLegendTextSize << "." << std::endl;
    }
  }

  // Read the legend text font for profile plots
  try {
    fProfileLegendTextFont =
        configuration
            .get_child(Form("jethtplot.%s.%s", fJsonCategoryNameProfile.c_str(), fJsonNameLegendTextFont.c_str()))
            .get_value<int>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value "
                << Form("jethtplot.%s.%s", fJsonCategoryNameProfile.c_str(), fJsonNameLegendTextFont.c_str())
                << " in configuration. Using default value " << fProfileLegendTextFont << "." << std::endl;
    }
  }

  // Read the legend text referring to all runs
  try {
    fLegendTextForAllRuns =
        configuration.get_child(Form("jethtplot.%s", fJsonNameLegendTextForAllRuns.c_str())).get_value<std::string>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s", fJsonNameLegendTextForAllRuns.c_str())
                << " in configuration. Using default value " << fLegendTextForAllRuns << "." << std::endl;
    }
  }

  // Read the configuration for plotting individual IOV:s for dxy, dx and profile plots
  try {
    fDrawPlotsForEachIOV =
        configuration.get_child(Form("jethtplot.%s", fJsonDrawPlotsForEachIOV.c_str())).get_value<bool>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s", fJsonDrawPlotsForEachIOV.c_str())
                << " in configuration. Using default value " << fDrawPlotsForEachIOV << "." << std::endl;
    }
  }

  // Read the number of IOVs plotted in the same figure, if individual IOV are plotted for profiles
  try {
    fNIovInOnePlot =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameProfile.c_str(), fJsonNIovInOnePlot.c_str()))
            .get_value<int>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameProfile.c_str(), fJsonNIovInOnePlot.c_str())
                << " in configuration. Using default value " << fNIovInOnePlot << "." << std::endl;
    }
  }

  // Read the axis zoom values for profile plots
  for (int iProfile = 0; iProfile < knProfileTypes; iProfile++) {
    try {
      fProfileZoomLow[iProfile] =
          configuration
              .get_child(Form(
                  "jethtplot.%s.min%s", fJsonCategoryNameProfileZoom.c_str(), fJsonNameProfileZoom[iProfile].c_str()))
              .get_value<double>();
    } catch (const std::exception& e) {
      if (fDebugLevel > 0) {
        std::cout << "No value "
                  << Form("jethtplot.%s.min%s",
                          fJsonCategoryNameProfileZoom.c_str(),
                          fJsonNameProfileZoom[iProfile].c_str())
                  << " in configuration. Using default value " << fProfileZoomLow[iProfile] << "." << std::endl;
      }
    }

    try {
      fProfileZoomHigh[iProfile] =
          configuration
              .get_child(Form(
                  "jethtplot.%s.max%s", fJsonCategoryNameProfileZoom.c_str(), fJsonNameProfileZoom[iProfile].c_str()))
              .get_value<double>();
    } catch (const std::exception& e) {
      if (fDebugLevel > 0) {
        std::cout << "No value "
                  << Form("jethtplot.%s.max%s",
                          fJsonCategoryNameProfileZoom.c_str(),
                          fJsonNameProfileZoom[iProfile].c_str())
                  << " in configuration. Using default value " << fProfileZoomHigh[iProfile] << "." << std::endl;
      }
    }
  }

  // Read the drawing configuration for trend plots
  for (int iTrend = 0; iTrend < knTrendTypes; iTrend++) {
    try {
      fDrawTrend[iTrend] =
          configuration
              .get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameTrend[iTrend].c_str()))
              .get_value<bool>();
    } catch (const std::exception& e) {
      if (fDebugLevel > 0) {
        std::cout << "No value "
                  << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameTrend[iTrend].c_str())
                  << " in configuration. Using default value " << fDrawTrend[iTrend] << "." << std::endl;
      }
    }
  }

  // Read the total legend shift in x-direction for legends in trend plots
  try {
    fTrendLegendShiftTotalX =
        configuration
            .get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameLegendShiftTotalX.c_str()))
            .get_value<double>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value "
                << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameLegendShiftTotalX.c_str())
                << " in configuration. Using default value " << fTrendLegendShiftTotalX << "." << std::endl;
    }
  }

  // Read the total legend shift in y-direction for legends in trend plots
  try {
    fTrendLegendShiftTotalY =
        configuration
            .get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameLegendShiftTotalY.c_str()))
            .get_value<double>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value "
                << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameLegendShiftTotalY.c_str())
                << " in configuration. Using default value " << fTrendLegendShiftTotalY << "." << std::endl;
    }
  }

  // Read the legend text size for trend plots
  try {
    fTrendLegendTextSize =
        configuration
            .get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameLegendTextSize.c_str()))
            .get_value<double>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value "
                << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameLegendTextSize.c_str())
                << " in configuration. Using default value " << fTrendLegendTextSize << "." << std::endl;
    }
  }

  // Read the legend text font for trend plots
  try {
    fTrendLegendTextFont =
        configuration
            .get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameLegendTextFont.c_str()))
            .get_value<int>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value "
                << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameLegendTextFont.c_str())
                << " in configuration. Using default value " << fTrendLegendTextFont << "." << std::endl;
    }
  }

  // Read the flag for drawing vertical lines to trend plots
  try {
    fDrawYearLines =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonDrawYearLines.c_str()))
            .get_value<bool>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonDrawYearLines.c_str())
                << " in configuration. Using default value " << fDrawYearLines << "." << std::endl;
    }
  }

  // Read the line color for year lines in trend plots
  try {
    fYearLineColor =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonYearLineColor.c_str()))
            .get_value<int>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonYearLineColor.c_str())
                << " in configuration. Using default value " << fYearLineColor << "." << std::endl;
    }
  }

  // Read the line witdh for year lines in trend plots
  try {
    fYearLineWidth =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonYearLineWidth.c_str()))
            .get_value<int>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonYearLineWidth.c_str())
                << " in configuration. Using default value " << fYearLineWidth << "." << std::endl;
    }
  }

  // Read the line style for year lines in trend plots
  try {
    fYearLineStyle =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonYearLineStyle.c_str()))
            .get_value<int>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonYearLineStyle.c_str())
                << " in configuration. Using default value " << fYearLineStyle << "." << std::endl;
    }
  }

  // Read run positions to which vertical lines are drawn in trend plots
  try {
    for (auto& item :
         configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonRunsForLines.c_str()))) {
      fRunsForLines.push_back(item.second.get_value<int>());
    }
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonRunsForLines.c_str())
                << " in configuration. Using default values 290543 and 314881." << std::endl;
    }
    fRunsForLines.push_back(290543);
    fRunsForLines.push_back(314881);
  }

  // Read the flag for drawing manual tags to trend plots
  try {
    fDrawTrendTag =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameDrawTag.c_str()))
            .get_value<bool>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameDrawTag.c_str())
                << " in configuration. Using default value " << fDrawTrendTag << "." << std::endl;
    }
  }

  // Read the configuration for tags in trend plots
  try {
    int itemIndex = 0;
    for (auto& configurationArray :
         configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameTagInfo.c_str()))) {
      itemIndex = 0;
      for (auto& item : configurationArray.second) {
        if (itemIndex == 0) {
          fTrendTagText.push_back(item.second.get_value<std::string>());
        }
        if (itemIndex == 1) {
          fTrendTagPositionX.push_back(item.second.get_value<double>());
        }
        if (itemIndex == 2) {
          fTrendTagPositionY.push_back(item.second.get_value<double>());
        }
        itemIndex++;
      }
    }
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "There was either no value for "
                << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameTagInfo.c_str())
                << " in configuration, or the input was in wrong format. Using default values 2016 at (0.105,0.855), "
                   "2017 at (0.305,0.855) and 2018 at (0.563,0.855)."
                << std::endl;
    }
    fTrendTagText.clear();
    fTrendTagText.push_back("2016");
    fTrendTagText.push_back("2017");
    fTrendTagText.push_back("2018");
    fTrendTagPositionX.clear();
    fTrendTagPositionX.push_back(0.105);
    fTrendTagPositionX.push_back(0.305);
    fTrendTagPositionX.push_back(0.563);
    fTrendTagPositionY.clear();
    fTrendTagPositionY.push_back(0.855);
    fTrendTagPositionY.push_back(0.855);
    fTrendTagPositionY.push_back(0.855);
  }

  // Read the tag text size for trend plots
  try {
    fTrendTagTextSize =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameTagTextSize.c_str()))
            .get_value<double>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameTagTextSize.c_str())
                << " in configuration. Using default value " << fTrendTagTextSize << "." << std::endl;
    }
  }

  // Read the tag text font for trend plots
  try {
    fTrendTagTextFont =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameTagTextFont.c_str()))
            .get_value<int>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameTagTextFont.c_str())
                << " in configuration. Using default value " << fTrendTagTextFont << "." << std::endl;
    }
  }

  // Read the trend plot canvas height
  try {
    fTrendCanvasHeight =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameCanvasHeight.c_str()))
            .get_value<int>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameCanvasHeight.c_str())
                << " in configuration. Using default value " << fTrendCanvasHeight << "." << std::endl;
    }
  }

  // Read the trend plot canvas width
  try {
    fTrendCanvasWidth =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameCanvasWidth.c_str()))
            .get_value<int>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameCanvasWidth.c_str())
                << " in configuration. Using default value " << fTrendCanvasWidth << "." << std::endl;
    }
  }

  // Read the left margin for trend plots
  try {
    fTrendMarginLeft =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameMarginLeft.c_str()))
            .get_value<double>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameMarginLeft.c_str())
                << " in configuration. Using default value " << fTrendMarginLeft << "." << std::endl;
    }
  }

  // Read the right margin for trend plots
  try {
    fTrendMarginRight =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameMarginRight.c_str()))
            .get_value<double>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameMarginRight.c_str())
                << " in configuration. Using default value " << fTrendMarginRight << "." << std::endl;
    }
  }

  // Read the top margin for trend plots
  try {
    fTrendMarginTop =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameMarginTop.c_str()))
            .get_value<double>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameMarginTop.c_str())
                << " in configuration. Using default value " << fTrendMarginTop << "." << std::endl;
    }
  }

  // Read the bottom margin for trend plots
  try {
    fTrendMarginBottom =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameMarginBottom.c_str()))
            .get_value<double>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameMarginBottom.c_str())
                << " in configuration. Using default value " << fTrendMarginBottom << "." << std::endl;
    }
  }

  // Read the offset of the x-axis title in trend plots
  try {
    fTrendTitleOffsetX =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameTitleOffsetX.c_str()))
            .get_value<double>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameTitleOffsetX.c_str())
                << " in configuration. Using default value " << fTrendTitleOffsetX << "." << std::endl;
    }
  }

  // Read the offset of the y-axis title in trend plots
  try {
    fTrendTitleOffsetY =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameTitleOffsetY.c_str()))
            .get_value<double>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameTitleOffsetY.c_str())
                << " in configuration. Using default value " << fTrendTitleOffsetY << "." << std::endl;
    }
  }

  // Read the size of the x-axis title in trend plots
  try {
    fTrendTitleSizeX =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameTitleSizeX.c_str()))
            .get_value<double>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameTitleSizeX.c_str())
                << " in configuration. Using default value " << fTrendTitleSizeX << "." << std::endl;
    }
  }

  // Read the size of the y-axis title in trend plots
  try {
    fTrendTitleSizeY =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameTitleSizeY.c_str()))
            .get_value<double>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameTitleSizeY.c_str())
                << " in configuration. Using default value " << fTrendTitleSizeY << "." << std::endl;
    }
  }

  // Read the offset of the x-axis label in trend plots
  try {
    fTrendLabelOffsetX =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameLabelOffsetX.c_str()))
            .get_value<double>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameLabelOffsetX.c_str())
                << " in configuration. Using default value " << fTrendLabelOffsetX << "." << std::endl;
    }
  }

  // Read the offset of the y-axis label in trend plots
  try {
    fTrendLabelOffsetY =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameLabelOffsetY.c_str()))
            .get_value<double>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameLabelOffsetY.c_str())
                << " in configuration. Using default value " << fTrendLabelOffsetY << "." << std::endl;
    }
  }

  // Read the size of the x-axis label in trend plots
  try {
    fTrendLabelSizeX =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameLabelSizeX.c_str()))
            .get_value<double>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameLabelSizeX.c_str())
                << " in configuration. Using default value " << fTrendLabelSizeX << "." << std::endl;
    }
  }

  // Read the size of the y-axis label in trend plots
  try {
    fTrendLabelSizeY =
        configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameLabelSizeY.c_str()))
            .get_value<double>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameLabelSizeY.c_str())
                << " in configuration. Using default value " << fTrendLabelSizeY << "." << std::endl;
    }
  }

  // Read the configuration for plotting trends plots as a function of luminosity
  try {
    fUseLuminosityForTrends =
        configuration
            .get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonUseLuminosityForTrends.c_str()))
            .get_value<bool>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value "
                << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonUseLuminosityForTrends.c_str())
                << " in configuration. Using default value " << fUseLuminosityForTrends << "." << std::endl;
    }
  }

  // Read the configuration for skipping runs with no data in trend plots
  try {
    fSkipRunsWithNoData =
        configuration
            .get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonSkipRunsWithNoData.c_str()))
            .get_value<bool>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value "
                << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonSkipRunsWithNoData.c_str())
                << " in configuration. Using default value " << fSkipRunsWithNoData << "." << std::endl;
    }
  }

  // Read the axis zoom values for trend plots
  for (int iTrend = 0; iTrend < knTrendTypes; iTrend++) {
    try {
      fTrendZoomLow[iTrend] =
          configuration
              .get_child(
                  Form("jethtplot.%s.min%s", fJsonCategoryNameTrendZoom.c_str(), fJsonNameTrendZoom[iTrend].c_str()))
              .get_value<double>();
    } catch (const std::exception& e) {
      if (fDebugLevel > 0) {
        std::cout << "No value "
                  << Form("jethtplot.%s.min%s", fJsonCategoryNameTrendZoom.c_str(), fJsonNameTrendZoom[iTrend].c_str())
                  << " in configuration. Using default value " << fTrendZoomLow[iTrend] << "." << std::endl;
      }
    }

    try {
      fTrendZoomHigh[iTrend] =
          configuration
              .get_child(
                  Form("jethtplot.%s.max%s", fJsonCategoryNameTrendZoom.c_str(), fJsonNameTrendZoom[iTrend].c_str()))
              .get_value<double>();
    } catch (const std::exception& e) {
      if (fDebugLevel > 0) {
        std::cout << "No value "
                  << Form("jethtplot.%s.max%s", fJsonCategoryNameTrendZoom.c_str(), fJsonNameTrendZoom[iTrend].c_str())
                  << " in configuration. Using default value " << fTrendZoomHigh[iTrend] << "." << std::endl;
      }
    }
  }

  // Read the configuration for normalizing QA plots by their integral
  try {
    fNormalizeQAplots = configuration.get_child(Form("jethtplot.%s", fJsonNormalizeQAplots.c_str())).get_value<bool>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s", fJsonNormalizeQAplots.c_str())
                << " in configuration. Using default value " << fNormalizeQAplots << "." << std::endl;
    }
  }

  // Read comment given to saved figures
  try {
    fSaveComment = configuration.get_child(Form("jethtplot.%s", fJsonSaveComment.c_str())).get_value<std::string>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s", fJsonSaveComment.c_str())
                << " in configuration. Using default value (empty string)." << std::endl;
    }
  }

  // Read the file name for luminosity per IOV
  try {
    fLumiPerIovFile =
        configuration.get_child(Form("jethtplot.%s", fJsonLumiPerIovFile.c_str())).get_value<std::string>();

    // From the file name, expand environment variables
    autoExpandEnvironmentVariables(fLumiPerIovFile);

    // Expand CMSSW_BASE even without preceding $-sign
    boost::replace_all(fLumiPerIovFile, "CMSSW_BASE", getenv("CMSSW_BASE"));

    // If file doesn't exist search elsewhere

  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s", fJsonLumiPerIovFile.c_str())
                << " in configuration. Using default value " << fLumiPerIovFile << "." << std::endl;
    }
  }

  // Read the IOV list mode. Either "run" or "IOV"
  try {
    fIovListMode = configuration.get_child(Form("jethtplot.%s", fJsonIovListMode.c_str())).get_value<std::string>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s", fJsonIovListMode.c_str())
                << " in configuration. Using default value " << fIovListMode << "." << std::endl;
    }
  }

  // Read the bin borders for the wide pT bims.
  try {
    for (auto& item : configuration.get_child(Form("jethtplot.%s", fJsonWidePtBinBorders.c_str()))) {
      fWidePtBinBorders.push_back(item.second.get_value<double>());
    }
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s", fJsonWidePtBinBorders.c_str())
                << " in configuration. Using default values 3, 5, 10, 20, 50 and 100." << std::endl;
    }
    fWidePtBinBorders.push_back(3.0);
    fWidePtBinBorders.push_back(5.0);
    fWidePtBinBorders.push_back(10.0);
    fWidePtBinBorders.push_back(20.0);
    fWidePtBinBorders.push_back(50.0);
    fWidePtBinBorders.push_back(100.0);
  }

  // Read the flag for creating an IOV list for the slides
  try {
    fMakeIovListForSlides =
        configuration.get_child(Form("jethtplot.%s", fJsonMakeIovListForSlides.c_str())).get_value<bool>();
  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s", fJsonMakeIovListForSlides.c_str())
                << " in configuration. Using default value " << fMakeIovListForSlides << "." << std::endl;
    }
  }

  // Read the output file name for the IOV list for the slides
  try {
    fIovListForSlides =
        configuration.get_child(Form("jethtplot.%s", fJsonIovListForSlides.c_str())).get_value<std::string>();

    // From the file name, expand environment variables
    autoExpandEnvironmentVariables(fIovListForSlides);

    // Expand CMSSW_BASE event without preceding $-sign
    boost::replace_all(fIovListForSlides, "CMSSW_BASE", getenv("CMSSW_BASE"));

  } catch (const std::exception& e) {
    if (fDebugLevel > 0) {
      std::cout << "No value " << Form("jethtplot.%s", fJsonIovListForSlides.c_str())
                << " in configuration. Using default value " << fIovListForSlides << "." << std::endl;
    }
  }
}

/*
 * Print the current configuration to console
 */
void JetHtPlotConfiguration::printConfiguration() const {
  // Print all input files
  std::cout << "Input files: " << std::endl;
  for (const std::string& fileName : fInputFileNames) {
    std::cout << fileName << std::endl;
  }

  // Print legend text related to input files
  std::cout << "Legend text related to input files: " << std::endl;
  for (const std::string& legendText : fLegendComments) {
    std::cout << legendText << std::endl;
  }

  // Print all marker colors
  std::cout << "Marker colors: " << std::endl;
  for (int color : fMarkerColor) {
    std::cout << color << std::endl;
  }

  // Print all marker styles
  std::cout << "Marker styles: " << std::endl;
  for (int style : fMarkerStyle) {
    std::cout << style << std::endl;
  }

  // Print all marker sizes
  std::cout << "Marker sizes: " << std::endl;
  for (int size : fMarkerSize) {
    std::cout << size << std::endl;
  }

  // Print all error color copying information
  std::cout << "Copy error colors: " << std::endl;
  for (bool color : fCopyErrorColor) {
    std::cout << color << std::endl;
  }

  // Print the configuration for QA plot drawing
  std::cout << fJsonTrackQAname << " : " << fDrawTrackQA << std::endl;

  // Print the configuration for histogram drawing
  for (int iHistogram = 0; iHistogram < knHistogramTypes; iHistogram++) {
    std::cout << fJsonCategoryNameHistogram << " " << fJsonNameHistogram[iHistogram] << " : "
              << fDrawHistogram[iHistogram] << std::endl;
  }

  // Print the configuration for profile plot drawing
  for (int iProfile = 0; iProfile < knProfileTypes; iProfile++) {
    std::cout << fJsonCategoryNameProfile << " " << fJsonNameProfile[iProfile] << " : " << fDrawProfile[iProfile]
              << std::endl;
  }

  // Print the configuration for reference profile drawing
  std::cout << fJsonCategoryNameProfile << " " << fJsonNameReferenceProfile << " : " << fDrawReferenceProfile
            << std::endl;
  std::cout << fJsonCategoryNameProfile << " " << fJsonNameCentralEtaSummaryProfile << " : "
            << fDrawCentralEtaSummaryProfile << std::endl;

  // Print the configuration for positioning the legends in profile plots
  std::cout << fJsonCategoryNameProfile << " " << fJsonNameLegendShiftTotalX << " : " << fProfileLegendShiftTotalX
            << std::endl;
  std::cout << fJsonCategoryNameProfile << " " << fJsonNameLegendShiftTotalY << " : " << fProfileLegendShiftTotalY
            << std::endl;
  for (int iColumn = 0; iColumn < kMaxLegendColumns; iColumn++) {
    std::cout << fJsonCategoryNameProfile << " " << fJsonNameLegendShiftColumnX << iColumn << " : "
              << fProfileLegendShiftColumnX[iColumn] << std::endl;
    std::cout << fJsonCategoryNameProfile << " " << fJsonNameLegendShiftColumnY << iColumn << " : "
              << fProfileLegendShiftColumnY[iColumn] << std::endl;
  }
  std::cout << fJsonCategoryNameProfile << " " << fJsonNameLegendTextSize << " : " << fProfileLegendTextSize
            << std::endl;
  std::cout << fJsonCategoryNameProfile << " " << fJsonNameLegendTextFont << " : " << fProfileLegendTextFont
            << std::endl;

  // Print the configuration for profile plot zoom
  for (int iProfile = 0; iProfile < knProfileTypes; iProfile++) {
    std::cout << fJsonCategoryNameProfileZoom << " min" << fJsonNameProfileZoom[iProfile] << " : "
              << fProfileZoomLow[iProfile] << std::endl;
    std::cout << fJsonCategoryNameProfileZoom << " max" << fJsonNameProfileZoom[iProfile] << " : "
              << fProfileZoomHigh[iProfile] << std::endl;
  }

  // Print the configuration for trend plot drawing
  for (int iTrend = 0; iTrend < knTrendTypes; iTrend++) {
    std::cout << fJsonCategoryNameTrend << " " << fJsonNameTrend[iTrend] << " : " << fDrawTrend[iTrend] << std::endl;
  }

  // Print the canvas information for trend plots
  std::cout << fJsonCategoryNameTrend << " " << fJsonNameCanvasHeight << " : " << fTrendCanvasHeight << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonNameCanvasWidth << " : " << fTrendCanvasWidth << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonNameMarginLeft << " : " << fTrendMarginLeft << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonNameMarginRight << " : " << fTrendMarginRight << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonNameMarginTop << " : " << fTrendMarginTop << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonNameMarginBottom << " : " << fTrendMarginBottom << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonNameTitleOffsetX << " : " << fTrendTitleOffsetX << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonNameTitleOffsetY << " : " << fTrendTitleOffsetY << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonNameTitleSizeX << " : " << fTrendTitleSizeX << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonNameTitleSizeY << " : " << fTrendTitleSizeY << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonNameLabelOffsetX << " : " << fTrendLabelOffsetX << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonNameLabelOffsetY << " : " << fTrendLabelOffsetY << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonNameLabelSizeX << " : " << fTrendLabelSizeX << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonNameLabelSizeY << " : " << fTrendLabelSizeY << std::endl;

  // Print the configuration for positioning the legends in trend plots
  std::cout << fJsonCategoryNameTrend << " " << fJsonNameLegendShiftTotalX << " : " << fTrendLegendShiftTotalX
            << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonNameLegendShiftTotalY << " : " << fTrendLegendShiftTotalY
            << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonNameLegendTextSize << " : " << fTrendLegendTextSize << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonNameLegendTextFont << " : " << fTrendLegendTextFont << std::endl;

  // Print the remaining configuration for trend plots
  std::cout << fJsonCategoryNameTrend << " " << fJsonDrawYearLines << " : " << fDrawYearLines << std::endl;
  std::cout << "Runs for vertical lines: " << std::endl;
  for (int myRun : fRunsForLines) {
    std::cout << myRun << " ";
  }
  std::cout << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonYearLineColor << " : " << fYearLineColor << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonYearLineWidth << " : " << fYearLineWidth << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonYearLineStyle << " : " << fYearLineStyle << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonUseLuminosityForTrends << " : " << fUseLuminosityForTrends
            << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonSkipRunsWithNoData << " : " << fSkipRunsWithNoData << std::endl;

  std::cout << fJsonCategoryNameTrend << " " << fJsonNameTagInfo << " : " << fDrawTrendTag << std::endl;
  for (std::vector<std::string>::size_type iTag = 0; iTag < fTrendTagText.size(); iTag++) {
    std::cout << "Tag" << iTag << ": " << fTrendTagText.at(iTag) << " x=" << fTrendTagPositionX.at(iTag)
              << " y=" << fTrendTagPositionY.at(iTag) << std::endl;
  }
  std::cout << fJsonCategoryNameTrend << " " << fJsonNameTagTextSize << " : " << fTrendTagTextSize << std::endl;
  std::cout << fJsonCategoryNameTrend << " " << fJsonNameTagTextFont << " : " << fTrendTagTextFont << std::endl;

  // Print the configuration for trend plot zoom
  for (int iTrend = 0; iTrend < knTrendTypes; iTrend++) {
    std::cout << fJsonCategoryNameTrendZoom << " min" << fJsonNameTrendZoom[iTrend] << " : " << fTrendZoomLow[iTrend]
              << std::endl;
    std::cout << fJsonCategoryNameTrendZoom << " max" << fJsonNameTrendZoom[iTrend] << " : " << fTrendZoomHigh[iTrend]
              << std::endl;
  }

  // Print plotting configuration values
  std::cout << fJsonLumiPerIovFile << " : " << fLumiPerIovFile << std::endl;
  std::cout << fJsonIovListMode << " : " << fIovListMode << std::endl;
  std::cout << fJsonNameLegendTextForAllRuns << " : " << fLegendTextForAllRuns << std::endl;

  std::cout << "Wide pT bin borders: " << std::endl;
  for (double myPt : fWidePtBinBorders) {
    std::cout << myPt << " ";
  }
  std::cout << std::endl;

  std::cout << fJsonDrawPlotsForEachIOV << " : " << fDrawPlotsForEachIOV << std::endl;
  std::cout << fJsonNIovInOnePlot << " : " << fNIovInOnePlot << std::endl;
  std::cout << fJsonNormalizeQAplots << " : " << fNormalizeQAplots << std::endl;

  std::cout << fJsonMakeIovListForSlides << " : " << fMakeIovListForSlides << std::endl;
  std::cout << fJsonIovListForSlides << " : " << fIovListForSlides << std::endl;

  // Print the save comment
  std::cout << "Saved files are given a comment: " << fSaveComment << std::endl;
}

// Getter for track QA histogram drawing flag
bool JetHtPlotConfiguration::drawTrackQA() const { return fDrawTrackQA; }

// Getter for dxy and dz histogram drawing flags
bool JetHtPlotConfiguration::drawHistogram(const int iHistogram) const { return fDrawHistogram[iHistogram]; }

// Getter for profile drawing flags
bool JetHtPlotConfiguration::drawProfile(const int iProfile) const { return fDrawProfile[iProfile]; }

// Getter for drawing reference profile
bool JetHtPlotConfiguration::drawReferenceProfile() const { return fDrawReferenceProfile; }

// Getter for drawing central eta histograms to the all runs summary plots
bool JetHtPlotConfiguration::drawCentralEtaSummaryProfile() const { return fDrawCentralEtaSummaryProfile; }

// Getter for total legend shift in x-direction for the profile plots
double JetHtPlotConfiguration::profileLegendShiftTotalX() const { return fProfileLegendShiftTotalX; }

// Getter for total legend shift in x-direction for the profile plots
double JetHtPlotConfiguration::profileLegendShiftTotalY() const { return fProfileLegendShiftTotalY; }

// Getter for columnwise legend shift in x-direction for iColumnth column in the profile plots
double JetHtPlotConfiguration::profileLegendShiftColumnX(const int iColumn) const {
  if (iColumn < 0 || iColumn >= kMaxLegendColumns)
    return 0;
  return fProfileLegendShiftColumnX[iColumn];
}

// Getter for columnwise legend shift in x-direction for iColumnth column in the profile plots
double JetHtPlotConfiguration::profileLegendShiftColumnY(const int iColumn) const {
  if (iColumn < 0 || iColumn >= kMaxLegendColumns)
    return 0;
  return fProfileLegendShiftColumnY[iColumn];
}

// Getter for text size in profile plots
double JetHtPlotConfiguration::profileLegendTextSize() const { return fProfileLegendTextSize; }

// Getter for text font in profile plots
int JetHtPlotConfiguration::profileLegendTextFont() const { return fProfileLegendTextFont; }

// Getter for text string used to describe all runs
std::string JetHtPlotConfiguration::legendTextForAllRuns() const { return fLegendTextForAllRuns; }

// Getter for low end of profile axis zooms
double JetHtPlotConfiguration::profileZoomLow(const int iProfile) const { return fProfileZoomLow[iProfile]; }

// Getter for high end of profile axis zooms
double JetHtPlotConfiguration::profileZoomHigh(const int iProfile) const { return fProfileZoomHigh[iProfile]; }

// Getter for trend drawing flags
bool JetHtPlotConfiguration::drawTrend(const int iTrend) const { return fDrawTrend[iTrend]; }

// Getter for total legend shift in x-direction for the trend plots
double JetHtPlotConfiguration::trendLegendShiftTotalX() const { return fTrendLegendShiftTotalX; }

// Getter for total legend shift in x-direction for the trend plots
double JetHtPlotConfiguration::trendLegendShiftTotalY() const { return fTrendLegendShiftTotalY; }

// Getter for text size in trend plots
double JetHtPlotConfiguration::trendLegendTextSize() const { return fTrendLegendTextSize; }

// Getter for text font in trend plots
int JetHtPlotConfiguration::trendLegendTextFont() const { return fTrendLegendTextFont; }

// Getter for drawing tags to the trend plots
bool JetHtPlotConfiguration::drawTrendTag() const { return fDrawTrendTag; }

// Getter for tag list for the trend plot
std::vector<std::string> JetHtPlotConfiguration::trendTagText() const { return fTrendTagText; }

// Getter for x-positions of the tags in the trend plots
std::vector<double> JetHtPlotConfiguration::trendTagPositionX() const { return fTrendTagPositionX; }

// Getter for y-positions of the tags in the trend plots
std::vector<double> JetHtPlotConfiguration::trendTagPositionY() const { return fTrendTagPositionY; }

// Getter for text size in tags of trend plots
double JetHtPlotConfiguration::trendTagTextSize() const { return fTrendTagTextSize; }

// Getter for text font in tags of trend plots
int JetHtPlotConfiguration::trendTagTextFont() const { return fTrendTagTextFont; }

// Getter for trend plot canvas height
int JetHtPlotConfiguration::trendCanvasHeight() const { return fTrendCanvasHeight; }

// Getter for trend plot canvas width
int JetHtPlotConfiguration::trendCanvasWidth() const { return fTrendCanvasWidth; }

// Getter for the left margin in trend plots
double JetHtPlotConfiguration::trendMarginLeft() const { return fTrendMarginLeft; }

// Getter for the right margin in trend plots
double JetHtPlotConfiguration::trendMarginRight() const { return fTrendMarginRight; }

// Getter for the top margin in trend plots
double JetHtPlotConfiguration::trendMarginTop() const { return fTrendMarginTop; }

// Getter for the bottom margin in trend plots
double JetHtPlotConfiguration::trendMarginBottom() const { return fTrendMarginBottom; }

// Getter for the offset of the x-axis title in trend plots
double JetHtPlotConfiguration::trendTitleOffsetX() const { return fTrendTitleOffsetX; }

// Getter for the offset of the y-axis title in trend plots
double JetHtPlotConfiguration::trendTitleOffsetY() const { return fTrendTitleOffsetY; }

// Getter for the size of the x-axis title in trend plots
double JetHtPlotConfiguration::trendTitleSizeX() const { return fTrendTitleSizeX; }

// Getter for the size of the y-axis title in trend plots
double JetHtPlotConfiguration::trendTitleSizeY() const { return fTrendTitleSizeY; }

// Getter for the offset of the x-axis label in trend plots
double JetHtPlotConfiguration::trendLabelOffsetX() const { return fTrendLabelOffsetX; }

// Getter for the offset of the y-axis label in trend plots
double JetHtPlotConfiguration::trendLabelOffsetY() const { return fTrendLabelOffsetY; }

// Getter for the size of the x-axis label in trend plots
double JetHtPlotConfiguration::trendLabelSizeX() const { return fTrendLabelSizeX; }

// Getter for the size of the y-axis label in trend plots
double JetHtPlotConfiguration::trendLabelSizeY() const { return fTrendLabelSizeY; }

// Getter for low end of trend axis zooms
double JetHtPlotConfiguration::trendZoomLow(const int iTrend) const { return fTrendZoomLow[iTrend]; }

// Getter for high end of trend axis zooms
double JetHtPlotConfiguration::trendZoomHigh(const int iTrend) const { return fTrendZoomHigh[iTrend]; }

// Getter for number of input files
int JetHtPlotConfiguration::nInputFiles() const { return fInputFileNames.size(); }

// Getter for input file of index iFile
std::string JetHtPlotConfiguration::inputFile(const int iFile) const {
  const int inputFileCount = fInputFileNames.size();
  if (iFile < 0 || iFile >= inputFileCount)
    return "";
  return fInputFileNames.at(iFile);
}

// Getter for input file vector
std::vector<std::string> JetHtPlotConfiguration::inputFiles() const { return fInputFileNames; }

// Getter for a comment added to the legend
std::string JetHtPlotConfiguration::legendComment(const int iComment) const {
  const int commentCount = fLegendComments.size();
  if (iComment < 0 || iComment >= commentCount)
    return "";
  return fLegendComments.at(iComment);
}

// Getter for the marker color related to an alignment
int JetHtPlotConfiguration::markerColor(const int iFile) const {
  if (iFile < 0)
    return kBlack;
  if (iFile > 10)
    return kBlack;
  const int markerColorCount = fMarkerColor.size();
  if (iFile >= markerColorCount)
    return fDefaultColors[iFile];
  return fMarkerColor.at(iFile);
}

// Getter for the marker style related to an alignment
int JetHtPlotConfiguration::markerStyle(const int iFile) const {
  const int markerStyleCount = fMarkerStyle.size();
  if (iFile < 0 || iFile >= markerStyleCount)
    return fDefaultStyle;
  return fMarkerStyle.at(iFile);
}

// Getter for the marker size related to an alignment
int JetHtPlotConfiguration::markerSize(const int iFile) const {
  const int markerSizeCount = fMarkerSize.size();
  if (iFile < 0 || iFile >= markerSizeCount)
    return fDefaultMarkerSize;
  return fMarkerSize.at(iFile);
}

// Getter for flags to copy error bar colors related to an alignment
bool JetHtPlotConfiguration::copyErrorColor(const int iFile) const {
  const int errorCopyCount = fCopyErrorColor.size();
  if (iFile < 0 || iFile >= errorCopyCount)
    return false;
  return fCopyErrorColor.at(iFile);
}

// Getter for the luminosity per IOV file
const char* JetHtPlotConfiguration::lumiPerIovFile() const { return fLumiPerIovFile.c_str(); }

// Getter for the IOV list mode
const char* JetHtPlotConfiguration::iovListMode() const { return fIovListMode.c_str(); }

// Getter for the flag for drawing vertical lines to trend plots
bool JetHtPlotConfiguration::drawYearLines() const { return fDrawYearLines; }

// Getter for color of the vertical lines drawn to trend plots
int JetHtPlotConfiguration::yearLineColor() const { return fYearLineColor; }

// Getter for color of the vertical lines drawn to trend plots
int JetHtPlotConfiguration::yearLineWidth() const { return fYearLineWidth; }

// Getter for color of the vertical lines drawn to trend plots
int JetHtPlotConfiguration::yearLineStyle() const { return fYearLineStyle; }

// Getter for the run positions to where vertical lines are drawn
std::vector<int> JetHtPlotConfiguration::runsForLines() const { return fRunsForLines; }

// Getter for bin borders used in the wide pT binned histogram
std::vector<double> JetHtPlotConfiguration::widePtBinBorders() const { return fWidePtBinBorders; }

// Getter for drawing plots for each IOV
bool JetHtPlotConfiguration::drawPlotsForEachIOV() const { return fDrawPlotsForEachIOV; }

// Getter for number of IOVs plotted in each figure
int JetHtPlotConfiguration::nIovInOnePlot() const { return fNIovInOnePlot; }

// Getter for drawing trends as a function of luminosity
bool JetHtPlotConfiguration::useLuminosityForTrends() const { return fUseLuminosityForTrends; }

// Getter for skipping runs with no data in trend plots
bool JetHtPlotConfiguration::skipRunsWithNoData() const { return fSkipRunsWithNoData; }

// Getter for normalizing QA plots
bool JetHtPlotConfiguration::normalizeQAplots() const { return fNormalizeQAplots; }

// Getter for comment given to saved figures
const char* JetHtPlotConfiguration::saveComment() const { return fSaveComment.c_str(); }

// Getter for flag to produce IOV list for slides
bool JetHtPlotConfiguration::makeIovListForSlides() const { return fMakeIovListForSlides; }

// Getter for the name given to the IOV list for slides
const char* JetHtPlotConfiguration::iovListForSlides() const { return fIovListForSlides.c_str(); }

// Expand environmental variables updating the input string
void JetHtPlotConfiguration::autoExpandEnvironmentVariables(std::string& text) const {
  static std::regex env("\\$\\{?([^}\\/]+)\\}?\\/");
  std::smatch match;
  while (std::regex_search(text, match, env)) {
    const char* s = getenv(match[1].str().c_str());
    const std::string var(s == nullptr ? "" : Form("%s/", s));
    text.replace(match[0].first, match[0].second, var);
  }
}

// Expand environmental variables to a new string
std::string JetHtPlotConfiguration::expandEnvironmentVariables(const std::string& input) const {
  std::string text = input;
  autoExpandEnvironmentVariables(text);
  return text;
}
