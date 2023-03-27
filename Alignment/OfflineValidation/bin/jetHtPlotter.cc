// framework includes
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// C++ includes
#include <iostream>  // Input/output stream. Needed for cout.
#include <vector>

// Root includes
#include "TFile.h"
#include "TString.h"
#include "TH1D.h"
#include "TProfile.h"
#include "TGraphErrors.h"
#include "TLegend.h"
#include "TLine.h"
#include "TSystem.h"
#include "TMath.h"
#include "TLatex.h"

// AllInOneTool includes
#include "Options.h"

// Include a drawing helper class
#include "JetHtPlotConfiguration.h"
#include "JDrawer.h"

// Maximum number of compared files implemented in current code
const int kMaxFiles = 4;

/*
 *  Draw a histogram to canvas
 *
 *  Arguments:
 *    TH1D *histogram[kMaxFiles] = Input histogram and comparison histogram
 *    const char *saveName = Name given for saved figures
 *    bool saveFigures = True: Save figures to file, False: Do not save figures
 *    TString comment[kMaxFiles] = Text written to legend
 *    int legendPosition = Position index of legend: 0 = Right top, 1 = Middle bottom
 *    bool normalize = Normalize the distributions to one so that shapes can be compared
 *    bool logScale = Use logarithmic scale for y-axis
 *    int color[kMaxFiles] = Color used for each files
 */
void drawSingleHistogram(TH1D *histogram[kMaxFiles],
                         const char *saveName,
                         bool saveFigures,
                         TString comment[kMaxFiles],
                         int legendPosition,
                         bool normalize,
                         bool logScale,
                         int color[kMaxFiles]) {
  // Create and setup the histogram drawer
  const auto &drawer = std::make_unique<JDrawer>();
  drawer->SetLogY(logScale);
  drawer->SetTopMargin(0.08);

  // The first histogram in the array is always included, draw it
  histogram[0]->SetLineColor(color[0]);
  if (normalize)
    histogram[0]->Scale(1.0 / histogram[0]->Integral());
  drawer->DrawHistogram(histogram[0]);

  // Draw all the remaining histograms in the array. Calculate how many were drawn
  int nHistograms = 1;
  for (int iFile = 1; iFile < kMaxFiles; iFile++) {
    if (histogram[iFile]) {
      nHistograms++;
      histogram[iFile]->SetLineColor(color[iFile]);
      if (normalize)
        histogram[iFile]->Scale(1.0 / histogram[iFile]->Integral());
      histogram[iFile]->Draw("Same");
    }
  }

  // Create a legend with size depending on the number of histograms
  double legendX1 = 0.6;
  double legendY1 = 0.9 - 0.07 * nHistograms;
  double legendX2 = 0.9;
  double legendY2 = 0.9;
  if (legendPosition == 1) {
    legendX1 = 0.35;
    legendY1 = 0.4 - 0.07 * nHistograms + 0.07 * (nHistograms / 4);
    legendX2 = 0.65;
    legendY2 = 0.4 + 0.07 * (nHistograms / 4);
  }
  TLegend *legend = new TLegend(legendX1, legendY1, legendX2, legendY2);
  legend->SetFillStyle(0);
  legend->SetBorderSize(0);
  legend->SetTextSize(0.05);
  legend->SetTextFont(62);

  // Add all the histograms to the legend and draw it
  legend->AddEntry(histogram[0], comment[0], "l");
  for (int iFile = 1; iFile < kMaxFiles; iFile++) {
    if (histogram[iFile]) {
      legend->AddEntry(histogram[iFile], comment[iFile], "l");
    }
  }
  legend->Draw();

  // Save the drawn figure to a file
  if (saveFigures)
    gPad->GetCanvas()->SaveAs(Form("output/%s.pdf", saveName));
}

/*
 *  Find a good range for the y-axes of the histograms
 *
 *  Arguments:
 *   TH1D *histogram[kMaxFiles] = Histogram array from which the Y-axis range is searched
 *
 *  return:
 *   Tuple containing the minimum and maximum zoom for the histogram Y-axis
 */
std::tuple<double, double> findHistogramYaxisRange(TH1D *histogram[kMaxFiles]) {
  // Find the smallest minimum and largest maximum from the histograms
  double minValue = histogram[0]->GetMinimum();
  double maxValue = histogram[0]->GetMaximum();

  double newMinValue, newMaxValue;

  for (int iFile = 1; iFile < kMaxFiles; iFile++) {
    if (histogram[iFile]) {
      newMinValue = histogram[iFile]->GetMinimum();
      newMaxValue = histogram[iFile]->GetMaximum();

      if (newMinValue < minValue)
        minValue = newMinValue;
      if (newMaxValue > maxValue)
        maxValue = newMaxValue;
    }
  }

  // Add some margin below the minimum and over the maximum
  double margin = 0.075;
  double totalSpan = maxValue - minValue;
  minValue = minValue - margin * totalSpan;
  if (minValue < 0)
    minValue = 0;
  maxValue = maxValue + margin * totalSpan;

  // Return the minimum and maximum values
  return std::make_tuple(minValue, maxValue);
}

/*
 *  Construct vectors of ptHat files and the ptHat value in each of those files
 *
 *  Arguments:
 *   const char* inputFile = File containing the list of ptHat separated files and ptHat bin in each file
 *
 *  return:
 *   Tuple containing list of runs, luminosities, and histogram names within the validation files
 */
std::tuple<std::vector<std::string>, std::vector<int>> ptHatFilesAndValues(const char *inputFile) {
  // Create vectors for ptHat files and values
  std::vector<std::string> ptHatFileList;
  std::vector<int> ptHatList;

  // Helper variables to read the file
  std::string lineInFile;

  // Open the input file for reading
  std::ifstream ptHatFile(inputFile);

  // Go through the file line by line. Each line has one file and information about the ptHat in the file.
  while (std::getline(ptHatFile, lineInFile)) {
    auto stringStream = std::istringstream{lineInFile};
    auto wordString = std::string{};

    // Find the file name and ptHat in that file
    stringStream >> wordString;
    ptHatFileList.push_back(wordString);
    stringStream >> wordString;
    ptHatList.push_back(std::stoi(wordString));
  }

  // After all the vectors are filled, return them
  return std::make_tuple(ptHatFileList, ptHatList);
}

/*
 * Get a selected histogram combining different pThat bins
 */
TH1 *ptHatCombinedHistogram(std::vector<TFile *> ptHatFiles, std::vector<int> ptHatValues, const char *histogramName) {
  // Weight for bins:       30to50     50to80      80to120    120to170    170to300    300to470    470to600
  double ptHatWeights[] = {
      138800000,
      19110000,
      2735000,
      466200,
      117200,
      7763,
      641.0,
      //                       600to800   800to1000  1000to1400  1400to1800  1800to2400  2400to3200  3200toInf
      185.7,
      32.02,
      9.375,
      0.8384,
      0.1133,
      0.006746,
      0.0001623};
  const int nPtHat = 14;
  int ptHatBoundaries[] = {30, 50, 80, 120, 170, 300, 470, 600, 800, 1000, 1400, 1800, 2400, 3200};

  TFile *openFile = ptHatFiles.at(0);
  TH1 *combinedHistogram = (TH1 *)openFile->Get(histogramName)->Clone(Form("%sClone", histogramName));
  TH1 *currentHistogram;
  combinedHistogram->Reset("ICES");
  int ptHatIndex = -1;

  const int nFiles = ptHatFiles.size();

  for (int iFile = 0; iFile < nFiles; iFile++) {
    ptHatIndex = -1;
    for (int iPtHat = 0; iPtHat < nPtHat; iPtHat++) {
      if (ptHatValues.at(iFile) == ptHatBoundaries[iPtHat]) {
        ptHatIndex = iPtHat;
        break;
      }
    }

    if (ptHatIndex < 0) {
      std::cout << "Could not find pT hat boundary " << ptHatValues.at(iFile) << " for file " << ptHatFiles.at(iFile)
                << std::endl;
      std::cout << "Please check your input! It needs to be in the form <fileName> <ptHatBoundary>" << std::endl;
      return nullptr;
    }

    openFile = ptHatFiles.at(iFile);
    currentHistogram = (TH1 *)openFile->Get(histogramName);
    combinedHistogram->Add(currentHistogram, ptHatWeights[ptHatIndex]);
  }

  return combinedHistogram;
}

/*
 *  Construct vectors of runs, luminosities, and histogram names within the validation files from an input file
 *  containing a list of runs and luminosities attached to each one
 *
 *  Arguments:
 *   const char* inputFile = File containing the run and luminosity lists
 *   const char* iovListMode = List mode for IOVs. Tells if every run in the IOV list is it's own IOV or a border run in a set of runs for IOV.
 *
 *  return:
 *   Tuple containing list of runs, luminosities, histogram names and legend strings within the validation files
 */
std::tuple<std::vector<int>, std::vector<double>, std::vector<TString>, std::vector<TString>> runAndLumiLists(
    const char *inputFile, const char *iovListMode) {
  // Create vectors for each list
  std::vector<int> iovVector;
  std::vector<double> lumiPerIov;
  std::vector<TString> iovNames;
  std::vector<TString> iovLegend;

  // Helper variables to read the file
  std::string lineInFile;
  int thisIov;
  double thisLumi;

  // Load the iovList
  std::ifstream iovList(edm::FileInPath(inputFile).fullPath().c_str());
  if (!iovList.good()) {
    edm::LogError("jetHTPlotter") << __PRETTY_FUNCTION__ << "\n Input file: " << inputFile
                                  << " is corrupt or not existing";
    return std::make_tuple(iovVector, lumiPerIov, iovNames, iovLegend);
  }

  // Go through the file line by line. Each line has an IOV boundary and luminosity for this IOV.
  while (std::getline(iovList, lineInFile)) {
    // Read the iov and luminosity from the file
    std::istringstream lineStream(lineInFile);
    lineStream >> thisIov;
    lineStream >> thisLumi;

    // Push the extracted numbers to vectors
    iovVector.push_back(thisIov);
    lumiPerIov.push_back(thisLumi);
  }

  // Create names for the different IOV:s
  for (std::vector<int>::size_type i = 1; i < iovVector.size(); i++) {
    iovNames.push_back(Form("iov%d-%d", iovVector.at(i - 1), iovVector.at(i) - 1));
  }

  // For the legend naming, use just the run if specified in iovListMode
  TString listModeParser = iovListMode;
  if (listModeParser.EqualTo("iov", TString::kIgnoreCase)) {
    for (std::vector<int>::size_type i = 1; i < iovVector.size(); i++) {
      iovLegend.push_back(Form("IOV %d-%d", iovVector.at(i - 1), iovVector.at(i) - 1));
    }
  } else {
    for (std::vector<int>::size_type i = 0; i < iovVector.size() - 1; i++) {
      iovLegend.push_back(Form("Run %d", iovVector.at(i)));
    }
  }

  // Add the iov integrated histograms after the histograms per IOV
  iovNames.push_back("all");
  iovNames.push_back("central");
  iovLegend.push_back("All");
  iovLegend.push_back("Central");

  // After all the vectors are filled, return them
  return std::make_tuple(iovVector, lumiPerIov, iovNames, iovLegend);
}

/*
 *  Scale the x-axis of a given graph by integrated luminosity. The error on x-axis represents the luminosity of each run.
 *
 *  Arguments:
 *   TGraphErrors *runGraph = Graph containing dxy or dz error trends as a function of run index
 *   std::vector<double> lumiPerIov = Vector showing luminosities for each run. Indices must match with the graph
 *   bool skipMissingRuns = If there is no data for a run in the runlist, do not assign luminosity for it
 *
 *  Return: std::vector<double> New luminosity list where skipped runs are set to zero
 *
 */
std::vector<double> scaleGraphByLuminosity(TGraphErrors *runGraph,
                                           std::vector<double> lumiPerIov,
                                           bool skipMissingRuns) {
  // Read the number of runs from the graph by run number
  int nRuns = runGraph->GetN();

  // Helper variables
  std::vector<double> xAxisValues, yAxisValues, xAxisErrors, yAxisErrors;

  int iRun = 0;
  int offset = 0;
  double lumiFactor = 1000;  // Scale factor value to have luminosity expressed in fb^-1

  double runIndex, yValue;
  double xValue = 0;
  double epsilon = 1e-5;

  std::vector<double> lumiPerIovWithSkips;

  // Loop over all runs, remove zeros and for runs with content, replace x-axis index by luminosity
  while (iRun < nRuns) {
    runGraph->GetPoint(iRun, runIndex, yValue);

    if (lumiPerIov.at(iRun + offset) == 0 || (yValue < epsilon && skipMissingRuns)) {
      nRuns--;
      runGraph->RemovePoint(iRun);
      offset++;

      // Fill vector where lumi for skipped runs is set to zero
      lumiPerIovWithSkips.push_back(0);
    } else {
      xValue += lumiPerIov.at(iRun + offset) / lumiFactor;
      xAxisValues.push_back(xValue - (lumiPerIov.at(iRun + offset) / (lumiFactor * 2)));
      xAxisErrors.push_back(lumiPerIov.at(iRun + offset) / (lumiFactor * 2));
      yAxisValues.push_back(yValue);
      yAxisErrors.push_back(runGraph->GetErrorY(iRun));

      // Fill vector where lumi for skipped runs is set to zero
      lumiPerIovWithSkips.push_back(lumiPerIov.at(iRun + offset));

      iRun++;
    }
  }  // Loop over all runs in original histogram

  // Delete remaining old content and replace it with new one calculated from luminosities
  runGraph->GetHistogram()->Delete();
  runGraph->SetHistogram(nullptr);
  for (int iRun = 0; iRun < nRuns; iRun++) {
    runGraph->SetPoint(iRun, xAxisValues.at(iRun), yAxisValues.at(iRun));
    runGraph->SetPointError(iRun, xAxisErrors.at(iRun), yAxisErrors.at(iRun));
  }

  return lumiPerIovWithSkips;
}

/*
 * Get the total luminosity upto a given run
 *
 *  std::vector<double> lumiPerIov = Vector containing luminosity information
 *  std::vector<int> iovVector = Vector containing run information
 *  in runNumber = Run upto which the luminosity is calculated
 */
double luminosityBeforeRun(std::vector<double> lumiPerIov, std::vector<int> iovVector, int runNumber) {
  double lumiFactor = 1000;  // Scale factor value to have luminosity expressed in fb^-1

  int nRuns = lumiPerIov.size();
  double luminosityBefore = 0;

  for (int iRun = 0; iRun < nRuns; iRun++) {
    if (runNumber <= iovVector.at(iRun))
      return luminosityBefore;

    luminosityBefore += lumiPerIov.at(iRun) / lumiFactor;
  }

  return luminosityBefore;
}

/*
 * Macro for plotting figures for the study of jet HT sample
 */
void jetHtPlotter(std::string configurationFileName) {
  // ======================================================
  // ================== Configuration =====================
  // ======================================================

  JetHtPlotConfiguration *configurationGiver = new JetHtPlotConfiguration();
  configurationGiver->readJsonFile(configurationFileName);
  //configurationGiver->printConfiguration();

  enum enumHistogramType { kDz, kDzError, kDxy, kDxyError, knHistogramTypes };
  TString histogramName[knHistogramTypes] = {"dz", "dzerr", "dxy", "dxyerr"};
  TString histogramXaxis[knHistogramTypes] = {
      "d_{z} (#mum)", "#sigma(d_{z}) (#mum)", "d_{xy} (#mum)", "#sigma(d_{xy}) (#mum)"};
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
  enum enumTrendType { kDzErrorTrend, kDxyErrorTrend, knTrendTypes };
  TString profileName[knProfileTypes] = {"dzErrVsPt",
                                         "dzErrVsPhi",
                                         "dzErrVsEta",
                                         "dxyErrVsPt",
                                         "dxyErrVsPhi",
                                         "dxyErrVsEta",
                                         "dzErrVsPtWide",
                                         "dxyErrVsPtWide"};
  TString profileXaxis[knProfileTypes] = {
      "p_{T} (GeV)", "#varphi", "#eta", "p_{T} (GeV)", "#varphi", "#eta", "p_{T} bin", "p_{T} bin"};
  TString profileYaxis[knProfileTypes] = {"d_{z}", "d_{z}", "d_{z}", "d_{xy}", "d_{xy}", "d_{xy}", "d_{z}", "d_{xy}"};
  TString trendName[] = {"dzErr", "dxyErr"};
  bool drawHistogram[knHistogramTypes];
  bool drawProfile[knProfileTypes];
  bool drawTrend[knTrendTypes];

  bool drawTrackQA = configurationGiver->drawTrackQA();                       // Draw track and vertex QA figures
  drawHistogram[kDz] = configurationGiver->drawHistogram(kDz);                // Draw the dz histograms
  drawHistogram[kDzError] = configurationGiver->drawHistogram(kDzError);      // Draw the dz error histograms
  drawProfile[kDzErrorVsPt] = configurationGiver->drawProfile(kDzErrorVsPt);  // Draw mean dz error as a function of pT
  drawProfile[kDzErrorVsPhi] =
      configurationGiver->drawProfile(kDzErrorVsPhi);  // Draw mean dz error as a function of phi
  drawProfile[kDzErrorVsEta] =
      configurationGiver->drawProfile(kDzErrorVsEta);  // Draw mean dz error as a function of eta
  drawProfile[kDzErrorVsPtWide] =
      configurationGiver->drawProfile(kDzErrorVsPtWide);                    // Draw mean dz error in wide pT bins
  drawHistogram[kDxy] = configurationGiver->drawHistogram(kDxy);            // Draw the dxy histograms
  drawHistogram[kDxyError] = configurationGiver->drawHistogram(kDxyError);  // Draw the dxy error histograms
  drawProfile[kDxyErrorVsPt] =
      configurationGiver->drawProfile(kDxyErrorVsPt);  // Draw the dxy error as a function of pT
  drawProfile[kDxyErrorVsPhi] =
      configurationGiver->drawProfile(kDxyErrorVsPhi);  // Draw the dxy error as a function of phi
  drawProfile[kDxyErrorVsEta] =
      configurationGiver->drawProfile(kDxyErrorVsEta);  // Draw the dxy error as a function of eta
  drawProfile[kDxyErrorVsPtWide] =
      configurationGiver->drawProfile(kDxyErrorVsPtWide);                  // Draw mean dxy error in wide pT bins
  bool drawReferenceProfile = configurationGiver->drawReferenceProfile();  // Draw reference profile to single IOV plots
  bool drawCentralEtaSummaryProfile =
      configurationGiver->drawCentralEtaSummaryProfile();  // Draw central eta histograms to all runs summary profiles
  drawTrend[kDzErrorTrend] = configurationGiver->drawTrend(kDzErrorTrend);    // Draw the trend plots for dz errors
  drawTrend[kDxyErrorTrend] = configurationGiver->drawTrend(kDxyErrorTrend);  // Draw the trend plots for dxy errors

  const int nMaxLegendColumns = 3;
  double profileLegendShiftTotalX =
      configurationGiver->profileLegendShiftTotalX();  // Total legend position shift in x-direction for profile plots
  double profileLegendShiftTotalY =
      configurationGiver->profileLegendShiftTotalY();  // Total legend position shift in y-direction for profile plots
  double profileLegendShiftColumnX[nMaxLegendColumns];
  double profileLegendShiftColumnY[nMaxLegendColumns];
  for (int iColumn = 0; iColumn < nMaxLegendColumns; iColumn++) {
    profileLegendShiftColumnX[iColumn] = configurationGiver->profileLegendShiftColumnX(
        iColumn);  // Columnwise legend position shift in x-direction for profile plots
    profileLegendShiftColumnY[iColumn] = configurationGiver->profileLegendShiftColumnY(
        iColumn);  // Columnwise legend position shift in x-direction for profile plots
  }
  double profileLegendTextSize = configurationGiver->profileLegendTextSize();  // Legend text size for profile plots
  int profileLegendTextFont = configurationGiver->profileLegendTextFont();     // Legend text font for profile plots
  TString legendTextForAllRuns =
      configurationGiver->legendTextForAllRuns();  // Legend text referring to all runs in the file

  double trendLegendShiftTotalX =
      configurationGiver->trendLegendShiftTotalX();  // Total legend position shift in x-direction for trend plots
  double trendLegendShiftTotalY =
      configurationGiver->trendLegendShiftTotalY();  // Total legend position shift in y-direction for trend plots
  double trendLegendTextSize = configurationGiver->trendLegendTextSize();  // Legend text size for trend plots
  int trendLegendTextFont = configurationGiver->trendLegendTextFont();     // Legend text font for trend plots

  bool drawTrendTag = configurationGiver->drawTrendTag();                      // Draw manual tags to the trend plots
  std::vector<std::string> trendTagText = configurationGiver->trendTagText();  // Drawn tag texts for trend plots
  std::vector<double> trendTagPositionX = configurationGiver->trendTagPositionX();  // Trend tag x-positions
  std::vector<double> trendTagPositionY = configurationGiver->trendTagPositionY();  // Trend tag y-positions
  double trendTagTextSize = configurationGiver->trendTagTextSize();                 // Tag text size for trend plots
  int trendTagTextFont = configurationGiver->trendTagTextFont();                    // Tag text font for trend plots

  int trendCanvasHeight = configurationGiver->trendCanvasHeight();     // Canvas height for trend plots
  int trendCanvasWidth = configurationGiver->trendCanvasWidth();       // Canvas width for trend plots
  double trendMarginLeft = configurationGiver->trendMarginLeft();      // Left margin in trend plots
  double trendMarginRight = configurationGiver->trendMarginRight();    // Right margin in trend plots
  double trendMarginTop = configurationGiver->trendMarginTop();        // Top margin in trend plots
  double trendMarginBottom = configurationGiver->trendMarginBottom();  // Bottom margin in trend plots
  double trendTitleOffsetX = configurationGiver->trendTitleOffsetX();  // x-axis title offset in trend plots
  double trendTitleOffsetY = configurationGiver->trendTitleOffsetY();  // y-axis title offset in trend plots
  double trendTitleSizeX = configurationGiver->trendTitleSizeX();      // x-axis title size in trend plots
  double trendTitleSizeY = configurationGiver->trendTitleSizeY();      // y-axis title size in trend plots
  double trendLabelOffsetX = configurationGiver->trendLabelOffsetX();  // x-axis label offset in trend plots
  double trendLabelOffsetY = configurationGiver->trendLabelOffsetY();  // y-axis label offset in trend plots
  double trendLabelSizeX = configurationGiver->trendLabelSizeX();      // x-axis label size in trend plots
  double trendLabelSizeY = configurationGiver->trendLabelSizeY();      // y-axis label size in trend plots

  bool drawPlotsForEachIOV =
      configurationGiver
          ->drawPlotsForEachIOV();  // True = Draw profile and histogram plots for every IOV. False = only draw average over all runs
  bool useLuminosityForTrends =
      configurationGiver
          ->useLuminosityForTrends();  // True = Draw trends as a function of luminosity. False = Draw trends as a function of run index
  bool skipRunsWithNoData =
      configurationGiver
          ->skipRunsWithNoData();  // True = Do not draw empty space if run in list is missing data. False = Draw empty space

  int colors[] = {kBlue, kRed, kGreen + 2, kMagenta, kCyan, kViolet + 3, kOrange, kPink - 7, kSpring + 3, kAzure - 7};
  int nIovInOnePlot = configurationGiver->nIovInOnePlot();  // Define how many iov:s are drawn to the same plot

  double profileZoomLow[knProfileTypes];
  double profileZoomHigh[knProfileTypes];
  double trendZoomLow[knTrendTypes];
  double trendZoomHigh[knTrendTypes];

  for (int iProfile = 0; iProfile < knProfileTypes; iProfile++) {
    profileZoomLow[iProfile] = configurationGiver->profileZoomLow(iProfile);
    profileZoomHigh[iProfile] = configurationGiver->profileZoomHigh(iProfile);
  }
  for (int iTrend = 0; iTrend < knTrendTypes; iTrend++) {
    trendZoomLow[iTrend] = configurationGiver->trendZoomLow(iTrend);
    trendZoomHigh[iTrend] = configurationGiver->trendZoomHigh(iTrend);
  }

  const std::vector<double> widePtBinBorders = configurationGiver->widePtBinBorders();
  const int nWidePtBins = widePtBinBorders.size();

  bool normalizeQAplots = configurationGiver->normalizeQAplots();  // Divide in QA plot yield by its integral
  bool saveFigures = true;
  const char *saveComment = configurationGiver->saveComment();

  int compareFiles = configurationGiver->nInputFiles();
  if (compareFiles == 0)
    return;  // Cannot do plotting without files!
  if (compareFiles > kMaxFiles)
    compareFiles = kMaxFiles;  // Four is maximum number of files in current implementation

  // If we have more then one file, we should not draw more than one IOV per plot and plot all the files instead
  if (compareFiles > 1)
    nIovInOnePlot = 1;

  // Read the input file names. If file names end with .txt, they are assumed to be lists of files in different ptHat bins
  TString inputFileName[kMaxFiles];
  bool loadFromList[kMaxFiles] = {false, false, false, false};
  TString legendComment[kMaxFiles];
  int fileColor[kMaxFiles];
  int fileMarkerStyle[kMaxFiles];
  int fileMarkerSize[kMaxFiles];
  bool copyErrorColor[kMaxFiles];

  for (int iFile = 0; iFile < kMaxFiles; iFile++) {
    inputFileName[iFile] = configurationGiver->inputFile(iFile);
    if (inputFileName[iFile].EndsWith("txt"))
      loadFromList[iFile] = true;
    legendComment[iFile] = configurationGiver->legendComment(iFile);  // Text written to the legend for each file
    fileColor[iFile] = configurationGiver->markerColor(iFile);        // Color of markers related to this file
    fileMarkerStyle[iFile] = configurationGiver->markerStyle(iFile);  // Style for markers related to this file
    fileMarkerSize[iFile] = configurationGiver->markerSize(iFile);    // Size for markers related to this file
    copyErrorColor[iFile] =
        configurationGiver->copyErrorColor(iFile);  // Copy the marker color for error bars for this file
  }

  // Prepare an IOV list that can be used with the slide generation script
  bool makeIovListForSlides = configurationGiver->makeIovListForSlides();
  const char *iovListForSlides = configurationGiver->iovListForSlides();

  // Helper variable to ensure that each IOV is added exactly once to the list
  int profileIndexForIovList = 0;
  for (int iProfile = 0; iProfile < knProfileTypes; iProfile++) {
    if (drawProfile[iProfile]) {
      profileIndexForIovList = iProfile;
      break;
    }
  }

  TString plotTitle = " ";  // Title given to the plot

  // Year boundary runs
  bool drawYearLines = configurationGiver->drawYearLines();            // Draw lines between data taking years
  std::vector<int> linePosition = configurationGiver->runsForLines();  // Positions of the above lines
  int yearLineColor = configurationGiver->yearLineColor();             // Color of the above lines
  int yearLineWidth = configurationGiver->yearLineWidth();             // Width of the above lines
  int yearLineStyle = configurationGiver->yearLineStyle();             // Style of the above lines

  // ======================================================
  // ================ Configuration done ==================
  // ======================================================

  // =======================================================
  //   Make sure that the output folder for figures exists
  // =======================================================

  if (saveFigures) {
    TString outputFolderStatus = gSystem->GetFromPipe("if [ -d output ]; then echo true; else echo false; fi");
    if (outputFolderStatus == "false") {
      gSystem->Exec("mkdir output");
    }
  }

  // ===============================================================
  //   Read different ptHat files for combining those with weights
  // ===============================================================

  // List for all the different ptHat files to be combined
  std::vector<std::string> ptHatFileNameList[kMaxFiles];
  std::vector<TFile *> ptHatFileList[kMaxFiles];
  std::vector<int> ptHatList[kMaxFiles];
  int nFiles;

  for (int iInput = 0; iInput < kMaxFiles; iInput++) {
    if (loadFromList[iInput]) {
      std::tie(ptHatFileNameList[iInput], ptHatList[iInput]) = ptHatFilesAndValues(inputFileName[iInput]);

      // Open all files
      nFiles = ptHatFileNameList[iInput].size();
      for (int iFile = 0; iFile < nFiles; iFile++) {
        ptHatFileList[iInput].push_back(TFile::Open(ptHatFileNameList[iInput].at(iFile).data()));
      }
    }
  }

  // ===============================================
  //        Read the run and luminosity list
  // ===============================================

  // Luminosity per run file
  const char *iovAndLumiFile = configurationGiver->lumiPerIovFile();
  const char *iovListMode = configurationGiver->iovListMode();

  // Create a vector for a new iovList
  std::vector<int> iovVector;
  std::vector<double> lumiPerIov;
  std::vector<double> lumiPerIovWithSkips;
  std::vector<TString> iovNames;
  std::vector<TString> iovLegend;

  std::tie(iovVector, lumiPerIov, iovNames, iovLegend) = runAndLumiLists(iovAndLumiFile, iovListMode);
  // protection against empty input
  if (iovVector.empty()) {
    edm::LogError("jetHTPlotter") << __PRETTY_FUNCTION__ << "\n The list of input IOVs is empty. Exiting!";
    return;
  }

  // For the IOV legend, remove the two last entries and replace them with user defined names
  iovLegend.pop_back();
  iovLegend.pop_back();
  iovLegend.push_back(legendTextForAllRuns);
  iovLegend.push_back(Form("%s  |#eta| < 1", legendTextForAllRuns.Data()));

  // ===============================================
  //                IOV list for slides
  // ===============================================

  // If we are preparing an iov list for slides, make a file for that
  std::ofstream iovFileForSlides;
  if (makeIovListForSlides)
    iovFileForSlides.open(iovListForSlides);

  // ===============================================
  //   Create histograms and read them from a file
  // ===============================================

  // Open the input files
  TFile *inputFile[kMaxFiles];
  for (int iFile = 0; iFile < kMaxFiles; iFile++) {
    if (compareFiles > iFile && !loadFromList[iFile])
      inputFile[iFile] = TFile::Open(inputFileName[iFile]);
  }

  // Define the histograms that will be read from the file
  const int nIov = iovNames.size();
  TH1D *hVertex[kMaxFiles];
  TH1D *hTracksPerVertex[kMaxFiles];
  TH1D *hTrackPt[kMaxFiles];
  TH1D *hTrackEta[kMaxFiles];
  TH1D *hTrackPhi[kMaxFiles];
  TH1D *jetHtHistograms[kMaxFiles][knHistogramTypes][nIov];
  TProfile *jetHtProfiles[kMaxFiles][knProfileTypes][nIov];
  TGraphErrors *gBigTrend[kMaxFiles][knTrendTypes][nWidePtBins];

  // Initialize everything to NULL
  for (int iFile = 0; iFile < kMaxFiles; iFile++) {
    hVertex[iFile] = nullptr;
    hTracksPerVertex[iFile] = nullptr;
    hTrackPt[iFile] = nullptr;
    hTrackEta[iFile] = nullptr;
    hTrackPhi[iFile] = nullptr;
    for (int iIov = 0; iIov < nIov; iIov++) {
      for (int iHistogramType = 0; iHistogramType < knHistogramTypes; iHistogramType++) {
        jetHtHistograms[iFile][iHistogramType][iIov] = nullptr;
      }  // histogram type loop
      for (int iProfileType = 0; iProfileType < knProfileTypes; iProfileType++) {
        jetHtProfiles[iFile][iProfileType][iIov] = nullptr;
      }  // profile type loop
    }    // iov loop for reading histograms from file
    for (int iTrend = 0; iTrend < knTrendTypes; iTrend++) {
      for (int iWidePt = 0; iWidePt < nWidePtBins; iWidePt++) {
        gBigTrend[iFile][iTrend][iWidePt] = nullptr;
      }  // Trend loop
    }    // Wide pT bin loop
  }      // file loop

  // Read the histograms from the file
  for (int iFile = 0; iFile < compareFiles; iFile++) {
    if (loadFromList[iFile]) {
      // Alternative loading for ptHat combined values

      hVertex[iFile] = (TH1D *)ptHatCombinedHistogram(ptHatFileList[iFile], ptHatList[iFile], "jetHTAnalyzer/all_nvtx");
      hTracksPerVertex[iFile] =
          (TH1D *)ptHatCombinedHistogram(ptHatFileList[iFile], ptHatList[iFile], "jetHTAnalyzer/h_ntrks");
      hTrackPt[iFile] =
          (TH1D *)ptHatCombinedHistogram(ptHatFileList[iFile], ptHatList[iFile], "jetHTAnalyzer/h_probePt");
      hTrackEta[iFile] =
          (TH1D *)ptHatCombinedHistogram(ptHatFileList[iFile], ptHatList[iFile], "jetHTAnalyzer/h_probeEta");
      hTrackPhi[iFile] =
          (TH1D *)ptHatCombinedHistogram(ptHatFileList[iFile], ptHatList[iFile], "jetHTAnalyzer/h_probePhi");
      for (int iIov = nIov - 2; iIov < nIov - 1; iIov++) {
        for (int iHistogramType = 0; iHistogramType < knHistogramTypes; iHistogramType++) {
          if (drawHistogram[iHistogramType]) {
            jetHtHistograms[iFile][iHistogramType][iIov] = (TH1D *)ptHatCombinedHistogram(
                ptHatFileList[iFile],
                ptHatList[iFile],
                Form("jetHTAnalyzer/%s_%s", iovNames.at(iIov).Data(), histogramName[iHistogramType].Data()));
          }  // if for drawing histogram
        }    // histogram type loop
        for (int iProfileType = 0; iProfileType < knProfileTypes; iProfileType++) {
          if (drawProfile[iProfileType] || (drawTrend[kDzErrorTrend] && iProfileType == kDzErrorVsPtWide) ||
              (drawTrend[kDxyErrorTrend] && iProfileType == kDxyErrorVsPtWide)) {
            jetHtProfiles[iFile][iProfileType][iIov] = (TProfile *)ptHatCombinedHistogram(
                ptHatFileList[iFile],
                ptHatList[iFile],
                Form("jetHTAnalyzer/%s_%s", iovNames.at(iIov).Data(), profileName[iProfileType].Data()));
          }  // if for drawing profile
        }    // profile type loop
      }      // iov loop for reading histograms from file

    } else {
      // Regular histogram loading

      hVertex[iFile] = (TH1D *)inputFile[iFile]->Get("jetHTAnalyzer/all_nvtx");
      hTracksPerVertex[iFile] = (TH1D *)inputFile[iFile]->Get("jetHTAnalyzer/h_ntrks");
      hTrackPt[iFile] = (TH1D *)inputFile[iFile]->Get("jetHTAnalyzer/h_probePt");
      hTrackEta[iFile] = (TH1D *)inputFile[iFile]->Get("jetHTAnalyzer/h_probeEta");
      hTrackPhi[iFile] = (TH1D *)inputFile[iFile]->Get("jetHTAnalyzer/h_probePhi");
      for (int iIov = 0; iIov < nIov; iIov++) {
        for (int iHistogramType = 0; iHistogramType < knHistogramTypes; iHistogramType++) {
          if (drawHistogram[iHistogramType]) {
            jetHtHistograms[iFile][iHistogramType][iIov] = (TH1D *)inputFile[iFile]->Get(
                Form("jetHTAnalyzer/%s_%s", iovNames.at(iIov).Data(), histogramName[iHistogramType].Data()));
          }  // if for drawing histogram
        }    // histogram type loop
        for (int iProfileType = 0; iProfileType < knProfileTypes; iProfileType++) {
          if (drawProfile[iProfileType] || (drawTrend[kDzErrorTrend] && iProfileType == kDzErrorVsPtWide) ||
              (drawTrend[kDxyErrorTrend] && iProfileType == kDxyErrorVsPtWide)) {
            jetHtProfiles[iFile][iProfileType][iIov] = (TProfile *)inputFile[iFile]->Get(
                Form("jetHTAnalyzer/%s_%s", iovNames.at(iIov).Data(), profileName[iProfileType].Data()));
          }  // if for drawing profile
        }    // profile type loop
      }      // iov loop for reading histograms from file
    }        // Regular histogram loading
  }          // Loop over files

  // Collect the information for the trend graphs
  const int nRuns = iovVector.size() - 1;
  double yValueDz[nRuns];
  double yErrorDz[nRuns];
  double yValueDxy[nRuns];
  double yErrorDxy[nRuns];
  double xValue[nRuns];
  double xError[nRuns];
  TString commentEntry = "";
  if (drawTrend[kDzErrorTrend] || drawTrend[kDxyErrorTrend]) {
    for (int iFile = 0; iFile < compareFiles; iFile++) {
      for (int iWidePt = 0; iWidePt < nWidePtBins; iWidePt++) {
        for (int iRun = 0; iRun < nRuns; iRun++) {
          xValue[iRun] = iRun;
          xError[iRun] = 0;
          if (jetHtProfiles[iFile][kDzErrorVsPtWide][iRun] == nullptr) {
            yValueDz[iRun] = 0;
            yErrorDz[iRun] = 0;
          } else {
            yValueDz[iRun] = jetHtProfiles[iFile][kDzErrorVsPtWide][iRun]->GetBinContent(iWidePt + 1);
            yErrorDz[iRun] = jetHtProfiles[iFile][kDzErrorVsPtWide][iRun]->GetBinError(iWidePt + 1);
          }
          if (jetHtProfiles[iFile][kDxyErrorVsPtWide][iRun] == nullptr) {
            yValueDxy[iRun] = 0;
            yErrorDxy[iRun] = 0;
          } else {
            yValueDxy[iRun] = jetHtProfiles[iFile][kDxyErrorVsPtWide][iRun]->GetBinContent(iWidePt + 1);
            yErrorDxy[iRun] = jetHtProfiles[iFile][kDxyErrorVsPtWide][iRun]->GetBinError(iWidePt + 1);
          }
        }  // Run loop

        gBigTrend[iFile][kDzErrorTrend][iWidePt] = new TGraphErrors(nRuns, xValue, yValueDz, xError, yErrorDz);
        gBigTrend[iFile][kDxyErrorTrend][iWidePt] = new TGraphErrors(nRuns, xValue, yValueDxy, xError, yErrorDxy);

        // Change x-axis to processed luminosity
        if (useLuminosityForTrends) {
          lumiPerIovWithSkips =
              scaleGraphByLuminosity(gBigTrend[iFile][kDzErrorTrend][iWidePt], lumiPerIov, skipRunsWithNoData);
          scaleGraphByLuminosity(gBigTrend[iFile][kDxyErrorTrend][iWidePt], lumiPerIov, skipRunsWithNoData);
        }

      }  // Wide pT bin loop
    }    // File loop
  }      // If for drawing trends

  // ===============================================
  //                  Draw the plots
  // ===============================================

  const auto &drawer = std::make_unique<JDrawer>();
  TLegend *legend[nMaxLegendColumns];
  int columnOrder[nMaxLegendColumns];
  bool noIovFound = true;
  bool canvasDrawn;
  bool doNotDrawEta;
  double minZoomY, maxZoomY;
  int nDrawnHistograms, nLeftLegend, nRightLegend;
  double legendX1, legendX2, legendY1, legendY2;
  int iLegend = 0;
  int skipColor = 0;
  int nNullHistograms;
  int iHistogram;
  TString legendString;
  TH1D *histogramSearchArray[kMaxFiles];
  for (int iFile = 0; iFile < kMaxFiles; iFile++) {
    histogramSearchArray[iFile] = nullptr;
  }

  // Draw track and vertex histograms
  if (drawTrackQA) {
    drawSingleHistogram(
        hVertex, Form("vertex%s", saveComment), saveFigures, legendComment, 0, normalizeQAplots, false, fileColor);
    drawSingleHistogram(hTracksPerVertex,
                        Form("tracksPerVertex%s", saveComment),
                        saveFigures,
                        legendComment,
                        0,
                        normalizeQAplots,
                        false,
                        fileColor);
    drawSingleHistogram(
        hTrackPt, Form("trackPt%s", saveComment), saveFigures, legendComment, 0, normalizeQAplots, true, fileColor);
    drawSingleHistogram(
        hTrackEta, Form("trackEta%s", saveComment), saveFigures, legendComment, 1, normalizeQAplots, false, fileColor);
    drawSingleHistogram(
        hTrackPhi, Form("trackPhi%s", saveComment), saveFigures, legendComment, 1, normalizeQAplots, false, fileColor);
  }

  // Draw dz and dxy histograms
  for (int iIov = 0; iIov < nIov; iIov++) {
    if (!drawPlotsForEachIOV && iIov < nIov - 2)
      continue;
    for (int iHistogramType = 0; iHistogramType < knHistogramTypes; iHistogramType++) {
      if (drawHistogram[iHistogramType]) {
        double legendX1 = 0.6;
        double legendY1 = 0.9 - 0.07 * compareFiles;
        double legendX2 = 0.9;
        double legendY2 = 0.9;
        if (iHistogramType == kDxy || iHistogramType == kDz) {
          legendX1 = 0.4;
          legendY1 = 0.4 - 0.07 * compareFiles + 0.07 * (compareFiles / 4);
          legendX2 = 0.7;
          legendY2 = 0.4 + 0.07 * (compareFiles / 4);
        }

        legend[0] = new TLegend(legendX1, legendY1, legendX2, legendY2);
        legend[0]->SetFillStyle(0);
        legend[0]->SetBorderSize(0);
        legend[0]->SetTextSize(0.05);
        legend[0]->SetTextFont(62);

        if (jetHtHistograms[0][iHistogramType][iIov] != nullptr) {
          for (int iFile = 0; iFile < compareFiles; iFile++) {
            histogramSearchArray[iFile] = jetHtHistograms[iFile][iHistogramType][iIov];
          }

          std::tie(minZoomY, maxZoomY) = findHistogramYaxisRange(histogramSearchArray);
          jetHtHistograms[0][iHistogramType][iIov]->GetYaxis()->SetRangeUser(minZoomY, maxZoomY);
          jetHtHistograms[0][iHistogramType][iIov]->SetLineColor(fileColor[0]);

          drawer->DrawHistogram(jetHtHistograms[0][iHistogramType][iIov],
                                histogramXaxis[iHistogramType],
                                "Tracks",
                                iovLegend.at(iIov).Data());
          legend[0]->AddEntry(jetHtHistograms[0][iHistogramType][iIov], legendComment[0], "l");

          for (int iFile = 1; iFile < compareFiles; iFile++) {
            if (jetHtHistograms[iFile][iHistogramType][iIov] != nullptr) {
              jetHtHistograms[iFile][iHistogramType][iIov]->SetLineColor(fileColor[iFile]);
              jetHtHistograms[iFile][iHistogramType][iIov]->Draw("same");
              legend[0]->AddEntry(jetHtHistograms[iFile][iHistogramType][iIov], legendComment[iFile], "l");
            }
          }

          legend[0]->Draw();

          // Save the figures
          if (saveFigures) {
            gPad->GetCanvas()->SaveAs(Form(
                "output/%s%s_%s.pdf", histogramName[iHistogramType].Data(), saveComment, iovNames.at(iIov).Data()));
          }

        } else {
          std::cout << "No histogram found for: "
                    << Form("%s_%s", iovNames.at(iIov).Data(), histogramName[iHistogramType].Data()) << std::endl;
        }
      }  // if for drawing histogram
    }    // histogram type loop
  }

  // Draw dz and dxy profiles
  for (int iProfileType = 0; iProfileType < knProfileTypes; iProfileType++) {
    if (drawProfile[iProfileType]) {
      // Set the style for IOV integrated histograms
      for (int iFile = 0; iFile < compareFiles; iFile++) {
        jetHtProfiles[iFile][iProfileType][nIov - 2]->SetLineColor(fileColor[iFile]);
        jetHtProfiles[iFile][iProfileType][nIov - 2]->SetLineWidth(2);
        jetHtProfiles[iFile][iProfileType][nIov - 2]->SetLineStyle(2);
        jetHtProfiles[iFile][iProfileType][nIov - 2]->GetYaxis()->SetRangeUser(profileZoomLow[iProfileType],
                                                                               profileZoomHigh[iProfileType]);
      }

      // Drawing plots for each IOV.
      if (drawPlotsForEachIOV) {
        for (int iIov = 0; iIov < nIov - 2; iIov = iIov + nIovInOnePlot) {
          noIovFound = true;
          iLegend = 0;
          canvasDrawn = false;
          nNullHistograms = 0;

          // If we have more than one file, draw all files to the same figure. Otherwise use the nIovInOnePlot variable for the single file.
          // Set up the IOV:s or files to be drawn to the current plot
          for (int iFile = 0; iFile < compareFiles; iFile++) {
            skipColor = 0;
            for (int iSamePlot = 0; iSamePlot < nIovInOnePlot; iSamePlot++) {
              if (iIov + iSamePlot >= nIov - 2)
                break;  // Do not draw again all or central references
              if (jetHtProfiles[iFile][iProfileType][iIov + iSamePlot] != nullptr) {
                if (nIovInOnePlot > 1) {
                  if (colors[iFile + iSamePlot] == fileColor[0])
                    skipColor++;
                  jetHtProfiles[iFile][iProfileType][iIov + iSamePlot]->SetLineColor(colors[skipColor + iSamePlot]);
                } else {
                  jetHtProfiles[iFile][iProfileType][iIov + iSamePlot]->SetLineColor(fileColor[iFile + iSamePlot]);
                }
                jetHtProfiles[iFile][iProfileType][iIov + iSamePlot]->SetLineWidth(2);
                noIovFound = false;
              } else {
                std::cout << "No histogram found for: "
                          << Form("%s_%s", iovNames.at(iIov).Data(), profileName[iProfileType].Data()) << std::endl;
                nNullHistograms++;
              }
            }
          }

          if (noIovFound)
            continue;

          // Setup legends. There are four different configuration.
          // 1) Draw several files in one plot, include reference distribution from all runs
          // 2) Draw several files in one plot, do not include reference distribution from all runs
          // 3) Draw several IOVs in one plot, include reference distribution from all runs
          // 4) Draw several IOVs in one plot, do not include reference distribution from all runs
          // Each setup needs slightly different configuration for the legends to look good

          // First determine the number of drawns histograms and how many of them should be drawn to left and right, if the legend is split
          nDrawnHistograms = drawReferenceProfile * compareFiles + nIovInOnePlot * compareFiles - nNullHistograms;
          nLeftLegend = TMath::Ceil(nDrawnHistograms / 2.0);
          nRightLegend = TMath::Floor(nDrawnHistograms / 2.0);

          // Make adjustments to the number of drawn histograms in some special cases
          if (!drawReferenceProfile) {
            if (compareFiles > 1) {
              nLeftLegend = 1;
              nRightLegend = nDrawnHistograms;
            }
            if (nIovInOnePlot > 1) {
              if (nLeftLegend > nRightLegend) {
                nRightLegend++;
              } else {
                nLeftLegend++;
              }
            }
          }

          // The order of columns changes for different configurations. Define it here
          columnOrder[0] = 0;
          columnOrder[1] = 1;
          columnOrder[2] = 2;
          if (compareFiles > 1) {
            columnOrder[0] = 1;
            columnOrder[1] = 2;
            columnOrder[2] = 0;
            if (!drawReferenceProfile) {
              columnOrder[0] = 1;
              columnOrder[1] = 0;
              columnOrder[2] = 2;
            }
          }

          // Define three different legends. Their position is determined by the specific configuration
          legendX1 = 0.19 + 0.24 * drawReferenceProfile * (compareFiles > 1) + profileLegendShiftTotalX +
                     profileLegendShiftColumnX[columnOrder[0]];
          legendY1 = 0.9 - (profileLegendTextSize + 0.02) * nLeftLegend + profileLegendShiftTotalY +
                     profileLegendShiftColumnY[columnOrder[0]];
          legendX2 = 0.41 + 0.24 * drawReferenceProfile * (compareFiles > 1) + profileLegendShiftTotalX +
                     profileLegendShiftColumnX[columnOrder[0]];
          legendY2 = 0.9 + profileLegendShiftTotalY + profileLegendShiftColumnY[columnOrder[0]];
          ;
          legend[0] = new TLegend(legendX1, legendY1, legendX2, legendY2);
          legend[0]->SetFillStyle(0);
          legend[0]->SetBorderSize(0);
          legend[0]->SetTextSize(profileLegendTextSize);
          legend[0]->SetTextFont(profileLegendTextFont);

          legendX1 = 0.55 - 0.24 * !drawReferenceProfile * (compareFiles > 1) + profileLegendShiftTotalX +
                     profileLegendShiftColumnX[columnOrder[1]];
          legendY1 = 0.9 - (profileLegendTextSize + 0.02) * nRightLegend + profileLegendShiftTotalY +
                     profileLegendShiftColumnY[columnOrder[1]];
          legendX2 = 0.77 - 0.24 * !drawReferenceProfile * (compareFiles > 1) + profileLegendShiftTotalX +
                     profileLegendShiftColumnX[columnOrder[1]];
          legendY2 = 0.9 + profileLegendShiftTotalY + profileLegendShiftColumnY[columnOrder[1]];
          legend[1] = new TLegend(legendX1, legendY1, legendX2, legendY2);
          legend[1]->SetFillStyle(0);
          legend[1]->SetBorderSize(0);
          legend[1]->SetTextSize(profileLegendTextSize);
          legend[1]->SetTextFont(profileLegendTextFont);

          legendX1 = 0.13 + profileLegendShiftTotalX + profileLegendShiftColumnX[columnOrder[2]];
          legendY1 = 0.9 - (profileLegendTextSize + 0.02) * nLeftLegend + profileLegendShiftTotalY +
                     profileLegendShiftColumnY[columnOrder[2]];
          legendX2 = 0.49 + profileLegendShiftTotalX + profileLegendShiftColumnX[columnOrder[2]];
          legendY2 = 0.9 + profileLegendShiftTotalY + profileLegendShiftColumnY[columnOrder[2]];
          legend[2] = new TLegend(legendX1, legendY1, legendX2, legendY2);
          legend[2]->SetFillStyle(0);
          legend[2]->SetBorderSize(0);
          legend[2]->SetTextSize(profileLegendTextSize);
          legend[2]->SetTextFont(profileLegendTextFont);

          // First, draw the reference over all runs to the plot
          if (drawReferenceProfile) {
            for (int iFile = 0; iFile < compareFiles; iFile++) {
              if (jetHtProfiles[iFile][iProfileType][nIov - 2] != nullptr) {
                if (!canvasDrawn) {
                  drawer->DrawHistogram(jetHtProfiles[iFile][iProfileType][nIov - 2],
                                        profileXaxis[iProfileType],
                                        Form("#LT#sigma(%s)#GT (#mum)", profileYaxis[iProfileType].Data()),
                                        " ",
                                        "HIST,C");
                  canvasDrawn = true;
                } else {
                  jetHtProfiles[iFile][iProfileType][nIov - 2]->Draw("same,HIST,C");
                }
                legendString = legendTextForAllRuns.Data();
                if (nIovInOnePlot > 1)
                  legendString.Append(Form(" (%s)", legendComment[iFile].Data()));
                legend[0]->AddEntry(jetHtProfiles[iFile][iProfileType][nIov - 2], legendString.Data(), "l");
                legend[2]->AddEntry((TObject *)nullptr, Form("%s:", legendComment[iFile].Data()), "");
              }
            }
          }

          // If we draw several histograms per IOV, add the alignment to the legend if not included in the reference
          if (!drawReferenceProfile && (nIovInOnePlot > 1)) {
            legend[0]->AddEntry((TObject *)nullptr, legendComment[0].Data(), "");
          }

          // Draw defined number of different IOVs to the plot
          for (int iFile = 0; iFile < compareFiles; iFile++) {
            iHistogram = 0;  // This variable takes into account null histograms in case of legend splitting
            for (int iSamePlot = 0; iSamePlot < nIovInOnePlot; iSamePlot++) {
              if (iIov + iSamePlot >= nIov - 2)
                break;  // Do not draw again all or central references
              if (jetHtProfiles[iFile][iProfileType][iIov + iSamePlot] != nullptr) {
                if (!canvasDrawn) {
                  drawer->DrawHistogram(jetHtProfiles[iFile][iProfileType][iIov + iSamePlot],
                                        profileXaxis[iProfileType],
                                        Form("#LT#sigma(%s)#GT (#mum)", profileYaxis[iProfileType].Data()),
                                        " ");
                  canvasDrawn = true;
                } else {
                  jetHtProfiles[iFile][iProfileType][iIov + iSamePlot]->Draw("same");
                }

                // Different legend texts if drawing several files or several IOVs in one plot
                if (compareFiles > 1) {
                  legendString = iovLegend.at(iIov + iSamePlot).Data();
                  if (!drawReferenceProfile)
                    legendString.Append(Form(" (%s)", legendComment[iFile].Data()));
                  legend[1]->AddEntry(jetHtProfiles[iFile][iProfileType][iIov + iSamePlot], legendString.Data(), "l");
                } else {
                  if (iHistogram + 1 == nLeftLegend)
                    iLegend = 1;
                  legend[iLegend]->AddEntry(
                      jetHtProfiles[iFile][iProfileType][iIov + iSamePlot], iovLegend.at(iIov + iSamePlot), "l");
                }
                iHistogram++;
              }
            }
          }

          // Draw the legends
          legend[0]->Draw();
          legend[1]->Draw();
          if (drawReferenceProfile && compareFiles > 1)
            legend[2]->Draw();

          // Save the figures
          if (saveFigures) {
            gPad->GetCanvas()->SaveAs(Form("output/%s%s_iov%d-%d.pdf",
                                           profileName[iProfileType].Data(),
                                           saveComment,
                                           iovVector.at(iIov),
                                           iovVector.at(std::min(iIov + nIovInOnePlot, nIov - 2)) - 1));
          }

          // Add the current IOV to the IOV list to be used with the slide generation script
          if (makeIovListForSlides && (iProfileType == profileIndexForIovList)) {
            iovFileForSlides << Form("%d-%d",
                                     iovVector.at(iIov),
                                     iovVector.at(std::min(iIov + nIovInOnePlot, nIov - 2)) - 1)
                             << "\n";
          }

        }  // iov loop for drawing
      }    // if for drawing profiles for each IOV

      // First, setup the legends for the all runs plots
      doNotDrawEta = (!drawCentralEtaSummaryProfile || iProfileType == kDxyErrorVsEta || iProfileType == kDzErrorVsEta);

      // The order of columns changes for different configurations. Define it here
      columnOrder[0] = 1;
      columnOrder[1] = 2;
      columnOrder[2] = 0;
      if (doNotDrawEta) {
        columnOrder[0] = 1;
        columnOrder[1] = 0;
        columnOrder[2] = 2;
      }

      // Define three different legends. Their position is determined by the specific configuration
      legendX1 = 0.48 + profileLegendShiftTotalX + profileLegendShiftColumnX[columnOrder[0]];
      legendY1 = 0.9 - (profileLegendTextSize + 0.02) * compareFiles + profileLegendShiftTotalY +
                 profileLegendShiftColumnY[columnOrder[0]];
      legendX2 = 0.7 + profileLegendShiftTotalX + profileLegendShiftColumnX[columnOrder[0]];
      legendY2 = 0.9 + profileLegendShiftTotalY + profileLegendShiftColumnY[columnOrder[0]];
      legend[0] = new TLegend(legendX1, legendY1, legendX2, legendY2);
      legend[0]->SetFillStyle(0);
      legend[0]->SetBorderSize(0);
      legend[0]->SetTextSize(profileLegendTextSize);
      legend[0]->SetTextFont(profileLegendTextFont);

      legendX1 = 0.65 - 0.25 * doNotDrawEta + profileLegendShiftTotalX + profileLegendShiftColumnX[columnOrder[1]];
      legendY1 = 0.9 - (profileLegendTextSize + 0.02) * compareFiles + profileLegendShiftTotalY +
                 profileLegendShiftColumnY[columnOrder[1]];
      legendX2 = 0.87 - 0.25 * doNotDrawEta + profileLegendShiftTotalX + profileLegendShiftColumnX[columnOrder[1]];
      legendY2 = 0.9 + profileLegendShiftTotalY + profileLegendShiftColumnY[columnOrder[1]];
      legend[1] = new TLegend(legendX1, legendY1, legendX2, legendY2);
      legend[1]->SetFillStyle(0);
      legend[1]->SetBorderSize(0);
      legend[1]->SetTextSize(profileLegendTextSize);
      legend[1]->SetTextFont(profileLegendTextFont);

      legendX1 = 0.18 + profileLegendShiftTotalX + profileLegendShiftColumnX[columnOrder[2]];
      legendY1 = 0.9 - (profileLegendTextSize + 0.02) * compareFiles + profileLegendShiftTotalY +
                 profileLegendShiftColumnY[columnOrder[2]];
      legendX2 = 0.54 + profileLegendShiftTotalX + profileLegendShiftColumnX[columnOrder[2]];
      legendY2 = 0.9 + profileLegendShiftTotalY + profileLegendShiftColumnY[columnOrder[2]];
      legend[2] = new TLegend(legendX1, legendY1, legendX2, legendY2);
      legend[2]->SetFillStyle(0);
      legend[2]->SetBorderSize(0);
      legend[2]->SetTextSize(profileLegendTextSize);
      legend[2]->SetTextFont(profileLegendTextFont);

      // First, draw histograms from all runs to the plot
      canvasDrawn = false;
      for (int iFile = 0; iFile < compareFiles; iFile++) {
        if (jetHtProfiles[iFile][iProfileType][nIov - 2] != nullptr) {
          jetHtProfiles[iFile][iProfileType][nIov - 2]->SetLineStyle(
              1);  // Reset the line style after IOV specific plots
          if (!canvasDrawn) {
            drawer->DrawHistogram(jetHtProfiles[iFile][iProfileType][nIov - 2],
                                  profileXaxis[iProfileType],
                                  Form("#LT#sigma(%s)#GT (#mum)", profileYaxis[iProfileType].Data()),
                                  plotTitle);
            canvasDrawn = true;
          } else {
            jetHtProfiles[iFile][iProfileType][nIov - 2]->Draw("same");
          }
          legendString = legendTextForAllRuns.Data();
          if (doNotDrawEta)
            legendString.Append(Form(" (%s)", legendComment[iFile].Data()));
          legend[1]->AddEntry(jetHtProfiles[iFile][iProfileType][nIov - 2], legendString.Data(), "l");
          legend[2]->AddEntry((TObject *)nullptr, Form("%s:", legendComment[iFile].Data()), "");
        }
      }

      // If there is no canvas, nothing can be done. Break out of the if-statement
      if (!canvasDrawn)
        break;

      // If we want to draw the central eta as reference, draw them as a dashed line
      if (!doNotDrawEta) {
        for (int iFile = 0; iFile < compareFiles; iFile++) {
          if (jetHtProfiles[iFile][iProfileType][nIov - 1] != nullptr) {
            jetHtProfiles[iFile][iProfileType][nIov - 1]->SetLineColor(fileColor[iFile]);
            jetHtProfiles[iFile][iProfileType][nIov - 1]->SetLineWidth(2);
            jetHtProfiles[iFile][iProfileType][nIov - 1]->SetLineStyle(2);
            jetHtProfiles[iFile][iProfileType][nIov - 1]->Draw("same,HIST,C");
            legend[0]->AddEntry(jetHtProfiles[iFile][iProfileType][nIov - 1], "|#eta| < 1", "l");
          }
        }
      }

      legend[1]->Draw();
      if (!doNotDrawEta) {
        legend[0]->Draw();
        legend[2]->Draw();
      }

      // Save the figures
      if (saveFigures) {
        gPad->GetCanvas()->SaveAs(Form("output/%s%s_allIovs.pdf", profileName[iProfileType].Data(), saveComment));
      }
    }  // if for drawing profile
  }    // profile type loop

  // Close the output file
  if (makeIovListForSlides)
    iovFileForSlides.close();

  // Trend plots
  TLegend *trendLegend;
  drawer->SetCanvasSize(trendCanvasWidth, trendCanvasHeight);
  drawer->SetLeftMargin(trendMarginLeft);
  drawer->SetRightMargin(trendMarginRight);
  drawer->SetTopMargin(trendMarginTop);
  drawer->SetBottomMargin(trendMarginBottom);
  drawer->SetTitleOffsetX(trendTitleOffsetX);
  drawer->SetTitleOffsetY(trendTitleOffsetY);
  drawer->SetTitleSizeX(trendTitleSizeX);
  drawer->SetTitleSizeY(trendTitleSizeY);
  drawer->SetLabelOffsetX(trendLabelOffsetX);
  drawer->SetLabelOffsetY(trendLabelOffsetY);
  drawer->SetLabelSizeX(trendLabelSizeX);
  drawer->SetLabelSizeY(trendLabelSizeY);
  double lumiX;
  TLine *lumiLine;

  // Writed for adding tags to trend plots
  TLatex *tagWriter = new TLatex();
  tagWriter->SetTextFont(trendTagTextFont);
  tagWriter->SetTextSize(trendTagTextSize);

  TString xTitle = "Run index";
  if (useLuminosityForTrends)
    xTitle = "Delivered luminosity  (1/fb)";

  for (int iTrend = 0; iTrend < knTrendTypes; iTrend++) {
    if (!drawTrend[iTrend])
      continue;
    for (int iWidePt = 0; iWidePt < nWidePtBins; iWidePt++) {
      legendX1 = 0.65 + trendLegendShiftTotalX;
      legendY1 = 0.83 - (profileLegendTextSize + 0.02) * compareFiles + trendLegendShiftTotalY;
      legendX2 = 0.9 + trendLegendShiftTotalX;
      legendY2 = 0.9 + trendLegendShiftTotalY;
      trendLegend = new TLegend(legendX1, legendY1, legendX2, legendY2);
      trendLegend->SetFillStyle(0);
      trendLegend->SetBorderSize(0);
      trendLegend->SetTextSize(trendLegendTextSize);
      trendLegend->SetTextFont(trendLegendTextFont);
      trendLegend->SetHeader(
          Form("%s error trend for p_{T} > %.0f GeV", profileYaxis[iTrend + 6].Data(), widePtBinBorders.at(iWidePt)));

      for (int iFile = 0; iFile < compareFiles; iFile++) {
        gBigTrend[iFile][iTrend][iWidePt]->SetMarkerColor(fileColor[iFile]);
        gBigTrend[iFile][iTrend][iWidePt]->SetMarkerStyle(fileMarkerStyle[iFile]);
        gBigTrend[iFile][iTrend][iWidePt]->SetMarkerSize(fileMarkerSize[iFile]);
        if (copyErrorColor[iFile])
          gBigTrend[iFile][iTrend][iWidePt]->SetLineColor(fileColor[iFile]);

        if (iFile == 0) {
          drawer->DrawGraphCustomAxes(gBigTrend[iFile][iTrend][iWidePt],
                                      0,
                                      nRuns,
                                      trendZoomLow[iTrend],
                                      trendZoomHigh[iTrend],
                                      xTitle,
                                      Form("#LT #sigma(%s) #GT", profileYaxis[iTrend + 6].Data()),
                                      " ",
                                      "ap");
        } else {
          gBigTrend[iFile][iTrend][iWidePt]->Draw("p,same");
        }

        trendLegend->AddEntry(gBigTrend[iFile][iTrend][iWidePt], legendComment[iFile].Data(), "p");

      }  // File loop

      trendLegend->Draw();

      // Draw lines for different data taking years
      if (drawYearLines) {
        for (int thisRun : linePosition) {
          lumiX = luminosityBeforeRun(lumiPerIovWithSkips, iovVector, thisRun);
          lumiLine = new TLine(lumiX, trendZoomLow[iTrend], lumiX, trendZoomHigh[iTrend]);
          lumiLine->SetLineColor(yearLineColor);
          lumiLine->SetLineWidth(yearLineWidth);
          lumiLine->SetLineStyle(yearLineStyle);
          lumiLine->Draw();
        }
      }

      // Draw all defined tags
      if (drawTrendTag) {
        for (std::vector<std::string>::size_type iTag = 0; iTag < trendTagText.size(); iTag++) {
          tagWriter->DrawLatexNDC(
              trendTagPositionX.at(iTag), trendTagPositionY.at(iTag), trendTagText.at(iTag).c_str());
        }
      }

      // Save the figures
      if (saveFigures) {
        gPad->GetCanvas()->SaveAs(Form(
            "output/%sTrendPtOver%.0f%s.pdf", trendName[iTrend].Data(), widePtBinBorders.at(iWidePt), saveComment));
      }

    }  // Wide pT loop
  }    // Trend type loop
}

/*
 * Main program
 */
int main(int argc, char **argv) {
  //==== Read command line arguments =====
  AllInOneConfig::Options options;
  options.helper(argc, argv);
  options.parser(argc, argv);

  // Run the program
  jetHtPlotter(options.config);
}
