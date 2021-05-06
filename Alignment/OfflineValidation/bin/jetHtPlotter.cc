// C++ includes
#include <iostream>   // Input/output stream. Needed for cout.
#include <vector>

// Root includes
#include <TFile.h>
#include <TString.h>
#include <TH1D.h>
#include <TProfile.h>
#include <TGraphErrors.h>
#include <TLegend.h>
#include <TLine.h>

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
void drawSingleHistogram(TH1D *histogram[kMaxFiles], const char *saveName, bool saveFigures, TString comment[kMaxFiles], int legendPosition, bool normalize, bool logScale, int color[kMaxFiles]){
  JDrawer *drawer = new JDrawer();
  drawer->SetLogY(logScale);
  double legendX1 = 0.6; double legendY1 = 0.75; double legendX2 = 0.9; double legendY2 = 0.9;
  if(legendPosition == 1){
    legendX1 = 0.35; legendY1 = 0.2; legendX2 = 0.65; legendY2 = 0.35;
  }
  TLegend *legend = new TLegend(legendX1,legendY1,legendX2,legendY2);
  legend->SetFillStyle(0);legend->SetBorderSize(0);legend->SetTextSize(0.05);legend->SetTextFont(62);
  histogram[0]->SetLineColor(color[0]);
  if(normalize) histogram[0]->Scale(1.0/histogram[0]->Integral());
  legend->AddEntry(histogram[0],comment[0],"l");
  drawer->DrawHistogram(histogram[0]);
  for(int iFile = 1; iFile < kMaxFiles; iFile++){
    if(histogram[iFile]){
      histogram[iFile]->SetLineColor(color[iFile]);
      if(normalize) histogram[iFile]->Scale(1.0/histogram[iFile]->Integral());
      histogram[iFile]->Draw("Same");
      legend->AddEntry(histogram[iFile],comment[iFile],"l");
    }
  }
  legend->Draw();
  if(saveFigures) gPad->GetCanvas()->SaveAs(Form("output/%s.pdf",saveName));
}

/*
 *  Construct vectors of ptHet files and the ptHat value in each of those files
 *
 *  Arguments:
 *   const char* inputFile = File containing the list of ptHat separated files and ptHat bin in each file
 *
 *  return:
 *   Tuple containing list of runs, luminosities, and histogram names within the validation files
 */
std::tuple<std::vector<std::string>, std::vector<int>> getPtHatFilesAndValues(const char* inputFile){
  
  // Create vectors for ptHat files and values
  std::vector<std::string> ptHatFileList;
  std::vector<int> ptHatList;
  
  // Helper variables to read the file
  std::string lineInFile;
  
  // Open the input file for reading
  std::ifstream ptHatFile(inputFile);
  
  // Go through the file line by line. Each line has one file and information about the ptHat in the file.
  while(std::getline(ptHatFile, lineInFile))
  {
    
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
TH1* getPtHatCombinedHistogram(std::vector<TFile*> ptHatFiles, std::vector<int> ptHatValues, const char* histogramName){
  
  // Weight for bins:       30to50     50to80      80to120    120to170    170to300    300to470    470to600
  double ptHatWeights[] = {138800000, 19110000,    2735000,    466200,     117200,      7763,      641.0,
  //                       600to800   800to1000  1000to1400  1400to1800  1800to2400  2400to3200  3200toInf
                            185.7,      32.02,      9.375 ,    0.8384,     0.1133,    0.006746,  0.0001623};
  const int nPtHat = 14;
  int ptHatBoundaries[] = {30, 50, 80, 120, 170, 300, 470, 600, 800, 1000, 1400, 1800, 2400, 3200};
  
  TFile *openFile = ptHatFiles.at(0);
  TH1* combinedHistogram = (TH1*) openFile->Get(histogramName)->Clone(Form("%sClone",histogramName));
  TH1* currentHistogram;
  combinedHistogram->Reset("ICES");
  int ptHatIndex = -1;
  
  const int nFiles = ptHatFiles.size();

  for(int iFile = 0; iFile < nFiles; iFile++){
    
    ptHatIndex = -1;
    for(int iPtHat = 0; iPtHat < nPtHat; iPtHat++){
      if(ptHatValues.at(iFile) == ptHatBoundaries[iPtHat]){
        ptHatIndex = iPtHat;
        break;
      }
    }
    
    if(ptHatIndex < 0){
      cout << "Could not find pT hat boundary " << ptHatValues.at(iFile) << " for file " << ptHatFiles.at(iFile) << endl;
      cout << "Please check your input! It needs to be in the form <fileName> <ptHatBoundary>" << endl;
      return NULL;
    }
    
    openFile = ptHatFiles.at(iFile);
    currentHistogram = (TH1*) openFile->Get(histogramName);
    combinedHistogram->Add(currentHistogram,ptHatWeights[ptHatIndex]);
    
  }
  
  return combinedHistogram;
  
}

/*
 *  Construct vectors of runs, luminosities, and histogram names within the validation files from an input file
 *  containing a list of runs and luminosities attached to each one
 *
 *  Arguments:
 *   const char* inputFile = File containing the run and luminosity lists
 *
 *  return:
 *   Tuple containing list of runs, luminosities, and histogram names within the validation files
 */
std::tuple<std::vector<int>, std::vector<double>, std::vector<TString>> getRunAndLumiLists(const char* inputFile){
  
  // Create vectors for each list
  std::vector<int> iovVector;
  std::vector<double> lumiPerIov;
  std::vector<TString> iovNames;
  
  // Helper variables to read the file
  std::string lineInFile;
  int thisIov;
  double thisLumi;
  
  // Load the iovList
  std::ifstream iovList(inputFile);
  
  // Go through the file line by line. Each line has an IOV boundary and luminosity for this IOV.
  while(std::getline(iovList, lineInFile))
  {
    
    // Read the iov and luminosity from the file
    std::istringstream lineStream(lineInFile);
    lineStream >> thisIov;
    lineStream >> thisLumi;
    
    // Push the extracted numbers to vectors
    iovVector.push_back(thisIov);
    lumiPerIov.push_back(thisLumi);
  }

  // Create names for the different IOV:s
  for(std::vector<int>::size_type i = 1; i < iovVector.size(); i++){
    iovNames.push_back(Form("iov%d-%d",iovVector.at(i-1),iovVector.at(i)-1));
  }
  
  // Add the iov integrated histograms after the histograms per IOV
  iovNames.push_back("all");
  iovNames.push_back("central");
  
  // After all the vectors are filled, return them
  return std::make_tuple(iovVector, lumiPerIov, iovNames);
  
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
std::vector<double> scaleGraphByLuminosity(TGraphErrors *runGraph, std::vector<double> lumiPerIov, bool skipMissingRuns) {
  
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

    if (lumiPerIov.at(iRun+offset) == 0 || (yValue < epsilon && skipMissingRuns)) {
      nRuns--;
      runGraph->RemovePoint(iRun);
      offset++;
      
      // Fill vector where lumi for skipped runs is set to zero
      lumiPerIovWithSkips.push_back(0);
    } else {
      
      xValue += lumiPerIov.at(iRun+offset) / lumiFactor;
      xAxisValues.push_back(xValue - (lumiPerIov.at(iRun+offset) / (lumiFactor * 2)));
      xAxisErrors.push_back(lumiPerIov.at(iRun+offset) / (lumiFactor * 2));
      yAxisValues.push_back(yValue);
      yAxisErrors.push_back(runGraph->GetErrorY(iRun));
      
      // Fill vector where lumi for skipped runs is set to zero
      lumiPerIovWithSkips.push_back(lumiPerIov.at(iRun+offset));
      
      iRun++;
    }
  } // Loop over all runs in original histogram
  
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
double getLuminosityBeforeRun(std::vector<double> lumiPerIov, std::vector<int> iovVector, int runNumber) {

  double lumiFactor = 1000;  // Scale factor value to have luminosity expressed in fb^-1
  
  int nRuns = lumiPerIov.size();
  double luminosityBefore = 0;
  
  for(int iRun = 0; iRun < nRuns; iRun++){
    
    if(runNumber <= iovVector.at(iRun)) return luminosityBefore;
    
    luminosityBefore += lumiPerIov.at(iRun) / lumiFactor;
    
  }
  
  return luminosityBefore;
  
}

/*
 * Macro for plotting figures for the study of jet HT sample
 */
void jetHtPlotter(std::string configurationFileName){

  // ======================================================
  // ================== Configuration =====================
  // ======================================================
  
  JetHtPlotConfiguration *configurationGiver = new JetHtPlotConfiguration();
  configurationGiver->ReadJsonFile(configurationFileName);
  configurationGiver->PrintConfiguration();
  
  enum enumHistogramType{kDz, kDzError, kDxy, kDxyError, knHistogramTypes};
  TString histogramName[knHistogramTypes] = {"dz","dzerr","dxy","dxyerr"};
  TString histogramXaxis[knHistogramTypes] = {"d_{z} (#mum)","#sigma(d_{z}) (#mum)","d_{xy} (#mum)","#sigma(d_{xy}) (#mum)"};
  enum enumProfileType{kDzErrorVsPt, kDzErrorVsPhi, kDzErrorVsEta, kDxyErrorVsPt, kDxyErrorVsPhi, kDxyErrorVsEta, kDzErrorVsPtWide, kDxyErrorVsPtWide, knProfileTypes};
  enum enumTrendType{kDzErrorTrend, kDxyErrorTrend, knTrendTypes};
  TString profileName[knProfileTypes] = {"dzErrVsPt", "dzErrVsPhi", "dzErrVsEta", "dxyErrVsPt", "dxyErrVsPhi", "dxyErrVsEta", "dzErrVsPtWide", "dxyErrVsPtWide"};
  TString profileXaxis[knProfileTypes] = {"p_{T} (GeV)","#varphi","#eta","p_{T} (GeV)","#varphi","#eta", "p_{T} bin", "p_{T} bin"};
  TString profileYaxis[knProfileTypes] = {"d_{z}","d_{z}","d_{z}","d_{xy}","d_{xy}","d_{xy}","d_{z}","d_{xy}"};
  TString trendName[] = {"dzErr","dxyErr"};
  bool drawHistogram[knHistogramTypes];
  bool drawProfile[knProfileTypes];
  bool drawTrend[knTrendTypes];
  
  bool drawTrackQA = configurationGiver->GetDrawTrackQA();                                // Draw track and vertex QA figures
  drawHistogram[kDz] = configurationGiver->GetDrawHistogram(kDz);                         // Draw the dz histograms
  drawHistogram[kDzError] = configurationGiver->GetDrawHistogram(kDzError);               // Draw the dz error histograms
  drawProfile[kDzErrorVsPt] = configurationGiver->GetDrawProfile(kDzErrorVsPt);           // Draw mean dz error as a function of pT
  drawProfile[kDzErrorVsPhi] = configurationGiver->GetDrawProfile(kDzErrorVsPhi);         // Draw mean dz error as a function of phi
  drawProfile[kDzErrorVsEta] = configurationGiver->GetDrawProfile(kDzErrorVsEta);         // Draw mean dz error as a function of eta
  drawProfile[kDzErrorVsPtWide] = configurationGiver->GetDrawProfile(kDzErrorVsPtWide);   // Draw mean dz error in wide pT bins
  drawHistogram[kDxy] = configurationGiver->GetDrawHistogram(kDxy);                       // Draw the dxy histograms
  drawHistogram[kDxyError] = configurationGiver->GetDrawHistogram(kDxyError);             // Draw the dxy error histograms
  drawProfile[kDxyErrorVsPt] = configurationGiver->GetDrawProfile(kDxyErrorVsPt);         // Draw the dxy error as a function of pT
  drawProfile[kDxyErrorVsPhi] = configurationGiver->GetDrawProfile(kDxyErrorVsPhi);       // Draw the dxy error as a function of phi
  drawProfile[kDxyErrorVsEta] = configurationGiver->GetDrawProfile(kDxyErrorVsEta);       // Draw the dxy error as a function of eta
  drawProfile[kDxyErrorVsPtWide] = configurationGiver->GetDrawProfile(kDxyErrorVsPtWide); // Draw mean dxy error in wide pT bins
  drawTrend[kDzErrorTrend] = configurationGiver->GetDrawTrend(kDzErrorTrend);             // Draw the trend plots for dz errors
  drawTrend[kDxyErrorTrend] = configurationGiver->GetDrawTrend(kDxyErrorTrend);           // Draw the trend plots for dxy errors
  
  bool drawPlotsForEachIOV = configurationGiver->GetDrawPlotsForEachIOV();        // True = Draw profile and histogram plots for every IOV. False = only draw average over all runs
  bool useLuminosityForTrends = configurationGiver->GetUseLuminosityForTrends();  // True = Draw trends as a function of luminosity. False = Draw trends as a function of run index
  bool skipRunsWithNoData = configurationGiver->GetSkipRunsWithNoData();          // True = Do not draw empty space if run in list is missing data. False = Draw empty space
  
  int colors[] = {kRed,kGreen+3,kMagenta,kCyan,kViolet+3,kOrange,kPink-7,kSpring+3,kAzure-7};
  int nIovInOnePlot = configurationGiver->GetNIovInOnePlot();  // Define how many iov:s are drawn to the same plot
  
  double profileZoomLow[knProfileTypes];
  double profileZoomHigh[knProfileTypes];
  double trendZoomLow[knTrendTypes];
  double trendZoomHigh[knTrendTypes];
  
  for(int iProfile = 0; iProfile < knProfileTypes; iProfile++){
    profileZoomLow[iProfile] = configurationGiver->GetProfileZoomLow(iProfile);
    profileZoomHigh[iProfile] = configurationGiver->GetProfileZoomHigh(iProfile);
  }
  for(int iTrend = 0; iTrend < knTrendTypes; iTrend++){
    trendZoomLow[iTrend] = configurationGiver->GetTrendZoomLow(iTrend);
    trendZoomHigh[iTrend] = configurationGiver->GetTrendZoomHigh(iTrend);
  }
  
  const std::vector<double> widePtBinBorders = configurationGiver->GetWidePtBinBorders();
  const int nWidePtBins = widePtBinBorders.size();
  
  bool normalizeQAplots = configurationGiver->GetNormalizeQAplots();  // Divide in QA plot yield by its integral
  bool saveFigures = true;
  const char *saveComment = configurationGiver->GetSaveComment();
  
  int compareFiles = configurationGiver->GetNInputFiles();
  if(compareFiles == 0) return; // Cannot do plotting without files!
  if(compareFiles > kMaxFiles) compareFiles = kMaxFiles;  // Four is maximum number of files in current implementation
  
  // Read the input file names. If file names end with .txt, they are assumed to be lists of files in different ptHat bins
  TString inputFileName[kMaxFiles];
  bool loadFromList[kMaxFiles] = {false, false, false, false};
  TString legendComment[kMaxFiles];
  int fileColor[kMaxFiles];
  int fileMarker[kMaxFiles];
  
  for(int iFile = 0; iFile < kMaxFiles; iFile++){
    inputFileName[iFile] = configurationGiver->GetInputFile(iFile);
    if(inputFileName[iFile].EndsWith("txt")) loadFromList[iFile] = true;
    legendComment[iFile] = configurationGiver->GetLegendComment(iFile); // Text written to the legend for each file
    fileColor[iFile] = configurationGiver->GetMarkerColor(iFile);       // Color of markers related to this file
    fileMarker[iFile] = configurationGiver->GetMarkerStyle(iFile);      // Style for markers related to this file
  }
  
  // Prepare an IOV list that can be used with the slide generation script
  bool makeIovListForSlides = configurationGiver->GetMakeIovListForSlides();
  const char *iovListForSlides = configurationGiver->GetIovListForSlides();
  
  // Helper variable to ensure that each IOV is added exactly once to the list
  int profileIndexForIovList = 0;
  for(int iProfile = 0; iProfile < knProfileTypes; iProfile++){
    if(drawProfile[iProfile]){
      profileIndexForIovList = iProfile;
      break;
    }
  }
  
  TString plotTitle = " ";  // Title given to the plot
  
  // Year boundary runs
  bool drawYearLines = configurationGiver->GetDrawYearLines();           // Draw lines between data taking years
  std::vector<int> linePosition = configurationGiver->GetRunsForLines(); // Positions of the above lines

  // ======================================================
  // ================ Configuration done ==================
  // ======================================================
  
  // ===============================================================
  //   Read different ptHat files for combining those with weights
  // ===============================================================
  
  // List for all the different ptHat files to be combined
  std::vector<std::string> ptHatFileNameList[kMaxFiles];
  std::vector<TFile*> ptHatFileList[kMaxFiles];
  std::vector<int> ptHatList[kMaxFiles];
  int nFiles;
  
  for(int iInput = 0; iInput < kMaxFiles; iInput++){
    if(loadFromList[iInput]){
      std::tie(ptHatFileNameList[iInput], ptHatList[iInput]) = getPtHatFilesAndValues(inputFileName[iInput]);
      
      // Open all files
      nFiles = ptHatFileNameList[iInput].size();
      for(int iFile = 0; iFile < nFiles; iFile++){
        ptHatFileList[iInput].push_back(TFile::Open(ptHatFileNameList[iInput].at(iFile).data()));
      }
    }
  }
  
  // ===============================================
  //        Read the run and luminosity list
  // ===============================================
  
  // Luminosity per run file
  const char* iovAndLumiFile = configurationGiver->GetLumiPerIovFile();
  
  // Create a vector for a new iovList
  std::vector<int> iovVector;
  std::vector<double> lumiPerIov;
  std::vector<double> lumiPerIovWithSkips;
  std::vector<TString> iovNames;
  
  std::tie(iovVector, lumiPerIov, iovNames) = getRunAndLumiLists(iovAndLumiFile);
  
  // ===============================================
  //                IOV list for slides
  // ===============================================
  
  // If we are preparing an iov list for slides, make a file for that
  std::ofstream iovFileForSlides;
  if(makeIovListForSlides) iovFileForSlides.open(iovListForSlides);
  
  // ===============================================
  //   Create histograms and read them from a file
  // ===============================================
  
  // Open the input files
  TFile *inputFile[kMaxFiles];
  for(int iFile = 0; iFile < kMaxFiles; iFile++){
    if(compareFiles > iFile && !loadFromList[iFile]) inputFile[iFile] = TFile::Open(inputFileName[iFile]);
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
  int legendSplitter = 0;
  
  // Initialize everything to NULL
  for(int iFile = 0; iFile < kMaxFiles; iFile++){
    hVertex[iFile] = NULL;
    hTracksPerVertex[iFile] = NULL;
    hTrackPt[iFile] = NULL;
    hTrackEta[iFile] = NULL;
    hTrackPhi[iFile] = NULL;
    for(int iIov = 0; iIov < nIov; iIov++){
      for(int iHistogramType = 0; iHistogramType < knHistogramTypes; iHistogramType++){
        jetHtHistograms[iFile][iHistogramType][iIov] = NULL;
      } // histogram type loop
      for(int iProfileType = 0; iProfileType < knProfileTypes; iProfileType++){
        jetHtProfiles[iFile][iProfileType][iIov] = NULL;
      } // profile type loop
    } // iov loop for reading histograms from file
    for(int iTrend = 0; iTrend < knTrendTypes; iTrend++){
      for(int iWidePt = 0; iWidePt < nWidePtBins; iWidePt++){
        gBigTrend[iFile][iTrend][iWidePt] = NULL;
      } // Trend loop
    } // Wide pT bin loop
  } // file loop
  
  // Read the histograms from the file
  for(int iFile = 0; iFile < compareFiles; iFile++){
    
    if(loadFromList[iFile]){
      
      // Alternative loading for ptHat combined values
      
      hVertex[iFile] = (TH1D*) getPtHatCombinedHistogram(ptHatFileList[iFile],ptHatList[iFile],"jetHTAnalyzer/all_nvtx");
      hTracksPerVertex[iFile] = (TH1D*) getPtHatCombinedHistogram(ptHatFileList[iFile],ptHatList[iFile],"jetHTAnalyzer/h_ntrks");
      hTrackPt[iFile] = (TH1D*) getPtHatCombinedHistogram(ptHatFileList[iFile],ptHatList[iFile],"jetHTAnalyzer/h_probePt");
      hTrackEta[iFile] = (TH1D*) getPtHatCombinedHistogram(ptHatFileList[iFile],ptHatList[iFile],"jetHTAnalyzer/h_probeEta");
      hTrackPhi[iFile] = (TH1D*) getPtHatCombinedHistogram(ptHatFileList[iFile],ptHatList[iFile],"jetHTAnalyzer/h_probePhi");
      for(int iIov = nIov-2; iIov < nIov-1; iIov++){
        for(int iHistogramType = 0; iHistogramType < knHistogramTypes; iHistogramType++){
          if(drawHistogram[iHistogramType]){
            jetHtHistograms[iFile][iHistogramType][iIov] = (TH1D*) getPtHatCombinedHistogram(ptHatFileList[iFile], ptHatList[iFile], Form("jetHTAnalyzer/%s_%s",iovNames.at(iIov).Data(),histogramName[iHistogramType].Data()));
          } // if for drawing histogram
        } // histogram type loop
        for(int iProfileType = 0; iProfileType < knProfileTypes; iProfileType++){
          if(drawProfile[iProfileType] || (drawTrend[kDzErrorTrend] && iProfileType == kDzErrorVsPtWide) || (drawTrend[kDxyErrorTrend] && iProfileType == kDxyErrorVsPtWide)){
            jetHtProfiles[iFile][iProfileType][iIov] = (TProfile*) getPtHatCombinedHistogram(ptHatFileList[iFile], ptHatList[iFile], Form("jetHTAnalyzer/%s_%s",iovNames.at(iIov).Data(),profileName[iProfileType].Data()));
          } // if for drawing profile
        } // profile type loop
      } // iov loop for reading histograms from file
      
    } else {
      
      // Regular histogram loading
      
      hVertex[iFile] = (TH1D*) inputFile[iFile]->Get("jetHTAnalyzer/all_nvtx");
      hTracksPerVertex[iFile] = (TH1D*) inputFile[iFile]->Get("jetHTAnalyzer/h_ntrks");
      hTrackPt[iFile] = (TH1D*) inputFile[iFile]->Get("jetHTAnalyzer/h_probePt");
      hTrackEta[iFile] = (TH1D*) inputFile[iFile]->Get("jetHTAnalyzer/h_probeEta");
      hTrackPhi[iFile] = (TH1D*) inputFile[iFile]->Get("jetHTAnalyzer/h_probePhi");
      for(int iIov = 0; iIov < nIov; iIov++){
        for(int iHistogramType = 0; iHistogramType < knHistogramTypes; iHistogramType++){
          if(drawHistogram[iHistogramType]){
            jetHtHistograms[iFile][iHistogramType][iIov] = (TH1D*) inputFile[iFile]->Get(Form("jetHTAnalyzer/%s_%s",iovNames.at(iIov).Data(),histogramName[iHistogramType].Data()));
          } // if for drawing histogram
        } // histogram type loop
        for(int iProfileType = 0; iProfileType < knProfileTypes; iProfileType++){
          if(drawProfile[iProfileType] || (drawTrend[kDzErrorTrend] && iProfileType == kDzErrorVsPtWide) || (drawTrend[kDxyErrorTrend] && iProfileType == kDxyErrorVsPtWide)){
            jetHtProfiles[iFile][iProfileType][iIov] = (TProfile*) inputFile[iFile]->Get(Form("jetHTAnalyzer/%s_%s",iovNames.at(iIov).Data(),profileName[iProfileType].Data()));
          } // if for drawing profile
        } // profile type loop
      } // iov loop for reading histograms from file
    } // Regular histogram loading
  } // Loop over files
  
  // Collect the information for the trend graphs
  const int nRuns = iovVector.size()-1;
  double yValueDz[nRuns];
  double yErrorDz[nRuns];
  double yValueDxy[nRuns];
  double yErrorDxy[nRuns];
  double xValue[nRuns];
  double xError[nRuns];
  TString commentEntry = "";
  if(drawTrend[kDzErrorTrend] || drawTrend[kDxyErrorTrend]){
    for(int iFile = 0; iFile < compareFiles; iFile++){
      for(int iWidePt = 0; iWidePt < nWidePtBins; iWidePt++){
        for(int iRun = 0; iRun < nRuns; iRun++){
          xValue[iRun] = iRun;
          xError[iRun] = 0;
          if(jetHtProfiles[iFile][kDzErrorVsPtWide][iRun] == NULL){
            yValueDz[iRun] = 0;
            yErrorDz[iRun] = 0;
          } else {
            yValueDz[iRun] = jetHtProfiles[iFile][kDzErrorVsPtWide][iRun]->GetBinContent(iWidePt+1);
            yErrorDz[iRun] = jetHtProfiles[iFile][kDzErrorVsPtWide][iRun]->GetBinError(iWidePt+1);
          }
          if(jetHtProfiles[iFile][kDxyErrorVsPtWide][iRun] == NULL){
            yValueDxy[iRun] = 0;
            yErrorDxy[iRun] = 0;
          } else {
            yValueDxy[iRun] = jetHtProfiles[iFile][kDxyErrorVsPtWide][iRun]->GetBinContent(iWidePt+1);
            yErrorDxy[iRun] = jetHtProfiles[iFile][kDxyErrorVsPtWide][iRun]->GetBinError(iWidePt+1);
          }
        } // Run loop
        
        gBigTrend[iFile][kDzErrorTrend][iWidePt] = new TGraphErrors(nRuns, xValue, yValueDz, xError, yErrorDz);
        gBigTrend[iFile][kDxyErrorTrend][iWidePt] = new TGraphErrors(nRuns, xValue, yValueDxy, xError, yErrorDxy);
        
        // Change x-axis to processed luminosity
        if(useLuminosityForTrends){
          lumiPerIovWithSkips = scaleGraphByLuminosity(gBigTrend[iFile][kDzErrorTrend][iWidePt], lumiPerIov, skipRunsWithNoData);
          scaleGraphByLuminosity(gBigTrend[iFile][kDxyErrorTrend][iWidePt], lumiPerIov, skipRunsWithNoData);
        }
        
        
      } // Wide pT bin loop
    } // File loop
  } // If for drawing trends
  
  // ===============================================
  //                  Draw the plots
  // ===============================================
  
  JDrawer *drawer = new JDrawer();
  TLegend *legend[2];
  bool noIovFound = true;
  
  // Draw track and vertex histograms
  if(drawTrackQA){
    drawSingleHistogram(hVertex, Form("vertex%s",saveComment), saveFigures, legendComment, 0, normalizeQAplots, false, fileColor);
    drawSingleHistogram(hTracksPerVertex, Form("tracksPerVertex%s",saveComment), saveFigures, legendComment, 0, normalizeQAplots, false, fileColor);
    drawSingleHistogram(hTrackPt, Form("trackPt%s",saveComment), saveFigures, legendComment, 0, normalizeQAplots, true, fileColor);
    drawSingleHistogram(hTrackEta, Form("trackEta%s",saveComment), saveFigures, legendComment, 1, normalizeQAplots, false, fileColor);
    drawSingleHistogram(hTrackPhi, Form("trackPhi%s",saveComment), saveFigures, legendComment, 1, normalizeQAplots, false, fileColor);
  }
  
  // Draw dz and dxy histograms
  for(int iIov= 0; iIov < nIov; iIov++){
    if(!drawPlotsForEachIOV && iIov < nIov-2) continue;
    for(int iHistogramType = 0; iHistogramType < knHistogramTypes; iHistogramType++){
      if(drawHistogram[iHistogramType]){
        
        legend[0] = new TLegend(0.7,0.85-0.05*compareFiles,0.95,0.9);
        legend[0]->SetFillStyle(0);legend[0]->SetBorderSize(0);
        legend[0]->SetTextSize(0.05);legend[0]->SetTextFont(62);
        
        if(jetHtHistograms[0][iHistogramType][iIov] != NULL){
          drawer->DrawHistogram(jetHtHistograms[0][iHistogramType][iIov], histogramXaxis[iHistogramType], "tracks", iovNames.at(iIov).Data());
          legend[0]->AddEntry(jetHtHistograms[0][iHistogramType][iIov],legendComment[0],"l");
          
          for(int iFile = 1; iFile < compareFiles; iFile++){
            if(jetHtHistograms[iFile][iHistogramType][iIov] != NULL){
              jetHtHistograms[iFile][iHistogramType][iIov]->SetLineColor(colors[iFile-1]);
              jetHtHistograms[iFile][iHistogramType][iIov]->Draw("same");
              legend[0]->AddEntry(jetHtHistograms[iFile][iHistogramType][iIov],legendComment[iFile],"l");
            }
          }
          
          legend[0]->Draw();
          
          // Save the figures
          if(saveFigures){
            gPad->GetCanvas()->SaveAs(Form("output/%s%s_%s.pdf", histogramName[iHistogramType].Data(), saveComment, iovNames.at(iIov).Data()));
          }
          
        } else {
          cout << "No histogram found for: " << Form("%s_%s",iovNames.at(iIov).Data(),histogramName[iHistogramType].Data()) << endl;
        }
      } // if for drawing histogram
    } // histogram type loop
  }
  
  // Draw dz and dxy profiles
  for(int iProfileType = 0; iProfileType < knProfileTypes; iProfileType++){
    if(drawProfile[iProfileType]){
      
      // Set the style for IOV integrated histograms
      jetHtProfiles[0][iProfileType][nIov-2]->SetLineColor(fileColor[0]);
      jetHtProfiles[0][iProfileType][nIov-2]->SetLineWidth(2);
      jetHtProfiles[0][iProfileType][nIov-2]->GetYaxis()->SetRangeUser(profileZoomLow[iProfileType], profileZoomHigh[iProfileType]);
      
      for(int iFile = 1; iFile < compareFiles; iFile++){
        jetHtProfiles[iFile][iProfileType][nIov-2]->SetLineColor(fileColor[iFile]);
        jetHtProfiles[iFile][iProfileType][nIov-2]->SetLineWidth(2);
      }
      
      // NOTE: IOV drawing only works for one or two files! Using more files will not give good results!!!
      if(drawPlotsForEachIOV){
        for(int iIov= 0; iIov < nIov-2; iIov = iIov + nIovInOnePlot){
          
          noIovFound = true;
          
          // Set up the IOV:s to be drawn to the current plot
          for(int iSamePlot = 0; iSamePlot < nIovInOnePlot; iSamePlot++){
            if(iIov + iSamePlot >= nIov - 2) break; // Do not draw again all or central references
            if(jetHtProfiles[0][iProfileType][iIov+iSamePlot] != NULL){
              jetHtProfiles[0][iProfileType][iIov+iSamePlot]->SetLineColor(colors[iSamePlot]);
              jetHtProfiles[0][iProfileType][iIov+iSamePlot]->SetLineWidth(2);
              noIovFound = false;
            } else {
              cout << "No histogram found for: " << Form("%s_%s",iovNames.at(iIov).Data(),profileName[iProfileType].Data()) << endl;
            }
            
            if(jetHtProfiles[1][iProfileType][iIov+iSamePlot] != NULL){
              jetHtProfiles[1][iProfileType][iIov+iSamePlot]->SetLineColor(colors[iSamePlot+nIovInOnePlot]);
              jetHtProfiles[1][iProfileType][iIov+iSamePlot]->SetLineWidth(2);
            }
          }
          
          if(noIovFound) continue;
          
          // First, draw the reference over all runs to the plot
          drawer->DrawHistogram(jetHtProfiles[0][iProfileType][nIov-2], profileXaxis[iProfileType], Form("#LT#sigma(%s)#GT (#mum)",profileYaxis[iProfileType].Data()));
          
          // Define a new legend for the plot
          for(int iFile = 0; iFile < compareFiles; iFile++){
            legend[iFile] = new TLegend(0.56-0.37*iFile,0.8-0.05*nIovInOnePlot,0.86-0.37*iFile,0.9);
            legend[iFile]->SetFillStyle(0); legend[iFile]->SetBorderSize(0);
            legend[iFile]->SetTextSize(0.05); legend[iFile]->SetTextFont(62);
          }
          legend[0]->AddEntry(jetHtProfiles[0][iProfileType][nIov-2],Form("All (%s)",legendComment[0].Data()),"l");
          
          if(jetHtProfiles[1][iProfileType][nIov-2] != NULL){
            jetHtProfiles[1][iProfileType][nIov-2]->Draw("same");
            legend[1]->AddEntry(jetHtProfiles[1][iProfileType][nIov-2],Form("All (%s)",legendComment[1].Data()),"l");
          }
          
          // Draw defined number of different IOVs to the plot
          for(int iFile = 0; iFile < compareFiles; iFile++){
            for(int iSamePlot = 0; iSamePlot < nIovInOnePlot; iSamePlot++){
              if(iIov + iSamePlot >= nIov - 2) break; // Do not draw again all or central references
              if(jetHtProfiles[iFile][iProfileType][iIov+iSamePlot] != NULL){
                jetHtProfiles[iFile][iProfileType][iIov+iSamePlot]->Draw("same");
                legend[iFile]->AddEntry(jetHtProfiles[iFile][iProfileType][iIov+iSamePlot],iovNames.at(iIov+iSamePlot),"l");
              }
            }
            legend[iFile]->Draw();
          }
          
          
          
          // Save the figures
          if(saveFigures){
            gPad->GetCanvas()->SaveAs(Form("output/%s%s_iov%d-%d.pdf", profileName[iProfileType].Data(), saveComment, iovVector.at(iIov),iovVector.at(std::min(iIov+nIovInOnePlot,nIov-2))-1));
          }
          
          // Add the current IOV to the IOV list to be used with the slide generation script
          if(makeIovListForSlides && (iProfileType == profileIndexForIovList)){
            iovFileForSlides << Form("%d-%d",iovVector.at(iIov),iovVector.at(std::min(iIov+nIovInOnePlot,nIov-2))-1) << "\n";
          }
          
        } // iov loop for drawing
      } // if for drawing profiles for each IOV
      
      
      // After all IOV:s, draw summary with all and central curves
      drawer->DrawHistogram(jetHtProfiles[0][iProfileType][nIov-2], profileXaxis[iProfileType], Form("#LT#sigma(%s)#GT (#mum)",profileYaxis[iProfileType].Data()), plotTitle);
      
      // Add a reference to all runs in central eta range, except for plots as a function of eta
      // Only draw central histogram is comparing only 2 files
      if(jetHtProfiles[0][iProfileType][nIov-1] != NULL && compareFiles < 3){
        jetHtProfiles[0][iProfileType][nIov-1]->SetLineColor(kRed);
        jetHtProfiles[0][iProfileType][nIov-1]->SetLineWidth(2);
        jetHtProfiles[0][iProfileType][nIov-1]->Draw("same");
      }
      
      
      for(int iFile = 1; iFile < compareFiles; iFile++){
        
        // Add all and central also from the comparison file
        if(jetHtProfiles[iFile][iProfileType][nIov-2] != NULL){
          jetHtProfiles[iFile][iProfileType][nIov-2]->Draw("same");
          if(jetHtProfiles[iFile][iProfileType][nIov-1] != NULL && compareFiles < 3){ // Only add central if comparing just two files
            jetHtProfiles[iFile][iProfileType][nIov-1]->SetLineColor(kMagenta);
            jetHtProfiles[iFile][iProfileType][nIov-1]->SetLineWidth(2);
            jetHtProfiles[iFile][iProfileType][nIov-1]->Draw("same");
          }
        }
      }
      
      // Define legends for the plots
      legendSplitter = 2;
      if(compareFiles < legendSplitter) legendSplitter = compareFiles;
      for(int iFile = 0; iFile < legendSplitter; iFile++){
        if(legendSplitter < 2){
          legend[iFile] = new TLegend(0.56,0.75,0.86,0.9);
        } else {
          legend[iFile] = new TLegend(0.19+0.37*iFile,0.75,0.49+0.37*iFile,0.9);
        }
        legend[iFile]->SetFillStyle(0); legend[iFile]->SetBorderSize(0);
        legend[iFile]->SetTextSize(0.05); legend[iFile]->SetTextFont(62);
        if(drawPlotsForEachIOV){
          commentEntry = Form("All (%s)",legendComment[iFile].Data());
        } else {
          commentEntry = legendComment[iFile];
        }
        legend[iFile]->AddEntry(jetHtProfiles[iFile][iProfileType][nIov-2],commentEntry.Data(),"l");
        if(jetHtProfiles[iFile][iProfileType][nIov-1] != NULL && compareFiles < 3) legend[iFile]->AddEntry(jetHtProfiles[iFile][iProfileType][nIov-1],Form("Central (%s)",legendComment[iFile].Data()),"l");
      }
      
      if(compareFiles > 2){
        for(int iFile = 2; iFile < compareFiles; iFile++){
          if(drawPlotsForEachIOV){
            commentEntry = Form("All (%s)",legendComment[iFile].Data());
          } else {
            commentEntry = legendComment[iFile];
          }
          legend[iFile-2]->AddEntry(jetHtProfiles[iFile][iProfileType][nIov-2],commentEntry.Data(),"l");
        }
      }
      
      for(int iFile = 0; iFile < legendSplitter; iFile++){
        legend[iFile]->Draw();
      }
      
      // Save the figures
      if(saveFigures){
        gPad->GetCanvas()->SaveAs(Form("output/%s%s_allIovs.pdf", profileName[iProfileType].Data(),saveComment));
      }
    } // if for drawing profile
  } // profile type loop
  
  // Close the output file
  if(makeIovListForSlides) iovFileForSlides.close();
  
  // Trend plots
  TLegend *trendLegend;
  drawer->SetCanvasSize(1000,400);
  drawer->SetLeftMargin(0.08);
  drawer->SetRightMargin(0.03);
  drawer->SetTitleOffsetY(0.55);
  drawer->SetLabelOffsetY(0.007);
  double lumiX;
  TLine *lumiLine;
  
  TString xTitle = "Run index";
  if(useLuminosityForTrends) xTitle = "Delivered luminosity  (1/fb)";
  
  for(int iTrend = 0; iTrend < knTrendTypes; iTrend++){
    if(!drawTrend[iTrend]) continue;
    for(int iWidePt = 0; iWidePt < nWidePtBins; iWidePt++){
      
      trendLegend = new TLegend(0.65,0.6,0.9,0.9);
      trendLegend->SetFillStyle(0); trendLegend->SetBorderSize(0);
      trendLegend->SetTextSize(0.05); trendLegend->SetTextFont(62);
      trendLegend->SetHeader(Form("%s error trend for p_{T} > %.0f GeV", profileYaxis[iTrend+6].Data(), widePtBinBorders.at(iWidePt)));
      
      for(int iFile = 0; iFile < compareFiles; iFile++){
        gBigTrend[iFile][iTrend][iWidePt]->SetMarkerColor(fileColor[iFile]);
        gBigTrend[iFile][iTrend][iWidePt]->SetMarkerStyle(fileMarker[iFile]);
        
        if(iFile == 0){
          drawer->DrawGraphCustomAxes(gBigTrend[iFile][iTrend][iWidePt], 0, nRuns, trendZoomLow[iTrend], trendZoomHigh[iTrend], xTitle, Form("#LT #sigma(%s) #GT", profileYaxis[iTrend+6].Data()), " ", "ap");
        } else {
          gBigTrend[iFile][iTrend][iWidePt]->Draw("p,same");
        }
        
        trendLegend->AddEntry(gBigTrend[iFile][iTrend][iWidePt], legendComment[iFile].Data(), "p");
        
      } // File loop
      
      trendLegend->Draw();
      
      // Draw lines for different data taking year
      if(drawYearLines){
        
        for(int thisRun : linePosition){
          
          lumiX = getLuminosityBeforeRun(lumiPerIovWithSkips, iovVector, thisRun);
          lumiLine = new TLine(lumiX, trendZoomLow[iTrend], lumiX, trendZoomHigh[iTrend]);
          lumiLine->Draw();
          
        }
        
      }
      
      // Save the figures
      if(saveFigures){
        gPad->GetCanvas()->SaveAs(Form("output/%sTrendPtOver%.0f%s.pdf", trendName[iTrend].Data(), widePtBinBorders.at(iWidePt), saveComment));
      }
      
    } // Wide pT loop
  } // Trend type loop
  
}

/*
 * Main program
 */
int main(int argc, char **argv){
  
  //==== Read command line arguments =====
  AllInOneConfig::Options options;
  options.helper(argc, argv);
  options.parser(argc, argv);

  // Run the program
  jetHtPlotter(options.config);
}
