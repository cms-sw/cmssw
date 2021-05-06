#include "JetHtPlotConfiguration.h"

/*
 * Default constructor
 */
JetHtPlotConfiguration::JetHtPlotConfiguration() :
  fDebugLevel(0),
  fDrawTrackQA(false),
  fInputFileNames(0),
  fLegendComments(0),
  fMarkerColor(0),
  fMarkerStyle(0),
  fDrawPlotsForEachIOV(false),
  fNIovInOnePlot(1),
  fUseLuminosityForTrends(true),
  fSkipRunsWithNoData(false),
  fNormalizeQAplots(true),
  fSaveComment(""),
  fLumiPerIovFile("lumiPerRun_Run2.txt"),
  fDrawYearLines(false),
  fRunsForLines(0),
  fWidePtBinBorders(0),
  fMakeIovListForSlides(false),
  fIovListForSlides("iovListForSlides.txt")
{
  
  // Default zoom values
  double defaultProfileZoomLow[knProfileTypes] = {28,45,30,7,40,20,25,20};
  double defaultProfileZoomHigh[knProfileTypes] = {60,80,95,40,70,90,90,80};
  double defaultTrendZoomLow[knTrendTypes] = {20,10};
  double defaultTrendZoomHigh[knTrendTypes] = {95,90};
  
  // Initialize arrays
  for(int iHistogram = 0; iHistogram < knHistogramTypes; iHistogram++){
    fDrawHistogram[iHistogram] = false;
  }
  for(int iProfile = 0; iProfile < knProfileTypes; iProfile++){
    fDrawProfile[iProfile] = false;
    fProfileZoomLow[iProfile] = defaultProfileZoomLow[iProfile];
    fProfileZoomHigh[iProfile] = defaultProfileZoomHigh[iProfile];
  }
  for(int iTrend = 0; iTrend < knTrendTypes; iTrend++){
    fDrawTrend[iTrend] = false;
    fTrendZoomLow[iTrend] = defaultTrendZoomLow[iTrend];
    fTrendZoomHigh[iTrend] = defaultTrendZoomHigh[iTrend];
  }
  
}

/*
 * Copy constructor
 */
JetHtPlotConfiguration::JetHtPlotConfiguration(const JetHtPlotConfiguration& in) :
  fDebugLevel(in.fDebugLevel),
  fDrawTrackQA(in.fDrawTrackQA),
  fInputFileNames(in.fInputFileNames),
  fLegendComments(in.fLegendComments),
  fMarkerColor(in.fMarkerColor),
  fMarkerStyle(in.fMarkerStyle),
  fDrawPlotsForEachIOV(in.fDrawPlotsForEachIOV),
  fNIovInOnePlot(in.fNIovInOnePlot),
  fUseLuminosityForTrends(in.fUseLuminosityForTrends),
  fSkipRunsWithNoData(in.fSkipRunsWithNoData),
  fNormalizeQAplots(in.fNormalizeQAplots),
  fSaveComment(in.fSaveComment),
  fLumiPerIovFile(in.fLumiPerIovFile),
  fDrawYearLines(in.fDrawYearLines),
  fRunsForLines(in.fRunsForLines),
  fWidePtBinBorders(in.fWidePtBinBorders),
  fMakeIovListForSlides(in.fMakeIovListForSlides),
  fIovListForSlides(in.fIovListForSlides)
{
  // Copy arrays
  for(int iHistogram = 0; iHistogram < knHistogramTypes; iHistogram++){
    fDrawHistogram[iHistogram] = in.fDrawHistogram[iHistogram];
  }
  for(int iProfile = 0; iProfile < knProfileTypes; iProfile++){
    fDrawProfile[iProfile] = in.fDrawProfile[iProfile];
    fProfileZoomLow[iProfile] = in.fProfileZoomLow[iProfile];
    fProfileZoomHigh[iProfile] = in.fProfileZoomHigh[iProfile];
  }
  for(int iTrend = 0; iTrend < knTrendTypes; iTrend++){
    fDrawTrend[iTrend] = in.fDrawTrend[iTrend];
    fTrendZoomLow[iTrend] = in.fTrendZoomLow[iTrend];
    fTrendZoomHigh[iTrend] = in.fTrendZoomHigh[iTrend];
  }
}

/*
 * Assingment operator
 */
JetHtPlotConfiguration& JetHtPlotConfiguration::operator=(const JetHtPlotConfiguration& in){
  
  if (&in==this) return *this;
  
  fDebugLevel = in.fDebugLevel;
  fDrawTrackQA = in.fDrawTrackQA;
  fInputFileNames = in.fInputFileNames;
  fLegendComments = in.fLegendComments;
  fMarkerColor = in.fMarkerColor;
  fMarkerStyle = in.fMarkerStyle;
  fDrawPlotsForEachIOV = in.fDrawPlotsForEachIOV;
  fNIovInOnePlot = in.fNIovInOnePlot;
  fUseLuminosityForTrends = in.fUseLuminosityForTrends;
  fSkipRunsWithNoData = in.fSkipRunsWithNoData;
  fNormalizeQAplots = in.fNormalizeQAplots;
  fSaveComment = in.fSaveComment;
  fLumiPerIovFile = in.fLumiPerIovFile;
  fDrawYearLines = in.fDrawYearLines;
  fRunsForLines = in.fRunsForLines;
  fWidePtBinBorders = in.fWidePtBinBorders;
  fMakeIovListForSlides = in.fMakeIovListForSlides;
  fIovListForSlides = in.fIovListForSlides;

  // Copy arrays
  for(int iHistogram = 0; iHistogram < knHistogramTypes; iHistogram++){
    fDrawHistogram[iHistogram] = in.fDrawHistogram[iHistogram];
  }
  for(int iProfile = 0; iProfile < knProfileTypes; iProfile++){
    fDrawProfile[iProfile] = in.fDrawProfile[iProfile];
    fProfileZoomLow[iProfile] = in.fProfileZoomLow[iProfile];
    fProfileZoomHigh[iProfile] = in.fProfileZoomHigh[iProfile];
  }
  for(int iTrend = 0; iTrend < knTrendTypes; iTrend++){
    fDrawTrend[iTrend] = in.fDrawTrend[iTrend];
    fTrendZoomLow[iTrend] = in.fTrendZoomLow[iTrend];
    fTrendZoomHigh[iTrend] = in.fTrendZoomHigh[iTrend];
  }
  
  return *this;
}

/*
 * Destructor
 */
JetHtPlotConfiguration::~JetHtPlotConfiguration(){}

/*
 * Read the configuration from json file
 */
void JetHtPlotConfiguration::ReadJsonFile(const std::string fileName){
  
  // Read the file to property tree
  namespace pt = boost::property_tree;
  pt::ptree configuration;
  pt::read_json(fileName, configuration);
    
  // Read the alignments and styles for these
  int indexForDefault = 0;
  std::string alignmentName;
  std::string thisValue;
  int thisNumber;
  for (pt::ptree::value_type &result : configuration.get_child("jethtplot.alignments")){
    alignmentName = result.first;
    
    // Input file for the alignment
    try{
      thisValue = configuration.get_child(Form("jethtplot.alignments.%s.inputFile", alignmentName.c_str())).get_value<std::string>();

      // From the file name, replace CMSSW_BASE string with this shell variable
      boost::replace_all(thisValue, "CMSSW_BASE", getenv("CMSSW_BASE"));

      fInputFileNames.push_back(thisValue);
    } catch(const std::exception& e){
      if(fDebugLevel > 0){
        std::cout << "No input file given for alignment " << alignmentName << "! All the configuration for this alignment is skipped." << std::endl;
      }
      continue;
    }
    
    // Legend text for the alignment
    try{
      thisValue = configuration.get_child(Form("jethtplot.alignments.%s.legendText", alignmentName.c_str())).get_value<std::string>();
      fLegendComments.push_back(thisValue);
    } catch(const std::exception& e){
      if(fDebugLevel > 0){
        std::cout << "No legend text given for alignment " << alignmentName << "! Using default value " << Form("Alignment%d",indexForDefault) << "." << std::endl;
        fLegendComments.push_back(Form("Alignment%d",indexForDefault));
      }
    }
    
    // Marker color related to the alignment
    try{
      thisNumber = configuration.get_child(Form("jethtplot.alignments.%s.color", alignmentName.c_str())).get_value<int>();
      fMarkerColor.push_back(thisNumber);
    } catch(const std::exception& e){
      if(fDebugLevel > 0){
        std::cout << "No marker color given for alignment " << alignmentName << "! Using default value " << fDefaultColors[indexForDefault] << "." << std::endl;
        fMarkerColor.push_back(fDefaultColors[indexForDefault]);
      }
    }
    
    // Marker style related to the alignment
    try{
      thisNumber = configuration.get_child(Form("jethtplot.alignments.%s.style", alignmentName.c_str())).get_value<int>();
      fMarkerStyle.push_back(thisNumber);
    } catch(const std::exception& e){
      if(fDebugLevel > 0){
        std::cout << "No marker style given for alignment " << alignmentName << "! Using default value " << fDefaultStyle << "." << std::endl;
        fMarkerStyle.push_back(fDefaultStyle);
      }
    }
    
    indexForDefault++;
  }
  
  // Read the drawing configuration for track QA
  try{
    fDrawTrackQA = configuration.get_child(Form("jethtplot.%s", fJsonTrackQAname.c_str())).get_value<bool>();
  } catch(const std::exception& e){
    if(fDebugLevel > 0){
      std::cout << "No value " << Form("jethtplot.%s", fJsonTrackQAname.c_str()) << " in configuration. Using default value." << std::endl;
    }
  }
  
  // Read the drawing configuration for dz and dxy value and error histograms
  for(int iHistogram = 0; iHistogram < knHistogramTypes; iHistogram++){
    try{
      fDrawHistogram[iHistogram] = configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameHistogram.c_str(), fJsonNameHistogram[iHistogram].c_str())).get_value<bool>();
    } catch(const std::exception& e){
      if(fDebugLevel > 0){
        std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameHistogram.c_str(), fJsonNameHistogram[iHistogram].c_str()) << " in configuration. Using default value." << std::endl;
      }
    }
  }
  
  // Read the drawing configuration for profile plots
  for(int iProfile = 0; iProfile < knProfileTypes; iProfile++){
    try{
      fDrawProfile[iProfile] = configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameProfile.c_str(), fJsonNameProfile[iProfile].c_str())).get_value<bool>();
    } catch(const std::exception& e){
      if(fDebugLevel > 0){
        std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameProfile.c_str(), fJsonNameProfile[iProfile].c_str()) << " in configuration. Using default value." << std::endl;
      }
    }
  }
  
  // Read the axis zoom values for profile plots
  for(int iProfile = 0; iProfile < knProfileTypes; iProfile++){
    try{
      fProfileZoomLow[iProfile] = configuration.get_child(Form("jethtplot.%s.min%s", fJsonCategoryNameProfileZoom.c_str(), fJsonNameProfileZoom[iProfile].c_str())).get_value<double>();
    } catch(const std::exception& e){
      if(fDebugLevel > 0){
        std::cout << "No value " << Form("jethtplot.%s.min%s", fJsonCategoryNameProfileZoom.c_str(), fJsonNameProfileZoom[iProfile].c_str()) << " in configuration. Using default value." << std::endl;
      }
    }
    
    try{
      fProfileZoomHigh[iProfile] = configuration.get_child(Form("jethtplot.%s.max%s", fJsonCategoryNameProfileZoom.c_str(), fJsonNameProfileZoom[iProfile].c_str())).get_value<double>();
    } catch(const std::exception& e){
      if(fDebugLevel > 0){
        std::cout << "No value " << Form("jethtplot.%s.max%s", fJsonCategoryNameProfileZoom.c_str(), fJsonNameProfileZoom[iProfile].c_str()) << " in configuration. Using default value." << std::endl;
      }
    }
  }
  
  // Read the drawing configuration for trend plots
  for(int iTrend = 0; iTrend < knTrendTypes; iTrend++){
    try{
      fDrawTrend[iTrend] = configuration.get_child(Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameTrend[iTrend].c_str())).get_value<bool>();
    } catch(const std::exception& e){
      if(fDebugLevel > 0){
        std::cout << "No value " << Form("jethtplot.%s.%s", fJsonCategoryNameTrend.c_str(), fJsonNameTrend[iTrend].c_str()) << " in configuration. Using default value." << std::endl;
      }
    }
  }
  
  // Read the axis zoom values for trend plots
  for(int iTrend = 0; iTrend < knTrendTypes; iTrend++){
    try{
      fTrendZoomLow[iTrend] = configuration.get_child(Form("jethtplot.%s.min%s", fJsonCategoryNameTrendZoom.c_str(), fJsonNameTrendZoom[iTrend].c_str())).get_value<double>();
    } catch(const std::exception& e){
      if(fDebugLevel > 0){
        std::cout << "No value " << Form("jethtplot.%s.min%s", fJsonCategoryNameTrendZoom.c_str(), fJsonNameTrendZoom[iTrend].c_str()) << " in configuration. Using default value." << std::endl;
      }
    }
    
    try{
      fTrendZoomHigh[iTrend] = configuration.get_child(Form("jethtplot.%s.max%s", fJsonCategoryNameTrendZoom.c_str(), fJsonNameTrendZoom[iTrend].c_str())).get_value<double>();
    } catch(const std::exception& e){
      if(fDebugLevel > 0){
        std::cout << "No value " << Form("jethtplot.%s.max%s", fJsonCategoryNameTrendZoom.c_str(), fJsonNameTrendZoom[iTrend].c_str()) << " in configuration. Using default value." << std::endl;
      }
    }
  }
  
  // Read the configuration for plotting individual IOV plots
  try{
    fDrawPlotsForEachIOV = configuration.get_child(Form("jethtplot.%s", fJsonDrawPlotsForEachIOV.c_str())).get_value<bool>();
  } catch(const std::exception& e){
    if(fDebugLevel > 0){
      std::cout << "No value " << Form("jethtplot.%s", fJsonDrawPlotsForEachIOV.c_str()) << " in configuration. Using default value." << std::endl;
    }
  }
  
  // Read the number of IOVs plotted in the same figure, if individual IOV are plotted
  try{
    fNIovInOnePlot = configuration.get_child(Form("jethtplot.%s", fJsonNIovInOnePlot.c_str())).get_value<int>();
  } catch(const std::exception& e){
    if(fDebugLevel > 0){
      std::cout << "No value " << Form("jethtplot.%s", fJsonNIovInOnePlot.c_str()) << " in configuration. Using default value." << std::endl;
    }
  }
  
  // Read the configuration for plotting trends plots as a function of luminosity
  try{
    fUseLuminosityForTrends = configuration.get_child(Form("jethtplot.%s", fJsonUseLuminosityForTrends.c_str())).get_value<bool>();
  } catch(const std::exception& e){
    if(fDebugLevel > 0){
      std::cout << "No value " << Form("jethtplot.%s", fJsonUseLuminosityForTrends.c_str()) << " in configuration. Using default value." << std::endl;
    }
  }
  
  // Read the configuration for skipping runs with no data in trend plots
  try{
    fSkipRunsWithNoData = configuration.get_child(Form("jethtplot.%s", fJsonSkipRunsWithNoData.c_str())).get_value<bool>();
  } catch(const std::exception& e){
    if(fDebugLevel > 0){
      std::cout << "No value " << Form("jethtplot.%s", fJsonSkipRunsWithNoData.c_str()) << " in configuration. Using default value." << std::endl;
    }
  }
  
  // Read the configuration for normalizing QA plots by their integral
  try{
    fNormalizeQAplots = configuration.get_child(Form("jethtplot.%s", fJsonNormalizeQAplots.c_str())).get_value<bool>();
  } catch(const std::exception& e){
    if(fDebugLevel > 0){
      std::cout << "No value " << Form("jethtplot.%s", fJsonNormalizeQAplots.c_str()) << " in configuration. Using default value." << std::endl;
    }
  }
  
  // Read comment given to saved figures
  try{
    fSaveComment = configuration.get_child(Form("jethtplot.%s", fJsonSaveComment.c_str())).get_value<std::string>();
  } catch(const std::exception& e){
    if(fDebugLevel > 0){
      std::cout << "No value " << Form("jethtplot.%s", fJsonSaveComment.c_str()) << " in configuration. Using default value." << std::endl;
    }
  }
  
  // Read the file name for luminosity per IOV
  try{
    fLumiPerIovFile = configuration.get_child(Form("jethtplot.%s", fJsonLumiPerIovFile.c_str())).get_value<std::string>();

    // From the file name, replace CMSSW_BASE string with this shell variable
    boost::replace_all(fLumiPerIovFile, "CMSSW_BASE", getenv("CMSSW_BASE"));

  } catch(const std::exception& e){
    if(fDebugLevel > 0){
      std::cout << "No value " << Form("jethtplot.%s", fJsonLumiPerIovFile.c_str()) << " in configuration. Using default value " << fLumiPerIovFile << "." << std::endl;
    }
  }
  
  // Read the flag for drawing vertical lines to trend plots
  try{
    fDrawYearLines = configuration.get_child(Form("jethtplot.%s", fJsonDrawYearLines.c_str())).get_value<bool>();
  } catch(const std::exception& e){
    if(fDebugLevel > 0){
      std::cout << "No value " << Form("jethtplot.%s", fJsonDrawYearLines.c_str()) << " in configuration. Using default value " << fDrawYearLines << "." << std::endl;
    }
  }
  
  // Read run positions to which vertical lines are drawn
  try{
    for (auto& item : configuration.get_child(Form("jethtplot.%s",fJsonRunsForLines.c_str()))){
      fRunsForLines.push_back(item.second.get_value<int>());
    }
  } catch(const std::exception& e){
    if(fDebugLevel > 0){
      std::cout << "No value " << Form("jethtplot.%s", fJsonRunsForLines.c_str()) << " in configuration. Using default values 290543 and 314881." << std::endl;
    }
    fRunsForLines.push_back(290543);
    fRunsForLines.push_back(314881);
  }
  
  // Read the bin borders for the wide pT bims.
  try{
    for (auto& item : configuration.get_child(Form("jethtplot.%s",fJsonWidePtBinBorders.c_str()))){
      fWidePtBinBorders.push_back(item.second.get_value<double>());
    }
  } catch(const std::exception& e){
    if(fDebugLevel > 0){
      std::cout << "No value " << Form("jethtplot.%s", fJsonWidePtBinBorders.c_str()) << " in configuration. Using default values 3, 5, 10, 20, 50 and 100." << std::endl;
    }
    fWidePtBinBorders.push_back(3.0);
    fWidePtBinBorders.push_back(5.0);
    fWidePtBinBorders.push_back(10.0);
    fWidePtBinBorders.push_back(20.0);
    fWidePtBinBorders.push_back(50.0);
    fWidePtBinBorders.push_back(100.0);
  }

  // Read the flag for creating an IOV list for the slides
  try{
    fMakeIovListForSlides = configuration.get_child(Form("jethtplot.%s", fJsonMakeIovListForSlides.c_str())).get_value<bool>();
  } catch(const std::exception& e){
    if(fDebugLevel > 0){
      std::cout << "No value " << Form("jethtplot.%s", fJsonMakeIovListForSlides.c_str()) << " in configuration. Using default value " << fMakeIovListForSlides << "." << std::endl;
    }
  }
  
  // Read the output file name for the IOV list for the slides
  try{
    fIovListForSlides = configuration.get_child(Form("jethtplot.%s", fJsonIovListForSlides.c_str())).get_value<std::string>();

    // From the file name, replace CMSSW_BASE string with this shell variable
    boost::replace_all(fIovListForSlides, "CMSSW_BASE", getenv("CMSSW_BASE"));

  } catch(const std::exception& e){
    if(fDebugLevel > 0){
      std::cout << "No value " << Form("jethtplot.%s", fJsonIovListForSlides.c_str()) << " in configuration. Using default value " << fIovListForSlides << "." << std::endl;
    }
  }

}

/*
 * Print the current configuration to console
 */
void JetHtPlotConfiguration::PrintConfiguration() const{
  
  // Print all input files
  std::cout << "Input files: " << std::endl;
  for(std::string fileName : fInputFileNames){
    std::cout << fileName << std::endl;
  }
  
  // Print legend text related to input files
  std::cout << "Legend text related to input files: " << std::endl;
  for(std::string legendText : fLegendComments){
    std::cout << legendText << std::endl;
  }
  
  // Print all marker colors
  std::cout << "Marker colors: " << std::endl;
  for(int color : fMarkerColor){
    std::cout << color << std::endl;
  }
  
  // Print all marker styles
  std::cout << "Marker styles: " << std::endl;
  for(int style : fMarkerStyle){
    std::cout << style << std::endl;
  }
  
  // Print the configuration for QA plot drawing
  std::cout << fJsonTrackQAname << " : " << fDrawTrackQA << std::endl;
  
  // Print the configuration for histogram drawing
  for(int iHistogram = 0; iHistogram < knHistogramTypes; iHistogram++){
    std::cout << fJsonCategoryNameHistogram << " " << fJsonNameHistogram[iHistogram] << " : " << fDrawHistogram[iHistogram] << std::endl;
  }
  
  // Print the configuration for profile plot drawing
  for(int iProfile = 0; iProfile < knProfileTypes; iProfile++){
    std::cout << fJsonCategoryNameProfile << " " << fJsonNameProfile[iProfile] << " : " << fDrawProfile[iProfile] << std::endl;
  }
  
  // Print the configuration for profile plot zoom
  for(int iProfile = 0; iProfile < knProfileTypes; iProfile++){
    std::cout << fJsonCategoryNameProfileZoom << " min" << fJsonNameProfileZoom[iProfile] << " : " << fProfileZoomLow[iProfile] << std::endl;
    std::cout << fJsonCategoryNameProfileZoom << " max" << fJsonNameProfileZoom[iProfile] << " : " << fProfileZoomHigh[iProfile] << std::endl;
  }
  
  // Print the configuration for trend plot drawing
  for(int iTrend = 0; iTrend < knTrendTypes; iTrend++){
    std::cout << fJsonCategoryNameTrend << " " << fJsonNameTrend[iTrend] << " : " << fDrawTrend[iTrend] << std::endl;
  }
  
  // Print the configuration for trend plot zoom
  for(int iTrend = 0; iTrend < knTrendTypes; iTrend++){
    std::cout << fJsonCategoryNameTrendZoom << " min" << fJsonNameTrendZoom[iTrend] << " : " << fTrendZoomLow[iTrend] << std::endl;
    std::cout << fJsonCategoryNameTrendZoom << " max" << fJsonNameTrendZoom[iTrend] << " : " << fTrendZoomHigh[iTrend] << std::endl;
  }
  
  // Print plotting configuration values
  std::cout << fJsonLumiPerIovFile << " : " << fLumiPerIovFile << std::endl;
  std::cout << fJsonDrawYearLines << " : " << fDrawYearLines << std::endl;
  std::cout << "Runs for vertical lines: " << std::endl;
  for(int myRun : fRunsForLines){
    std::cout << myRun << " ";
  }
  std::cout << std::endl;
  
  std::cout << "Wide pT bin borders: " << std::endl;
  for(double myPt : fWidePtBinBorders){
    std::cout << myPt << " ";
  }
  std::cout << std::endl;

  std::cout << fJsonDrawPlotsForEachIOV << " : " << fDrawPlotsForEachIOV << std::endl;
  std::cout << fJsonNIovInOnePlot << " : " << fNIovInOnePlot << std::endl;
  std::cout << fJsonUseLuminosityForTrends << " : " << fUseLuminosityForTrends << std::endl;
  std::cout << fJsonSkipRunsWithNoData<< " : " << fSkipRunsWithNoData << std::endl;
  std::cout << fJsonNormalizeQAplots << " : " << fNormalizeQAplots << std::endl;
  
  std::cout << fJsonMakeIovListForSlides << " : " << fMakeIovListForSlides << std::endl;
  std::cout << fJsonIovListForSlides << " : " << fIovListForSlides << std::endl;
  
  // Print the save comment
  std::cout << "Saved files are given a comment: " << fSaveComment << std::endl;

}

// Getter for track QA histogram drawing flag
bool JetHtPlotConfiguration::GetDrawTrackQA() const{
  return fDrawTrackQA;
}

// Getter for dxy and dz histogram drawing flags
bool JetHtPlotConfiguration::GetDrawHistogram(const int iHistogram) const{
  return fDrawHistogram[iHistogram];
}

// Getter for profile drawing flags
bool JetHtPlotConfiguration::GetDrawProfile(const int iProfile) const{
  return fDrawProfile[iProfile];
}

// Getter for low end of profile axis zooms
double JetHtPlotConfiguration::GetProfileZoomLow(const int iProfile) const{
  return fProfileZoomLow[iProfile];
}

// Getter for high end of profile axis zooms
double JetHtPlotConfiguration::GetProfileZoomHigh(const int iProfile) const{
  return fProfileZoomHigh[iProfile];
}

// Getter for trend drawing flags
bool JetHtPlotConfiguration::GetDrawTrend(const int iTrend) const{
  return fDrawTrend[iTrend];
}

// Getter for low end of trend axis zooms
double JetHtPlotConfiguration::GetTrendZoomLow(const int iTrend) const{
  return fTrendZoomLow[iTrend];
}

// Getter for high end of trend axis zooms
double JetHtPlotConfiguration::GetTrendZoomHigh(const int iTrend) const{
  return fTrendZoomHigh[iTrend];
}

// Getter for number of input files
int JetHtPlotConfiguration::GetNInputFiles() const{
  return fInputFileNames.size();
}

// Getter for input file of index iFile
std::string JetHtPlotConfiguration::GetInputFile(const int iFile) const{
  const int inputFileCount = fInputFileNames.size();
  if(iFile < 0 || iFile >= inputFileCount) return "";
  return fInputFileNames.at(iFile);
}

// Getter for input file vector
std::vector<std::string> JetHtPlotConfiguration::GetInputFiles() const{
  return fInputFileNames;
}

// Getter for a comment added to the legend
std::string JetHtPlotConfiguration::GetLegendComment(const int iComment) const{
  const int commentCount = fLegendComments.size();
  if(iComment < 0 || iComment >= commentCount) return "";
  return fLegendComments.at(iComment);
}

// Getter for the marker color related to an alignment
int JetHtPlotConfiguration::GetMarkerColor(const int iFile) const{
  if(iFile < 0) return kBlack;
  if(iFile > 10) return kBlack;
  const int markerColorCount = fMarkerColor.size();
  if(iFile >= markerColorCount) return fDefaultColors[iFile];
  return fMarkerColor.at(iFile);
}

// Getter for the marker style related to an alignment
int JetHtPlotConfiguration::GetMarkerStyle(const int iFile) const{
  const int markerStyleCount = fMarkerStyle.size();
  if(iFile < 0 || iFile >= markerStyleCount) return fDefaultStyle;
  return fMarkerStyle.at(iFile);
}

// Getter for the luminosity per IOV file
const char* JetHtPlotConfiguration::GetLumiPerIovFile() const{
  return fLumiPerIovFile.c_str();
}

// Getter for the flag for drawing vertical lines to trend plots
bool JetHtPlotConfiguration::GetDrawYearLines() const{
  return fDrawYearLines;
}

// Getter for the run positions to where vertical lines are drawn
std::vector<int> JetHtPlotConfiguration::GetRunsForLines() const{
  return fRunsForLines;
}

// Getter for bin borders used in the wide pT binned histogram
std::vector<double> JetHtPlotConfiguration::GetWidePtBinBorders() const{
  return fWidePtBinBorders;
}

// Getter for drawing plots for each IOV
bool JetHtPlotConfiguration::GetDrawPlotsForEachIOV() const{
  return fDrawPlotsForEachIOV;
}

// Getter for number of IOVs plotted in each figure
int JetHtPlotConfiguration::GetNIovInOnePlot() const{
  return fNIovInOnePlot;
}

// Getter for drawing trends as a function of luminosity
bool JetHtPlotConfiguration::GetUseLuminosityForTrends() const{
  return fUseLuminosityForTrends;
}

// Getter for skipping runs with no data in trend plots
bool JetHtPlotConfiguration::GetSkipRunsWithNoData() const{
  return fSkipRunsWithNoData;
}

// Getter for normalizing QA plots
bool JetHtPlotConfiguration::GetNormalizeQAplots() const{
  return fNormalizeQAplots;
}

// Getter for comment given to saved figures
const char* JetHtPlotConfiguration::GetSaveComment() const{
  return fSaveComment.c_str();
}

// Getter for flag to produce IOV list for slides
bool JetHtPlotConfiguration::GetMakeIovListForSlides() const{
  return fMakeIovListForSlides;
}

// Getter for the name given to the IOV list for slides
const char* JetHtPlotConfiguration::GetIovListForSlides() const{
  return fIovListForSlides.c_str();
}

