#include "DQMOffline/RecoB/interface/Tools.h"

#include "TROOT.h"
#include "TSystem.h"
#include "TStyle.h"
#include <cmath>

using namespace std;
using namespace RecoBTag;

//
//
// TOOLS
//
//

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double RecoBTag::HistoBinWidth(const TH1F* theHisto, const int& iBin) {
  const int& nBins = theHisto->GetSize();  // includes underflow/overflow
  // return 0.0 , if invalid bin
  if (iBin < 0 || iBin >= nBins)
    return 0.0;
  // return first binwidth, if underflow bin
  if (iBin == 0)
    return theHisto->GetBinWidth(1);
  // return last real binwidth, if overflow bin
  if (iBin == nBins - 1)
    return theHisto->GetBinWidth(nBins - 2);
  // return binwidth from histo, if within range
  return theHisto->GetBinWidth(iBin);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double RecoBTag::IntegrateHistogram(const TH1F* theHisto) {
  // include underflow and overflow: assign binwidth of first/last bin to them!!
  // integral = sum ( entry_i * binwidth_i )
  //
  double histoIntegral = 0.0;
  const int& nBins = theHisto->GetSize();
  //
  // loop over bins:
  // bin 0       : underflow
  // bin nBins-1 : overflow
  for (int iBin = 0; iBin != nBins; ++iBin) {
    const double& binWidth = HistoBinWidth(theHisto, iBin);
    histoIntegral += (*theHisto)[iBin] * binWidth;
  }
  //
  return histoIntegral;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void RecoBTag::HistoToNormalizedArrays(const TH1F* theHisto,
                                       TArrayF& theNormalizedArray,
                                       TArrayF& theLeftOfBinArray,
                                       TArrayF& theBinWidthArray) {
  const int& nBins = theHisto->GetSize();

  // check that all arrays/histo have the same size
  if (nBins == theNormalizedArray.GetSize() && nBins == theLeftOfBinArray.GetSize() &&
      nBins == theBinWidthArray.GetSize()) {
    const double& histoIntegral = IntegrateHistogram(theHisto);

    for (int iBin = 0; iBin != nBins; ++iBin) {
      theNormalizedArray[iBin] = (*theHisto)[iBin] / histoIntegral;
      theLeftOfBinArray[iBin] = theHisto->GetBinLowEdge(iBin);
      theBinWidthArray[iBin] = HistoBinWidth(theHisto, iBin);
    }

  } else {
    cout << "============>>>>>>>>>>>>>>>>" << endl
         << "============>>>>>>>>>>>>>>>>" << endl
         << "============>>>>>>>>>>>>>>>>" << endl
         << "============>>>>>>>>>>>>>>>>" << endl
         << "============>>>>>>>>>>>>>>>> HistoToNormalizedArrays failed: not equal sizes of all arrays!!" << endl
         << "============>>>>>>>>>>>>>>>>" << endl
         << "============>>>>>>>>>>>>>>>>" << endl
         << "============>>>>>>>>>>>>>>>>" << endl
         << "============>>>>>>>>>>>>>>>>" << endl;
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double RecoBTag::IntegrateArray(const TArrayF& theArray, const TArrayF& theBinWidth) {
  double arrayIntegral = 0.0;
  const int& nBins = theArray.GetSize();
  //
  for (int iBin = 0; iBin != nBins; ++iBin) {
    arrayIntegral += theArray[iBin] * theBinWidth[iBin];
  }
  //
  return arrayIntegral;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void RecoBTag::PrintCanvasHistos(TCanvas* canvas,
                                 const std::string& psFile,
                                 const std::string& epsFile,
                                 const std::string& gifFile) {
  //
  //
  // to create gif in 'batch mode' (non-interactive) see
  // http://root.cern.ch/cgi-bin/print_hit_bold.pl/root/roottalk/roottalk00/0402.html?gifbatch#first_hit
  //
  // ROOT 4 can do it!!??
  //
  // if string = "" don't print to corresponding file
  //
  if (!psFile.empty())
    canvas->Print(psFile.c_str());
  if (!epsFile.empty())
    canvas->Print(epsFile.c_str(), "eps");
  // if in batch: use a converter tool
  const std::string& rootVersion(gROOT->GetVersion());
  const bool& rootCanGif = rootVersion.find('4') == 0 || rootVersion.find('5') == 0;
  if (!gifFile.empty()) {
    if (!(gROOT->IsBatch()) || rootCanGif) {  // to find out if running in batch mode
      cout << "--> Print directly gif!" << endl;
      canvas->Print(gifFile.c_str(), "gif");
    } else {
      if (!epsFile.empty()) {  // eps file must have been created before
        cout << "--> Print gif via scripts!" << endl;
        const std::string& executeString1 = "pstopnm -ppm -xborder 0 -yborder 0 -portrait " + epsFile;
        gSystem->Exec(executeString1.c_str());
        const std::string& ppmFile = epsFile + "001.ppm";
        const std::string& executeString2 = "ppmtogif " + ppmFile + " > " + gifFile;
        gSystem->Exec(executeString2.c_str());
        const std::string& executeString3 = "rm " + ppmFile;
        gSystem->Exec(executeString3.c_str());  // delete the intermediate file
      }
    }
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TObjArray RecoBTag::getHistArray(TFile* histoFile, const std::string& baseName) {
  //
  // return the TObjArray built from the basename
  //
  //
  TObjArray histos(3);  // reserve 3
  //
  const std::string nameB(baseName + "B");
  const std::string nameC(baseName + "C");
  const std::string nameDUSG(baseName + "DUSG");
  //
  histos.Add((TH1F*)histoFile->Get(nameB.c_str()));
  histos.Add((TH1F*)histoFile->Get(nameC.c_str()));
  histos.Add((TH1F*)histoFile->Get(nameDUSG.c_str()));
  //
  return histos;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string RecoBTag::flavour(const int& flav) {
  switch (flav) {
    case 1:
      return "d";
    case 2:
      return "u";
    case 3:
      return "s";
    case 4:
      return "c";
    case 5:
      return "b";
    case 21:
      return "g";
    default:
      return "";
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool RecoBTag::flavourIsD(const int& flav) { return flav == 1; }
bool RecoBTag::flavourIsU(const int& flav) { return flav == 2; }
bool RecoBTag::flavourIsS(const int& flav) { return flav == 3; }
bool RecoBTag::flavourIsC(const int& flav) { return flav == 4; }
bool RecoBTag::flavourIsB(const int& flav) { return flav == 5; }
bool RecoBTag::flavourIsG(const int& flav) { return flav == 21; }

bool RecoBTag::flavourIsDUS(const int& flav) { return (flavourIsD(flav) || flavourIsU(flav) || flavourIsS(flav)); }
bool RecoBTag::flavourIsDUSG(const int& flav) { return (flavourIsDUS(flav) || flavourIsG(flav)); }

bool RecoBTag::flavourIsNI(const int& flav) {
  return !(flavourIsD(flav) || flavourIsU(flav) || flavourIsS(flav) || flavourIsC(flav) || flavourIsB(flav) ||
           flavourIsG(flav));
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int RecoBTag::checkCreateDirectory(const std::string& directory) {
  cout << "====>>>> ToolsC:checkCreateDirectory() : " << endl;
  int exists = gSystem->Exec(("ls -d " + directory).c_str());
  // create it if it doesn't exist
  if (exists != 0) {
    cout << "====>>>> ToolsC:checkCreateDirectory() : The directory does not exist : " << directory << endl;
    cout << "====>>>> ToolsC:checkCreateDirectory() : I'll try to create it" << endl;
    const int& create = gSystem->Exec(("mkdir " + directory).c_str());
    if (create != 0) {
      cout << "====>>>> ToolsC:checkCreateDirectory() : Creation of directory failed : " << directory << endl
           << "====>>>> ToolsC:checkCreateDirectory() : Please check your write permissions!" << endl;
    } else {
      cout << "====>>>> ToolsC:checkCreateDirectory() : Creation of directory successful!" << endl;
      // check again if it exists now
      cout << "====>>>> ToolsC:checkCreateDirectory() : " << endl;
      exists = gSystem->Exec(("ls -d " + directory).c_str());
      if (exists != 0)
        cout << "ToolsC:checkCreateDirectory() : However, it still doesn't exist!?" << endl;
    }
  }
  cout << endl;
  return exists;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int RecoBTag::findBinClosestYValue(const TH1F* histo, const float& yVal, const float& yLow, const float& yHigh) {
  //
  // Find the bin in a 1-dim. histogram which has its y-value closest to
  // the given value yVal where the value yVal has to be in the range yLow < yVal < yHigh.
  // If it is outside this range the corresponding bin number is returned as negative value.
  // Currently, there is no protection if there are many bins with the same value!
  // The user has to take care to interpret the output correctly.
  //

  // init
  const int& nBins = histo->GetNbinsX() - 2;  // -2 because we don't include under/overflow alos in this loop
  int iBinClosestInit = 0;
  // init start value properly: must avoid that the real one is not filled
  float yClosestInit;
  //
  const float& maxInHisto = histo->GetMaximum();
  const float& minInHisto = histo->GetMinimum();
  //
  // if yVal is smaller than max -> take any value well above the maximum
  if (yVal <= maxInHisto) {
    yClosestInit = maxInHisto + 1;
  } else {
    // if yVal is greater than max value -> take a value < minimum
    yClosestInit = minInHisto - 1.0;
  }

  int iBinClosest = iBinClosestInit;
  float yClosest = yClosestInit;

  // loop over bins of histogram
  for (int iBin = 1; iBin <= nBins; ++iBin) {
    const float& yBin = histo->GetBinContent(iBin);
    if (fabs(yBin - yVal) < fabs(yClosest - yVal)) {
      yClosest = yBin;
      iBinClosest = iBin;
    }
  }

  // check if in interval
  if (yClosest < yLow || yClosest > yHigh) {
    iBinClosest *= -1;
  }

  // check that not the initialization bin (would mean that init value was the closest)
  if (iBinClosest == iBinClosestInit) {
    cout << "====>>>> ToolsC=>findBinClosestYValue() : WARNING: returned bin is the initialization bin!!" << endl;
  }

  return iBinClosest;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
vector<int> RecoBTag::findBinClosestYValueAtFixedZ(const TH2F* histoY,
                                                   const float& yVal,
                                                   const float& yLow,
                                                   const float& yHigh,
                                                   const TH2F* histoZ,
                                                   const vector<double>& zVal) {
  //
  // Find the bin in a 2-dim. histogram which has its y-value closest to
  // the given value yVal where the value yVal has to be in the range yLow < yVal < yHigh.
  // If it is outside this range the corresponding bin number is returned as negative value.
  // The bin should also correspond to a value of z=zVal within the same precision as yVal.
  // Currently, there is no protection if there are many bins with the same value!
  // The user has to take care to interpret the output correctly.
  //

  // init
  const int& nBinsX = histoY->GetNbinsX() - 2;  // -2 because we don't include under/overflow alos in this loop
  const int& nBinsY = histoY->GetNbinsY() - 2;  // -2 because we don't include under/overflow alos in this loop
  int iBinClosestInit = 0;
  // init start value properly: must avoid that the real one is not filled
  vector<float> yClosestInit(zVal.size());
  //
  const float& maxInHisto = histoY->GetMaximum();
  const float& minInHisto = histoY->GetMinimum();
  //
  // if yVal is smaller than max -> take any value well above the maximum
  for (unsigned int i = 0; i < yClosestInit.size(); i++) {
    if (yVal <= maxInHisto) {
      yClosestInit[i] = maxInHisto + 1;
    } else {
      // if yVal is greater than max value -> take a value < minimum
      yClosestInit[i] = minInHisto - 1.0;
    }
  }

  vector<int> iBinClosest(zVal.size(), iBinClosestInit);
  vector<float> yClosest(yClosestInit);

  // loop over bins of histogram
  for (int iBinX = 1; iBinX <= nBinsX; ++iBinX) {
    for (int iBinY = 1; iBinY <= nBinsY; ++iBinY) {
      const float& yBin = histoY->GetBinContent(iBinX, iBinY);
      for (unsigned int i = 0; i < zVal.size(); i++) {
        if (fabs(yBin - yVal) < fabs(yClosest[i] - yVal)) {
          const float& zLow = zVal[i] - (yVal - yLow);
          const float& zHigh = zVal[i] + (yHigh - yVal);
          const float& zBin = histoZ->GetBinContent(iBinX, iBinY);
          if (zBin < zLow || zBin > zHigh)
            continue;
          yClosest[i] = yBin;
          iBinClosest[i] = histoY->GetBin(iBinX, iBinY);
        }
      }
    }
  }
  // check if in interval
  for (unsigned int i = 0; i < yClosest.size(); i++) {
    if (yClosest[i] < yLow || yClosest[i] > yHigh)
      iBinClosest[i] *= -1;
  }

  return iBinClosest;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TStyle* RecoBTag::setTDRStyle() {
  TStyle* tdrStyle = new TStyle("tdrStyle", "Style for P-TDR");

  // For the canvas:
  tdrStyle->SetCanvasBorderMode(0);
  tdrStyle->SetCanvasColor(kWhite);
  tdrStyle->SetCanvasDefH(600);  //Height of canvas
  tdrStyle->SetCanvasDefW(600);  //Width of canvas
  tdrStyle->SetCanvasDefX(0);    //POsition on screen
  tdrStyle->SetCanvasDefY(0);

  // For the Pad:
  tdrStyle->SetPadBorderMode(0);
  // tdrStyle->SetPadBorderSize(Width_t size = 1);
  tdrStyle->SetPadColor(kWhite);
  tdrStyle->SetPadGridX(false);
  tdrStyle->SetPadGridY(false);
  tdrStyle->SetGridColor(0);
  tdrStyle->SetGridStyle(3);
  tdrStyle->SetGridWidth(1);

  // For the frame:
  tdrStyle->SetFrameBorderMode(0);
  tdrStyle->SetFrameBorderSize(1);
  tdrStyle->SetFrameFillColor(0);
  tdrStyle->SetFrameFillStyle(0);
  tdrStyle->SetFrameLineColor(1);
  tdrStyle->SetFrameLineStyle(1);
  tdrStyle->SetFrameLineWidth(1);

  // For the histo:
  // tdrStyle->SetHistFillColor(1);
  // tdrStyle->SetHistFillStyle(0);
  tdrStyle->SetHistLineColor(1);
  tdrStyle->SetHistLineStyle(0);
  tdrStyle->SetHistLineWidth(1);
  // tdrStyle->SetLegoInnerR(Float_t rad = 0.5);
  // tdrStyle->SetNumberContours(Int_t number = 20);

  tdrStyle->SetEndErrorSize(15);
  //   tdrStyle->SetErrorMarker(20);
  tdrStyle->SetErrorX(1);

  tdrStyle->SetMarkerStyle(21);
  tdrStyle->SetMarkerSize(1.);

  //For the fit/function:
  tdrStyle->SetOptFit(0);
  tdrStyle->SetFitFormat("5.4g");
  tdrStyle->SetFuncColor(2);
  tdrStyle->SetFuncStyle(1);
  tdrStyle->SetFuncWidth(1);

  //For the date:
  tdrStyle->SetOptDate(0);
  // tdrStyle->SetDateX(Float_t x = 0.01);
  // tdrStyle->SetDateY(Float_t y = 0.01);

  // For the statistics box:
  tdrStyle->SetOptFile(1111);
  tdrStyle->SetOptStat(0);  // To display the mean and RMS:   SetOptStat("mr");
  tdrStyle->SetStatColor(kWhite);
  tdrStyle->SetStatFont(42);
  tdrStyle->SetStatFontSize(0.025);
  tdrStyle->SetStatTextColor(1);
  tdrStyle->SetStatFormat("6.4g");
  tdrStyle->SetStatBorderSize(1);
  tdrStyle->SetStatH(0.2);
  tdrStyle->SetStatW(0.15);
  // tdrStyle->SetStatStyle(Style_t style = 1001);
  // tdrStyle->SetStatX(Float_t x = 0);
  // tdrStyle->SetStatY(Float_t y = 0);

  // Margins:
  tdrStyle->SetPadTopMargin(0.05);
  tdrStyle->SetPadBottomMargin(0.13);
  tdrStyle->SetPadLeftMargin(0.16);
  tdrStyle->SetPadRightMargin(0.02);

  // For the Global title:

  tdrStyle->SetOptTitle(0);
  tdrStyle->SetTitleW(0.8);  // Set the width of the title box

  tdrStyle->SetTitleFont(42);
  tdrStyle->SetTitleColor(1);
  tdrStyle->SetTitleTextColor(1);
  tdrStyle->SetTitleFillColor(10);
  tdrStyle->SetTitleFontSize(0.05);
  // tdrStyle->SetTitleH(0); // Set the height of the title box
  // tdrStyle->SetTitleX(0); // Set the position of the title box
  // tdrStyle->SetTitleY(0.985); // Set the position of the title box
  // tdrStyle->SetTitleStyle(Style_t style = 1001);
  // tdrStyle->SetTitleBorderSize(2);

  // For the axis titles:

  tdrStyle->SetTitleColor(1, "XYZ");
  tdrStyle->SetTitleFont(42, "XYZ");
  tdrStyle->SetTitleSize(0.06, "XYZ");
  // tdrStyle->SetTitleXSize(Float_t size = 0.02); // Another way to set the size?
  // tdrStyle->SetTitleYSize(Float_t size = 0.02);
  tdrStyle->SetTitleXOffset(0.75);
  tdrStyle->SetTitleYOffset(0.75);
  // tdrStyle->SetTitleOffset(1.1, "Y"); // Another way to set the Offset

  // For the axis labels:

  tdrStyle->SetLabelColor(1, "XYZ");
  tdrStyle->SetLabelFont(42, "XYZ");
  tdrStyle->SetLabelOffset(0.007, "XYZ");
  tdrStyle->SetLabelSize(0.05, "XYZ");

  // For the axis:

  tdrStyle->SetAxisColor(1, "XYZ");
  tdrStyle->SetStripDecimals(kTRUE);
  tdrStyle->SetTickLength(0.03, "XYZ");
  tdrStyle->SetNdivisions(510, "XYZ");
  tdrStyle->SetPadTickX(1);  // To get tick marks on the opposite side of the frame
  tdrStyle->SetPadTickY(1);

  // Change for log plots:
  tdrStyle->SetOptLogx(0);
  tdrStyle->SetOptLogy(0);
  tdrStyle->SetOptLogz(0);

  // Postscript options:
  tdrStyle->SetPaperSize(21., 28.);
  //  tdrStyle->SetPaperSize(20.,20.);
  // tdrStyle->SetLineScalePS(Float_t scale = 3);
  // tdrStyle->SetLineStyleString(Int_t i, const char* text);
  // tdrStyle->SetHeaderPS(const char* header);
  // tdrStyle->SetTitlePS(const char* pstitle);

  // tdrStyle->SetBarOffset(Float_t baroff = 0.5);
  // tdrStyle->SetBarWidth(Float_t barwidth = 0.5);
  // tdrStyle->SetPaintTextFormat(const char* format = "g");
  // tdrStyle->SetPalette(Int_t ncolors = 0, Int_t* colors = 0);
  // tdrStyle->SetTimeOffset(Double_t toffset);
  // tdrStyle->SetHistMinimumZero(kTRUE);

  tdrStyle->cd();
  return tdrStyle;
}
// tdrGrid: Turns the grid lines on (true) or off (false)

void RecoBTag::tdrGrid(const bool& gridOn) {
  TStyle* tdrStyle = setTDRStyle();
  tdrStyle->SetPadGridX(gridOn);
  tdrStyle->SetPadGridY(gridOn);
  tdrStyle->cd();
}

// fixOverlay: Redraws the axis

void RecoBTag::fixOverlay() { gPad->RedrawAxis(); }

string RecoBTag::itos(const int& i)  // convert int to string
{
  ostringstream s;
  s << i;
  return s.str();
}
