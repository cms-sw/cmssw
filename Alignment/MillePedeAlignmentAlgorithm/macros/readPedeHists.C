//
//    |-------------------------------------------------------------------|
//    ||-----------------------------------------------------------------||
//    ||   ROOT script to read the millepede.his file produced by pede   ||
//    ||-----------------------------------------------------------------||
//    |-------------------------------------------------------------------|
//
// Author     : Gero Flucke
// Date       : July 2007
// Last update: $Date: 2009/01/20 20:22:27 $ by $Author: flucke $
//
//
// Usage:
// ======
//
// Start ROOT and compile (!) the script:
//
// root [0] .L readPedeHists.C+
// Info in <TUnixSystem::ACLiC>: creating shared library ./readPedeHists_C.so
//
// If the millepede.his file is in the directory that ROOT was started in, just call
//
// root [1] readPedeHists()
//
// ROOT will display the histograms (TH1) and XY-data objects (TGraph).
//
// The following options and their combinations can be given as first argument:
// - print: produce a postscript file millepede.his.ps 
// - write: write the histograms and graphs into the ROOT file millepede.his.root
// - nodraw: skip displaying (write/print work still fine)
//
// Note that both options 'print' and 'write' will overwrite existing files.
//
// If the millepede.his file has been renamed or is not in the local directory,
// its name can be given as second argument. The names of the postscript or ROOT files
// will be adjusted to the given name, too.
// 
// The following example will read the file '../adir/millepede_result5.his' and directly
// produce the postscript file '../adir/millepede_result5.his.ps' without displaying and
// without producing a ROOT file:
//
// root [1] readPedeHists("print nodraw", "../adir/millepede_result5.his")
// Info in <TCanvas::Print>: ps file ../adir/millepede_result5.hisps has been created
// Info in <TCanvas::Print>: Current canvas added to ps file ../adir/millepede_result5.his.ps
// Info in <TCanvas::Print>: Current canvas added to ps file ../adir/millepede_result5.his.ps
// Info in <TCanvas::Print>: Current canvas added to ps file ../adir/millepede_result5.his.ps
// Info in <TCanvas::Print>: Current canvas added to ps file ../adir/millepede_result5.his.ps
//
//
// Possible modifications:
// =======================
// - The size of the canvases is defined in ReadPedeHists::Draw() via 'nPixelX' and 'nPixelY'.
// - The number of histograms/graphs per canvas is defined in ReadPedeHists::Draw() as 
//   'nHistX' and 'nHistY'.
// - The position of the corners of the boxes giving the minimum or maximum value of a
//   histogrammed distribution is defined as the first four arguments after 'new TPaveText'
//   at the end of the method ReadPedeHists::readNextHist.
// - gStyle->SetOptStat(...), executed before readPedeHists(), defines whether you see all
//   relevant information in the statistics. Try e.g.:
//   root [0] gStyle->SetOptStat("emrou"); // or "nemrou"
//

#include <fstream>
#include <vector>
#include <utility> //  for std::pair
#include <iostream>

#include <TROOT.h>
#include <TFile.h>
#include <TDirectory.h>
#include <TError.h>
#include <TH1.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TCanvas.h>
#include <TMath.h>
#include <TPaveText.h>

//__________________________________________________________________________
void readPedeHists(Option_t *option = "", const char *txtFile = "millepede.his");//"nodraw" skips drawing, "print" produces ps file, "write" writes object in ROOT file 

//__________________________________________________________________________
//__________________________________________________________________________
//__________________________________________________________________________

class ReadPedeHists
{
 public:
  explicit ReadPedeHists(const char *txtFile);
  ~ReadPedeHists() {} // non virtual: do not inherit from this class
  void Draw(); // draw hists and graphs read from file
  void Write(const char *rootFileName); // write hists/graphs to file
  void Print(const char *printFileName); // if drawn, print into file, e.g. ps

 private:
  template<class T> 
  bool readBasic(std::ifstream &aStream, T &outValue);
  template<class T> 
  bool readBasicVector(std::ifstream &aStream, std::vector<T> &outVector);
  bool proceedTo(std::ifstream &aStream, const TString &pattern);
  void readNumVersTypeTitle(std::ifstream &file, Int_t &num, Int_t &version, Int_t &type,
			    TString &title);
  TH1 *readNextHist(std::ifstream &file);
  std::pair<TGraph *,Option_t*> readNextGraph(std::ifstream &file);
  bool readNext(std::ifstream &file, TH1 *&hist, std::pair<TGraph*,Option_t*> &graphOpt);
  void read(std::ifstream &stream);

  std::ifstream theStream;
  std::vector<TH1*> theHists;
  std::vector<std::pair<TGraph*, Option_t*> > theGraphOpts;
  std::vector<TCanvas*> theCanvases;
};

//__________________________________________________________________________
ReadPedeHists::ReadPedeHists(const char *txtFile) :
  theStream(txtFile, ios::in)
{
  if (!theStream.is_open()) {
    ::Error("ReadPedeHists::ReadPedeHists", "file %s could not be opened", txtFile);
  } else {
    this->read(theStream);
  }
}


//__________________________________________________________________________
template<class T> 
bool ReadPedeHists::readBasic(std::ifstream &aStream, T &outValue)
{
  while (true) {
    const int aChar = aStream.get();
    if (!aStream.good()) return false;

    switch(aChar) {
    case ' ':
    case '\t':
    case '\n':
      if (aStream.eof()) return false;
      continue; // to next character
    default:
      aStream.unget();
      aStream >> outValue;
      if (aStream.fail()) {// not correct type 'T' (!aStream.good() is true also in case of EOF)
        aStream.clear();
        return false; 
      } else {
        return true;
      }
    } // switch
  } // while

  ::Error("ReadPedeHists::readBasic", "Should never come here!");
  return false;
}

//__________________________________________________________________________
template<class T> 
bool ReadPedeHists::readBasicVector(std::ifstream &aStream, std::vector<T> &outVector)
{
  // vector must have desired size
  for (unsigned int i = 0; i < outVector.size(); ++i) {
    if (!readBasic(aStream, outVector[i])) return false;
  }

  return true;
}

//__________________________________________________________________________
bool ReadPedeHists::proceedTo(std::ifstream &aStream, const TString &pattern)
{
  if (pattern.IsNull()) return true;
  const char *method = "ReadPedeHists::proceedTo";

  TString line;
  do {
    line.ReadLine(aStream);
    if (line.Contains(pattern)) {
      line.ReplaceAll(pattern, "");
      line.ReplaceAll(" ", "");
      if (!line.IsNull()) {
	::Warning(method, "line contains also '%s'", line.Data());
      }
      return true;
    } else {
      ::Warning(method, "skipping line '%s'", line.Data());
    }
  } while (!aStream.eof());
  
  ::Error(method, "pattern '%s' not found", pattern.Data());
  return false; // did not find pattern
}

//__________________________________________________________________________
void ReadPedeHists::readNumVersTypeTitle(std::ifstream &file, Int_t &num,
					 Int_t &version, Int_t &type, TString &title)
{
  std::string key; // key word

  const char *method = "ReadPedeHists::readNumVersTypeTitle";
  if (!readBasic(file, num)) {
    ::Error(method, "failed reading hist number");
  }

  if (!readBasic(file, key) || key != "version") {
    ::Error(method, "expect key 'version', got '%s'", key.c_str());
  }
  if (!readBasic(file, version)) {
    ::Error(method, "failed reading version");
  }

  if (!readBasic(file, key) || key != "type") {
    ::Error(method, "expect key 'type', got '%s'", key.c_str());
  }
  if (!readBasic(file, type)) ::Error(method, "failed reading type");

  title.ReadLine(file); // Title is a full line without key after the type!
  Ssiz_t len = title.Length();
  while (len != kNPOS && len > 0 && title[--len] == ' ') {} // empty loop
  title.Resize(len+1); // remove trailing space
  title += Form(" (version %d)", version);
}

//__________________________________________________________________________
TH1 *ReadPedeHists::readNextHist(std::ifstream &file)
{
  // Key 'Histogram' assumed to be already read!

  // Until histogram title we have a fixed order to read in these numbers:
  Int_t num = -1; // hist number
  Int_t version = -1; // version (is it iteration?)
  Int_t type = -1; // type, e.g. x-axis in log scale 
  TString title; // skip spaces??
  
  const char *method = "ReadPedeHists::readNextHist"; // for errors/warnings

  readNumVersTypeTitle(file, num, version, type, title);
  if (num == -1 || version == -1 || type == -1) {
    ::Error(method, "Problems reading hist number, version or type, so skip it.");
    proceedTo(file, "end of histogram");
    return 0;
  }
  //   type 1: normal 1D histogram
  //        2: 1D histogram with bins in log_10

  // For the remaining information we accept any order, but there will be problems 
  // in case number of bins (key 'bins,') comes after 'bincontent'...
  std::vector<Float_t> nBinsUpLow(3, -1.); // nBins (int...), lower and upper edge
  std::vector<Int_t> underInOver(3, -1); // underflow : between lower/upper : overflow
  std::vector<Float_t> binContent; // do not yet know length
  Float_t min = 0., max = 0., mean = 0., sigma = 0.; // min/max of x-axis, mean/sigma of distrib.

  std::string key; // key word
  while (readBasic(file, key)) {
    if (key == "bins,") {
      // read nBins with borders
      if (!readBasic(file, key) || key != "limits") {
	::Error(method, "expect key 'limits', got (%s)", key.c_str());
      } else if (!readBasicVector(file, nBinsUpLow)) {
	::Error(method, "failed reading nBins, xLow, xUp (%f %f %f)", 
		nBinsUpLow[0], nBinsUpLow[1], nBinsUpLow[2]);
      } else {
	binContent.resize(static_cast<unsigned int>(nBinsUpLow[0]));
      }
    } else if (key == "out-low") {
      // read under-/overflow with what is 'in between'
      if (!readBasic(file, key) || key != "inside"
	  || !readBasic(file, key) || key != "out-high") {
	::Error(method, "expected keys 'inside' and 'out-high', got (%s)", key.c_str());
      } else if (!readBasicVector(file, underInOver) || underInOver[0] == -1 
		 || underInOver[1] == -1 || underInOver[2] == -1) {
	::Error(method,	"failed reading under-, 'in-' and overflow (%d %d %d)",
		underInOver[0], underInOver[1], underInOver[2]);
      }
    } else if (key == "bincontent") {
      // read bin content - problem if lenght not yet set!
      if (nBinsUpLow[0] == -1.) {
	::Error(method, "n(bins) (key 'bins') not yet set, bin content cannot be read");
      } else if (!readBasicVector(file, binContent)) {
	::Error(method, "failed reading bincontent ");
      }
    } else if (key ==  "minmax") {
      // read minimal and maximal x-value
      if (!readBasic(file, min) || !readBasic(file, max)) {
	::Error(method, "failed reading min or max (%f %f)", min, max);
      }
    } else if (key == "meansigma") {
      // read mean and sigma as calculated in pede
      if (!readBasic(file, mean) || !readBasic(file, sigma)) {
	::Error(method, "failed reading mean or sigma (%f %f)", mean, sigma);
      }
    } else if (key == "end") {
      // reached end - hopefully all has been read...
      proceedTo(file, "of histogram");
      break; // ...the while reading the next key
    } else {
      ::Error(method, "unknown key '%s', try next word", key.c_str());
    }
  }

  // now create histogram
  if (nBinsUpLow[1] == nBinsUpLow[2]) { // causes ROOT drawing errors
    nBinsUpLow[2] = nBinsUpLow[1] + 1.;
    ::Error(method, "Hist %d (version %d): same upper and lower edge (%f), set upper %f.",
	    num, version, nBinsUpLow[1], nBinsUpLow[2]);
  }
  TH1 *h = new TH1F(Form("hist%d_version%d", num, version), title,
		    binContent.size(), nBinsUpLow[1], nBinsUpLow[2]);
  h->SetBinContent(0, underInOver[0]);
  for (UInt_t iBin = 1; iBin <= binContent.size(); ++iBin) {
    h->SetBinContent(iBin, binContent[iBin - 1]);
  }
  h->SetBinContent(binContent.size() + 1, underInOver[2]);
  h->SetEntries(underInOver[0] + underInOver[1] + underInOver[2]);

  if (type == 2) {
    // could do more fancy stuff for nicer display...
    h->SetXTitle("log_{10}");
  } else if (type != 1) {
    ::Warning(method, "Unknown histogram type %d.", type);
  }

  if (mean || sigma) { // overwrite ROOT's approximations from bin contents
    Double_t stats[11] = {0.}; // no way to get this '11' from TH1... :-(
    h->GetStats(stats);
    stats[0] = stats[1] = h->GetEntries();// sum w and w^2
    stats[2] = mean * stats[0]; // sum wx
    stats[3] = (sigma * sigma + mean * mean) * stats[0]; // sum wx^2
    h->PutStats(stats);
  }
  if (min || max) {
    TPaveText *text = new TPaveText(.175, .675, .45, .875, "NDC");
    text->AddText(Form("min = %g", min));
    text->AddText(Form("max = %g", max));
    text->SetTextAlign(12);
    text->SetBorderSize(1);
    h->GetListOfFunctions()->Add(text);// 'hack' to get it drawn with the hist
  }

  return h;
}

//__________________________________________________________________________
std::pair<TGraph*, Option_t*> ReadPedeHists::readNextGraph(std::ifstream &file)
{
  // graph and drawing option...
  // Key 'XY-Data' assumed to be already read!

  TGraph *graph = 0;
  Option_t *drawOpt = 0; // fine to use simple pointer since assigned only hardcoded strings

  // Until graph title we have a fixed order to read in these numbers:
  Int_t num = -1; // graph number
  Int_t version = -1; // version (is it iteration?)
  Int_t type = -1; // cf. below
  TString title;

  const char *method = "ReadPedeHists::readNextGraph"; // for errors/warnings

  readNumVersTypeTitle(file, num, version, type, title);
  if (num == -1 || version == -1 || type == -1) {
    ::Error(method, "Problems reading graph number, version or type, so skip it.");
    proceedTo(file, "end of xy-data");
    return std::make_pair(graph, drawOpt);
  }
  // graph types: 1   dots              (x,y)
  //              2   polyline
  //              3   dots and polyline
  //              4   symbols with (x,y) and dx, dy
  //              5   same as 5
  if (type < 1 || type > 5) {
    ::Error(method, "Unknown xy-data type %d, so skip graph.", type);
    proceedTo(file, "end of xy-data");
  }

  // now read number of points and content
  UInt_t numPoints = 0;
  std::vector<Float_t> content; // do not yet know length (need two/four values per point!)

  std::string key;
  while (readBasic(file, key)) {
    if (key == "stored") {
      if (!readBasic(file, key) || key != "not-stored") {
	::Error(method, "expected key 'not-stored', got '%s'", key.c_str());
      } else if (!readBasic(file, numPoints)) {
	::Error(method, "failed reading number of points (%d)", numPoints);
      }
    } else if (key == "x-y") {
      if (type < 1 || type > 3) {
	::Error(method, "expect key x-y-dx-dy for type %d, found x-y", type);
      }
      content.resize(numPoints * 2);
      if (!readBasicVector(file, content) || !numPoints) {
	::Error(method, "failed reading x-y content%s",
		(!numPoints ? " since n(points) (key 'stored') not yet set" : ""));
      }
    } else if (key == "x-y-dx-dy") {
      if (type < 4 || type > 5) {
	::Error(method, "expect key x-y for type %d, found x-y-dx-dy", type);
      }
      content.resize(numPoints * 4);
      if (!readBasicVector(file, content) || !numPoints) {
	::Error(method, "failed reading x-y-dx-dy content%s",
		(!numPoints ? " since n(points) (key 'stored') not yet set" : ""));
      }
    } else if (key == "end") {
      proceedTo(file, "of xy-data");
      break;
    } else {
      break; 
      ::Error(method, "unknown key '%s', try next word", key.c_str());
    }
  }

  // now create TGraph(Error) and fill drawOpt
  if (type == 4 || type == 5) {
    TGraphErrors *graphE = new TGraphErrors(numPoints);
    for (unsigned int i = 0; i < content.size(); i += 4) {
      graphE->SetPoint     (i/4, content[i]  , content[i+1]);
      graphE->SetPointError(i/4, content[i+2], content[i+3]);
    }
    drawOpt = "AP";
    graph = graphE;
  } else if (type >= 1 && type <= 3) {
    graph = new TGraph(numPoints);
    for (unsigned int i = 0; i < content.size(); i += 2) {
      graph->SetPoint(i/2, content[i], content[i+1]);
    }
    if (type == 1) {
      drawOpt = "AP";
    } else if (type == 2) {
      drawOpt = "AL";
    } else if (type == 3) {
      drawOpt = "ALP";
    }
    if (TString(drawOpt).Contains("P")) graph->SetMarkerStyle(20); // 
  } // 'else' not needed, tested above

  if (graph) graph->SetNameTitle(Form("graph%d_version%d", num, version), title);
  return std::make_pair(graph, drawOpt);
}

//__________________________________________________________________________
bool ReadPedeHists::readNext(std::ifstream &file, TH1 *&hist,
			     std::pair<TGraph*,Option_t*> &graphOpt)
{
  hist = 0;
  graphOpt.first = 0;
  graphOpt.second = 0;

  TString type;
  while (true) {
    if (file.eof()) break;
    file >> type;
    if (file.fail() || (type != "Histogram" && type != "XY-Data")) {
      TString line;
      line.ReadLine(file);
      if (line != "" && line.Length() != line.CountChar(' ')) { // not empty
	::Error("ReadPedeHists::readNext", 
		"Expect 'Histogram' or 'XY-Data', but failed, line is '%s'",
		line.Data());
	if (proceedTo(file, "end of")) line.ReadLine(file); // just skip rest of line...
      }
    }

    if (type == "Histogram") hist = readNextHist(file);
    if (type == "XY-Data")  graphOpt = readNextGraph(file);
    if (hist || graphOpt.first) break;
  }
  
  return (hist || graphOpt.first);
}

//__________________________________________________________________________
void ReadPedeHists::read(std::ifstream &file)
{
  theHists.clear();
  theGraphOpts.clear();

  TH1 *hist = 0;
  std::pair<TGraph*, Option_t*> graphOpt(0,0); // graph and its drawing option
  while (readNext(file, hist, graphOpt)) {
    if (hist) theHists.push_back(hist);
    if (graphOpt.first) theGraphOpts.push_back(graphOpt);
  }
}

//__________________________________________________________________________
void ReadPedeHists::Draw()
{
  theCanvases.clear(); // memory leak?

  const Int_t nHistX = 3;
  const Int_t nPixelX = 700;
  const Int_t nHistY = 2;
  const Int_t nPixelY = 500;
  Int_t last = nHistX * nHistY;
  unsigned int iH = 0;

  while (iH < theHists.size()) {
    if (last >= nHistX * nHistY) {
      unsigned int canCorner = theCanvases.size() * 20;
      theCanvases.push_back(new TCanvas(Form("hists%d", iH), "", 
					canCorner, canCorner, nPixelX, nPixelY));
      theCanvases.back()->Divide(nHistX, nHistY);
      last = 0;
    }
    theCanvases.back()->cd(++last);
    theHists[iH]->Draw();
    ++iH;
  }

  last = nHistX * nHistY;
  iH = 0;
  while (iH < theGraphOpts.size()) {
    if (last >= nHistX * nHistY) {
      unsigned int canCorner = theCanvases.size() * 20;
      theCanvases.push_back(new TCanvas(Form("graphs%d", iH), "",
					canCorner, canCorner, nPixelX, nPixelY));
      theCanvases.back()->Divide(nHistX, nHistY);
      last = 0;
    }
    theCanvases.back()->cd(++last);
    theGraphOpts[iH].first->Draw(theGraphOpts[iH].second);
    ++iH;
  }
}

//__________________________________________________________________________
void ReadPedeHists::Print(const char *printFileName)
{
  std::vector<TCanvas*>::iterator iC = theCanvases.begin(), iE = theCanvases.end();
  if (iC == iE) return; // empty...

  theCanvases.front()->Print(Form("%s[", printFileName)); // just open ps
  while(iC != iE) {
    (*iC)->Print(printFileName);
    ++iC;
  }
  theCanvases.front()->Print(Form("%s]", printFileName)); // just close ps
       
}

//__________________________________________________________________________
void ReadPedeHists::Write(const char *rootFileName)
{
  if (theHists.empty() && theGraphOpts.empty()) return;

  ::Info("ReadPedeHists::Write", "(Re-)Creating ROOT file %s.", rootFileName);

  TDirectory *oldDir = gDirectory;
  TFile *rootFile = TFile::Open(rootFileName, "RECREATE");

  for (std::vector<TH1*>::iterator iH = theHists.begin(), iE = theHists.end();
       iH != iE; ++iH) {
    (*iH)->Write();
  }

  for (std::vector<std::pair<TGraph*,Option_t*> >::iterator iG = theGraphOpts.begin(),
	 iE = theGraphOpts.end(); iG != iE; ++iG) {
    (*iG).first->Write();
  }

  delete rootFile;
  oldDir->cd();
}

//__________________________________________________________________________
//__________________________________________________________________________
//__________________________________________________________________________
void readPedeHists(Option_t *option, const char *txtFile)
{
  ReadPedeHists reader(txtFile);
  TString opt(option);
  opt.ToLower();
  
  const bool oldBatch = gROOT->IsBatch();
  if (opt.Contains("nodraw")) {
    opt.ReplaceAll("nodraw", "");
    gROOT->SetBatch(true);
  }

  reader.Draw();

  if (opt.Contains("print")) {
    opt.ReplaceAll("print", "");
    reader.Print(TString(Form("%s.ps", txtFile)));
  }

  if (opt.Contains("write")) {
    opt.ReplaceAll("write", "");
    reader.Write(TString(Form("%s.root", txtFile)));
  }

  gROOT->SetBatch(oldBatch);
  opt.ReplaceAll(" ", "");
  if (!opt.IsNull()) {
    ::Warning("readPedeHists", "Unknown option '%s', know 'nodraw', 'print' and 'write'.",
	      opt.Data());
  }
}
