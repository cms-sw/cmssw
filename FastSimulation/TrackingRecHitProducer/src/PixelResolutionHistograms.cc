//
//  The implementation of the PixelResolutinHistograms.cc class.  Please
//  look at PixelResolutionHistograms.h header file for the interface.
//
//------------------------------------------------------------------------------

// The switch, undefined in CMSSW release, and defined by standalone compilation script:

#ifdef SI_PIXEL_TEMPLATE_STANDALONE
//
//--- Stand-alone: Include a the header file from the local directory, as well as
//    dummy implementations of SimpleHistogramGenerator, LogInfo, LogError and LogDebug...
//
#include "PixelResolutionHistograms.h"
//
class TH1F;
class TH2F;
class SimpleHistogramGenerator {
public:
  SimpleHistogramGenerator(TH1F* hist) : hist_(hist){};

private:
  TH1F* hist_;  // we don't own it
};
#define LOGDEBUG std::cout
#define LOGERROR std::cout
#define LOGINFO std::cout
//
#else
//--- We're inside a CMSSW release: Include the real thing.
//
#include "FastSimulation/TrackingRecHitProducer/interface/PixelResolutionHistograms.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "FastSimulation/Utilities/interface/SimpleHistogramGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
#define LOGDEBUG LogDebug("")
#define LOGERROR edm::LogError("Error")
#define LOGINFO edm::LogInfo("Info")
//
#endif

// Generic C stuff
#include <cmath>
#include <iostream>
#include <string>

// ROOT
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>

// Global definitions
const float cmtomicron = 10000.0;

//------------------------------------------------------------------------------
//  Constructor: Books the FastSim Histograms, given the input parameters
//  which are provided as arguments. These variables are then const inside
//  the class. (That is, once we make the histograms, we can't change the
//  definition of the binning.)
//------------------------------------------------------------------------------
PixelResolutionHistograms::PixelResolutionHistograms(std::string filename,   // ROOT file for histograms
                                                     std::string rootdir,    // Subdirectory in the file, "" if none
                                                     std::string descTitle,  // Descriptive title
                                                     unsigned int detType,   // Where we are... (&&& do we need this?)
                                                     std::vector<double> cotbetaEdges,
                                                     std::vector<double> cotalphaEdges)
    : weOwnHistograms_(true),  // we'll be making some histos
      detType_(detType),
      qbinWidth_(1),
      qbins_(4),
      binningHisto_(nullptr),
      resMultiPixelXHist_(),
      resSinglePixelXHist_(),  // all to nullptr
      resMultiPixelYHist_(),
      resSinglePixelYHist_(),  // all to nullptr
      qbinHist_(),             // all to nullptr
      file_(nullptr),
      status_(0),
      resMultiPixelXGen_(),
      resSinglePixelXGen_(),
      resMultiPixelYGen_(),
      resSinglePixelYGen_(),
      qbinGen_() {
  file_ = std::make_unique<TFile>(filename.c_str(), "RECREATE");

  // Dummy 2D histogram to store binning:
  binningHisto_ = new TH2F("ResHistoBinning",
                           descTitle.c_str(),
                           cotbetaEdges.size() - 1,
                           cotbetaEdges.data(),
                           cotalphaEdges.size() - 1,
                           cotalphaEdges.data());
  cotbetaAxis_ = binningHisto_->GetXaxis();
  cotalphaAxis_ = binningHisto_->GetYaxis();

  // Store detType in the underflow bin
  binningHisto_->SetBinContent(0, 0, detType_);

  // All other histograms:
  Char_t histo[200];
  Char_t title[200];
  Char_t binstr[50];

  for (int ii = 0; ii < cotbetaAxis_->GetNbins(); ii++) {
    for (int jj = 0; jj < cotalphaAxis_->GetNbins(); jj++) {
      sprintf(binstr,
              "cotbeta %.1f-%.1f cotalpha %.2f-%.2f",
              cotbetaEdges[ii],
              cotbetaEdges[ii + 1],
              cotalphaEdges[jj],
              cotalphaEdges[jj + 1]);
      //
      //--- Histograms for clusters with multiple pixels hit in a given direction.
      //
      for (size_t kk = 0; kk < qbins_; kk++) {
        //information of bits of histogram names
        //--- First bit 1/0 barrel/forward, second 1/0 multi/single, cotbeta, cotalpha, qbins
        sprintf(histo, "hx%d1%02d%d%zu", detType_, ii + 1, jj + 1, kk + 1);
        sprintf(title, "%s qbin %zu npixel>1 X", binstr, kk + 1);
        resMultiPixelXHist_[ii][jj][kk] = new TH1F(histo, title, 1000, -0.05, 0.05);

        sprintf(histo, "hy%d1%02d%d%zu", detType_, ii + 1, jj + 1, kk + 1);
        sprintf(title, "%s qbin %zu npixel>1 Y", binstr, kk + 1);
        resMultiPixelYHist_[ii][jj][kk] = new TH1F(histo, title, 1000, -0.05, 0.05);
      }
      //
      //--- Histograms for clusters where only a single pixel was hit in a given direction.
      //
      sprintf(histo, "hx%d0%02d%d", detType_, ii + 1, jj + 1);
      sprintf(title, "%s npixel=1 X", binstr);
      resSinglePixelXHist_[ii][jj] = new TH1F(histo, title, 1000, -0.05, 0.05);

      sprintf(histo, "hy%d0%02d%d", detType_, ii + 1, jj + 1);
      sprintf(title, "%s npixel=1 Y", binstr);
      resSinglePixelYHist_[ii][jj] = new TH1F(histo, title, 1000, -0.05, 0.05);

      sprintf(histo, "hqbin%d%02d%d", detType_, ii + 1, jj + 1);
      sprintf(title, "%s qbin", binstr);
      qbinHist_[ii][jj] = new TH1F(histo, title, 4, -0.49, 3.51);
    }
  }
}

//------------------------------------------------------------------------------
//  Another constructor: consistent bin width scheme.
//
//  The other parameters are the same (needed later) and must correspond
//  to the histograms we are loading from the file.
//------------------------------------------------------------------------------
PixelResolutionHistograms::PixelResolutionHistograms(std::string filename,   // ROOT file for histograms
                                                     std::string rootdir,    // Subdirectory in the file, "" if none
                                                     std::string descTitle,  // Descriptive title
                                                     unsigned int detType,   // Where we are... (&&& do we need this?)
                                                     double cotbetaBinWidth,
                                                     double cotbetaLowEdge,
                                                     int cotbetaBins,
                                                     double cotalphaBinWidth,
                                                     double cotalphaLowEdge,
                                                     int cotalphaBins)
    : PixelResolutionHistograms(filename,
                                rootdir,
                                descTitle,
                                detType,
                                getBinEdges(cotbetaBinWidth, cotbetaLowEdge, cotbetaBins),
                                getBinEdges(cotalphaBinWidth, cotalphaLowEdge, cotalphaBins)) {}

//------------------------------------------------------------------------------
// Method to create a list of bin edges based on input width, lower edge, and
// number of bins.
//------------------------------------------------------------------------------
std::vector<double> PixelResolutionHistograms::getBinEdges(double width, double lowEdge, int nbins) {
  std::vector<double> binedges;
  for (int ibin = 0; ibin <= nbins; ibin++) {
    binedges.emplace_back(lowEdge + ibin * width);
  }
  return binedges;
}

std::vector<double> PixelResolutionHistograms::getBinEdges(TAxis* axis) {
  return getBinEdges(axis->GetBinWidth(1), axis->GetBinLowEdge(1), axis->GetNbins());
}

//------------------------------------------------------------------------------
// Sanity check for histograms
//------------------------------------------------------------------------------
inline bool PixelResolutionHistograms::histCheck(TH1F* hist, std::string histname, bool ignore_single, bool ignore_qBit, const int& statusToSet) {
  std::cout << histname << std::endl;
  if (!hist) {
    if (!ignore_single && !ignore_qBit) {
      status_ = statusToSet;
      LOGERROR << "Failed to find histogram=" << std::string(histname);
      return true;
    }
    return false;
  } else {
    LOGDEBUG << "Found histo " << std::string(histname) << " with title = " << std::string(hist->GetTitle()) << std::endl;
    if (hist->GetEntries() < 5) {
      LOGINFO << "Histogram " << std::string(histname) << " has only " << hist->GetEntries() << " entries. Trouble ahead."
              << std::endl;
    }
    return true;
  }
}

//------------------------------------------------------------------------------
//  Another constructor: load the histograms from one file.
//     filename = full path to filename
//     rootdir  = ROOT directory inside the file
//
//  The other parameters are the same (needed later) and must correspond
//  to the histograms we are loading from the file.
//------------------------------------------------------------------------------
PixelResolutionHistograms::PixelResolutionHistograms(std::string filename,
                                                     std::string rootdir,
                                                     int detType,
                                                     bool ignore_multi,
                                                     bool ignore_single,
                                                     bool ignore_qBin)
    : weOwnHistograms_(false),  // resolution histograms are owned by the ROOT file
      detType_(-1),
      qbinWidth_(1),
      qbins_(4),
      binningHisto_(nullptr),
      resMultiPixelXHist_(),
      resSinglePixelXHist_(),  // all to nullptr
      resMultiPixelYHist_(),
      resSinglePixelYHist_(),  // all to nullptr
      qbinHist_(),             // all to nullptr
      file_(nullptr),
      status_(0),
      resMultiPixelXGen_(),
      resSinglePixelXGen_(),
      resMultiPixelYGen_(),
      resSinglePixelYGen_(),
      qbinGen_() {
  Char_t histo[200];  // the name of the histogram
  Char_t title[200];  // histo title, for debugging and sanity checking (compare inside file)
  Char_t binstr[50];
  TH1F* tmphist = nullptr;  // cache for histo pointer

  //--- Open the file for reading.
  file_ = std::make_unique<TFile>(filename.c_str(), "READ");
  if (!file_) {
    status_ = 1;
    LOGERROR << "PixelResolutionHistograms:: Error, file " << filename << " not found.";
    return;  // PixelTemplateSmearerBase will throw an exception upon our return.
  }

  //--- The dummy 2D histogram with the binning of cot\beta and cot\alpha:
  binningHisto_ = (TH2F*)file_->Get(Form("%s%s", rootdir.c_str(), "ResHistoBinning"));
  if (!binningHisto_) {
    status_ = 11;
    LOGERROR << "PixelResolutionHistograms:: Error, binning histogrram ResHistoBinning not found.";
    return;  // PixelTemplateSmearerBase will throw an exception upon our return.
  }

  if (detType == -1) {  //--- Fish out detType from the underflow bin:
    detType_ = binningHisto_->GetBinContent(0, 0);
  } else {
    detType_ = detType;  // constructor's argument overrides what's in ResHistoBinning histogram.
  }

  //--- Now we fill the binning variables:
  cotbetaAxis_ = binningHisto_->GetXaxis();
  cotalphaAxis_ = binningHisto_->GetYaxis();
  std::vector<double> cotalphaEdges, cotbetaEdges;

  if (cotbetaAxis_->GetXbins()->GetSize() > 0) {
    for (int iedge = 0; iedge < cotbetaAxis_->GetXbins()->GetSize(); iedge++) {
      cotbetaEdges.push_back(cotbetaAxis_->GetXbins()->GetAt(iedge));
    }
  } else {
    cotbetaEdges = getBinEdges(cotbetaAxis_);
  }

  if (cotalphaAxis_->GetXbins()->GetSize() > 0) {
    for (int iedge = 0; iedge < cotalphaAxis_->GetXbins()->GetSize(); iedge++) {
      cotalphaEdges.push_back(cotalphaAxis_->GetXbins()->GetAt(iedge));
    }
  } else {
    cotalphaEdges = getBinEdges(cotalphaAxis_);
  }

  if (!ignore_multi) {
    //--- Histograms for clusters with multiple pixels hit in a given direction.
    for (size_t ii = 0; ii < cotbetaEdges.size()-1; ii++) {
      for (size_t jj = 0; jj < cotalphaEdges.size()-1; jj++) {
        sprintf(binstr,
                "cotbeta %.1f-%.1f cotalpha %.2f-%.2f",
                cotbetaEdges[ii],
                cotbetaEdges[ii + 1],
                cotalphaEdges[jj],
                cotalphaEdges[jj + 1]);
        for (size_t kk = 0; kk < qbins_; kk++) {
          //information of bits of histogram names
          //--- First bit 1/0 barrel/forward, second 1/0 multi/single, cotbeta, cotalpha, qbins
          sprintf(histo, "hx%d1%02zu%zu%zu", detType_, ii + 1, jj + 1, kk + 1);
          sprintf(title, "%s qbin %zu npixel>1 X", binstr, kk + 1);
          tmphist = (TH1F*)file_->Get(Form("%s%s", rootdir.c_str(), histo));

          if (!histCheck(tmphist, histo, false, false, 2)) {return;}
          resMultiPixelXHist_[ii][jj][kk] = tmphist;
          resMultiPixelXGen_[ii][jj][kk] = new SimpleHistogramGenerator(tmphist);

          sprintf(histo, "hy%d1%02zu%zu%zu", detType_, ii + 1, jj + 1, kk + 1);
          sprintf(title, "%s qbin %zu npixel>1 Y", binstr, kk + 1);
          tmphist = (TH1F*)file_->Get(Form("%s%s", rootdir.c_str(), histo));

          if (!histCheck(tmphist, histo, false, false, 3)) {return;}
          resMultiPixelYHist_[ii][jj][kk] = tmphist;
          resMultiPixelYGen_[ii][jj][kk] = new SimpleHistogramGenerator(tmphist);
        }

        //--- Histograms for clusters where only a single pixel was hit in a given direction.
        //
        //--- Single pixel, along X.
        sprintf(histo, "hx%d0%02zu%zu", detType_, ii + 1, jj + 1);
        sprintf(title, "%s npixel=1 X", binstr);
        tmphist = (TH1F*)file_->Get(Form("%s%s", rootdir.c_str(), histo));

        if (!histCheck(tmphist, histo, ignore_single, false, 4)) {return;}
        resSinglePixelXHist_[ii][jj] = tmphist;
        resSinglePixelXGen_[ii][jj] = new SimpleHistogramGenerator(tmphist);

        //--- Single pixel, along Y.
        sprintf(histo, "hy%d0%02zu%zu", detType_, ii + 1, jj + 1);
        sprintf(title, "%s npixel=1 Y", binstr);
        tmphist = (TH1F*)file_->Get(Form("%s%s", rootdir.c_str(), histo));

        if (!histCheck(tmphist, histo, ignore_single, false, 5)) {return;}
        resSinglePixelYHist_[ii][jj] = tmphist;
        resSinglePixelYGen_[ii][jj] = new SimpleHistogramGenerator(tmphist);

        //--- qBin distribution, for this (cotbeta, cotalpha) bin.
        sprintf(histo, "hqbin%d%02zu%zu", detType_, ii + 1, jj + 1);
        sprintf(title, "%s qbin", binstr);
        tmphist = (TH1F*)file_->Get(Form("%s%s", rootdir.c_str(), histo));

        if (!histCheck(tmphist, histo, false, ignore_qBin, 6)) {return;}
        qbinHist_[ii][jj] = tmphist;
        qbinGen_[ii][jj] = new SimpleHistogramGenerator(tmphist);
      }
    }
  }
}

//------------------------------------------------------------------------------
//  Destructor.  Use file_ pointer to tell whether we loaded the histograms
//  from a file (and do not own them), or we built them ourselves and thus need
//  to delete them.
//------------------------------------------------------------------------------
PixelResolutionHistograms::~PixelResolutionHistograms() {
  //--- Delete histograms, but only if we own them. If
  //--- they came from a file, let them be.
  //
  if (!weOwnHistograms_) {
    //--- Read the histograms from the TFile, the file will take care of them.
    file_->Close();
    /// delete file_ ;   // no need to delete if unique_ptr<>
    /// file_ = 0;
  } else {
    //--- We made the histograms, so first write them inthe output ROOT file and close it.
    LOGINFO << "PixelResHistoStore: Writing the histograms to the output file. "  // << filename
            << std::endl;
    file_->Write();
    file_->Close();

    // ROOT file has the ownership, and once the file is closed,
    // all of these guys are deleted.  So, we don't need to do anything.
  }  // else

  //--- Delete FastSim generators. (It's safe to delete a nullptr.)
  for (int ii = 0; ii < cotbetaAxis_->GetNbins(); ii++) {
    for (int jj = 0; jj < cotalphaAxis_->GetNbins(); jj++) {
      for (size_t kk = 0; kk < qbins_; kk++) {
        delete resMultiPixelXGen_[ii][jj][kk];
        delete resMultiPixelYGen_[ii][jj][kk];
      }
      delete resSinglePixelXGen_[ii][jj];
      delete resSinglePixelYGen_[ii][jj];
      delete qbinGen_[ii][jj];
    }
  }
}

//------------------------------------------------------------------------------
//  Fills the appropriate FastSim histograms.
//  Returns 0 if the relevant histogram(s) were found and filled, 1 if not.
//------------------------------------------------------------------------------
int PixelResolutionHistograms::Fill(
    double dx, double dy, double cotalpha, double cotbeta, int qbin, int nxpix, int nypix) {
  int icotalpha, icotbeta, iqbin;
  icotalpha = cotalphaAxis_->FindFixBin(cotalpha) - 1;
  icotbeta = cotbetaAxis_->FindFixBin(cotbeta) - 1;
  iqbin = qbin > 2 ? 3 : qbin;
  if (icotalpha >= 0 && icotalpha < cotalphaAxis_->GetNbins() && icotbeta >= 0 && icotbeta < cotbetaAxis_->GetNbins()) {
    qbinHist_[icotbeta][icotalpha]->Fill((double)iqbin);
    if (nxpix == 1)
      resSinglePixelXHist_[icotbeta][icotalpha]->Fill(dx / cmtomicron);
    else
      resMultiPixelXHist_[icotbeta][icotalpha][iqbin]->Fill(dx / cmtomicron);
    if (nypix == 1)
      resSinglePixelYHist_[icotbeta][icotalpha]->Fill(dy / cmtomicron);
    else
      resMultiPixelYHist_[icotbeta][icotalpha][iqbin]->Fill(dy / cmtomicron);
  }

  return 0;
}

const SimpleHistogramGenerator* PixelResolutionHistograms::getGenerator(
    double cotalpha, double cotbeta, int qbin, bool single, bool isX) {
  int icotalpha, icotbeta, iqbin;
  icotalpha = std::max(0, std::min(cotalphaAxis_->GetNbins() - 1, cotalphaAxis_->FindFixBin(cotalpha) - 1));
  icotbeta = std::max(0, std::min(cotbetaAxis_->GetNbins() - 1, cotbetaAxis_->FindFixBin(cotbeta) - 1));
  iqbin = qbin > 2 ? 3 : qbin;  // if (qbin>2) then = 3, else return qbin

  // At this point we are sure to return *some bin* from the 3D histogram
  if (single) {
    if (isX) {
      return resSinglePixelXGen_[icotbeta][icotalpha];
    } else {
      return resSinglePixelYGen_[icotbeta][icotalpha];
    }
  } else {
    if (isX) {
      return resMultiPixelXGen_[icotbeta][icotalpha][iqbin];
    } else {
      return resMultiPixelYGen_[icotbeta][icotalpha][iqbin];
    }
  }
}

//------------------------------------------------------------------------------
//  Return the histogram generator for resolution in X.  A generator contains
//  both the histogram and knows how to throw a random number off it.  It is
//  called from FastSim (from PixelTemplateSmearerBase).
//  If cotalpha or cotbeta are outside of the range, return the end of the range.
//------------------------------------------------------------------------------
const SimpleHistogramGenerator* PixelResolutionHistograms::getGeneratorX(double cotalpha,
                                                                         double cotbeta,
                                                                         int qbin,
                                                                         bool single) {
  return getGenerator(cotalpha, cotbeta, qbin, single, true);
}

//------------------------------------------------------------------------------
//  Return the histogram generator for resolution in Y.  A generator contains
//  both the histogram and knows how to throw a random number off it.  It is
//  called from FastSim (from PixelTemplateSmearerBase).
//  If cotalpha or cotbeta are outside of the range, return the end of the range.
//------------------------------------------------------------------------------
const SimpleHistogramGenerator* PixelResolutionHistograms::getGeneratorY(double cotalpha,
                                                                         double cotbeta,
                                                                         int qbin,
                                                                         bool single) {
  return getGenerator(cotalpha, cotbeta, qbin, single, false);
}
