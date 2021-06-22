#ifndef DQM_TRACKERREMAPPER_PHASE1PIXELMAPS_H
#define DQM_TRACKERREMAPPER_PHASE1PIXELMAPS_H

#include "TH2Poly.h"
#include "TGraph.h"
#include "TH1.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"

#include <fmt/printf.h>
#include <fstream>
#include <boost/tokenizer.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

#ifndef PHASE1PIXELMAP_STANDALONE
#define LOGDEBUG(x) LogDebug(x)
#else
#define LOGDEBUG(x) std::cout << x << ": "
#endif

using indexedCorners = std::map<unsigned int, std::pair<std::vector<float>, std::vector<float>>>;

/*--------------------------------------------------------------------
/ Ancillary class to build pixel phase-1 tracker maps
/--------------------------------------------------------------------*/
class Phase1PixelMaps {
public:
  Phase1PixelMaps(const char* option)
      : m_option{option},
        m_isBooked{std::make_pair(false, false)},
        m_knownNames{{}},
        m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
            edm::FileInPath("Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml").fullPath())} {
    // set the rescale to true by default
    m_autorescale = true;

    // store the file in path for the corners (BPIX)
    for (unsigned int i = 1; i <= 4; i++) {
      m_cornersBPIX.push_back(edm::FileInPath(Form("DQM/SiStripMonitorClient/data/Geometry/vertices_barrel_%i", i)));
    }

    // store the file in path for the corners (BPIX)
    for (int j : {-3, -2, -1, 1, 2, 3}) {
      m_cornersFPIX.push_back(edm::FileInPath(Form("DQM/SiStripMonitorClient/data/Geometry/vertices_forward_%i", j)));
    }
  }

  ~Phase1PixelMaps() = default;

  // set of no rescale
  inline void setNoRescale() { m_autorescale = false; }

  // set option, but only if not already set
  void resetOption(const char* option);

  // book them all
  void book(const std::string& currentHistoName, const char* what, const char* zaxis);

  // booking methods
  void bookBarrelHistograms(const std::string& currentHistoName, const char* what, const char* zaxis);
  void bookForwardHistograms(const std::string& currentHistoName, const char* what, const char* zaxis);

  // fill them all
  void fill(const std::string& currentHistoName, unsigned int id, double value);

  // filling methods
  void fillBarrelBin(const std::string& currentHistoName, unsigned int id, double value);
  void fillForwardBin(const std::string& currentHistoName, unsigned int id, double value);

  // beautification methods
  void beautifyAllHistograms();
  void setBarrelScale(const std::string& currentHistoName, std::pair<float, float> extrema);
  void setForwardScale(const std::string& currentHistoName, std::pair<float, float> extrema);

  // drawing methos
  void drawBarrelMaps(const std::string& currentHistoName, TCanvas& canvas, const char* drawOption = nullptr);
  void drawForwardMaps(const std::string& currentHistoName, TCanvas& canvas, const char* drawOption = nullptr);
  void drawSummaryMaps(const std::string& currentHistoName, TCanvas& canvas);

private:
  Option_t* m_option;
  bool m_autorescale;
  std::pair<bool, bool> m_isBooked;
  std::vector<std::string> m_knownNames;

  TrackerTopology m_trackerTopo;

  std::map<uint32_t, std::shared_ptr<TGraph>> bins, binsSummary;
  std::map<std::string, std::vector<std::shared_ptr<TH2Poly>>> pxbTh2PolyBarrel;
  std::map<std::string, std::shared_ptr<TH2Poly>> pxbTh2PolyBarrelSummary;
  std::map<std::string, std::vector<std::shared_ptr<TH2Poly>>> pxfTh2PolyForward;
  std::map<std::string, std::shared_ptr<TH2Poly>> pxfTh2PolyForwardSummary;

  std::vector<edm::FileInPath> m_cornersBPIX;
  std::vector<edm::FileInPath> m_cornersFPIX;

  const indexedCorners retrieveCorners(const std::vector<edm::FileInPath>& cornerFiles, const unsigned int reads);

  // called by book histograms
  void bookBarrelBins(const std::string& currentHistoName);
  void bookForwardBins(const std::string& currentHistoName);

  // graphics
  void makeNicePlotStyle(TH1* hist);
  void adjustCanvasMargins(TVirtualPad* pad, float top, float bottom, float left, float right);
  void rescaleAllBarrel(const std::string& currentHistoName);
  void rescaleAllForward(const std::string& currentHistoName);
};

#endif
