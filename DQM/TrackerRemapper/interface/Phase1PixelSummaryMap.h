#ifndef DQM_TRACKERREMAPPER_PHASE1PIXELSUMMARYMAP_H
#define DQM_TRACKERREMAPPER_PHASE1PIXELSUMMARYMAP_H

#include "TArrow.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TH1.h"
#include "TH2.h"
#include "TH2Poly.h"
#include "TLatex.h"
#include "TStyle.h"

#include <fmt/printf.h>
#include <fstream>
#include <memory>
#include <boost/tokenizer.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

#ifndef PH1PSUMMARYMAP_STANDALONE
#define LOGDEBUG(x) LogDebug(x)
#define LOGINFO(x) edm::LogInfo(x)
#define LOGPRINT(x) edm::LogPrint(x)
#else
#define LOGDEBUG(x) std::cout << x << " Debug : "
#define LOGINFO(x) std::cout << x << " Info : "
#define LOGPRINT(x) std::cout << x << " : "
#endif

using indexedCorners = std::map<unsigned int, std::pair<std::vector<float>, std::vector<float>>>;

namespace Ph1PMapSummaryHelper {
  //============================================================================
  // utility to tokenize std::string
  //============================================================================
  inline std::vector<std::string> tokenize(std::string line, char delimiter) {
    // Vector of string to save tokens
    std::vector<std::string> tokens;
    std::stringstream check1(line);
    std::string intermediate;

    // Tokenizing w.r.t. delimiter
    while (getline(check1, intermediate, delimiter)) {
      tokens.push_back(intermediate);
    }
    return tokens;
  }
}  // namespace Ph1PMapSummaryHelper

/*--------------------------------------------------------------------
/ Ancillary class to build pixel phase-1 tracker maps
/--------------------------------------------------------------------*/
class Phase1PixelSummaryMap {
public:
  Phase1PixelSummaryMap(const char* option, std::string title, std::string zAxisTitle)
      : m_option{option},
        m_title{title},
        m_zAxisTitle{zAxisTitle},
        m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
            edm::FileInPath("Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml").fullPath())} {
    // store the file in path for the corners (BPIX)
    for (unsigned int i = 1; i <= 4; i++) {
      m_cornersBPIX.push_back(edm::FileInPath(Form("DQM/SiStripMonitorClient/data/Geometry/vertices_barrel_%i", i)));
    }

    // store the file in path for the corners (BPIX)
    for (int j : {-3, -2, -1, 1, 2, 3}) {
      m_cornersFPIX.push_back(edm::FileInPath(Form("DQM/SiStripMonitorClient/data/Geometry/vertices_forward_%i", j)));
    }
  }

  ~Phase1PixelSummaryMap() = default;

  void resetOption(const char* option);
  void createTrackerBaseMap();
  void printTrackerMap(TCanvas& canvas);
  bool fillTrackerMap(unsigned int id, double value);

protected:
  void addNamedBins(edm::FileInPath geoFile, int tX, int tY, int sX, int sY, bool applyModuleRotation = false);

private:
  Option_t* m_option;
  const std::string m_title;
  const std::string m_zAxisTitle;

  TrackerTopology m_trackerTopo;
  std::shared_ptr<TH2Poly> m_BaseTrackerMap;
  std::map<uint32_t, std::shared_ptr<TGraph>> bins;

  std::vector<edm::FileInPath> m_cornersBPIX;
  std::vector<edm::FileInPath> m_cornersFPIX;

  static const unsigned int maxPxBarrel = 4;
  static const unsigned int maxPxForward = 3;
  const std::array<int, maxPxBarrel> barrelLadderShift = {{0, 14, 44, 90}};
  const std::array<int, maxPxForward> forwardDiskXShift = {{25, 75, 125}};

  const int forwardDiskYShift = 45;  //# to make +DISK on top in the 'strip-like' layout

  const int plotWidth = 3000;
  const int plotHeight = 2000;

  TArrow arrow, phiArrow, xArrow, yArrow;
};

#endif
