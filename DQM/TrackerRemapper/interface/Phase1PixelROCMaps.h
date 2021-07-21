#ifndef DQM_TRACKERREMAPPER_PHASE1PIXELROCMAPS_H
#define DQM_TRACKERREMAPPER_PHASE1PIXELROCMAPS_H

#include "TH1.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TLatex.h"

#include <bitset>
#include <fmt/printf.h>
#include <fstream>
#include <boost/tokenizer.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"

#ifndef PHASE1PIXELMAP_STANDALONE
#define LOGDEBUG(x) LogDebug(x)
#else
#define LOGDEBUG(x) std::cout << x << ": "
#endif

/*--------------------------------------------------------------------
/ helper functions to dress plots
/--------------------------------------------------------------------*/
namespace PixelROCMapHelper {
  void draw_line(double x1, double x2, double y1, double y2, int width, int style, int color);
  void dress_plot(
      TPad*& canv, TH2* h, int lay, int ring, int phase, bool half_shift, bool mark_zero, bool standard_palette);
}  // namespace PixelROCMapHelper

/*--------------------------------------------------------------------
/ Ancillary struct to contain detector topology coordinates
/--------------------------------------------------------------------*/
struct DetCoordinates {
  int m_layer;
  int m_s_ladder;
  int m_s_module;
  int m_ring;
  int m_s_blade;
  int m_s_disk;
  int m_panel;
  bool m_isFlipped;

public:
  DetCoordinates() {
    m_layer = -1;
    m_s_ladder = -1;
    m_s_module = -1;
    m_ring = -1;
    m_s_blade = -1;
    m_s_disk = -1;
    m_panel = -1;
    m_isFlipped = false;
  }

  void printCoordinates() {
    if (this->isBarrel()) {
      edm::LogPrint("DetCoordinates") << "layer: " << m_layer << " ladder: " << m_s_ladder << " module: " << m_s_module
                                      << std::endl;
    } else {
      edm::LogPrint("DetCoordinates") << "ring: " << m_ring << " blade: " << m_s_blade << " panel: " << m_panel
                                      << " disk: " << m_s_disk << std::endl;
    }
  }

  bool isBarrel() { return (m_layer > 0); }
  bool isEndcap() { return (m_ring > 0); }
  bool isFlipped() { return m_isFlipped; }
};

/*--------------------------------------------------------------------
/ Ancillary class to build pixel phase-1 tracker maps
/--------------------------------------------------------------------*/
class Phase1PixelROCMaps {
public:
  Phase1PixelROCMaps(const char* option, const std::string& zAxisTitle = "")
      : m_option{option},
        m_zAxisTitle{zAxisTitle},
        m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
            edm::FileInPath("Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml").fullPath())} {
    // ---------------------    BOOK HISTOGRAMS
    // barrel
    for (unsigned int lay = 1; lay <= n_layers; lay++) {
      int nlad = nlad_list[lay - 1];
      std::string name = "occ_Layer_" + std::to_string(lay);
      std::string title = "; Module # ; Ladder #";

      // if a z-axis title is specified, add the z-axis title
      if (!m_zAxisTitle.empty()) {
        title += fmt::sprintf(" ;%s", m_zAxisTitle.c_str());
      }

      h_bpix_maps[lay - 1] = std::make_shared<TH2D>(
          name.c_str(), title.c_str(), 72, -n_layers - 0.5, n_layers + 0.5, (nlad * 4 + 2), -nlad - 0.5, nlad + 0.5);
    }

    // endcaps
    for (unsigned int ring = 1; ring <= n_rings; ring++) {
      int n = nybins_list[ring - 1];
      float y = nxbins_list[ring - 1] + 0.5;
      std::string name = "occ_ring_" + std::to_string(ring);
      std::string title = "; Disk # ; Blade/Panel #";

      // if a z-axis title is specified, add the z-axis title
      if (!m_zAxisTitle.empty()) {
        title += fmt::sprintf(" ;%s", m_zAxisTitle.c_str());
      }

      h_fpix_maps[ring - 1] =
          std::make_shared<TH2D>(name.c_str(), title.c_str(), 56, -n_rings - 1.5, n_rings + 1.5, n, -y, y);
    }
  }

  ~Phase1PixelROCMaps() {}

  // Forward declarations
  DetCoordinates findDetCoordinates(const uint32_t& t_detid);
  void fillWholeModule(const uint32_t& detid, double value);
  void fillSelectedRocs(const uint32_t& detid, const std::bitset<16>& theROCs, double value);
  void drawBarrelMaps(TCanvas& canvas, const std::string& text = "");
  void drawForwardMaps(TCanvas& canvas, const std::string& text = "");
  void drawMaps(TCanvas& canvas, const std::string& text = "");

  inline std::array<std::shared_ptr<TH2D>, 4> getLayerMaps() { return h_bpix_maps; }
  inline std::array<std::shared_ptr<TH2D>, 2> getRingMaps() { return h_fpix_maps; }

private:
  Option_t* m_option;
  std::string m_zAxisTitle;
  TrackerTopology m_trackerTopo;

  // tough luck, we can only do phase-1...
  static constexpr int numColumns = 416;
  static constexpr int numRows = 160;
  static constexpr int n_rings = 2;
  static constexpr int n_layers = 4;

  const int nlad_list[n_layers] = {6, 14, 22, 32};
  const int nybins_list[n_rings] = {92, 140};
  const int nxbins_list[n_rings] = {11, 17};

  // maps
  std::array<std::shared_ptr<TH2D>, n_layers> h_bpix_maps;
  std::array<std::shared_ptr<TH2D>, n_rings> h_fpix_maps;

  // options
  static constexpr const char* kVerbose = "verbose";

  // Forward declarations of private methods
  std::vector<std::pair<int, int> > maskedBarrelRocsToBins(DetCoordinates coord);
  std::vector<std::tuple<int, int, int> > maskedBarrelRocsToBins(DetCoordinates coord, std::bitset<16> myRocs);
  std::vector<std::pair<int, int> > maskedForwardRocsToBins(DetCoordinates coord);
  std::vector<std::tuple<int, int, int> > maskedForwardRocsToBins(DetCoordinates coord, std::bitset<16> myRocs);

protected:
  //============================================================================
  // Taken from pixel naming classes
  // BmO (-z-x) = 1, BmI (-z+x) = 2 , BpO (+z-x) = 3 , BpI (+z+x) = 4
  inline int quadrant(const DetId& detid, bool phase_) {
    if (detid.subdetId() == PixelSubdetector::PixelBarrel) {
      return PixelBarrelName(detid, &m_trackerTopo, phase_).shell();
    } else {
      return PixelEndcapName(detid, &m_trackerTopo, phase_).halfCylinder();
    }
  }

  //============================================================================
  // Online ladder convention taken from pixel naming class for barrel
  // Apply sign convention (- sign for BmO and BpO)
  inline int signed_ladder(const DetId& detid, bool phase_) {
    if (detid.subdetId() != PixelSubdetector::PixelBarrel)
      return -9999;
    int signed_ladder = PixelBarrelName(detid, &m_trackerTopo, phase_).ladderName();
    if (quadrant(detid, phase_) % 2)
      signed_ladder *= -1;
    return signed_ladder;
  }

  //============================================================================
  // Online mdoule convention taken from pixel naming class for barrel
  // Apply sign convention (- sign for BmO and BmI)
  inline int signed_module(const DetId& detid, bool phase_) {
    if (detid.subdetId() != PixelSubdetector::PixelBarrel)
      return -9999;
    int signed_module = PixelBarrelName(detid, &m_trackerTopo, phase_).moduleName();
    if (quadrant(detid, phase_) < 3)
      signed_module *= -1;
    return signed_module;
  }

  //============================================================================
  // Phase 0: Ring was not an existing convention
  //   but the 7 plaquettes were split by HV group
  //   --> Derive Ring 1/2 for them
  //   Panel 1 plq 1-2, Panel 2, plq 1   = Ring 1
  //   Panel 1 plq 3-4, Panel 2, plq 2-3 = Ring 2
  // Phase 1: Using pixel naming class for endcap
  inline int ring(const DetId& detid, bool phase_) {
    if (detid.subdetId() != PixelSubdetector::PixelEndcap)
      return -9999;
    int ring = -9999;
    if (phase_ == 0) {
      ring = 1 + (m_trackerTopo.pxfPanel(detid) + m_trackerTopo.pxfModule(detid) > 3);
    } else if (phase_ == 1) {
      ring = PixelEndcapName(detid, &m_trackerTopo, phase_).ringName();
    }
    return ring;
  }

  //============================================================================
  // Online blade convention taken from pixel naming class for endcap
  // Apply sign convention (- sign for BmO and BpO)
  inline int signed_blade(const DetId& detid, bool phase_) {
    if (detid.subdetId() != PixelSubdetector::PixelEndcap)
      return -9999;
    int signed_blade = PixelEndcapName(detid, &m_trackerTopo, phase_).bladeName();
    if (quadrant(detid, phase_) % 2)
      signed_blade *= -1;
    return signed_blade;
  }

  //============================================================================
  inline int signed_blade_panel(const DetId& detid, bool phase_) {
    if (detid.subdetId() != PixelSubdetector::PixelEndcap)
      return -9999;
    int signed_blade_panel = signed_blade(detid, phase_) + (m_trackerTopo.pxfPanel(detid) - 1);
    return signed_blade_panel;
  }

  //============================================================================
  // Online disk convention
  // Apply sign convention (- sign for BmO and BmI)
  inline int signed_disk(const DetId& detid, bool phase_) {
    if (detid.subdetId() != PixelSubdetector::PixelEndcap)
      return -9999;
    int signed_disk = m_trackerTopo.pxfDisk(DetId(detid));
    if (quadrant(detid, phase_) < 3)
      signed_disk *= -1;
    return signed_disk;
  }

  //============================================================================
  // Determines if the BPix ldder is inner or outer
  inline bool isBPixOuterLadder(const DetId& detid, bool isPhase0) {
    bool isOuter = false;
    int layer = m_trackerTopo.pxbLayer(detid.rawId());
    bool odd_ladder = m_trackerTopo.pxbLadder(detid.rawId()) % 2;
    if (isPhase0) {
      if (layer == 2)
        isOuter = !odd_ladder;
      else
        isOuter = odd_ladder;
    } else {
      if (layer == 4)
        isOuter = odd_ladder;
      else
        isOuter = !odd_ladder;
    }
    return isOuter;
  }
};

#endif
