#ifndef DQM_TRACKERREMAPPER_SISTRIPTKMAPS_H
#define DQM_TRACKERREMAPPER_SISTRIPTKMAPS_H

// CMSSW includes
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

// ROOT includes
#include "TArrow.h"
#include "TPaletteAxis.h"
#include "TGaxis.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TGraph.h"
#include "TLatex.h"
#include "TH2Poly.h"
#include "TStyle.h"

// STL includes
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// boost includes
#include <boost/tokenizer.hpp>
#include <boost/range/adaptor/indexed.hpp>

#define MYOUT LogDebug("SiStripTkMaps")

/*--------------------------------------------------------------------
/ Ancillary class to build SiStrip Tracker maps
/--------------------------------------------------------------------*/
class SiStripTkMaps {
public:
  SiStripTkMaps(const char* option)
      : m_option{option},
        m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
            edm::FileInPath("Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml").fullPath())} {}

  ~SiStripTkMaps() = default;

  void bookMap(const std::string mapTitle, const std::string zAxisTitle);
  void fill(long rawid, double val);
  void drawMap(TCanvas& canvas, std::string option = "");

  //============================================================================
  inline const TH2Poly* getTheMap() { return m_trackerMap; }

  //============================================================================
  inline const std::string& getTheMapTitle() { return m_mapTitle; }

  //============================================================================
  inline const std::string& getTheZAxisTitle() { return m_zAxisTitle; }

  //============================================================================
  inline const std::vector<unsigned int>& getTheFilledIds() { return m_detIdVector; }

  //============================================================================
  inline const std::vector<double>& getTheFilledValues() { return m_values; }

  //============================================================================
  inline void setZAxisRange(double xmin, double xmax) { m_trackerMap->GetZaxis()->SetRangeUser(xmin, xmax); }

private:
  // private members
  Option_t* m_option;
  std::string m_mapTitle = "";
  std::string m_zAxisTitle = "";
  double m_axmin, m_axmax;
  std::map<long, std::shared_ptr<TGraph>> m_bins;
  std::vector<unsigned int> m_detIdVector;
  std::vector<double> m_values;
  TrackerTopology m_trackerTopo;
  TH2Poly* m_trackerMap{nullptr};

  // private methods
  //============================================================================
  void dressMap(TCanvas& canv);
  void drawArrows(
      const float x_X1, const float x_X2, const float x_Y1, const float y_Y2, const char* x_label, const char* y_label);
  void adjustCanvasMargins(TVirtualPad* pad, const float top, const float bottom, const float left, const float right);
  void readVertices(double& minx, double& maxx, double& miny, double& maxy);
};

#endif
