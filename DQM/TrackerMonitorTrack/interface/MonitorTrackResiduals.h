#ifndef MonitorTrackResiduals_H
#define MonitorTrackResiduals_H

// -*- C++ -*-
//
// Package:    TrackerMonitorTrack
// Class:      MonitorTrackResiduals
//
/**\class MonitorTrackResiduals MonitorTrackResiduals.h
DQM/TrackerMonitorTrack/interface/MonitorTrackResiduals.cc Monitoring source for
track residuals on each detector module
*/
// Original Author:  Israel Goitom
//         Created:  Fri May 26 14:12:01 CEST 2006
// Author:  Marcel Schneider
//         Extended to Pixel Residuals.
#include "Alignment/OfflineValidation/interface/TrackerValidationVariables.h"
#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"
#include "DQM/SiStripCommon/interface/TkHistoMap.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include <fstream>
#include <memory>

class GenericTriggerEventFlag;
namespace edm {
  class Event;
}

enum TrackerType { TRACKERTYPE_STRIP, TRACKERTYPE_PIXEL };

template <TrackerType pixel_or_strip>
class MonitorTrackResidualsBase : public DQMEDAnalyzer {
public:
  // constructors and EDAnalyzer Methods
  explicit MonitorTrackResidualsBase(const edm::ParameterSet &);
  ~MonitorTrackResidualsBase() override;
  void dqmBeginRun(const edm::Run &, const edm::EventSetup &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:
  // Own methods
  void createMEs(DQMStore::IBooker &, const edm::EventSetup &);
  std::pair<std::string, int32_t> findSubdetAndLayer(uint32_t ModuleID, const TrackerTopology *tTopo);

  struct HistoPair {
    HistoPair() {
      base = nullptr;
      normed = nullptr;
    };
    MonitorElement *base;
    MonitorElement *normed;
  };
  struct HistoXY {
    HistoPair x;
    HistoPair y;
  };
  typedef std::map<std::pair<std::string, int32_t>, HistoXY> HistoSet;

  HistoSet m_SubdetLayerResiduals;
  HistoSet m_ModuleResiduals;
  std::unique_ptr<TkHistoMap> tkhisto_ResidualsMean;

  edm::ParameterSet conf_;
  edm::ParameterSet Parameters;
  edm::EDGetTokenT<reco::VertexCollection> offlinePrimaryVerticesToken_;

  edm::ESGetToken<TkDetMap, TrackerTopologyRcd> tkDetMapToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyRunToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyEventToken_;

  unsigned long long m_cacheID_;
  bool ModOn;

  GenericTriggerEventFlag *genTriggerEventFlag_;
  TrackerValidationVariables avalidator_;
};

// Naming is for legacy reasons.
typedef MonitorTrackResidualsBase<TRACKERTYPE_STRIP> MonitorTrackResiduals;
typedef MonitorTrackResidualsBase<TRACKERTYPE_PIXEL> SiPixelMonitorTrackResiduals;

#endif
