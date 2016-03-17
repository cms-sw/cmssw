#ifndef MonitorTrackResiduals_H
#define MonitorTrackResiduals_H

// -*- C++ -*-
//
// Package:    TrackerMonitorTrack
// Class:      MonitorTrackResiduals
// 
/**\class MonitorTrackResiduals MonitorTrackResiduals.h DQM/TrackerMonitorTrack/interface/MonitorTrackResiduals.cc
Monitoring source for track residuals on each detector module
*/
// Original Author:  Israel Goitom
//         Created:  Fri May 26 14:12:01 CEST 2006
// Author:  Marcel Schneider
//         Extended to Pixel Residuals. 
#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Run.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>
#include "Alignment/OfflineValidation/interface/TrackerValidationVariables.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class MonitorElement;
class DQMStore;
class GenericTriggerEventFlag;
namespace edm { class Event; }

enum TrackerType {
  TRACKERTYPE_STRIP, TRACKERTYPE_PIXEL
};

template<TrackerType pixel_or_strip>
class MonitorTrackResidualsBase : public DQMEDAnalyzer {
 public:
  // constructors and EDAnalyzer Methods
  explicit MonitorTrackResidualsBase(const edm::ParameterSet&);
  ~MonitorTrackResidualsBase();
  void dqmBeginRun(const edm::Run& , const edm::EventSetup& ) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
 private:

  // Own methods 
  void createMEs( DQMStore::IBooker & , const edm::EventSetup&);
  std::pair<std::string, int32_t> findSubdetAndLayer(uint32_t ModuleID, const TrackerTopology* tTopo);
  
  struct HistoPair {
    HistoPair() {base = nullptr; normed = nullptr;};
    MonitorElement* base;
    MonitorElement* normed;
  };
  struct HistoXY {
    HistoPair x;
    HistoPair y;
  };
  typedef std::map<std::pair<std::string, int32_t>, HistoXY> HistoSet;

  HistoSet m_SubdetLayerResiduals;
  HistoSet m_ModuleResiduals;
  
  edm::ParameterSet conf_;
  edm::ParameterSet Parameters;
  edm::EDGetTokenT<reco::VertexCollection> offlinePrimaryVerticesToken_;

  unsigned long long m_cacheID_;
  bool ModOn;

  GenericTriggerEventFlag* genTriggerEventFlag_;
  TrackerValidationVariables avalidator_;
};

// Naming is for legacy reasons.
typedef MonitorTrackResidualsBase<TRACKERTYPE_STRIP> MonitorTrackResiduals;
typedef MonitorTrackResidualsBase<TRACKERTYPE_PIXEL> SiPixelMonitorTrackResiduals;

#endif
