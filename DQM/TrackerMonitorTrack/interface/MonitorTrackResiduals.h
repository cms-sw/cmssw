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
// $Id: MonitorTrackResiduals.h,v 1.18 2010/05/12 06:43:53 dutta Exp $
#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Run.h"


class MonitorElement;
class DQMStore;
class GenericTriggerEventFlag;
namespace edm { class Event; }

typedef std::map<int32_t, MonitorElement *> HistoClass;

class MonitorTrackResiduals : public edm::EDAnalyzer {
 public:
  // constructors and EDAnalyzer Methods
  explicit MonitorTrackResiduals(const edm::ParameterSet&);
  ~MonitorTrackResiduals();
  virtual void beginRun(edm::Run const& run, edm::EventSetup const& iSetup);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);
  virtual void beginJob(void);
  virtual void endJob(void);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  // Own methods 
  void createMEs(const edm::EventSetup&);
  void resetModuleMEs(int32_t modid);
  void resetLayerMEs(const std::pair<std::string, int32_t>&);
 private:
      
  DQMStore * dqmStore_;
  edm::ParameterSet conf_;
  edm::ParameterSet Parameters;
  std::map< std::pair<std::string,int32_t>, MonitorElement* > m_SubdetLayerResiduals;
  std::map< std::pair<std::string,int32_t>, MonitorElement* > m_SubdetLayerNormedResiduals;
  HistoClass HitResidual;
  HistoClass NormedHitResiduals;
  SiStripFolderOrganizer folder_organizer;
  unsigned long long m_cacheID_;
  bool reset_me_after_each_run;
  bool ModOn;
  GenericTriggerEventFlag* genTriggerEventFlag_;
};
#endif
