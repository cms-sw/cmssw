#ifndef TrackingMonitor_TrackFoldedOccupancyClient_h
#define TrackingMonitor_TrackFoldedOccupancyClient_h
// -*- C++ -*-
//
// Package:    TrackingMonitor
// Class  :    TrackFoldedOccupancyClient
//
//DQM class to plot occupancy in eta phi

#include <string>

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMStore.h"

class TrackFoldedOccupancyClient : public DQMEDHarvester {
public:
  /// Constructor
  TrackFoldedOccupancyClient(const edm::ParameterSet& ps);

  /// Destructor
  ~TrackFoldedOccupancyClient() override;

protected:
  /// BeginJob
  void beginJob(void) override;

  /// BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup) override;

  /// EndJob
  void dqmEndJob(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_) override;

private:
  /// book MEs
  void bookMEs(DQMStore::IBooker& ibooker_);

  edm::ParameterSet conf_;
  std::string algoName_;
  std::string quality_;
  std::string state_;
  std::string histTag_;
  std::string TopFolder_;

  MonitorElement* TkEtaPhi_RelativeDifference_byFoldingmap = nullptr;
  MonitorElement* TkEtaPhi_RelativeDifference_byFoldingmap_op = nullptr;
  MonitorElement* TkEtaPhi_Ratio_byFoldingmap = nullptr;
  MonitorElement* TkEtaPhi_Ratio_byFoldingmap_op = nullptr;
};
#endif
