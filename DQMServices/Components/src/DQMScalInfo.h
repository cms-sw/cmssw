#ifndef DQMSCALINFO_H
#define DQMSCALINFO_H

/*
 * \file DQMDcsInfo.h
 *
 * \author A.Meyer - DESY
 *
*/

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/Run.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/ParameterSet/interface/Registry.h>
#include <FWCore/ServiceRegistry/interface/Service.h>

#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/MonitorElement.h>
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>
//DataFormats
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"

class DQMScalInfo: public DQMEDAnalyzer{

public:

  /// Constructor
  DQMScalInfo(const edm::ParameterSet& ps);

  /// Destructor
  virtual ~DQMScalInfo();

protected:

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:

  void makeL1Scalars(const edm::Event& e);

  edm::ParameterSet parameters_;
  std::string scalfolder_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> gtCollection_;
  edm::EDGetTokenT<DcsStatusCollection> dcsStatusCollection_;
  edm::EDGetTokenT<Level1TriggerScalersCollection> l1tscollectionToken_;
   // histograms
  MonitorElement * hlresync_;
  MonitorElement * hlOC0_;
  MonitorElement * hlTE_;
  MonitorElement * hlstart_;
  MonitorElement * hlEC0_;
  MonitorElement * hlHR_;

};

#endif
