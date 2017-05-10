#ifndef DQMDCSINFO_H
#define DQMDCSINFO_H

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
#include <FWCore/ServiceRegistry/interface/Service.h>

#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/MonitorElement.h>
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>
//DataFormats
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

class DQMDcsInfo: public DQMEDAnalyzer{

public:

  /// Constructor
  DQMDcsInfo(const edm::ParameterSet& ps);

  /// Destructor
  virtual ~DQMDcsInfo();

protected:

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void endLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c);

private:

  void makeDcsInfo(const edm::Event& e);
  void makeGtInfo(const edm::Event& e);

  edm::ParameterSet parameters_;
  std::string subsystemname_;
  std::string dcsinfofolder_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> gtCollection_;
  edm::EDGetTokenT<DcsStatusCollection> dcsStatusCollection_;

  bool dcs[25];
   // histograms
  MonitorElement * DCSbyLS_ ;

};

#endif
