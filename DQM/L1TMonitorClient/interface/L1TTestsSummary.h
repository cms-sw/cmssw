#ifndef DQM_L1TMONITORCLIENT_L1TOCCUPANCYCLIENT_H
#define DQM_L1TMONITORCLIENT_L1TOCCUPANCYCLIENT_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <TH1F.h>
#include <TH1D.h>
#include <TH2F.h>
#include <TF1.h>
#include <TProfile2D.h>

class L1TTestsSummary : public DQMEDHarvester {
public:
  // Constructor
  L1TTestsSummary(const edm::ParameterSet &ps);

  // Destructor
  ~L1TTestsSummary() override;

protected:
  void dqmEndLuminosityBlock(DQMStore::IBooker &ibooker,
                             DQMStore::IGetter &igetter,
                             const edm::LuminosityBlock &lumiSeg,
                             const edm::EventSetup &c) override;  // DQM Client Diagnostic

  void dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) override;
  virtual void book(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter);

private:
  edm::ParameterSet mParameters;  //parameter set from python

  // bool
  bool mVerbose;              // verbose mode
  bool mMonitorL1TRate;       // If we are going to monitor the L1TRate Module
  bool mMonitorL1TSync;       // If we are going to monitor the L1TSync Module
  bool mMonitorL1TOccupancy;  // If we are going to monitor the L1TOccupancy Module

  // int
  int binYRate, binYSync, binYOccpancy;  // What bin in Y corresponds to which test in L1TSummary

  // string
  std::string mL1TRatePath;       // Path to histograms produced by L1TRate Module
  std::string mL1TSyncPath;       // Path to histograms produced by L1TSync Module
  std::string mL1TOccupancyPath;  // Path to histograms produced by L1TOccupancy Module

  // vector
  std::vector<int> mProcessedLS;  // Already processed Luminosity Blocks

  // MonitorElement
  MonitorElement *mL1TRateMonitor;
  MonitorElement *mL1TSyncMonitor;
  MonitorElement *mL1TOccupancyMonitor;
  MonitorElement *mL1TSummary;

  // Private Functions
private:
  void updateL1TRateMonitor(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter);
  void updateL1TSyncMonitor(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter);
  void updateL1TOccupancyMonitor(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter);
  void updateL1TSummary(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter);
};

#endif
