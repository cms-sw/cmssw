#ifndef DQM_L1TMONITORCLIENT_L1TOCCUPANCYCLIENT_H
#define DQM_L1TMONITORCLIENT_L1TOCCUPANCYCLIENT_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DQM/L1TMonitorClient/interface/L1TOccupancyClientHistogramService.h"
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
#include <TNamed.h>
#include <TRandom3.h>
#include <TDirectory.h>

class L1TOccupancyClient : public DQMEDHarvester {
public:
  /// Constructor
  L1TOccupancyClient(const edm::ParameterSet& ps);

  /// Destructor
  ~L1TOccupancyClient() override;

protected:
  void dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) override;
  void book(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter);
  void dqmEndLuminosityBlock(DQMStore::IBooker& ibooker,
                             DQMStore::IGetter& igetter,
                             const edm::LuminosityBlock& lumiSeg,
                             const edm::EventSetup& c) override;  // DQM Client Diagnostic

  //DQM test routines
  double xySymmetry(const edm::ParameterSet& ps,
                    std::string test_name,
                    std::vector<std::pair<int, double> >& deadChannels,
                    std::vector<std::pair<int, double> >& statDev,
                    bool& enoughStats);  // Performs the checking of enough statistics and invokes compareWithStrip()

private:
  edm::ParameterSet parameters_;                  //parameter set from python
  L1TOccupancyClientHistogramService* hservice_;  //histogram service
  TFile* file_;                                   //output file for test results

  // bool
  bool verbose_;  //verbose mode

  // vector
  std::vector<edm::ParameterSet> tests_;        // all tests defined in python file
  std::vector<edm::ParameterSet*> mValidTests;  // Valid tests
  // map
  std::map<std::string, MonitorElement*> meResults;
  std::map<std::string, MonitorElement*> meDifferential;
  std::map<std::string, MonitorElement*> meCertification;

private:
  // performs the actual test
  int compareWithStrip(TH2F* histo,
                       std::string test,
                       int binStrip,
                       int nBins,
                       int axis,
                       double avg,
                       const edm::ParameterSet& ps,
                       std::vector<std::pair<int, double> >& deadChannels);

  // Gets the bin-number of a bin with content and on axis
  void getBinCoordinateOnAxisWithValue(TH2F* h2f, double content, int& coord, int axis);

  // Puts out the bad and masked channels of a specific test to h2f
  void printDeadChannels(const std::vector<std::pair<int, double> >& deadChannels,
                         TH2F* h2f,
                         const std::vector<std::pair<int, double> >& statDev,
                         std::string test_name);

  // Gets the average (avrgMode=1 arithmetic, avrgMode=2 median) for a specific binStrip in histo h2f for a specific test
  double getAvrg(TH2F* h2f, std::string test, int axis, int nBins, int binStrip, int avrgMode);
};

#endif
