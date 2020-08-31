#include "DQM/L1TMonitorClient/interface/L1TCSCTPGClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "TRandom.h"
using namespace edm;
using namespace std;

L1TCSCTPGClient::L1TCSCTPGClient(const edm::ParameterSet &ps) { monitorDir_ = ps.getParameter<string>("monitorDir"); }

L1TCSCTPGClient::~L1TCSCTPGClient() {}

void L1TCSCTPGClient::dqmEndLuminosityBlock(DQMStore::IBooker &,
                                            DQMStore::IGetter &igetter,
                                            const edm::LuminosityBlock &lumiSeg,
                                            const edm::EventSetup &c) {}

//--------------------------------------------------------
void L1TCSCTPGClient::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  ibooker.setCurrentFolder(monitorDir_);
}

//--------------------------------------------------------
void L1TCSCTPGClient::processHistograms(DQMStore::IGetter &igetter) {}
