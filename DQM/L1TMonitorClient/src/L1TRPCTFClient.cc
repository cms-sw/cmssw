#include "DQM/L1TMonitorClient/interface/L1TRPCTFClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "TRandom.h"

#include <TF1.h>
#include <cstdio>
#include <sstream>
#include <cmath>
#include <TProfile.h>
#include <TProfile2D.h>

using namespace edm;
using namespace std;

L1TRPCTFClient::L1TRPCTFClient(const edm::ParameterSet &ps) {
  parameters_ = ps;
  initialize();
}

L1TRPCTFClient::~L1TRPCTFClient() { LogInfo("TriggerDQM") << "[TriggerDQM]: ending... "; }

//--------------------------------------------------------
void L1TRPCTFClient::initialize() {
  counterLS_ = 0;
  counterEvt_ = 0;

  // base folder for the contents of this job
  monitorName_ = parameters_.getUntrackedParameter<string>("monitorName", "");
  //  cout << "Monitor name = " << monitorName_ << endl;
  prescaleLS_ = parameters_.getUntrackedParameter<int>("prescaleLS", -1);
  //  cout << "DQM lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< endl;
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  //  cout << "DQM event prescale = " << prescaleEvt_ << " events(s)"<< endl;
  output_dir_ = parameters_.getUntrackedParameter<string>("output_dir", "");
  //  cout << "DQM output dir = " << output_dir_ << endl;
  input_dir_ = parameters_.getUntrackedParameter<string>("input_dir", "");
  //  cout << "DQM input dir = " << input_dir_ << endl;

  verbose_ = parameters_.getUntrackedParameter<bool>("verbose", false);

  m_runInEventLoop = parameters_.getUntrackedParameter<bool>("runInEventLoop", false);
  m_runInEndLumi = parameters_.getUntrackedParameter<bool>("runInEndLumi", false);
  m_runInEndRun = parameters_.getUntrackedParameter<bool>("runInEndRun", false);
  m_runInEndJob = parameters_.getUntrackedParameter<bool>("runInEndJob", false);

  LogInfo("TriggerDQM");
}

//--------------------------------------------------------
void L1TRPCTFClient::book(DQMStore::IBooker &ibooker) {
  LogInfo("TriggerDQM") << "[TriggerDQM]: Begin Job";

  ibooker.setCurrentFolder(output_dir_);

  m_deadChannels = ibooker.book2D("RPCTF_deadchannels", "RPCTF deadchannels", 33, -16.5, 16.5, 144, -0.5, 143.5);
  m_noisyChannels = ibooker.book2D("RPCTF_noisychannels", "RPCTF noisy channels", 33, -16.5, 16.5, 144, -0.5, 143.5);
}

//--------------------------------------------------------

void L1TRPCTFClient::dqmEndLuminosityBlock(DQMStore::IBooker &ibooker,
                                           DQMStore::IGetter &igetter,
                                           const edm::LuminosityBlock &lumiSeg,
                                           const edm::EventSetup &c) {
  if (verbose_)
    std::cout << "L1TRPCTFClient::endLuminosityBlock" << std::endl;

  if (m_runInEndLumi) {
    book(ibooker);
    processHistograms(igetter);
  }
}

void L1TRPCTFClient::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  book(ibooker);
  processHistograms(igetter);
}

//--------------------------------------------------------
void L1TRPCTFClient::processHistograms(DQMStore::IGetter &igetter) {
  igetter.setCurrentFolder(input_dir_);

  {
    MonitorElement *me = igetter.get(input_dir_ + "/RPCTF_muons_eta_phi_bx0");

    if (me) {
      const QReport *qreport;

      qreport = me->getQReport("DeadChannels_RPCTF_2D");
      if (qreport) {
        vector<dqm::me_util::Channel> badChannels = qreport->getBadChannels();
        for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); channel != badChannels.end();
             ++channel) {
          m_deadChannels->setBinContent((*channel).getBinX(), (*channel).getBinY(), 100);
        }  // for(badchannels)
      }    //if (qreport)

      qreport = me->getQReport("HotChannels_RPCTF_2D");
      if (qreport) {
        vector<dqm::me_util::Channel> badChannels = qreport->getBadChannels();
        for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); channel != badChannels.end();
             ++channel) {
          // (*channel).getBinY() == 0 for NoisyChannels QTEST
          m_noisyChannels->setBinContent((*channel).getBinX(), 100);
        }  // for(badchannels)
      }    //if (qreport)
           //      else std::cout << "dupa" << std::endl;
    }      // if (me)
  }

  if (verbose_) {
    std::vector<string> meVec = igetter.getMEs();
    for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end(); it++) {
      std::string full_path = input_dir_ + "/" + (*it);
      MonitorElement *me = igetter.get(full_path);

      // for this MEs, get list of associated QTs
      std::vector<QReport *> Qtest_map = me->getQReports();

      if (!Qtest_map.empty()) {
        std::cout << "Test: " << full_path << std::endl;
        for (std::vector<QReport *>::const_iterator it = Qtest_map.begin(); it != Qtest_map.end(); ++it) {
          std::cout << " Name " << (*it)->getQRName() << " Status " << (*it)->getStatus() << std::endl;

          std::vector<dqm::me_util::Channel> badChannels = (*it)->getBadChannels();

          vector<dqm::me_util::Channel>::iterator badchsit = badChannels.begin();
          while (badchsit != badChannels.end()) {
            int ix = (*badchsit).getBinX();
            int iy = (*badchsit).getBinY();
            std::cout << "(" << ix << "," << iy << ") ";
            ++badchsit;
          }
          std::cout << std::endl;
        }
      }

    }  //
  }
}
