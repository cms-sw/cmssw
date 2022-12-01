#ifndef DQM_L1TMONITORCLIENT_L1TGMTCLIENT_H
#define DQM_L1TMONITORCLIENT_L1TGMTCLIENT_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <string>

class L1TGMTClient : public DQMEDHarvester {
public:
  /// Constructor
  L1TGMTClient(const edm::ParameterSet &);

  /// Destructor
  ~L1TGMTClient() override;

protected:
  void dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) override;
  void dqmEndLuminosityBlock(DQMStore::IBooker &ibooker,
                             DQMStore::IGetter &igetter,
                             const edm::LuminosityBlock &lumiSeg,
                             const edm::EventSetup &evSetup) override;

private:
  void initialize();
  void processHistograms(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter);
  void makeRatio1D(DQMStore::IBooker &ibooker,
                   DQMStore::IGetter &igetter,
                   MonitorElement *mer,
                   std::string h1Name,
                   std::string h2Name);
  void makeEfficiency1D(DQMStore::IBooker &ibooker,
                        DQMStore::IGetter &igetter,
                        MonitorElement *meeff,
                        std::string heName,
                        std::string hiName);
  void makeEfficiency2D(DQMStore::IBooker &ibooker,
                        DQMStore::IGetter &igetter,
                        MonitorElement *meeff,
                        std::string heName,
                        std::string hiName);
  TH1F *get1DHisto(std::string meName, DQMStore::IGetter &igetter);
  TH2F *get2DHisto(std::string meName, DQMStore::IGetter &igetter);

  MonitorElement *bookClone1D(DQMStore::IBooker &ibooker,
                              DQMStore::IGetter &igetter,
                              const std::string &name,
                              const std::string &title,
                              const std::string &hrefName);
  MonitorElement *bookClone1DVB(DQMStore::IBooker &ibooker,
                                DQMStore::IGetter &igetter,
                                const std::string &name,
                                const std::string &title,
                                const std::string &hrefName);
  MonitorElement *bookClone2D(DQMStore::IBooker &ibooker,
                              DQMStore::IGetter &igetter,
                              const std::string &name,
                              const std::string &title,
                              const std::string &hrefName);

  edm::ParameterSet parameters_;
  std::string monitorName_;
  std::string input_dir_;
  std::string output_dir_;

  bool m_runInEventLoop;
  bool m_runInEndLumi;
  bool m_runInEndRun;
  bool m_runInEndJob;

  // -------- member data --------
  MonitorElement *eff_eta_dtcsc;
  MonitorElement *eff_eta_rpc;
  MonitorElement *eff_phi_dtcsc;
  MonitorElement *eff_phi_rpc;
  MonitorElement *eff_etaphi_dtcsc;
  MonitorElement *eff_etaphi_rpc;
};

#endif
