#ifndef ESIntegrityClient_H
#define ESIntegrityClient_H

#include "DQM/EcalPreshowerMonitorClient/interface/ESClient.h"

#include "TH1F.h"
#include "TH2F.h"

class ESIntegrityClient : public ESClient {
public:
  /// Constructor
  ESIntegrityClient(const edm::ParameterSet &ps);

  /// Destructor
  ~ESIntegrityClient() override;

  /// Analyze
  void endLumiAnalyze(DQMStore::IGetter &) override;

private:
  void book(DQMStore::IBooker &) override;

  int fed_[2][2][40][40];
  int kchip_[2][2][40][40];
  int fiber_[2][2][40][40];
  int fedStatus_[56];
  int fiberStatus_[56][36];
  int syncStatus_[56];
  int slinkCRCStatus_[56];

  MonitorElement *meFED_[2][2];
  MonitorElement *meKCHIP_[2][2];

  TH1F *hFED_;
  TH2F *hFiberOff_;
  TH2F *hFiberBadStatus_;
  TH2F *hKF1_;
  TH2F *hKF2_;
  TH1F *hKBC_;
  TH1F *hKEC_;
  TH1F *hL1ADiff_;
  TH1F *hBXDiff_;
  TH1F *hOrbitNumberDiff_;
  TH1F *hSLinkCRCErr_;
};

#endif
