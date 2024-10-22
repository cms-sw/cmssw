#ifndef DQM_L1TMONITORCLIENT_L1TDTTPG_H
#define DQM_L1TMONITORCLIENT_L1TDTTPG_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <TH1F.h>
#include <TH2F.h>
#include <TProfile2D.h>

class L1TDTTPGClient : public DQMEDHarvester {
public:
  /// Constructor
  L1TDTTPGClient(const edm::ParameterSet &ps);

  /// Destructor
  ~L1TDTTPGClient() override;

protected:
  void dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) override;

private:
  void initialize();
  void makeRatioHisto(DQMStore::IGetter &igetter, MonitorElement *ratioME, std::string &nName, std::string &dName);
  void setMapPhLabel(MonitorElement *me);
  void setMapThLabel(MonitorElement *me);
  TH1F *get1DHisto(std::string meName, DQMStore::IGetter &igetter);
  TH2F *get2DHisto(std::string meName, DQMStore::IGetter &igetter);
  TProfile2D *get2DProfile(std::string meName, DQMStore::IGetter &igetter);
  TProfile *get1DProfile(std::string meName, DQMStore::IGetter &igetter);

  edm::ParameterSet parameters_;
  std::string monitorName_;
  std::string input_dir_;
  std::string output_dir_;
  int counterLS_;    ///counter
  int counterEvt_;   ///counter
  int prescaleLS_;   ///units of lumi sections
  int prescaleEvt_;  ///prescale on number of events

  // -------- member data --------
  //  MonitorElement * clientHisto;
  MonitorElement *dttpgphmapcorrf;
  MonitorElement *dttpgphmap2ndf;
  MonitorElement *dttpgphmapbxf[3];
  MonitorElement *dttpgthmaphf;
  MonitorElement *dttpgthmapbxf[3];
};

#endif
