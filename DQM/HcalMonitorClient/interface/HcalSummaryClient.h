#ifndef HcalSummaryClient_GUARD_H
#define HcalSummaryClient_GUARD_H

#include "DQM/HcalMonitorClient/interface/HcalBaseDQClient.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class EtaPhiHists;
class MonitorElement;
class HcalBaseDQClient;

class HcalSummaryClient : public HcalBaseDQClient {

 public:

  /// Constructors
  HcalSummaryClient(){name_="";};
  HcalSummaryClient(std::string myname);//{ name_=myname;};
  HcalSummaryClient(std::string myname, const edm::ParameterSet& ps);

  void analyze(DQMStore::IBooker &, DQMStore::IGetter &, int LS=-1);
  void updateChannelStatus(std::map<HcalDetId, unsigned int>& myqual);
  void endJob(void);
  void beginRun(void);
  //void endRun(void); 
  void setup(DQMStore::IBooker &, DQMStore::IGetter &);  
  void cleanup(void);
  
  void fillReportSummary(DQMStore::IBooker &, DQMStore::IGetter &, int LS);
  void fillReportSummaryLSbyLS(DQMStore::IBooker &, DQMStore::IGetter &, int LS);

  bool hasErrors_Temp(void);  
  bool hasWarnings_Temp(void);
  bool hasOther_Temp(void);
  bool test_enabled(void);

  void getFriends(const std::vector<HcalBaseDQClient*>& clients){clients_=clients;};

  /// Destructor
  ~HcalSummaryClient();

 private:
  int nevts_;
  EtaPhiHists* SummaryMapByDepth; // used to store & calculate problems
  MonitorElement* StatusVsLS_;
  MonitorElement* EnoughEvents_;
  MonitorElement* MinEvents_;
  MonitorElement* MinErrorRate_;
  MonitorElement* reportMapShift_;
  MonitorElement* reportMap_;
  MonitorElement* certificationMap_;

  // minEvents and TaskLists_ were added as private members during the
  // MT migration.
  // Originally this functionality was performed by the HcalLSbyLSMonitor
  // under DQM/HcalMonitorTasks
  int minEvents_;
  std::vector<std::string> TaskList_;

  double status_global_, status_HB_, status_HE_, status_HO_, status_HF_;
  double status_HO0_, status_HO12_, status_HFlumi_;
  int NLumiBlocks_;
  bool UseBadChannelStatusInSummary_;  // if turned on, the channel status DB output is checked and any channels reporting 'NaN' are counted as bad in the summary.

  std::vector<HcalBaseDQClient*> clients_;
  std::map<std::string, int> subdetCells_;
  int HBpresent_, HEpresent_, HOpresent_, HFpresent_;

  bool doSetup_; // defaults to true in constructor
};

#endif
