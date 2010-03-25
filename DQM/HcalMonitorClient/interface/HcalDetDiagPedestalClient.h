#ifndef HcalDetDiagPedestalClient_GUARD_H
#define HcalDetDiagPedestalClient_GUARD_H

#include "DQM/HcalMonitorClient/interface/HcalBaseDQClient.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class HcalDetDiagPedestalClient : public HcalBaseDQClient {

 public:

  /// Constructors
  HcalDetDiagPedestalClient(){name_="";};
  HcalDetDiagPedestalClient(std::string myname);//{ name_=myname;};
  HcalDetDiagPedestalClient(std::string myname, const edm::ParameterSet& ps);

  void analyze(void);
  void calculateProblems(void); // calculates problem histogram contents
  void updateChannelStatus(std::map<HcalDetId, unsigned int>& myqual);
  void beginJob(void);
  void endJob(void);
  void beginRun(void);
  void endRun(void); 
  void setup(void);  
  void cleanup(void);
  bool hasErrors_Temp(void);  
  bool hasWarnings_Temp(void);
  bool hasOther_Temp(void);
  bool test_enabled(void);
  

  void htmlOutput(std::string);
  bool validHtmlOutput();

  /// Destructor
  ~HcalDetDiagPedestalClient();

 private:
  int nevts_;
  int status;
};

#endif
