#ifndef EcalPreshowerMonitorClient_H
#define EcalPreshowerMonitorClient_H


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DQM/EcalPreshowerMonitorClient/interface/ESClient.h"

#include "OnlineDB/ESCondDB/interface/ESMonRunIOV.h"
#include "OnlineDB/ESCondDB/interface/ESMonRunDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunDat.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "DQM/EcalPreshowerMonitorClient/interface/ESClient.h"
#include "OnlineDB/ESCondDB/interface/ESCondDBInterface.h"

class DQMOldReceiver;
class DQMStore;

class RunIOV;
class ESMonRunIOV;

class EcalPreshowerMonitorClient : public edm::EDAnalyzer{

 public:

  EcalPreshowerMonitorClient(const edm::ParameterSet& ps);
  virtual ~EcalPreshowerMonitorClient();
  
 private:
  
  void analyze(const edm::Event &, const edm::EventSetup &);
  void analyze();
  
  void beginJob();
  void endJob() ;
  void beginRun() ;
  void beginRun(const edm::Run & r, const edm::EventSetup & c);
  void endRun() ;
  void endRun(const edm::Run & r, const edm::EventSetup & c);
  // BeginRunDB
  int beginRunDb(void);
  void endRunDb(void);
  // WriteDB
  void writeDb();
  void htmlOutput(int);
  void cleanup();

  void beginLuminosityBlock(const edm::LuminosityBlock & l, const edm::EventSetup & c);
  void endLuminosityBlock(const edm::LuminosityBlock & l, const edm::EventSetup & c);

  // ----------member data ---------------------------
 private:
  
  std::string outputFile_;
  std::string inputFile_;
  std::string prefixME_;
  
  bool enableMonitorDaemon_;
  
  std::string clientName_;
  std::string hostName_;
  int hostPort_;
  
  DQMOldReceiver* mui_;
  DQMStore* dqmStore_;
  
  int run_;
  int evt_;
  std::string status_;
  bool begin_run_;
  bool end_run_;
  bool debug_;
  bool verbose_;
  bool cloneME_;
  bool enableCleanup_;

  bool PlusSide;
  bool MinusSide;

  int runType_;
  int evtType_;
  
  int last_run_;
  int prescaleFactor_;		
  int ievt_;
  int jevt_;
  int Side_;
 
  int side;
  std::string location;
  
  std::string dbName_;
  std::string dbHostName_;
  int         dbHostPort_;
  std::string dbUserName_;
  std::string dbPassword_;
  std::string dbTagName_;
  
  std::vector<std::string> enabledClients_;
  std::vector<ESClient*> clients_;
  std::vector<std::string> runTypes_;
  std::vector<std::string> clientsNames_; 
  std::map<std::string,int> clientsStatus_;
  
  std::string location_;
  int nLines_, runNum_;
  int runtype_, seqtype_, dac_, gain_, precision_;
  int firstDAC_, nDAC_, isPed_, vDAC_[5], layer_;
  
  int senZ_[4288], senP_[4288], senX_[4288], senY_[4288];
  int qt[40][40], qtCriteria;
  
  RunIOV runiov_;
  ESMonRunIOV moniov_;
  
  int subrun_;
  
  time_t current_time_;
  
  time_t last_time_update_;
  time_t last_time_db_;
  
  time_t updateTime_;
  time_t dbUpdateTime_;
  
  TH1F* h_;
};

#endif
