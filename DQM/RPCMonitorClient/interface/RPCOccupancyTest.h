#ifndef RPCOccupancyTest_H
#define RPCOccupancyTest_H

#include "DQM/RPCMonitorClient/interface/RPCClient.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <map>
#include <memory>
#include <string>
#include <vector>


class RPCOccupancyTest:public RPCClient {
public:

  RPCOccupancyTest(const edm::ParameterSet& ps);
  virtual ~RPCOccupancyTest();

  void beginJob(DQMStore *);
  void beginRun(const edm::Run& , const edm::EventSetup& , std::vector<MonitorElement *> , std::vector<RPCDetId>); 
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& c);
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);
 void endRun(const edm::Run& , const edm::EventSetup& ); 		
  void endJob();
 protected:
  // void OccupancyDist();
  void fillGlobalME(RPCDetId & , MonitorElement *);
 private:
  
  std::string globalFolder_;

 
  std::vector<MonitorElement *>  myOccupancyMe_;
  std::vector<RPCDetId>   myDetIds_;
  int prescaleFactor_;
 
  DQMStore* dbe_;
  int numberOfDisks_, numberOfRings_;
 
  float rpcevents_;

  MonitorElement * AsyMeWheel[5];      //Left Right Asymetry 
  MonitorElement * NormOccupWheel[5];
 
  MonitorElement * AsyMeDWheel[5];      //Left Right Asymetry 
  MonitorElement * NormOccupDWheel[5];

  MonitorElement * AsyMeDisk[10];      //Left Right Asymetry 
  MonitorElement * NormOccupDisk[10];
 
  MonitorElement * AsyMeDDisk[10];      //Left Right Asymetry 
  MonitorElement * NormOccupDDisk[10];
};

#endif
