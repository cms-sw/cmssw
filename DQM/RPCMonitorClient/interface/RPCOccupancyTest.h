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

 void clientOperation();
 void getMonitorElements(std::vector<MonitorElement *> &, std::vector<RPCDetId> &, std::string &);
 void beginJob(std::string & );
 void myBooker(DQMStore::IBooker & );


 protected:
  // void OccupancyDist();
  void fillGlobalME(RPCDetId & , MonitorElement *);
 
private:
  
  std::string globalFolder_, prefixDir_;
  bool testMode_;
  bool  useRollInfo_;
  std::vector<MonitorElement *>  myOccupancyMe_;
  std::vector<RPCDetId>   myDetIds_;
  int prescaleFactor_;
 
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
  MonitorElement * Barrel_OccBySt;
  MonitorElement * EndCap_OccByRng;
  MonitorElement * EndCap_OccByDisk ;

};

#endif
