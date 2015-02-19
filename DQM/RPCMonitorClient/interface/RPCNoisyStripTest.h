#ifndef RPCNoisyStipTest_H
#define RPCNoisyStipTest_H


#include "DQM/RPCMonitorClient/interface/RPCClient.h"

#include <vector>


class RPCNoisyStripTest:public RPCClient{
public:


  RPCNoisyStripTest(const edm::ParameterSet& ps);
  virtual ~RPCNoisyStripTest();
 void clientOperation();
 void getMonitorElements(std::vector<MonitorElement *> &, std::vector<RPCDetId> &, std::string &);
 void beginJob(std::string & );
 void myBooker(DQMStore::IBooker & );


 protected:

  void fillGlobalME(RPCDetId & , MonitorElement * );


 private:
  
  std::string globalFolder_;
  int  numberOfRings_;
  int prescaleFactor_;
  bool testMode_;
 
  std::vector<MonitorElement *>  myOccupancyMe_;
  std::vector<RPCDetId>   myDetIds_;
  bool useRollInfo_;
  MonitorElement * NOISEWheel[5];
  MonitorElement * NOISEDWheel[5];
  MonitorElement * DEVDWheel[5]; 

  MonitorElement * NOISEDisk[10];
  MonitorElement * NOISEDDisk[10];
  MonitorElement * DEVDDisk[10]; 
  int numberOfDisks_;
  
};

#endif
