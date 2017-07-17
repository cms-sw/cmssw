#ifndef RPCDeadChannelTest_H
#define RPCDeadChannelTest_H


#include "DQM/RPCMonitorClient/interface/RPCClient.h"

//#include "DQMServices/Core/interface/DQMStore.h"


class RPCDeadChannelTest:public RPCClient{

public:


  /// Constructor
  RPCDeadChannelTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~RPCDeadChannelTest();

 void clientOperation();
 void getMonitorElements(std::vector<MonitorElement *> &, std::vector<RPCDetId> &, std::string &);
 void beginJob(std::string & );
 void myBooker(DQMStore::IBooker &);
   
 private:
  int prescaleFactor_;
  std::string globalFolder_;
  std::vector<MonitorElement *>  myOccupancyMe_;
  std::vector<RPCDetId>   myDetIds_;
  bool useRollInfo_;
 
  int numberOfDisks_;
  int  numberOfRings_;
  MonitorElement * DEADWheel[5];
  MonitorElement * DEADDisk[10]; 


  
};

#endif
