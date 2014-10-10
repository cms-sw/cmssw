#ifndef RPCNoisyStipTest_H
#define RPCNoisyStipTest_H


#include "DQM/RPCMonitorClient/interface/RPCClient.h"

#include <vector>


class RPCNoisyStripTest:public RPCClient{
public:


  RPCNoisyStripTest(const edm::ParameterSet& ps);
  virtual ~RPCNoisyStripTest();
  void beginJob(DQMStore *, std::string);
  void endRun(const edm::Run& , const edm::EventSetup& );
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& c);
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);
  void beginRun(const edm::Run& , const edm::EventSetup& ); 		
  void endJob();
  void clientOperation(edm::EventSetup const& c);
  void getMonitorElements(std::vector<MonitorElement *> & , std::vector<RPCDetId> &);
 protected:

  void fillGlobalME(RPCDetId & , MonitorElement * ,edm::EventSetup const& );


 private:
  
  std::string globalFolder_;
  int  numberOfRings_;
  int prescaleFactor_;
  bool testMode_;
  DQMStore* dbe_;
 
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
