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

  /// BeginJob
  void beginJob(DQMStore *, std::string);

  //Begin Run
   void endRun(const edm::Run& , const edm::EventSetup& );
  
  
  /// Begin Lumi block 
  void beginLuminosityBlock(edm::LuminosityBlock const& , edm::EventSetup const& ) ;

  /// Analyze  
  void analyze(const edm::Event& , const edm::EventSetup& );

  /// End Lumi Block
  void endLuminosityBlock(edm::LuminosityBlock const& , edm::EventSetup const& );
 
  //End Run
  void beginRun(const edm::Run& , const edm::EventSetup& ); 		
  
  /// Endjob
  void endJob();

  void clientOperation(edm::EventSetup const& c);
  void getMonitorElements(std::vector<MonitorElement *> &, std::vector<RPCDetId> &);

 protected:

  // void CalculateDeadChannelPercentage(RPCDetId & , MonitorElement *  , edm::EventSetup const& );
   
 private:
  int prescaleFactor_;
  std::string globalFolder_;
  std::vector<MonitorElement *>  myOccupancyMe_;
  std::vector<RPCDetId>   myDetIds_;
  bool useRollInfo_;
  DQMStore* dbe_;
 
 
  int numberOfDisks_;
  int  numberOfRings_;
  MonitorElement * DEADWheel[5];
  MonitorElement * DEADDisk[10]; 


  
};

#endif
