#ifndef RPCDeadChannelTest_H
#define RPCDeadChannelTest_H


#include "DQM/RPCMonitorClient/interface/RPCClient.h"

#include "DQMServices/Core/interface/DQMStore.h"


class RPCDeadChannelTest:public RPCClient{

public:

  /// Constructor
  RPCDeadChannelTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~RPCDeadChannelTest();

  /// BeginJob
  void beginJob(DQMStore *);

  //Begin Run
   void beginRun(const edm::Run& , const edm::EventSetup& ,std::vector<MonitorElement *> , std::vector<RPCDetId>);
  
  
  /// Begin Lumi block 
  void beginLuminosityBlock(edm::LuminosityBlock const& , edm::EventSetup const& ) ;

  /// Analyze  
  void analyze(const edm::Event& , const edm::EventSetup& );

  /// End Lumi Block
  void endLuminosityBlock(edm::LuminosityBlock const& , edm::EventSetup const& );
 
  //End Run
  void endRun(const edm::Run& , const edm::EventSetup& ); 		
  
  /// Endjob
  void endJob();

 protected:
  void CalculateDeadChannelPercentage(RPCDetId & , MonitorElement *  , edm::EventSetup const& );
   
 private:
  int prescaleFactor_;
  std::string globalFolder_;
  std::vector<MonitorElement *>  myOccupancyMe_;
  std::vector<RPCDetId>   myDetIds_;
 
  DQMStore* dbe_;
 
  int numberOfDisks_,numberOfRings_ ;

  MonitorElement * DEADWheel[5];
  MonitorElement * DEADDisk[10]; 


  
};

#endif
