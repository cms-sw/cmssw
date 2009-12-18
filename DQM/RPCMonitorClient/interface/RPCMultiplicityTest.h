#ifndef RPCMultiplicityTest_H
#define RPCMultiplicityTest_H

#include "DQM/RPCMonitorClient/interface/RPCClient.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <memory>
#include <string>
#include <vector>

class  RPCMultiplicityTest:public RPCClient{

public:

  /// Constructor
  RPCMultiplicityTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~RPCMultiplicityTest();

  /// BeginJob
  void beginJob(DQMStore * );

  //Begin Run
   void beginRun(const edm::Run& , const edm::EventSetup& , std::vector<MonitorElement *> , std::vector<RPCDetId>);
  
  
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
  void fillGlobalME(RPCDetId & detId, MonitorElement * myMe);

 private:
  int prescaleFactor_;
  std::string globalFolder_;
  int numberOfDisks_, numberOfRings_;

  std::vector<MonitorElement *>  myNumDigiMe_;
  std::vector<RPCDetId>   myDetIds_;

  MonitorElement * MULTWheel[5];
  MonitorElement * MULTDWheel[5];
  MonitorElement * MULTDisk[10]; 
  MonitorElement * MULTDDisk[10];

  DQMStore* dbe_;
  // std:: map<int, std::map< int ,  std::pair<float,float> > >  barrelMap_, endcapMap_;
  
};
#endif
