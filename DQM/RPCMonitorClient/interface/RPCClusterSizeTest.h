#ifndef RPCClusterSizeTest_H
#define RPCClusterSizeTest_H

#include "DQM/RPCMonitorClient/interface/RPCClient.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <map>
#include <memory>
#include <string>
#include <vector>


class RPCClusterSizeTest:public RPCClient{
 public:

  /// Constructor
  RPCClusterSizeTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~RPCClusterSizeTest();

  /// BeginJob
  void beginJob(DQMStore *);

  //Begin Run
  void endRun(const edm::Run& r, const edm::EventSetup& c );
  
  
  /// Begin Lumi block 
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// Analyze  
  void analyze(const edm::Event& iEvent, const edm::EventSetup& c);

  /// End Lumi Block
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);
  
  void  endJob(void);

  void  beginRun(const edm::Run& r, const edm::EventSetup& c);

  void  clientOperation(edm::EventSetup const& c);

  void  bookHisto(std::vector<MonitorElement *> , std::vector<RPCDetId>);

 private:
  
  std::string globalFolder_;
  int numberOfDisks_;
  int numberOfRings_;
  int prescaleFactor_;
 
  DQMStore* dbe_;
   
  std::vector<MonitorElement *>  myClusterMe_;
  std::vector<RPCDetId>   myDetIds_;

 
  MonitorElement * CLSWheel[5];          // ClusterSize in 1 bin, Roll vs Sector
  MonitorElement * CLSDWheel[5];         // ClusterSize in 1 bin, Distribution
  MonitorElement * MEANWheel[5];         // Mean ClusterSize, Roll vs Sector
  MonitorElement * MEANDWheel[5];        // Mean ClusterSize, Distribution

  MonitorElement * CLSDisk[10];          // ClusterSize in 1 bin, Roll vs Sector
  MonitorElement * CLSDDisk[10];         // ClusterSize in 1 bin, Distribution
  MonitorElement * MEANDisk[10];         // Mean ClusterSize, Roll vs Sector
  MonitorElement * MEANDDisk[10];        // Mean ClusterSize, Distribution

};

#endif
