#ifndef RPCBxTest_H
#define RPCBxTest_H

#include "DQM/RPCMonitorClient/interface/RPCClient.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <map>
#include <memory>
#include <string>
#include <vector>


class RPCBxTest:public RPCClient{
public:

  /// Constructor
  RPCBxTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~RPCBxTest();

  /// BeginJob
  void beginJob(DQMStore *);

  //Begin Run
   void beginRun(const edm::Run& r, const edm::EventSetup& c);
  
  void getMonitorElements(std::vector<MonitorElement *> & , std::vector<RPCDetId>& );
  
  /// Begin Lumi block 
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// Analyze  
  void analyze(const edm::Event& iEvent, const edm::EventSetup& c);

  /// End Lumi Block
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);
  
  virtual void  endJob(void);

  virtual void  endRun(const edm::Run& r, const edm::EventSetup& c);

 private:
  double distanceMean_;
  std::string globalFolder_;
  int numberOfDisks_,numberOfRings_;
  int prescaleFactor_;
  double rmsCut_;
  int entriesCut_;
  DQMStore* dbe_;
  int nLumiSegs_;
  
  std::vector<MonitorElement *>  myBXMe_;
  std::vector<RPCDetId>   myDetIds_;

  MonitorElement * BXEntriesEndcapN;     
  MonitorElement * BXEntriesEndcapP;     
  MonitorElement * BXEntriesBarrel;  
 
  MonitorElement * BXMeanEndcapN;     
  MonitorElement * BXMeanEndcapP;     
  MonitorElement * BXMeanBarrel;         // ClusterSize in 1 bin, Distribution
  MonitorElement * BXMeanWheel[5];         // Mean ClusterSize, Roll vs Sector
  MonitorElement * BXMeanDisk[10];        // Mean ClusterSize, Distribution
 
  MonitorElement * BXRmsEndcapN;     
  MonitorElement * BXRmsEndcapP;
  MonitorElement * BXRmsBarrel;
  MonitorElement * BXRmsDisk[10];         // Mean ClusterSize, Roll vs Sector
  MonitorElement * BXRmsWheel[5];        // Mean ClusterSize, Distribution

};

#endif
