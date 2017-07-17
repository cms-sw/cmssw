#ifndef RPCClusterSizeTest_H
#define RPCClusterSizeTest_H

#include "DQM/RPCMonitorClient/interface/RPCClient.h"
#include "DQMServices/Core/interface/DQMStore.h"


class RPCClusterSizeTest:public RPCClient{
 public:

  /// Constructor
  RPCClusterSizeTest(const edm::ParameterSet& ps);

  /// Destructor
  virtual ~RPCClusterSizeTest();
 void clientOperation();
 void getMonitorElements(std::vector<MonitorElement *> &, std::vector<RPCDetId> &, std::string &);
 void beginJob(std::string & );
 void myBooker(DQMStore::IBooker & );


 private:

  std::string globalFolder_;
  int numberOfDisks_;
  int numberOfRings_;
  int prescaleFactor_;
  bool testMode_;
  bool useRollInfo_;
  std::vector<MonitorElement *>  myClusterMe_;
  std::vector<RPCDetId>   myDetIds_;
  enum MEArraySizes {
    kWheels = 5,
    kDisks = 10
  };

  MonitorElement * CLSWheel[kWheels];          // ClusterSize in 1 bin, Roll vs Sector
  MonitorElement * CLSDWheel[kWheels];         // ClusterSize in 1 bin, Distribution
  MonitorElement * MEANWheel[kWheels];         // Mean ClusterSize, Roll vs Sector
  MonitorElement * MEANDWheel[kWheels];        // Mean ClusterSize, Distribution

  MonitorElement * CLSDisk[kDisks];          // ClusterSize in 1 bin, Roll vs Sector
  MonitorElement * CLSDDisk[kDisks];         // ClusterSize in 1 bin, Distribution
  MonitorElement * MEANDisk[kDisks];         // Mean ClusterSize, Roll vs Sector
  MonitorElement * MEANDDisk[kDisks];        // Mean ClusterSize, Distribution

  void resetMEArrays(void);
};

#endif
