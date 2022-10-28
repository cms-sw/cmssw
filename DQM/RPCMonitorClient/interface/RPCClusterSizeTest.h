#ifndef RPCClusterSizeTest_H
#define RPCClusterSizeTest_H

#include "DQM/RPCMonitorClient/interface/RPCClient.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <vector>
#include <string>

class RPCClusterSizeTest : public RPCClient {
public:
  /// Constructor
  RPCClusterSizeTest(const edm::ParameterSet &ps);

  /// Destructor
  ~RPCClusterSizeTest() override = default;
  void clientOperation() override;
  void getMonitorElements(std::vector<MonitorElement *> &, std::vector<RPCDetId> &, std::string &) override;
  void beginJob(std::string &) override;
  void myBooker(DQMStore::IBooker &) override;

private:
  std::string globalFolder_;
  int numberOfDisks_;
  int numberOfRings_;
  int prescaleFactor_;
  bool testMode_;
  bool useRollInfo_;
  std::vector<MonitorElement *> myClusterMe_;
  std::vector<RPCDetId> myDetIds_;
  enum MEArraySizes { kWheels = 5, kDisks = 10 };

  MonitorElement *MEANWheel[kWheels];   // Mean ClusterSize, Roll vs Sector
  MonitorElement *MEANDWheel[kWheels];  // Mean ClusterSize, Distribution

  MonitorElement *MEANDisk[kDisks];   // Mean ClusterSize, Roll vs Sector
  MonitorElement *MEANDDisk[kDisks];  // Mean ClusterSize, Distribution

  void resetMEArrays(void);
};

#endif
