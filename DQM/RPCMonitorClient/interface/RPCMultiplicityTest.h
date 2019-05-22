#ifndef RPCMultiplicityTest_H
#define RPCMultiplicityTest_H

#include "DQM/RPCMonitorClient/interface/RPCClient.h"
#include "DQMServices/Core/interface/DQMStore.h"

class RPCMultiplicityTest : public RPCClient {
public:
  /// Constructor
  RPCMultiplicityTest(const edm::ParameterSet &ps);

  /// Destructor
  ~RPCMultiplicityTest() override;

  void clientOperation() override;
  void getMonitorElements(std::vector<MonitorElement *> &, std::vector<RPCDetId> &, std::string &) override;
  void beginJob(std::string &) override;
  void myBooker(DQMStore::IBooker &) override;

protected:
  void fillGlobalME(RPCDetId &detId, MonitorElement *myMe);

private:
  int prescaleFactor_;
  std::string globalFolder_;
  int numberOfDisks_;
  int numberOfRings_;
  bool useRollInfo_;
  std::vector<MonitorElement *> myNumDigiMe_;
  std::vector<RPCDetId> myDetIds_;
  bool testMode_;
  MonitorElement *MULTWheel[5];
  MonitorElement *MULTDWheel[5];
  MonitorElement *MULTDisk[10];
  MonitorElement *MULTDDisk[10];
};
#endif
