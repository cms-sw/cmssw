#ifndef EcalPreshowerMonitorClient_H
#define EcalPreshowerMonitorClient_H

#include "DQMServices/Core/interface/DQMEDHarvester.h"

class ESClient;

class EcalPreshowerMonitorClient : public DQMEDHarvester {

 public:

  EcalPreshowerMonitorClient(const edm::ParameterSet&);
  ~EcalPreshowerMonitorClient() override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

 private:
  
  void dqmEndLuminosityBlock(DQMStore::IBooker&, DQMStore::IGetter&, const edm::LuminosityBlock &, const edm::EventSetup &) override;
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;
  
  // ----------member data ---------------------------
  bool debug_;
  bool verbose_;
  
  std::vector<ESClient*> clients_;

  // Data members existed below could not have been used in any way, yet was consuming O(100kB) of memory.
  // Removed together with htmlOutput, which was a private function that was not called from anywhere. (yiiyama, Sep 18 2014)
};

#endif
