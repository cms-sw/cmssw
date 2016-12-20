#ifndef RPCEventSummary_H
#define RPCEventSummary_H


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <string>


class RPCEventSummary:public DQMEDHarvester{

public:

  /// Constructor
  RPCEventSummary(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~RPCEventSummary();


 protected:
  void beginJob();
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&); //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob

 
 private:
  void clientOperation( DQMStore::IGetter & igetter);
  std::string eventInfoPath_, prefixDir_;

  //  bool tier0_;  
  bool enableReportSummary_;
  int prescaleFactor_, minimumEvents_;

  bool init_, isIn_;
   bool offlineDQM_;
  int lumiCounter_;
  std::string globalFolder_, prefixFolder_;
  
  int numberDisk_;
  bool   doEndcapCertification_;
  std::pair<int, int> FEDRange_;
  int NumberOfFeds_;

  enum RPCQualityFlags{DEAD = 6, PARTIALLY_DEAD=5};

};

#endif
