#ifndef RPCMonitorClient_RPCDCSSummary_H
#define RPCMonitorClient_RPCDCSSummary_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <map>

class DQMStore;
class MonitorElement;

class RPCDCSSummary : public edm::EDAnalyzer {
public:
  /// Constructor
  RPCDCSSummary(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~RPCDCSSummary();

  // Operations

protected:
  
private:
  virtual void beginJob();
  void beginRun(const edm::Run& , const edm::EventSetup& );
  virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumi, const  edm::EventSetup& setup);
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumi, const  edm::EventSetup& setup);
  virtual void endJob() ;
  
  DQMStore *theDbe;  
  MonitorElement*    DCSMap_;
  MonitorElement*  totalDCSFraction;
  MonitorElement* dcsWheelFractions[5];
  MonitorElement* dcsDiskFractions[10];
  std::pair<int, int> FEDRange_;
  int numberOfDisks_;  
  int NumberOfFeds_;

};


#endif
