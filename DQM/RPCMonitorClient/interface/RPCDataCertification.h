#ifndef RPCMonitorClient_RPCDataCertification_H
#define RPCMonitorClient_RPCDataCertification_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <map>

class DQMStore;
class MonitorElement;

class RPCDataCertification : public edm::EDAnalyzer {
public:
  /// Constructor
  RPCDataCertification(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~RPCDataCertification();

  // Operations

protected:
  
private:
  virtual void beginJob();
  virtual void beginRun(const edm::Run& r, const edm::EventSetup& setup);
  virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumi, const  edm::EventSetup& setup);
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumi, const  edm::EventSetup& setup);
  virtual void endJob() ;
  
  DQMStore *theDbe;  
  MonitorElement*    CertMap_;
  MonitorElement*  totalCertFraction;
  MonitorElement* certWheelFractions[5];
  MonitorElement* certDiskFractions[10];
 std::pair<int, int> FEDRange_;
  int numberOfDisks_;  
  int NumberOfFeds_;

};


#endif
