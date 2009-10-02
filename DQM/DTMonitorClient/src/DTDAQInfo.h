#ifndef DTMonitorClient_DTDAQInfo_H
#define DTMonitorClient_DTDAQInfo_H

/** \class DTDAQInfo
 *  No description available.
 *
 *  $Date: 2008/12/12 18:04:17 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <map>

class DQMStore;
class MonitorElement;
class DTReadOutMapping;

class DTDAQInfo : public edm::EDAnalyzer {
public:
  /// Constructor
  DTDAQInfo(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTDAQInfo();

  // Operations

protected:
  
private:
  virtual void beginJob(const edm::EventSetup& setup);
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& setup);
  virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumi, const  edm::EventSetup& setup);
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumi, const  edm::EventSetup& setup);
  virtual void endJob() ;
  
  DQMStore *theDbe;  
  
  MonitorElement*  totalDAQFraction;
  std::map<int, MonitorElement*> daqFractions;
  edm::ESHandle<DTReadOutMapping> mapping;

};


#endif

