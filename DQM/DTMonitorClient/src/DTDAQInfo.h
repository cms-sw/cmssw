#ifndef DTMonitorClient_DTDAQInfo_H
#define DTMonitorClient_DTDAQInfo_H

/** \class DTDAQInfo
 *  No description available.
 *
 *  $Date: 2010/01/05 10:15:46 $
 *  $Revision: 1.4 $
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
  virtual void beginJob();
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& setup);
  virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumi, const  edm::EventSetup& setup);
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumi, const  edm::EventSetup& setup);
  virtual void endJob() ;
  
  DQMStore *theDbe;  
  
  MonitorElement*  totalDAQFraction;
  MonitorElement*  daqMap;
  std::map<int, MonitorElement*> daqFractions;
  edm::ESHandle<DTReadOutMapping> mapping;

};


#endif

