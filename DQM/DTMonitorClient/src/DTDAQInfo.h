#ifndef DTMonitorClient_DTDAQInfo_H
#define DTMonitorClient_DTDAQInfo_H

/** \class DTDAQInfo
 *  No description available.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <map>

class DQMStore;
class MonitorElement;

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
  virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumi, const  edm::EventSetup& setup);
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumi, const  edm::EventSetup& setup);
  virtual void endJob() ;
  
  DQMStore *theDbe;  
  
  MonitorElement*  totalDAQFraction;
  std::map<int, MonitorElement*> daqFractions;

};


#endif

