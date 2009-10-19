#ifndef DTMonitorClient_DTCertificationSummary_H
#define DTMonitorClient_DTCertificationSummary_H

/** \class DTCertificationSummary
 *  No description available.
 *
 *  $Date: 2009/03/02 17:03:50 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <map>

class DQMStore;
class MonitorElement;

class DTCertificationSummary : public edm::EDAnalyzer {
public:
  /// Constructor
  DTCertificationSummary(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTCertificationSummary();

  // Operations

protected:
  
private:
  virtual void beginJob(const edm::EventSetup& setup);
  virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumi, const  edm::EventSetup& setup);
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumi, const  edm::EventSetup& setup);
  virtual void endJob() ;
  
  DQMStore *theDbe;  
  
  MonitorElement*  totalCertFraction;
  MonitorElement*  certMap;
  std::map<int, MonitorElement*> certFractions;

};


#endif

