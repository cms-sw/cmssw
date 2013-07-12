#ifndef DTMonitorClient_DTDCSSummary_H
#define DTMonitorClient_DTDCSSummary_H

/** \class DTDCSSummary
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

class DTDCSSummary : public edm::EDAnalyzer {
public:
  /// Constructor
  DTDCSSummary(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTDCSSummary();

  // Operations

protected:
  
private:
  virtual void beginJob();
  virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumi, const  edm::EventSetup& setup);
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumi, const  edm::EventSetup& setup);
  virtual void endJob() ;
  
  DQMStore *theDbe;  
  
  MonitorElement*  totalDCSFraction;
  std::map<int, MonitorElement*> dcsFractions;

};


#endif

