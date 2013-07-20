#ifndef DTMonitorClient_DTDCSByLumiSummary_H
#define DTMonitorClient_DTDCSByLumiSummary_H

/** \class DTDCSByLumiSummary
 *  No description available.
 *
 *  $Date: 2011/03/02 14:00:06 $
 *  $Revision: 1.1 $
 *  \author C. Battilana - CIEMAT
 *  \author P. Bellan - INFN PD
 *  \author A. Branca = INFN PD

 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <FWCore/Framework/interface/LuminosityBlock.h>

#include <map>

class DQMStore;
class MonitorElement;
class DTTimeEvolutionHisto;

class DTDCSByLumiSummary : public edm::EDAnalyzer {

public:

  /// Constructor
  DTDCSByLumiSummary(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTDCSByLumiSummary();


private:

  // Operations
  virtual void beginJob();
  virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumi, const  edm::EventSetup& setup);
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumi, const  edm::EventSetup& setup);
  virtual void endRun(const edm::Run& run, const edm::EventSetup& setup);
  virtual void endJob() ;
  
  DQMStore *theDQMStore;  
  
  MonitorElement*       totalDCSFraction;
  MonitorElement*       globalHVSummary;

  std::vector<DTTimeEvolutionHisto*> hDCSFracTrend;
  std::vector<MonitorElement*> totalDCSFractionWh;
  
 std::map<int, std::vector<float> > dcsFracPerLumi;

};


#endif

