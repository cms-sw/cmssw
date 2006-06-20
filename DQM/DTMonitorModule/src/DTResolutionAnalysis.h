#ifndef DTResolutionAnalysis_H
#define DTResolutionAnalysis_H

/** \class DTResolutionAnalysis
 *  DQM Analysis of 4D DT segments, it produces plots about: <br>
 *      - number of segments per event <br>
 *      - position of the segments in chamber RF <br>
 *      - direction of the segments (theta and phi projections) <br>
 *      - reduced chi-square <br>
 *  All histos are produce per Chamber
 *
 *
 *  $Date: 2006/06/01 11:09:27 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"

#include <string>
#include <map>
#include <vector>

class DaqMonitorBEInterface;
class MonitorElement;


class DTResolutionAnalysis {
public:
  /// Constructor
  DTResolutionAnalysis(const edm::ParameterSet& pset, DaqMonitorBEInterface* dbe);

  /// Destructor
  virtual ~DTResolutionAnalysis();

  // Operations
  void analyze(const edm::Event& event, const edm::EventSetup& setup);

protected:

private:
  DaqMonitorBEInterface* theDbe;

  bool debug;
  // Lable of 4D segments in the event
  std::string theRecHits4DLabel;
  // Lable of 1D rechits in the event
  std::string theRecHitLabel;
  

  // Book a set of histograms for a give chamber
  void bookHistos(DTSuperLayerId slId);
  // Fill a set of histograms for a give chamber 
  void fillHistos(DTSuperLayerId slId,
		  float distExtr,
		  float residual);
  
  std::map<DTSuperLayerId, std::vector<MonitorElement*> > histosPerSL;
};
#endif

