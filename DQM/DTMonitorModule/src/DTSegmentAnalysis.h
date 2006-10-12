#ifndef DTSegmentAnalysis_H
#define DTSegmentAnalysis_H

/** \class DTSegmentAnalysis
 *  DQM Analysis of 4D DT segments, it produces plots about: <br>
 *      - number of segments per event <br>
 *      - number of hits per segment <br>
 *      - position of the segments in chamber RF <br>
 *      - direction of the segments (theta and phi projections) <br>
 *      - reduced chi-square <br>
 *  All histos are produce per Chamber
 *
 *
 *  $Date: 2006/10/08 16:04:20 $
 *  $Revision: 1.3 $
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <map>
#include <vector>
//#include <pair>

class DaqMonitorBEInterface;
class MonitorElement;

class DTSegmentAnalysis {
public:
  /// Constructor
  DTSegmentAnalysis(const edm::ParameterSet& pset, DaqMonitorBEInterface* dbe);

  /// Destructor
  virtual ~DTSegmentAnalysis();

  // Operations
  void analyze(const edm::Event& event, const edm::EventSetup& setup);

protected:

private:
  DaqMonitorBEInterface* theDbe;

  bool debug;
  // Lable of 4D segments in the event
  std::string theRecHits4DLabel;

  edm::ParameterSet parameters;
  int DTTrig;
  int CSCTrig;
  int RBC1Trig;
  int RBC2Trig;
  int RPCTBTrig;

  // Book a set of histograms for a give chamber
  void bookHistos(int w, int sec);
  void bookHistos(DTChamberId chamberId);
  // Fill a single histogram
  void fillHistos(int nsegm, int w, int sec) ;
  // Fill a set of histograms for a give chamber 
  void fillHistos(DTChamberId chamberId, int nsegm);
  void fillHistos(DTChamberId chamberId,
		  int nHits,
		  float posX,
		  float posY,
		  float phi,
		  float theta,
		  float chi2);
  
  //   std::map<DTChamberId, MonitorElement*> numSegmentPerCh;
  std::map<DTChamberId, std::vector<MonitorElement*> > histosPerCh;
  std::map<std::pair<int,int>, MonitorElement* > histosPerSec;

};
#endif

