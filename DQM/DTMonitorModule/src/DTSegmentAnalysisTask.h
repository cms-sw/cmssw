#ifndef DTSegmentAnalysisTask_H
#define DTSegmentAnalysisTask_H

/** \class DTSegmentAnalysisTask
 *  DQM Analysis of 4D DT segments, it produces plots about: <br>
 *      - number of segments per event <br>
 *      - number of hits per segment <br>
 *      - position of the segments in chamber RF <br>
 *      - direction of the segments (theta and phi projections) <br>
 *      - reduced chi-square <br>
 *  All histos are produce per Chamber
 *
 *
 *  $Date: 2007/11/28 10:31:33 $
 *  $Revision: 1.4 $
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>

#include <string>
#include <map>
#include <vector>
//#include <pair>

class DQMStore;
class MonitorElement;

class DTSegmentAnalysisTask: public edm::EDAnalyzer{
public:
  /// Constructor
  DTSegmentAnalysisTask(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTSegmentAnalysisTask();

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// Endjob
  void endJob();

  // Operations
  void analyze(const edm::Event& event, const edm::EventSetup& setup);

protected:

private:

  // The BE interface
  DQMStore* theDbe;

  // Switch for verbosity
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

