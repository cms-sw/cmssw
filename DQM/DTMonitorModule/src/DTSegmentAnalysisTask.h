
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
 *  $Date: 2008/05/27 15:24:00 $
 *  $Revision: 1.6 $
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>

#include <string>
#include <map>
#include <vector>


class DTGeometry;
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

  // Switch for detailed analysis
  bool detailedAnalysis;

   // Get the DT Geometry
  edm::ESHandle<DTGeometry> dtGeom;

  // Lable of 4D segments in the event
  std::string theRecHits4DLabel;

  // Get the map of noisy channels
  bool checkNoisyChannels;

  edm::ParameterSet parameters;
 
  // book the histos
  void bookHistos(DTChamberId chamberId);
  // Fill a set of histograms for a given chamber 
  void fillHistos(DTChamberId chamberId,
		  int nHits,
		  float posX,
		  float posY,
		  float phi,
		  float theta,
		  float chi2);
  
  //  the histos
  std::map<DTChamberId, std::vector<MonitorElement*> > histosPerCh;
  std::map< int, MonitorElement* > summaryHistos;

};
#endif

