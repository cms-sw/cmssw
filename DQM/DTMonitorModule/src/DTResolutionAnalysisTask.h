#ifndef DTResolutionAnalysisTask_H
#define DTResolutionAnalysisTask_H

/** \class DTResolutionAnalysis
 *  DQM Analysis of 4D DT segments, it produces plots about: <br>
 *      - number of segments per event <br>
 *      - position of the segments in chamber RF <br>
 *      - direction of the segments (theta and phi projections) <br>
 *      - reduced chi-square <br>
 *  All histos are produce per Chamber
 *
 *
 *  $Date: 2012/02/17 16:05:19 $
 *  $Revision: 1.11 $
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/Framework/interface/ESHandle.h"


#include <string>
#include <map>
#include <vector>

class DQMStore;
class MonitorElement;
class DTGeometry;

class DTResolutionAnalysisTask: public edm::EDAnalyzer{
public:
  /// Constructor
  DTResolutionAnalysisTask(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTResolutionAnalysisTask();

  /// BeginRun
  void beginRun(const edm::Run&, const edm::EventSetup&);

  /// To reset the MEs
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// Endjob
  void endJob();

  // Operations
  void analyze(const edm::Event& event, const edm::EventSetup& setup);
  
 
protected:

private:
  DQMStore* theDbe;

  edm::ESHandle<DTGeometry> dtGeom;
  
  int prescaleFactor;
  int resetCycle;

  u_int32_t thePhiHitsCut;
  u_int32_t theZHitsCut;

  // Lable of 4D segments in the event
  std::string theRecHits4DLabel;
  
  // Book a set of histograms for a give chamber
  void bookHistos(DTSuperLayerId slId);
  // Fill a set of histograms for a give chamber 
  void fillHistos(DTSuperLayerId slId,
		  float distExtr,
		  float residual);

  std::map<DTSuperLayerId, std::vector<MonitorElement*> > histosPerSL;

  // top folder for the histograms in DQMStore
  std::string topHistoFolder;

};
#endif

