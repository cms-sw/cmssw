
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
 *  $Date: 2012/06/28 07:59:01 $
 *  $Revision: 1.13 $
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
class DTTimeEvolutionHisto;

class DTSegmentAnalysisTask: public edm::EDAnalyzer{


public:
  /// Constructor
  DTSegmentAnalysisTask(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTSegmentAnalysisTask();

  /// BeginRun
  void beginRun(const edm::Run& , const edm::EventSetup&);

  /// Endjob
  void endJob();

  // Operations
  void analyze(const edm::Event& event, const edm::EventSetup& setup);

  /// Summary
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup);
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup);


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
		  float chi2);
  
  //  the histos
  std::map<DTChamberId, std::vector<MonitorElement*> > histosPerCh;
  std::map< int, MonitorElement* > summaryHistos;
  std::map<int, std::map<int, DTTimeEvolutionHisto*> > histoTimeEvol;

  int nevents;
  int nEventsInLS;
  DTTimeEvolutionHisto*hNevtPerLS;

  // # of bins in the time histos
  int nTimeBins;
  // # of LS per bin in the time histos
  int nLSTimeBin; 
  // switch on/off sliding bins in time histos
  bool slideTimeBins;
  // top folder for the histograms in DQMStore
  std::string topHistoFolder;
  // hlt DQM mode
  bool hltDQMMode;
  // max phi angle of reconstructed segments 
  double phiSegmCut;
  // min # hits of segment used to validate a segment in WB+-2/SecX/MB1 
  int nhitsCut; 

  MonitorElement* nEventMonitor;

};
#endif

