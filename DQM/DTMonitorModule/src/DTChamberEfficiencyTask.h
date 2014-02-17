#ifndef DTChamberEfficiencyTask_H
#define DTChamberEfficiencyTask_H


/** \class DTChamberEfficiencyTask
 *  DQM Analysis of 4D DT segments, it produces plots about: <br>
 *      - single chamber efficiency 
 *  All histos are produced per Chamber
 *
 *  Class based on the code written by S. Lacaprara :
 *  RecoLocalMuon / DTSegment / test / DTEffAnalyzer.h
 *
 *  $Date: 2010/01/05 10:14:40 $
 *  $Revision: 1.7 $
 *  \author G. Mila - INFN Torino
 */



#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <string>
#include <map>
#include <vector>

class DQMStore;
class MonitorElement;


class DTChamberEfficiencyTask: public edm::EDAnalyzer{
public:
  /// Constructor
  DTChamberEfficiencyTask(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTChamberEfficiencyTask();

  /// BeginJob
  void beginJob();

  /// BeginRun
  void beginRun(const edm::Run& run, const edm::EventSetup& setup);

  /// To reset the MEs
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// Endjob
  void endJob();

  // Operations
  void analyze(const edm::Event& event, const edm::EventSetup& setup);

protected:


private:

  const DTRecSegment4D& getBestSegment(const DTRecSegment4DCollection::range& segs) const;
  const DTRecSegment4D* getBestSegment(const DTRecSegment4D* s1,
				       const DTRecSegment4D* s2) const;
  bool isGoodSegment(const DTRecSegment4D& seg) const;
  LocalPoint interpolate(const DTRecSegment4D& seg1,
			 const DTRecSegment4D& seg3,
			 const DTChamberId& MB2) const;
  void bookHistos(DTChamberId chId); 


  DQMStore* theDbe;
  
  // Switch for verbosity
  bool debug;
  // The running mode
  bool onlineMonitor;
  // The analysis mode
  bool detailedAnalysis;

  // Lable of 4D segments in the event
  std::string theRecHits4DLabel;

  edm::ParameterSet parameters;

  std::map<DTChamberId, std::vector<MonitorElement*> > histosPerCh;

  unsigned int theMinHitsSegment;
  double theMinChi2NormSegment;
  double theMinCloseDist;

  edm::ESHandle<DTGeometry> dtGeom;
  edm::Handle<DTRecSegment4DCollection> segs;

};
#endif
