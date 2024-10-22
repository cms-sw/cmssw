
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
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

//RecHit
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"

#include <string>
#include <map>
#include <vector>

class DTGeometry;
class DTStatusFlag;

class DTSegmentAnalysisTask : public DQMEDAnalyzer {
public:
  /// Constructor
  DTSegmentAnalysisTask(const edm::ParameterSet& pset);

  /// Destructor
  ~DTSegmentAnalysisTask() override;

  /// BeginRun
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;

  // Operations
  void analyze(const edm::Event& event, const edm::EventSetup& setup) override;

protected:
  // Book the histograms
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  // Switch for detailed analysis
  bool detailedAnalysis;

  // Get the DT Geometry
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> muonGeomToken_;
  const DTGeometry* dtGeom;

  // Get the status Map
  edm::ESGetToken<DTStatusFlag, DTStatusFlagRcd> statusMapToken_;
  const DTStatusFlag* statusMap;

  // Label of 4D segments in the event
  edm::EDGetTokenT<DTRecSegment4DCollection> recHits4DToken_;

  // Get the map of noisy channels
  bool checkNoisyChannels;

  // book the histos
  void bookHistos(DQMStore::IBooker& ibooker, DTChamberId chamberId);
  // Fill a set of histograms for a given chamber
  void fillHistos(DTChamberId chamberId, int nHits, float chi2);

  //  the histos
  std::map<DTChamberId, std::vector<MonitorElement*> > histosPerCh;
  std::map<int, MonitorElement*> summaryHistos;

  int nevents;
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

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
