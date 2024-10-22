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
 *  \author G. Cerminara - INFN Torino
 */

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include <string>
#include <map>
#include <vector>

class DTGeometry;

class DTResolutionAnalysisTask : public DQMOneEDAnalyzer<> {
public:
  /// Constructor
  DTResolutionAnalysisTask(const edm::ParameterSet& pset);

  /// Destructor
  ~DTResolutionAnalysisTask() override;

  /// BookHistograms
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  /// BeginRun
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;

  /// To reset the MEs
  //  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) override;
  //  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) final {}

  // Operations
  void analyze(const edm::Event& event, const edm::EventSetup& setup) override;

protected:
private:
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> muonGeomToken_;
  const DTGeometry* dtGeom;

  int prescaleFactor;
  int resetCycle;

  u_int32_t thePhiHitsCut;
  u_int32_t theZHitsCut;

  // Lable of 4D segments in the event
  edm::EDGetTokenT<DTRecSegment4DCollection> recHits4DToken_;

  // Book a set of histograms for a give chamber
  void bookHistos(DQMStore::IBooker& ibooker, DTSuperLayerId slId);
  // Fill a set of histograms for a give chamber
  void fillHistos(DTSuperLayerId slId, float distExtr, float residual);

  std::map<DTSuperLayerId, std::vector<MonitorElement*> > histosPerSL;

  // top folder for the histograms in DQMStore
  std::string topHistoFolder;
};
#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
