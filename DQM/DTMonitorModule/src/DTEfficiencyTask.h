#ifndef DTEfficiencyTask_H
#define DTEfficiencyTask_H

/** \class DTEfficiencyTask
 *  DQM Analysis of 4D DT segments, it produces plots about: <br>
 *      - single cell efficiency
 *  All histos are produced per Layer
 *
 *
 *  \author G. Mila - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

#include <string>
#include <map>
#include <vector>

class DTGeometry;

class DTEfficiencyTask : public DQMOneEDAnalyzer<edm::one::WatchLuminosityBlocks> {
public:
  /// Constructor
  DTEfficiencyTask(const edm::ParameterSet& pset);

  /// Destructor
  ~DTEfficiencyTask() override;

  /// To reset the MEs
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) final {}

  // Operations
  void analyze(const edm::Event& event, const edm::EventSetup& setup) override;

protected:
  /// BeginRun
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;

  // Book the histograms
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> muonGeomToken_;
  const DTGeometry* muonGeom;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomToken_;
  const DTGeometry* dtGeom;

  // Switch for verbosity
  bool debug;

  // Lable of 4D segments in the event
  edm::EDGetTokenT<DTRecSegment4DCollection> recHits4DToken_;

  // Lable of 1D rechits in the event
  edm::EDGetTokenT<DTRecHitCollection> recHitToken_;

  edm::ParameterSet parameters;

  // Fill a set of histograms for a given L
  void fillHistos(DTLayerId lId, int firstWire, int lastWire, int numWire);
  void fillHistos(DTLayerId lId, int firstWire, int lastWire, int missingWire, bool UnassHit);

  std::map<DTLayerId, std::vector<MonitorElement*> > histosPerL;
};
#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
