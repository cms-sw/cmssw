#ifndef DTTriggerEfficiencyTask_H
#define DTTriggerEfficiencyTask_H

/*
 * \file DTTriggerEfficiencyTask.h
 *
 * \author C. Battilana - CIEMAT
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/DTDigi/interface/DTLocalTriggerCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include <vector>
#include <string>
#include <map>

class DTGeometry;
class DTChamberId;
class DTTrigGeomUtils;

class DTTriggerEfficiencyTask : public DQMEDAnalyzer {
public:
  /// Constructor
  DTTriggerEfficiencyTask(const edm::ParameterSet& ps);

  /// Destructor
  ~DTTriggerEfficiencyTask() override;

protected:
  /// BeginRun
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  /// Book chamber granularity histograms
  void bookChamberHistos(DQMStore::IBooker& ibooker,
                         const DTChamberId& dtCh,
                         std::string histoTag,
                         std::string folder = "");

  /// Book wheel granularity histograms
  void bookWheelHistos(DQMStore::IBooker& ibooker, int wheel, std::string histoTag, std::string folder = "");

  /// checks for RPC Triggers
  bool hasRPCTriggers(const edm::Event& e);

  /// return the top folder
  std::string topFolder() {
    return "DT/03-LocalTrigger-TM/";  //DDU no longer existing, leaving folder here for past references
  }

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  /// To reset the MEs

private:
  int nevents;

  std::string SegmArbitration;

  bool processTM, detailedPlots, checkRPCtriggers;
  std::vector<std::string> processTags;

  float phiAccRange;
  int nMinHitsPhi;

  edm::EDGetTokenT<reco::MuonCollection> muons_Token_;
  edm::EDGetTokenT<L1MuDTChambPhContainer> tm_Token_;
  edm::InputTag inputTagSEG;
  edm::EDGetTokenT<L1MuGMTReadoutCollection> gmt_Token_;

  edm::ParameterSet parameters;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> muonGeomToken_;
  const DTGeometry* muonGeom;
  DTTrigGeomUtils* trigGeomUtils;
  std::map<uint32_t, std::map<std::string, MonitorElement*> > chamberHistos;
  std::map<int, std::map<std::string, MonitorElement*> > wheelHistos;
};

#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
