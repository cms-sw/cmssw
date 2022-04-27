#ifndef DTLocalTriggerBaseTask_H
#define DTLocalTriggerBaseTask_H

/*
 * \file DTLocalTriggerBaseTask.h
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

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DataFormats/DTDigi/interface/DTLocalTriggerCollection.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhContainer.h"

#include <vector>
#include <string>
#include <map>

class DTGeometry;
class DTTrigGeomUtils;
class DTChamberId;
class DTRecSegment4D;
class L1MuDTChambPhDigi;
class L1MuDTChambThDigi;
class L1Phase2MuDTPhDigi;
class DTTPGCompareUnit;
class DTTimeEvolutionHisto;

class DTLocalTriggerBaseTask : public DQMOneEDAnalyzer<edm::one::WatchLuminosityBlocks> {
  friend class DTMonitorModule;

public:
  /// Constructor
  DTLocalTriggerBaseTask(const edm::ParameterSet& ps);

  /// Destructor
  ~DTLocalTriggerBaseTask() override;

protected:
  ///Beginrun
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  /// To reset the MEs
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) override;

  /// Perform trend plot operations
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) override;

private:
  /// Run analysis on TM data
  void runTMAnalysis(std::vector<L1MuDTChambPhDigi> const* phInTrigs,
                     std::vector<L1MuDTChambPhDigi> const* phOutTrigs,
                     std::vector<L1MuDTChambThDigi> const* thTrigs);

  /// Run analysis on Phase2 readout for SliceTest
  void runAB7Analysis(std::vector<L1Phase2MuDTPhDigi> const* phTrigs);

  /// Get the Top folder (different between Physics and TP and TM)
  std::string& topFolder(std::string const& type) { return m_baseFolder[type == "TM"]; }

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  /// Book the histograms
  void bookHistos(DQMStore::IBooker&, const DTChamberId& chamb);

  /// Book the histograms
  void bookHistos(DQMStore::IBooker&, int wh);

  /// Set Quality labels
  void setQLabels(MonitorElement* me, short int iaxis);
  void setQLabelsTheta(MonitorElement* me, short int iaxis);
  void setQLabelsPh2(MonitorElement* me, short int iaxis);

  int m_nEvents;
  int m_nEventsInLS;
  int m_nLumis;

  std::string m_baseFolder[2];
  bool m_tpMode;
  bool m_detailedAnalysis;

  bool m_processTM;
  bool m_processAB7;

  int m_targetBXTM;
  int m_bestAccRange;

  edm::ParameterSet m_params;
  DTTrigGeomUtils* m_trigGeomUtils;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> muonGeomToken_;
  const DTGeometry* geom;

  std::vector<std::string> m_types;

  std::map<uint32_t, DTTPGCompareUnit> m_compMapIn;
  std::map<uint32_t, DTTPGCompareUnit> m_compMapOut;
  std::map<uint32_t, std::map<std::string, MonitorElement*> > m_chamberHistos;
  std::map<uint32_t, DTTimeEvolutionHisto*> m_trendHistos;
  MonitorElement* m_nEventMonitor;

  edm::EDGetTokenT<L1MuDTChambPhContainer> m_tm_phiIn_Token;
  edm::EDGetTokenT<L1MuDTChambPhContainer> m_tm_phiOut_Token;
  edm::EDGetTokenT<L1MuDTChambThContainer> m_tm_theta_Token;
  edm::EDGetTokenT<L1Phase2MuDTPhContainer> m_ab7_phi_Token;
};

#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
