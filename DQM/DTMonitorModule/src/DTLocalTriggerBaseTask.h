#ifndef DTLocalTriggerBaseTask_H
#define DTLocalTriggerBaseTask_H

/*
 * \file DTLocalTriggerBaseTask.h
 *
 * \author C. Battilana - CIEMAT
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <DQMServices/Core/interface/DQMOneEDAnalyzer.h>

#include "DataFormats/DTDigi/interface/DTLocalTriggerCollection.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

#include <vector>
#include <string>
#include <map>

class DTGeometry;
class DTTrigGeomUtils;
class DTChamberId;
class DTRecSegment4D;
class L1MuDTChambPhDigi;
class L1MuDTChambThDigi;
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

  /// Get the Top folder (different between Physics and TP and TM)
  std::string& topFolder(std::string const& type) { return baseFolder[type == "TM"]; }

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  /// Book the histograms
  void bookHistos(DQMStore::IBooker&, const DTChamberId& chamb);

  /// Book the histograms
  void bookHistos(DQMStore::IBooker&, int wh);

  /// Set Quality labels
  void setQLabels(MonitorElement* me, short int iaxis);
  void setQLabelsTheta(MonitorElement* me, short int iaxis);

  int nEvents;
  int nEventsInLS;
  int nLumis;

  std::string baseFolder[2];
  bool tpMode;
  bool detailedAnalysis;
  bool processTM;

  int targetBXTM;
  int bestAccRange;

  edm::ParameterSet theParams;
  DTTrigGeomUtils* theTrigGeomUtils;
  std::vector<std::string> theTypes;

  std::map<uint32_t, DTTPGCompareUnit> theCompMapIn;
  std::map<uint32_t, DTTPGCompareUnit> theCompMapOut;
  std::map<int, std::map<std::string, MonitorElement*> > wheelHistos;
  std::map<uint32_t, std::map<std::string, MonitorElement*> > chamberHistos;
  std::map<uint32_t, DTTimeEvolutionHisto*> trendHistos;
  MonitorElement* nEventMonitor;

  edm::EDGetTokenT<L1MuDTChambPhContainer> tm_phiIn_Token_;
  edm::EDGetTokenT<L1MuDTChambPhContainer> tm_phiOut_Token_;
  edm::EDGetTokenT<L1MuDTChambThContainer> tm_theta_Token_;
  edm::EDGetTokenT<DTLocalTriggerCollection> trig_Token_;
};

#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
