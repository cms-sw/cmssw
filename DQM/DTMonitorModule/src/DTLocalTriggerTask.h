#ifndef DTLocalTriggerTask_H
#define DTLocalTriggerTask_H

/*
 * \file DTLocalTriggerTask.h
 *
 * \author M. Zanetti - INFN Padova
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
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <DQMServices/Core/interface/DQMOneEDAnalyzer.h>

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/LTCDigi/interface/LTCDigi.h"
#include "DataFormats/DTDigi/interface/DTLocalTriggerCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

#include <vector>
#include <string>
#include <map>

class DTGeometry;
class DTTrigGeomUtils;
class DTChamberId;
class DTRecSegment4D;
class DTLocalTrigger;
class L1MuDTChambPhDigi;
class L1MuDTChambThDigi;

typedef std::array<std::array<std::array<int, 13>, 5>, 6> DTArr3int;
typedef std::array<std::array<std::array<const L1MuDTChambPhDigi*, 15>, 5>, 6> DTArr3PhDigi;
typedef std::array<std::array<std::array<const L1MuDTChambThDigi*, 15>, 5>, 6> DTArr3ThDigi;
typedef std::array<std::array<std::array<const DTLocalTrigger*, 15>, 5>, 6> DTArr3LocalTrigger;
typedef std::array<std::array<std::array<int, 2>, 13>, 6> DTArr3mapInt;

class DTLocalTriggerTask : public DQMOneEDAnalyzer<edm::one::WatchLuminosityBlocks> {
  friend class DTMonitorModule;

public:
  /// Constructor
  DTLocalTriggerTask(const edm::ParameterSet& ps);

  /// Destructor
  ~DTLocalTriggerTask() override;

protected:
  ///Beginrun
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;

  /// Book the histograms

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void bookHistos(DQMStore::IBooker&, const DTChamberId& dtCh, std::string folder, std::string histoTag);

  /// Book the histograms
  void bookWheelHistos(DQMStore::IBooker&, int wh, std::string histoTag);

  /// Book the histograms
  void bookBarrelHistos(DQMStore::IBooker&, std::string histoTag);

  /// Set Quality labels
  void setQLabels(MonitorElement* me, short int iaxis);
  void setQLabelsTheta(MonitorElement* me, short int iaxis);

  /// Run analysis on TM data
  void runTMAnalysis(std::vector<L1MuDTChambPhDigi> const* phTrigs, std::vector<L1MuDTChambThDigi> const* thTrigs);

  /// Run analysis using DT 4D segments
  void runSegmentAnalysis(edm::Handle<DTRecSegment4DCollection>& segments4D);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  /// To reset the MEs
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) override;
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) final {}

  /// Get the L1A source
  void triggerSource(const edm::Event& e);

  /// Get the Top folder (different between Physics and TP and TM)
  std::string& topFolder() { return baseFolderTM; }

  const int wheelArrayShift = 3;

private:
  edm::EDGetTokenT<L1MuDTChambPhContainer> tm_Token_;
  edm::EDGetTokenT<L1MuDTChambThContainer> tmTh_Token_;
  edm::EDGetTokenT<DTLocalTriggerCollection> ros_Token_;
  edm::EDGetTokenT<DTRecSegment4DCollection> seg_Token_;
  edm::EDGetTokenT<LTCDigiCollection> ltcDigiCollectionToken_;

  bool useTM, useSEG;
  std::string trigsrc;
  int nevents;
  bool tpMode;
  std::string baseFolderTM;
  bool doTMTheta;
  bool detailedAnalysis;

  DTArr3int phcode_best;
  DTArr3int thcode_best;
  DTArr3mapInt mapDTTF;
  DTArr3PhDigi iphbest;
  DTArr3ThDigi ithbest;
  bool track_ok[6][5][15];

  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;
  DTTrigGeomUtils* trigGeomUtils;
  std::map<uint32_t, std::map<std::string, MonitorElement*> > digiHistos;
  std::map<int, std::map<std::string, MonitorElement*> > wheelHistos;

  MonitorElement* tm_IDDataErrorPlot;

  bool isLocalRun;
};

#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
