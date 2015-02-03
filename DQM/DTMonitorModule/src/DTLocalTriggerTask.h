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
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

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


class DTLocalTriggerTask: public DQMEDAnalyzer{

  friend class DTMonitorModule;

 public:

  /// Constructor
  DTLocalTriggerTask(const edm::ParameterSet& ps );

  /// Destructor
  virtual ~DTLocalTriggerTask();

 protected:

  ///Beginrun
  void dqmBeginRun(const edm::Run& , const edm::EventSetup&);

  /// Book the histograms

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void bookHistos(DQMStore::IBooker &, const DTChamberId& dtCh, std::string folder, std::string histoTag );

  /// Book the histograms
  void bookWheelHistos(DQMStore::IBooker &, int wh, std::string histoTag );

  /// Book the histograms
  void bookBarrelHistos(DQMStore::IBooker &, std::string histoTag);

  /// Set Quality labels
  void setQLabels(MonitorElement* me, short int iaxis);

  /// Run analysis on DCC data
  void runDCCAnalysis(std::vector<L1MuDTChambPhDigi> const* phTrigs, std::vector<L1MuDTChambThDigi> const* thTrigs);

  /// Run analysis on ROS data
  void runDDUAnalysis(edm::Handle<DTLocalTriggerCollection>& trigsDDU);

  /// Run analysis using DT 4D segments
  void runSegmentAnalysis(edm::Handle<DTRecSegment4DCollection>& segments4D);

  /// Run analysis on ROS data
  void runDDUvsDCCAnalysis(std::string& trigsrc);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// To reset the MEs
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) ;

  /// Get the L1A source
  void triggerSource(const edm::Event& e);

  /// Get the Top folder (different between Physics and TP and DCC/DDU)
  std::string& topFolder(bool isDCC) { return isDCC ? baseFolderDCC : baseFolderDDU; }

 private:

  edm::EDGetTokenT<L1MuDTChambPhContainer> dcc_Token_;
  edm::EDGetTokenT<L1MuDTChambThContainer> dccTh_Token_;    // NEW (M.C Fouz July14) Needed, since at least version 710
  edm::EDGetTokenT<DTLocalTriggerCollection> ros_Token_;
  edm::EDGetTokenT<DTRecSegment4DCollection> seg_Token_;
  edm::EDGetTokenT<LTCDigiCollection> ltcDigiCollectionToken_;

  bool useDCC, useDDU, useSEG;
  std::string trigsrc;
  int nevents;
  bool tpMode;
  std::string baseFolderDCC;
  std::string baseFolderDDU;
  bool doDCCTheta;
  bool detailedAnalysis;


  int phcode_best[6][5][13];
  int dduphcode_best[6][5][13];
  int thcode_best[6][5][13];
  int dduthcode_best[6][5][13];
  int mapDTTF[6][13][2];
  const L1MuDTChambPhDigi* iphbest[6][5][13];
  const DTLocalTrigger*    iphbestddu[6][5][13];
  const L1MuDTChambThDigi* ithbest[6][5][13];
  bool track_ok[6][5][15];

  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;
  DTTrigGeomUtils* trigGeomUtils;
  std::map<uint32_t, std::map<std::string, MonitorElement*> > digiHistos;
  std::map<int, std::map<std::string, MonitorElement*> > wheelHistos;

  MonitorElement* dcc_IDDataErrorPlot;

  bool isLocalRun;
};

#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
