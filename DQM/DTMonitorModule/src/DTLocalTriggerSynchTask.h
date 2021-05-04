#ifndef DTLocalTriggerSynchTask_H
#define DTLocalTriggerSynchTask_H

/*
 * \file DTLocalTriggerSynchTask.h
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

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "DataFormats/DTDigi/interface/DTLocalTriggerCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

// DT trigger
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

#include <vector>
#include <string>
#include <map>

class DTGeometry;
class DTChamberId;
class DTRecSegment4D;
class DTTTrigBaseSync;
class DTLocalTrigger;
class L1MuDTChambPhDigi;
class L1MuDTChambThDigi;

typedef std::array<std::array<std::array<int, 13>, 5>, 6> DTArr3int;
typedef std::array<std::array<std::array<std::array<int, 3>, 13>, 5>, 6> DTArr4int;

class DTLocalTriggerSynchTask : public DQMEDAnalyzer {
  friend class DTMonitorModule;

public:
  /// Constructor
  DTLocalTriggerSynchTask(const edm::ParameterSet& ps);

  /// Destructor
  ~DTLocalTriggerSynchTask() override;

protected:
  /// Book the histograms
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  ///Beginrun
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;

  /// Book the histograms
  void bookHistos(DQMStore::IBooker&, const DTChamberId& dtCh);

  /// Analyze
  void analyze(const edm::Event& event, const edm::EventSetup& context) override;

  std::string& baseDir() { return baseDirectory; }

  const int wheelArrayShift = 3;

private:
  int nevents;

  DTArr3int phCodeBestTM;
  DTArr4int phCodeBXTM;
  DTArr3int segHitBest;

  float bxTime;
  bool rangeInBX;
  int nBXLow;
  int nBXHigh;
  float angleRange;
  float minHitsPhi;
  int fineDelay;
  std::unique_ptr<DTTTrigBaseSync> tTrigSync;

  std::string baseDirectory;

  edm::ESHandle<DTGeometry> muonGeom;
  std::map<uint32_t, std::map<std::string, MonitorElement*> > triggerHistos;
  MonitorElement* tm_IDDataErrorPlot;

  edm::EDGetTokenT<L1MuDTChambPhContainer> tm_Token_;
  edm::EDGetTokenT<DTRecSegment4DCollection> seg_Token_;
};

#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
