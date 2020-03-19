#ifndef DTLocalTriggerLutTask_H
#define DTLocalTriggerLutTask_H

/*
 * \file DTLocalTriggerLutTask.h
 *
 * \author D. Fasanella - INFN Bologna
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

#include <DQMServices/Core/interface/DQMOneEDAnalyzer.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"

#include <vector>
#include <string>
#include <map>
#include <array>

class DTGeometry;
class DTTrigGeomUtils;
class DTChamberId;
class L1MuDTChambPhDigi;

typedef std::array<std::array<std::array<int, 13>, 5>, 6> DTArr3int;
typedef std::array<std::array<std::array<int, 15>, 5>, 6> DTArr3bool;
typedef std::array<std::array<std::array<const L1MuDTChambPhDigi*, 15>, 5>, 6> DTArr3Digi;

class DTLocalTriggerLutTask : public DQMOneEDAnalyzer<edm::one::WatchLuminosityBlocks> {
  friend class DTMonitorModule;

public:
  /// Constructor
  DTLocalTriggerLutTask(const edm::ParameterSet& ps);

  /// Destructor
  ~DTLocalTriggerLutTask() override;

  /// bookHistograms
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

protected:
  ///BeginRun
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;

  /// Find best (highest qual) TM trigger segments
  void searchTMBestIn(std::vector<L1MuDTChambPhDigi> const* trigs);
  void searchTMBestOut(std::vector<L1MuDTChambPhDigi> const* trigs);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  /// To reset the MEs
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) override;
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) final {}

  const int wheelArrayShift = 3;

private:
  /// Get the top folder
  std::string& topFolder() { return baseFolder; }

  /// Book histos
  void bookHistos(DQMStore::IBooker& ibooker, DTChamberId chId);

private:
  int nEvents;
  int nLumis;
  int nPhiBins, nPhibBins;
  double rangePhi, rangePhiB;

  std::string baseFolder;
  bool detailedAnalysis;
  bool overUnderIn;

  edm::EDGetTokenT<L1MuDTChambPhContainer> tm_TokenIn_;
  edm::EDGetTokenT<L1MuDTChambPhContainer> tm_TokenOut_;
  edm::EDGetTokenT<DTRecSegment4DCollection> seg_Token_;

  DTArr3int trigQualBestIn;
  DTArr3int trigQualBestOut;
  DTArr3Digi trigBestIn;
  DTArr3Digi trigBestOut;
  DTArr3bool track_ok;  // CB controlla se serve

  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;
  std::string theGeomLabel;
  DTTrigGeomUtils* trigGeomUtils;

  std::map<uint32_t, std::map<std::string, MonitorElement*> > chHistos;
  std::map<int, std::map<std::string, MonitorElement*> > whHistos;
};

#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
