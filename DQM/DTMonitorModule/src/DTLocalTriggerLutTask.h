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

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"

#include <vector>
#include <string>
#include <map>

class DTGeometry;
class DTTrigGeomUtils;
class DTChamberId;
class L1MuDTChambPhDigi;


class DTLocalTriggerLutTask: public DQMEDAnalyzer{

  friend class DTMonitorModule;

 public:

  /// Constructor
  DTLocalTriggerLutTask(const edm::ParameterSet& ps );

  /// Destructor
  virtual ~DTLocalTriggerLutTask();

  /// bookHistograms
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

 protected:

  ///BeginRun
  void dqmBeginRun(const edm::Run& , const edm::EventSetup&);

  /// Find best (highest qual) DCC trigger segments
  void searchDccBest(std::vector<L1MuDTChambPhDigi> const* trigs);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// To reset the MEs
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) ;

 private:

  /// Get the top folder
  std::string& topFolder() { return  baseFolder; }

  /// Book histos
  void bookHistos(DQMStore::IBooker & ibooker,DTChamberId chId);

 private :

  int nEvents;
  int nLumis;
  int nPhiBins, nPhibBins;
  double rangePhi, rangePhiB;

  std::string baseFolder;
  bool detailedAnalysis;
  bool overUnderIn;

  edm::EDGetTokenT<L1MuDTChambPhContainer> dcc_Token_;
  edm::EDGetTokenT<DTRecSegment4DCollection> seg_Token_;

  int trigQualBest[6][5][13];
  const L1MuDTChambPhDigi* trigBest[6][5][13];
  bool track_ok[6][5][15]; // CB controlla se serve

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
