#ifndef DTnoiseDBValidation_H
#define DTnoiseDBValidation_H

/** \class DTnoiseDBValidation
 *  Plot the noise from the DB comparaison
 *
 *  \author G. Mila - INFN Torino
 */

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include <map>
#include <string>
#include <vector>

class DTGeometry;
class DTChamberId;
class DTStatusFlag;
class TFile;

class DTnoiseDBValidation : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchRuns> {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;
  /// Constructor
  DTnoiseDBValidation(const edm::ParameterSet &pset);

  /// Destructor
  ~DTnoiseDBValidation() override;

  /// Operations
  void beginRun(const edm::Run &run, const edm::EventSetup &setup) override;
  void endRun(edm::Run const &, edm::EventSetup const &) override;
  void endJob() override;
  void analyze(const edm::Event &event, const edm::EventSetup &setup) override {}

protected:
private:
  void bookHisto(const DTChamberId &);

  DQMStore *dbe_;
  // The DB label
  edm::ESGetToken<DTStatusFlag, DTStatusFlagRcd> labelDBRef_;
  edm::ESGetToken<DTStatusFlag, DTStatusFlagRcd> labelDB_;
  const DTStatusFlag *noiseRefMap;
  const DTStatusFlag *noiseMap;
  std::string diffTestName_, wheelTestName_, stationTestName_, sectorTestName_, layerTestName_;

  bool outputMEsInRootFile_;
  std::string outputFileName_;

  // The DTGeometry
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> muonGeomToken_;
  const DTGeometry *dtGeom;

  // The noise map
  const DTStatusFlag *noiseMap_;
  const DTStatusFlag *noiseRefMap_;

  // the total number of noisy cell
  int noisyCellsRef_;
  int noisyCellsValid_;
  // the histos
  MonitorElement *diffHisto_;
  MonitorElement *wheelHisto_;
  MonitorElement *stationHisto_;
  MonitorElement *sectorHisto_;
  MonitorElement *layerHisto_;
  std::map<DTChamberId, MonitorElement *> noiseHistoMap_;
};
#endif
