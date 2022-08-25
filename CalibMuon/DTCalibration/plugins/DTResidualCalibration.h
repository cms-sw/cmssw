#ifndef CalibMuon_DTCalibration_DTResidualCalibration_h
#define CalibMuon_DTCalibration_DTResidualCalibration_h

/** \class DTResidualCalibration
 *  Extracts DT segment residuals
 *
 */

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibMuon/DTCalibration/interface/DTSegmentSelector.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

#include <string>
#include <vector>
#include <map>

class TFile;
class TH1F;
class TH2F;
class DTGeometry;
class DTSuperLayerId;
class DTLayerId;

class DTResidualCalibration : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  /// Constructor
  DTResidualCalibration(const edm::ParameterSet& pset);
  /// Destructor
  ~DTResidualCalibration() override;

  void beginJob() override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endJob() override;
  void endRun(const edm::Run&, const edm::EventSetup&) override{};
  void analyze(const edm::Event& event, const edm::EventSetup& setup) override;

protected:
private:
  unsigned int nevent;
  unsigned int segmok, segmbad;

  float segmentToWireDistance(const DTRecHit1D& recHit1D, const DTRecSegment4D& segment);
  // Book a set of histograms for a given super-layer/layer
  void bookHistos(DTSuperLayerId slId);
  void bookHistos(DTLayerId slId);
  // Fill a set of histograms for a given super-layer/layer
  void fillHistos(DTSuperLayerId slId, float distance, float residualOnDistance);
  void fillHistos(DTLayerId slId, float distance, float residualOnDistance);

  DTSegmentSelector* select_;
  const double histRange_;
  const edm::EDGetTokenT<DTRecSegment4DCollection> segment4DToken_;
  const std::string rootBaseDir_;

  const bool detailedAnalysis_;
  TFile* rootFile_;
  // Geometry
  const DTGeometry* dtGeom_;
  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomToken_;

  // Histograms per super-layer
  std::map<DTSuperLayerId, TH1F*> histoMapTH1F_;
  std::map<DTSuperLayerId, TH2F*> histoMapTH2F_;
  // Histograms per layer
  std::map<DTLayerId, TH1F*> histoMapPerLayerTH1F_;
  std::map<DTLayerId, TH2F*> histoMapPerLayerTH2F_;
};
#endif
