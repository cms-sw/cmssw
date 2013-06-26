#ifndef CalibMuon_DTCalibration_DTResidualCalibration_h
#define CalibMuon_DTCalibration_DTResidualCalibration_h

/** \class DTResidualCalibration
 *  Extracts DT segment residuals
 *
 *  $Date: 2012/02/01 17:25:58 $
 *  $Revision: 1.2 $
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibMuon/DTCalibration/interface/DTSegmentSelector.h"

#include <string>
#include <vector>
#include <map>

class TFile;
class TH1F;
class TH2F;
class DTGeometry;
class DTSuperLayerId;
class DTLayerId;

class DTResidualCalibration: public edm::EDAnalyzer{
 public:
  /// Constructor
  DTResidualCalibration(const edm::ParameterSet& pset);
  /// Destructor
  virtual ~DTResidualCalibration();

  void beginJob();
  void beginRun(const edm::Run&, const edm::EventSetup&);
  void endJob();
  void analyze(const edm::Event& event, const edm::EventSetup& setup);

 protected:

 private:
  float segmentToWireDistance(const DTRecHit1D& recHit1D, const DTRecSegment4D& segment); 
  // Book a set of histograms for a given super-layer/layer
  void bookHistos(DTSuperLayerId slId);
  void bookHistos(DTLayerId slId);
  // Fill a set of histograms for a given super-layer/layer 
  void fillHistos(DTSuperLayerId slId, float distance, float residualOnDistance);
  void fillHistos(DTLayerId slId, float distance, float residualOnDistance);

  DTSegmentSelector select_;
  edm::InputTag segment4DLabel_;
  std::string rootBaseDir_;

  bool detailedAnalysis_;
  TFile* rootFile_;
  // Geometry
  const DTGeometry* dtGeom_;
  // Histograms per super-layer
  std::map<DTSuperLayerId, std::vector<TH1F*> > histoMapTH1F_;
  std::map<DTSuperLayerId, std::vector<TH2F*> > histoMapTH2F_;
  // Histograms per layer
  std::map<DTLayerId, std::vector<TH1F*> > histoMapPerLayerTH1F_;
  std::map<DTLayerId, std::vector<TH2F*> > histoMapPerLayerTH2F_;
};
#endif
