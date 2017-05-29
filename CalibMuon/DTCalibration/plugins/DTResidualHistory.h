#ifndef CalibMuon_DTCalibration_DTResidualHistory_h
#define CalibMuon_DTCalibration_DTResidualHistory_h

/** \class DTResidualHistory
 *  Extracts DT segment residual history vs run number
 *
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

class DTResidualHistory: public edm::EDAnalyzer{
 public:
  /// Constructor
  DTResidualHistory(const edm::ParameterSet& pset);
  /// Destructor
  virtual ~DTResidualHistory();

  void beginJob();
  void beginRun(const edm::Run&, const edm::EventSetup&);
  void endJob();
  void analyze(const edm::Event& event, const edm::EventSetup& setup);

 protected:

 private:
 
  unsigned int nevent;
  unsigned int segmok,segmbad;
   
  float segmentToWireDistance(const DTRecHit1D& recHit1D, const DTRecSegment4D& segment); 
  // Book a set of histograms for a given super-layer/layer
  void bookHistos(DTSuperLayerId slId, unsigned int run);
  void bookHistos(DTLayerId slId, unsigned int run);
  // Fill a set of histograms for a given super-layer/layer 
  void fillHistos(DTSuperLayerId slId, float residualOnDistance);
  void fillHistos(DTLayerId slId, float residualOnDistance);

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
  
  unsigned int lastrun;
  TH2F* histoResLs;
};
#endif
