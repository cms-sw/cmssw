#ifndef CalibMuon_DTCalibration_DTVDriftSegmentCalibration_h
#define CalibMuon_DTCalibration_DTVDriftSegmentCalibration_h

/** \class DTVDriftSegmentCalibration
 *  Produces histograms from v-drift computation in
 *  segment fit to be used for v-drift calibration
 *
 *  \author A. Vilela Pereira
 */

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CalibMuon/DTCalibration/interface/DTSegmentSelector.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include <map>

class DTChamberId;
class TFile;
class TH1F;
class TH2F;

class DTVDriftSegmentCalibration : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  // Constructor
  DTVDriftSegmentCalibration(const edm::ParameterSet& pset);
  // Destructor
  ~DTVDriftSegmentCalibration() override;

  void beginJob() override;
  void beginRun(const edm::Run& run, const edm::EventSetup& setup) override;
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;
  void endRun(const edm::Run& run, const edm::EventSetup& setup) override{};
  void endJob() override;

private:
  typedef std::map<DTChamberId, std::vector<TH1F*> > ChamberHistosMapTH1F;
  typedef std::map<DTChamberId, std::vector<TH2F*> > ChamberHistosMapTH2F;
  void bookHistos(DTChamberId);

  DTSegmentSelector* select_;

  const edm::EDGetTokenT<DTRecSegment4DCollection> theRecHits4DToken_;
  //bool writeVDriftDB_;
  const std::string theCalibChamber_;

  TFile* rootFile_;
  ChamberHistosMapTH1F theVDriftHistoMapTH1F_;
  ChamberHistosMapTH2F theVDriftHistoMapTH2F_;

  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomToken_;
};
#endif
