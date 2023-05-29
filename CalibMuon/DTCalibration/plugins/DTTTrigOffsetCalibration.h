#ifndef DTTTrigOffsetCalibration_H
#define DTTTrigOffsetCalibration_H

/** \class DTTTrigOffsetCalibration
 *  No description available.
 *
 *  \author A. Vilela Pereira
 */

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CalibMuon/DTCalibration/interface/DTSegmentSelector.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include <map>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class DTChamberId;
class DTTtrig;
class TFile;
class TH1F;

class DTTTrigOffsetCalibration : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  // Constructor
  DTTTrigOffsetCalibration(const edm::ParameterSet& pset);
  // Destructor
  ~DTTTrigOffsetCalibration() override;

  void beginRun(const edm::Run& run, const edm::EventSetup& setup) override;
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;
  void endRun(const edm::Run& run, const edm::EventSetup& setup) override{};
  void endJob() override;

private:
  typedef std::map<DTChamberId, std::vector<TH1F*> > ChamberHistosMap;
  void bookHistos(DTChamberId);

  DTSegmentSelector* select_;

  const edm::EDGetTokenT<DTRecSegment4DCollection> theRecHits4DToken_;
  const bool doTTrigCorrection_;
  const std::string theCalibChamber_;

  TFile* rootFile_;
  const DTTtrig* tTrigMap_;
  ChamberHistosMap theT0SegHistoMap_;

  const edm::ESGetToken<DTTtrig, DTTtrigRcd> ttrigToken_;
  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomToken_;
};
#endif
