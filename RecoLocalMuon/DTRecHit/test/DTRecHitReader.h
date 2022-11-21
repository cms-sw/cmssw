#ifndef RecoLocalMuon_DTRecHitReader_H
#define RecoLocalMuon_DTRecHitReader_H

/** \class DTRecHitReader
 *  Basic analyzer class which accesses 1D DTRecHits
 *  and plot resolution comparing them with muon simhits
 *
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/MuonDetId/interface/DTWireId.h"

#include "DTRecHitHistograms.h"

#include <vector>
#include <map>
#include <string>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class PSimHit;
class TFile;
class DTLayer;
class DTWireId;

class DTGeometry;
class MuonGeometryRecord;

class DTRecHitReader : public edm::one::EDAnalyzer<> {
public:
  /// Constructor
  DTRecHitReader(const edm::ParameterSet& pset);

  /// Destructor
  ~DTRecHitReader() override;

  // Operations

  /// Perform the real analysis
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;

protected:
private:
  // Select the mu simhit closest to the rechit
  const PSimHit* findBestMuSimHit(const DTLayer* layer,
                                  const DTWireId& wireId,
                                  const std::vector<const PSimHit*>& simhits,
                                  float recHitDistFromWire);

  // Map simhits per wireId
  std::map<DTWireId, std::vector<const PSimHit*> > mapSimHitsPerWire(const edm::Handle<edm::PSimHitContainer>& simhits);

  // Compute SimHit distance from wire
  double findSimHitDist(const DTLayer* layer, const DTWireId& wireId, const PSimHit* hit);

  // Histograms
  H1DRecHit* hRHitPhi;
  H1DRecHit* hRHitZ_W0;
  H1DRecHit* hRHitZ_W1;
  H1DRecHit* hRHitZ_W2;
  H1DRecHit* hRHitZ_All;

  // The file which will store the histos
  TFile* theFile;
  // Switch for debug output
  bool debug;
  // Root file name
  std::string rootFileName;
  std::string simHitLabel;
  std::string recHitLabel;

  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomToken_;
};

#endif
