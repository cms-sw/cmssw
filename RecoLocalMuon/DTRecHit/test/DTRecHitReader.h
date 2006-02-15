#ifndef RecoLocalMuon_DTRecHitReader_H
#define RecoLocalMuon_DTRecHitReader_H

/** \class DTRecHitReader
 *  Basic analyzer class which accesses 1D DTRecHits
 *  and plot resolution comparing them with muon simhits
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Handle.h"

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
}

class PSimHit;
class TFile;

class DTRecHitReader : public edm::EDAnalyzer {
public:
  /// Constructor
  DTRecHitReader(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTRecHitReader();

  // Operations

  /// Perform the real analysis
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);


protected:

private: 
  // Select the mu simhit closest to the rechit
  const PSimHit* findBestMuSimHit(const std::vector<const PSimHit*>& simhits,
				  float recHitDistFromWire);

  // Map simhits per wireId
  std::map<DTWireId,  std::vector<const PSimHit*> > 
  mapSimHitsPerWire(const edm::Handle<edm::PSimHitContainer >& simhits);

  // Histograms
  H1DRecHit *hRHitPhi;
  H1DRecHit *hRHitZ_W0;
  H1DRecHit *hRHitZ_W1;
  H1DRecHit *hRHitZ_W2;
  H1DRecHit *hRHitZ_All;

  // The file which will store the histos
  TFile *theFile;
  // Switch for debug output
  bool debug;
  // Root file name
  std::string rootFileName;
};


#endif




