#ifndef RecoLocalMuon_RPCRecHitReader_H
#define RecoLocalMuon_RPCRecHitReader_H

/** \class RPCRecHitReader
 *  Basic analyzer class which accesses 2D CSCRecHits
 *  and plot resolution comparing them with muon simhits
 *
 *  Author: D. Fortin  - UC Riverside
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Handle.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include <Geometry/RPCGeometry/interface/RPCRoll.h>

#include <vector>
#include <map>
#include <string>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

//class PSimHit;
//class RPCDetId;

class RPCRecHitReader : public edm::EDAnalyzer {
public:
  /// Constructor
  RPCRecHitReader(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~RPCRecHitReader();

  // Operations

  /// Perform the real analysis
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);


protected:

private: 

  std::string simHitLabel1;
  std::string simHitLabel2;
  std::string recHitLabel1;
  std::string recHitLabel2;
  std::string digiLabel;

};


#endif
