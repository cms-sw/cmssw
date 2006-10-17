#ifndef RecoMuon_L2MuonIsolationProducer_H
#define RecoMuon_L2MuonIsolationProducer_H

/**  \class L2MuonIsolationProducer
 * 
 *   L2 HLT muon producer:
 *
 *   \author  J.Alcaraz
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"

#include "RecoMuon/MuonIsolation/src/CaloExtractor.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class L2MuonIsolationProducer : public edm::EDProducer {

 public:

  /// constructor with config
  L2MuonIsolationProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~L2MuonIsolationProducer(); 
  
  /// Produce isolation maps
  virtual void produce(edm::Event&, const edm::EventSetup&);
  // ex virtual void reconstruct();

 private:
  
  // Muon track Collection Label
  std::string theSACollectionLabel;

  // Isolation cuts
  std::vector<double> coneCuts_;
  std::vector<double> edepCuts_;
  std::vector<double> etaBounds_;
  double ecalWeight_;

  // MuIsoExtractor settings
  muonisolation::CaloExtractor theMuIsoExtractor;

};

#endif
