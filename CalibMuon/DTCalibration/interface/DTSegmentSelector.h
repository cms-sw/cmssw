#ifndef CalibMuon_DTCalibration_DTSegmentSelector_h
#define CalibMuon_DTCalibration_DTSegmentSelector_h

/*
 *  \author A. Vilela Pereira
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include <vector>

class DTRecSegment4D;
class DTRecHit1D;
class DTStatusFlag;

class DTSegmentSelector {
public:
  DTSegmentSelector(edm::ParameterSet const& pset, edm::ConsumesCollector& iC);
  ~DTSegmentSelector() {}
  bool operator()(DTRecSegment4D const&, edm::Event const&, edm::EventSetup const&);

private:
  bool checkNoisySegment(edm::ESHandle<DTStatusFlag> const&, std::vector<DTRecHit1D> const&);

  edm::InputTag muonTags_;
  edm::EDGetTokenT<reco::MuonCollection> muonToken_;
  bool checkNoisyChannels_;
  int minHitsPhi_;
  int minHitsZ_;
  double maxChi2_;
  double maxAnglePhi_;
  double maxAngleZ_;
};

#endif
