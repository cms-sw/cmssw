#ifndef MuonSeedGenerator_MuonSeedVPatternRecognition_h
#define MuonSeedGenerator_MuonSeedVPatternRecognition_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

class MuonSeedVPatternRecognition
{
public:
  typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
  typedef MuonTransientTrackingRecHit::ConstMuonRecHitPointer ConstMuonRecHitPointer;
  typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;

  explicit MuonSeedVPatternRecognition(const edm::ParameterSet & pset);
  virtual ~MuonSeedVPatternRecognition() {}

  virtual void produce(const edm::Event& event, const edm::EventSetup& eSetup,
                       std::vector<MuonRecHitContainer> & result) = 0;

protected:
  /// the name of the DT rec hits collection
  edm::InputTag theDTRecSegmentLabel;

  /// the name of the CSC rec hits collection
  edm::InputTag theCSCRecSegmentLabel;

  ///Enable the DT measurement
  bool enableDTMeasurement;

  ///Enable the CSC measurement
  bool enableCSCMeasurement;

};

#endif

