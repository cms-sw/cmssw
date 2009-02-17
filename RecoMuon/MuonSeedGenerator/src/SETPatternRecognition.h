#ifndef MuonSeedGenerator_SETPatternRecognition_h
#define MuonSeedGenerator_SETPatternRecognition_h

#include "RecoMuon/MuonSeedGenerator/src/MuonSeedVPatternRecognition.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"


class SETPatternRecognition: public MuonSeedVPatternRecognition
{
public:
  explicit SETPatternRecognition(const edm::ParameterSet & pset);

  //FIXME deal with ServiceProxy destrction
  virtual void produce(const edm::Event& event, const edm::EventSetup& eSetup,
                       std::vector<MuonRecHitContainer> & result);

private:
  bool useSegmentsInTrajectory;
  bool useRPCs;

  edm::InputTag DTRecSegmentLabel;
  edm::InputTag CSCRecSegmentLabel;
  edm::InputTag RPCRecSegmentLabel;

  MuonServiceProxy *theService;

};

#endif

