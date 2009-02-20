#ifndef MuonSeedGenerator_SETPatternRecognition_h
#define MuonSeedGenerator_SETPatternRecognition_h

#include "RecoMuon/MuonSeedGenerator/src/MuonSeedVPatternRecognition.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"


class SETPatternRecognition: public MuonSeedVPatternRecognition
{
public:
  explicit SETPatternRecognition(const edm::ParameterSet & pset);
  virtual ~SETPatternRecognition() {}
  /** Output is a cluster, with possibly more than one hit per layer */
  virtual void produce(const edm::Event& event, const edm::EventSetup& eSetup,
                       std::vector<MuonRecHitContainer> & result);

  void setServiceProxy(MuonServiceProxy * service) {theService = service;}

private:
  bool useRPCs; 

  edm::InputTag DTRecSegmentLabel;
  edm::InputTag CSCRecSegmentLabel;
  edm::InputTag RPCRecSegmentLabel;

  MuonServiceProxy * theService;
};

#endif

