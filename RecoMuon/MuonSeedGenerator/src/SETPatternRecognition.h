#ifndef MuonSeedGenerator_SETPatternRecognition_h
#define MuonSeedGenerator_SETPatternRecognition_h

#include "RecoMuon/MuonSeedGenerator/src/MuonSeedVPatternRecognition.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"            
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"             
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"                


class SETPatternRecognition: public MuonSeedVPatternRecognition
{
public:
  explicit SETPatternRecognition(const edm::ParameterSet & pset, edm::ConsumesCollector& iC);
  ~SETPatternRecognition() override {}
  /** Output is a cluster, with possibly more than one hit per layer */
  void produce(const edm::Event& event, const edm::EventSetup& eSetup,
                       std::vector<MuonRecHitContainer> & result) override;

  void setServiceProxy(MuonServiceProxy * service) {theService = service;}
  // don't use "bad" segments
  bool segmentCleaning(const DetId & detId, 
		       const LocalPoint& localPosition, const LocalError& localError,
		       const LocalVector& localDirection, const LocalError& localDirectionError,
		       const double& chi2, const int& ndf);

private:
  int maxActiveChambers;
  bool useRPCs; 

  edm::InputTag DTRecSegmentLabel;
  edm::InputTag CSCRecSegmentLabel;
  edm::InputTag RPCRecSegmentLabel;

  edm::EDGetTokenT<DTRecSegment4DCollection> dtToken;
  edm::EDGetTokenT<CSCSegmentCollection> cscToken;
  edm::EDGetTokenT<RPCRecHitCollection> rpcToken;


  double outsideChamberErrorScale; 
  double minLocalSegmentAngle; 
  //----

  MuonServiceProxy * theService;


};

#endif

