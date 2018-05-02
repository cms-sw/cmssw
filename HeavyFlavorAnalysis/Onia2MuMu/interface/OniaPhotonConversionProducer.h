/**
   \file
   Declaration of OniaPhotonConversionProducer

   \author Stefano Argiro
   \modifier Alberto Sanchez
*/

#ifndef __OniaPhotonConversionProducer_h_
#define __OniaPhotonConversionProducer_h_

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <vector>

/**
   Select photon conversions and produce a conversion candidate collection

 */

class OniaPhotonConversionProducer : public edm::stream::EDProducer<> {

 public:
  explicit OniaPhotonConversionProducer(const edm::ParameterSet& ps);
 
 private:

  void produce(edm::Event& event, const edm::EventSetup& esetup) override;
  void endStream() override;
  void removeDuplicates(reco::ConversionCollection&);
  bool checkTkVtxCompatibility(const reco::Conversion&, const reco::VertexCollection&);
  bool foundCompatibleInnerHits(const reco::HitPattern& hitPatA, const reco::HitPattern& hitPatB); 
  bool HighpuritySubset(const reco::Conversion&, const reco::VertexCollection&);
  pat::CompositeCandidate* makePhotonCandidate(const reco::Conversion&);
  reco::Candidate::LorentzVector convertVector(const math::XYZTLorentzVectorF&);
  int PackFlags(const reco::Conversion&, bool, bool , bool, bool, bool);
  const reco::PFCandidateCollection selectPFPhotons(const reco::PFCandidateCollection&);
  bool CheckPi0( const reco::Conversion&, const reco::PFCandidateCollection&, bool &);

  edm::EDGetTokenT<reco::ConversionCollection> convCollectionToken_;
  edm::EDGetTokenT<reco::VertexCollection> thePVsToken_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfCandidateCollectionToken_;

  bool        wantTkVtxCompatibility_;
  uint32_t    sigmaTkVtxComp_;
  bool        wantCompatibleInnerHits_;
  uint32_t    TkMinNumOfDOF_;
  bool        wantHighpurity_;
  double _vertexChi2ProbCut;
  double _trackchi2Cut;
  double _minDistanceOfApproachMinCut;
  double _minDistanceOfApproachMaxCut;
  bool pi0OnlineSwitch_;
// low and high window limits
  std::vector<double> pi0SmallWindow_;
  std::vector<double> pi0LargeWindow_;

  int convAlgo_;
  std::vector<int>   convQuality_;
  
  std::string convSelectionCuts_;
  std::unique_ptr<StringCutObjectSelector<reco::Conversion>> convSelection_;
};

#endif
