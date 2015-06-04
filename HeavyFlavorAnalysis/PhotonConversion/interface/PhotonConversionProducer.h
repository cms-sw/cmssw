/**
   \file
   Declaration of PhotonConversionProducer

   \author Stefano Argiro
   \modifier Alberto Sanchez
*/

#ifndef __PhotonConversionProducer_h_
#define __PhotonConversionProducer_h_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"


#include <vector>

/**
   Select photon conversions and produce a conversion candidate collection

 */

class PhotonConversionProducer : public edm::EDProducer {

 public:
  explicit PhotonConversionProducer(const edm::ParameterSet& ps);
 
 private:

  virtual void produce(edm::Event& event, const edm::EventSetup& esetup);
  virtual void endJob() ;
  void removeDuplicates(reco::ConversionCollection&, std::vector<int> &);
  bool checkTkVtxCompatibility(const reco::Conversion&, const reco::VertexCollection&);
  bool foundCompatibleInnerHits(const reco::HitPattern& hitPatA, const reco::HitPattern& hitPatB); 
  bool HighpuritySubset(const reco::Conversion&, const reco::VertexCollection&);
  pat::CompositeCandidate* makePhotonCandidate(const reco::Conversion&);
  reco::Candidate::LorentzVector convertVector(const math::XYZTLorentzVectorF&);
  int PackFlags(const reco::Conversion&, bool, bool , bool, bool, bool);
  const reco::PFCandidateCollection selectPFPhotons(const reco::PFCandidateCollection&);
  bool CheckPi0( const reco::Conversion&, const reco::PFCandidateCollection&, bool &);

  edm::InputTag convCollection_;
  edm::InputTag pfPhotonCollection_;
  edm::InputTag thePVs_;
  edm::InputTag pfCandidateCollection_;
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
  
  int total_conversions;
  int selection_fail;
  int algo_fail;
  int flag_fail;
  int pizero_fail;
  int duplicates;
  int TkVtxC;
  int CInnerHits;
  int highpurity_count;
  int final_conversion;
  int store_conversion;

  std::string convSelectionCuts_;

};

#endif
