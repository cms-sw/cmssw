#ifndef CommonTools_PFCandProducer_PFPileUpAlgo_
#define CommonTools_PFCandProducer_PFPileUpAlgo_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class PFPileUpAlgo {
 public:


  typedef std::vector< edm::FwdPtr<reco::PFCandidate> >  PFCollection;

  PFPileUpAlgo():checkClosestZVertex_(true), verbose_(false) {;}
    
  PFPileUpAlgo( bool checkClosestZVertex, bool verbose=false):
    checkClosestZVertex_(checkClosestZVertex), verbose_(verbose) {;}

  ~PFPileUpAlgo(){;}

  // the last parameter is needed if you want to use the sourceCandidatePtr
  void process(const PFCollection & pfCandidates, 
	       const reco::VertexCollection & vertices)  ;

  inline void setVerbose(bool verbose) { verbose_ = verbose; }

  inline void setCheckClosestZVertex(bool val) { checkClosestZVertex_ = val;}

  const PFCollection & getPFCandidatesFromPU() const {return pfCandidatesFromPU_;}
  
  const PFCollection & getPFCandidatesFromVtx() const {return pfCandidatesFromVtx_;}

  int chargedHadronVertex(const reco::VertexCollection& vertices, 
			const reco::PFCandidate& pfcand ) const;


 private  :

  /// use the closest z vertex if a track is not in a vertex
  bool   checkClosestZVertex_;
  
  
  /// verbose ?
  bool   verbose_;

  PFCollection pfCandidatesFromVtx_;
  PFCollection pfCandidatesFromPU_;
  
};

#endif
