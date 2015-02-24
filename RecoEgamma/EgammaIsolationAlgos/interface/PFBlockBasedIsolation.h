//--------------------------------------------------------------------------------------------------
//
//  PfBlockBasedIsolationCalculator.cc
// Authors: N. Marinelli Univ. of Notre Dame
//--------------------------------------------------------------------------------------------------


#ifndef PFBlockBasedIsolation_H
#define PFBlockBasedIsolation_H


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidateFwd.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"


#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

namespace reco{
  class PFBlockElementCluster;
}

class PFBlockBasedIsolation{
 public:
  PFBlockBasedIsolation();


  ~PFBlockBasedIsolation();



  void setup(const edm::ParameterSet& conf);
  


 public:

  

    std::vector<reco::PFCandidateRef> calculate(math::XYZTLorentzVectorD p4,
		 const reco::PFCandidateRef pfEGCand,
		 const edm::Handle<reco::PFCandidateCollection> pfCandidateHandle);


private:  
  const reco::PFBlockElementCluster* getHighestEtECALCluster(const reco::PFCandidate& pfCand);
  bool passesCleaningPhoton(const  reco::PFCandidateRef& pfCand,const reco::PFCandidateRef& pfEGCand);
  bool passesCleaningNeutralHadron(const  reco::PFCandidateRef& pfCand,const reco::PFCandidateRef& pfEGCand);
  
  bool passesCleaningChargedHadron(const reco::PFCandidateRef& pfCand,const reco::PFCandidateRef& pfEGCand);
  bool elementPassesCleaning(const reco::PFCandidateRef& pfCand,const reco::PFCandidateRef& pfEGCand);
  
 private:

 double coneSize_;
     

};

#endif
