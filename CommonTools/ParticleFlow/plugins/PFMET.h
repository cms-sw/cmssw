#ifndef PhysicsTools_PFCandProducer_PFMET_
#define PhysicsTools_PFCandProducer_PFMET_

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CommonTools/ParticleFlow/interface/PFMETAlgo.h"

/**\class PFMET
\brief Computes the MET from a collection of PFCandidates. HF missing!

\todo Add HF energy to the MET calculation (access HF towers)

\author Colin Bernet
\date   february 2008
*/




class PFMET : public edm::EDProducer {
 public:

  explicit PFMET(const edm::ParameterSet&);

  ~PFMET();

  virtual void produce(edm::Event&, const edm::EventSetup&);

  virtual void beginJob();

 private:

  /// Input PFCandidates
  edm::InputTag       inputTagPFCandidates_;
  edm::EDGetTokenT<reco::PFCandidateCollection> tokenPFCandidates_;

  pf2pat::PFMETAlgo   pfMETAlgo_;
};

#endif
