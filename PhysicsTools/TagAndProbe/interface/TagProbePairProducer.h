#ifndef PhysicsTools_TagAndProbe_TagProbePairProducer_h
#define PhysicsTools_TagAndProbe_TagProbePairProducer_h
// -*- C++ -*-
//
// Package:     TagAndProbe
// Class  :     TagProbeProducer
// 
/**
 Description: <one line class summary>

 Usage:
    <usage>
*/
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

// forward declarations

class TagProbePairProducer : public edm::EDProducer 
{
   public:
      explicit TagProbePairProducer(const edm::ParameterSet&);
      ~TagProbePairProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      bool isPassingProbe (const unsigned int iprobe) const;
      void checkPassingProbes() const;

      // ----------member data ---------------------------
      
      edm::InputTag probeCollection_;
      edm::InputTag passingProbeCollection_;

      edm::Handle< reco::CandidateView > probes;
      edm::Handle< reco::CandidateView > passingProbes;
};

#endif
