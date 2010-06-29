#ifndef PhysicsTools_TagAndProbe_TagProbeProducer_h
#define PhysicsTools_TagAndProbe_TagProbeProducer_h
// -*- C++ -*-
//
// Package:     TagAndProbe
// Class  :     TagProbeProducer
// 
/**\class TagProbeProducer TagProbeProducer.h PhysicsTools/TagAndProbe/interface/TagProbeProducer.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Wed Apr 16 10:08:13 CDT 2008
// $Id: TagProbeProducer.h,v 1.3 2009/03/24 19:32:37 ahunt Exp $
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

// forward declarations

class TagProbeProducer : public edm::EDProducer 
{
   public:
      explicit TagProbeProducer(const edm::ParameterSet&);
      ~TagProbeProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      bool isPassingProbe (const unsigned int iprobe) const;

      // ----------member data ---------------------------
      
      edm::InputTag tagCollection_;
      edm::InputTag probeCollection_;
      edm::InputTag passingProbeCollection_;

      edm::Handle< reco::CandidateView > tags;
      edm::Handle< reco::CandidateView > probes;
      edm::Handle< reco::CandidateView > passingProbes;

      double massMinCut_;
      double massMaxCut_;
      double delRMinCut_;
      double delRMaxCut_;

      bool requireOS_;
};

#endif
