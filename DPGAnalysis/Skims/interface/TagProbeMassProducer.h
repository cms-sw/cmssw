#ifndef DPGAnalysis_Skim_TagProbeMassProducer_h
#define DPGAnalysis_Skim_TagProbeMassProducer_h
// -*- C++ -*-
//
// Package:     TagAndProbe
// Class  :     TagProbeMassProducer
// 
/**\class TagProbeMassProducer TagProbeMassProducer.h PhysicsTools/TagAndProbe/interface/TagProbeMassProducer.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Wed Apr 16 10:08:13 CDT 2008
// $Id: TagProbeMassProducer.h,v 1.2 2013/02/27 20:17:13 wmtan Exp $
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

class TagProbeMassProducer : public edm::EDProducer 
{
   public:
      explicit TagProbeMassProducer(const edm::ParameterSet&);
      ~TagProbeMassProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
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
