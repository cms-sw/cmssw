#ifndef PhysicsTools_TagAndProbe_TagProbeEDMNtuple_h
#define PhysicsTools_TagAndProbe_TagProbeEDMNtuple_h
// -*- C++ -*-
//
// Package:     TagAndProbe
// Class  :     TagProbeEDMNtuples
// 
/**\class TagProbeEDMNtuples TagProbeEDMNtuple.h PhysicsTools/TagAndProbe/interface/TagProbeEDMNtuple.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Mon May  5 09:05:35 CDT 2008
// $Id: TagProbeEDMNtuple.h,v 1.8 2009/06/22 21:39:17 ahunt Exp $
//
// Kalanand Mishra: October 7, 2008 
// Added vertex information of the tag & probe candidates in edm::TTree

//
// class decleration
//

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"  // reco::CandidateView
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"  // reco::GenParticleRef

class TagProbeEDMNtuple : public edm::EDProducer 
{
   public:
      typedef edm::Ref< reco::CandidateView > CandViewRef;

      explicit TagProbeEDMNtuple(const edm::ParameterSet&);
      ~TagProbeEDMNtuple();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // Functions to fill the varios event data sets
      void fillRunEventInfo();
      void fillTriggerInfo();
      void fillMCInfo();
      void fillTrackInfo();
      void fillTagProbeInfo();
      void fillTrueEffInfo();

      bool CandFromZ( reco::GenParticleRef mcRef );

      int ProbePassProbeOverlap( const reco::CandidateBaseRef& probe, 
				 edm::Handle<reco::CandidateView>& passprobes );

      bool MatchObjects( const reco::Candidate *hltObj, 
			 const reco::CandidateBaseRef& tagObj,
			 bool exact = true );

      int getBestProbe(int ptype, const reco::CandidateBaseRef &tag, std::vector< std::pair<reco::CandidateBaseRef,bool> > vprobes);
      
      // ----------member data ---------------------------
      edm::Event* m_event;
      const edm::EventSetup* m_setup;

      // Type of Cands (used for matching and PDGId)
      std::string candType_;

      // PDG id of Cands
      int candPDGId_;

      // MC particles to store
      std::vector<int> mcParticles_;
      std::vector<int> mcParents_;
  
      // Track Collection Tags
      std::vector<edm::InputTag> trackTags_;

      // Tag probe map tags
      std::vector<edm::InputTag> tagProbeMapTags_;
      
      // Jet Collection Tags
      std::string jetTags_;
      
      // Candidate collection tags
      edm::InputTag genParticlesTag_;
      std::vector<edm::InputTag> tagCandTags_;
      std::vector<edm::InputTag> allProbeCandTags_;
      std::vector<edm::InputTag> passProbeCandTags_;

      // Truth matching
      std::vector<edm::InputTag> tagTruthMatchMapTags_;
      std::vector<edm::InputTag> allProbeTruthMatchMapTags_;
      std::vector<edm::InputTag> passProbeTruthMatchMapTags_;

      // Trigger parameters
      edm::InputTag triggerEventTag_;
      edm::InputTag hltTag_;

      // Matching parameters
      double delRMatchingCut_;
      double delPtRelMatchingCut_;
      
      // MC parameter
      bool isMC_;      

      // whether to use exact Delta_R matching in ProbePassProbeOverlap 
      bool checkExactOverlap_;

      // Vectors of strings on the picking of the "best" probe
      std::vector<std::string> bestProbeCriteria_;
      std::vector<double> bestProbeInvMass_;
};

#endif
