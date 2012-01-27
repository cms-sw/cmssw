#ifndef PFTauVertexSelector_H
#define PFTauVertexSelector_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Error.h"

/* 
 * class PFTauVertexSelector
 * created : January 26 2012,
 * revised : Wed Jan 26 11:13:04 PDT 2012
 * Authors : Andreas Hinzmann (CERN)
 */

class PFTauVertexSelector : public edm::EDFilter  {
   public:
      explicit PFTauVertexSelector(const edm::ParameterSet& iConfig){   
         tauSrc_ = iConfig.getParameter<edm::InputTag>("tauSrc");
         vertexSrc_ = iConfig.getParameter<edm::InputTag>("vertexSrc");
         useLeadingTrack_ = iConfig.getParameter<bool>("useLeadingTrack");
         trackSrc_ = iConfig.getParameter<edm::InputTag>("trackSrc");
         useLeadingRecoCandidate_ = iConfig.getParameter<bool>("useLeadingRecoCandidate");
         recoCandidateSrc_ = iConfig.getParameter<edm::InputTag>("recoCandidateSrc");
         dZ_ = iConfig.getParameter<double>("dZ");
         filterOnNTaus_ = iConfig.getParameter<uint32_t>("filterOnNTaus");
         produces<reco::PFTauCollection>();
      }
      ~PFTauVertexSelector(){} 
   private:
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      edm::InputTag tauSrc_;
      edm::InputTag vertexSrc_;
      bool useLeadingTrack_;
      edm::InputTag trackSrc_;
      bool useLeadingRecoCandidate_;
      edm::InputTag recoCandidateSrc_;
      double dZ_;
      uint32_t filterOnNTaus_;
};

#endif