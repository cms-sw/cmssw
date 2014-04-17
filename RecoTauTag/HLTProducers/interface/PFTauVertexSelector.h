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
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

/* 
 * class PFTauVertexSelector
 * created : January 26 2012,
 * revised : Wed Jan 26 11:13:04 PDT 2012
 * Authors : Andreas Hinzmann (CERN)
 */

class PFTauVertexSelector : public edm::EDFilter  {
   public:
      explicit PFTauVertexSelector(const edm::ParameterSet& iConfig){   
         tauSrc_ = consumes<edm::View<reco::PFTau> >(iConfig.getParameter<edm::InputTag>("tauSrc"));
         useVertex_ = iConfig.getParameter<bool>("useVertex");
         vertexSrc_ = consumes<edm::View<reco::Vertex> >(iConfig.getParameter<edm::InputTag>("vertexSrc"));
         useBeamSpot_ = iConfig.getParameter<bool>("useBeamSpot");
         beamSpotSrc_ = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotSrc"));
         useLeadingTrack_ = iConfig.getParameter<bool>("useLeadingTrack");
         trackSrcIT_ = iConfig.getParameter<std::vector<edm::InputTag> >("trackSrc");
	 for( std::vector<edm::InputTag>::const_iterator it = trackSrcIT_.begin(); it != trackSrcIT_.end(); ++it ) {
	   edm::EDGetTokenT<edm::View<reco::Track> > aToken =  consumes<edm::View<reco::Track> >( *it );
	   trackSrc_.push_back(aToken);
	 }
         useLeadingRecoCandidate_ = iConfig.getParameter<bool>("useLeadingRecoCandidate");
         recoCandidateSrcIT_ = iConfig.getParameter<std::vector<edm::InputTag> >("recoCandidateSrc");
	 for( std::vector<edm::InputTag>::const_iterator it =  recoCandidateSrcIT_.begin(); it != recoCandidateSrcIT_.end(); ++it ) {
	   edm::EDGetTokenT<edm::View<reco::RecoCandidate> > aToken =  consumes<edm::View<reco::RecoCandidate> >( *it );
	   recoCandidateSrc_.push_back(aToken);
	 }
         useTriggerFilterElectrons_ = iConfig.getParameter<bool>("useTriggerFilterElectrons");
         triggerFilterElectronsSrc_ = consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<edm::InputTag>("triggerFilterElectronsSrc"));
         useTriggerFilterMuons_ = iConfig.getParameter<bool>("useTriggerFilterMuons");
         triggerFilterMuonsSrc_ = consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<edm::InputTag>("triggerFilterMuonsSrc"));
         dZ_ = iConfig.getParameter<double>("dZ");
         filterOnNTaus_ = iConfig.getParameter<uint32_t>("filterOnNTaus");
         produces<reco::PFTauCollection>();
      }
      ~PFTauVertexSelector(){} 
   private:
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      edm::EDGetTokenT<edm::View<reco::PFTau> > tauSrc_;
      bool useVertex_;
      edm::EDGetTokenT<edm::View<reco::Vertex> > vertexSrc_;
      bool useBeamSpot_;
      edm::EDGetTokenT<reco::BeamSpot> beamSpotSrc_;
      bool useLeadingTrack_;
      std::vector<edm::InputTag> trackSrcIT_;
      std::vector<edm::EDGetTokenT<edm::View<reco::Track> > > trackSrc_;
      bool useLeadingRecoCandidate_;
      std::vector<edm::InputTag> recoCandidateSrcIT_;
      std::vector<edm::EDGetTokenT<edm::View<reco::RecoCandidate> > > recoCandidateSrc_;
      bool useTriggerFilterElectrons_;
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> triggerFilterElectronsSrc_;
      bool useTriggerFilterMuons_;
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> triggerFilterMuonsSrc_;
      double dZ_;
      uint32_t filterOnNTaus_;
};

#endif
