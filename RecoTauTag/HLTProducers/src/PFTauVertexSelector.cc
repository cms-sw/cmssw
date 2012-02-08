#include "RecoTauTag/HLTProducers/interface/PFTauVertexSelector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

/* 
 * class PFTauVertexSelector
 * created : January 26 2012,
 * revised : Wed Jan 26 11:13:04 PDT 2012
 * Authors : Andreas Hinzmann (CERN)
 */

bool PFTauVertexSelector::filter(edm::Event& event, const edm::EventSetup& eventSetup) {

   math::XYZPoint vertexPoint;
   bool vertexAvailable=false;
   
   if(useBeamSpot_)
   {
       edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
       event.getByLabel(beamSpotSrc_,recoBeamSpotHandle);
       if (recoBeamSpotHandle.isValid()){
         vertexPoint = recoBeamSpotHandle->position();
	 vertexAvailable = true;
       }
   }
 
   if(useVertex_)
   {
       edm::Handle<edm::View<reco::Vertex> > recoVertexHandle;
       event.getByLabel(vertexSrc_,recoVertexHandle);
       if ((recoVertexHandle.isValid()) && (recoVertexHandle->size()>0)){
         vertexPoint = recoVertexHandle->at(0).position();
	 vertexAvailable = true;
       }
   }
 
   const reco::Track* track=0;
   double maxpt=0.;
   
   if (useLeadingTrack_)
   {
       edm::Handle<edm::View<reco::Track> > tracks;
       for( std::vector<edm::InputTag>::const_iterator trackSrc = trackSrc_.begin(); trackSrc != trackSrc_.end(); ++ trackSrc ) {
           event.getByLabel(*trackSrc, tracks);
           if ((tracks.isValid())&&(tracks->size()>0)){
               for (unsigned i = 0; i < tracks->size(); ++i) {
                  double pt=tracks->ptrAt(i)->pt();
                  if(pt>maxpt)
                  {
                      track = &*tracks->ptrAt(i);
                      maxpt=pt;
                  }
               }
           }
       }
   }
   
   if (useLeadingRecoCandidate_)
   {
       edm::Handle<edm::View<reco::RecoCandidate> > recocandidates;
       for( std::vector<edm::InputTag>::const_iterator recoCandidateSrc = recoCandidateSrc_.begin(); recoCandidateSrc != recoCandidateSrc_.end(); ++ recoCandidateSrc ) {
           event.getByLabel(*recoCandidateSrc, recocandidates);
           if ((recocandidates.isValid())&&(recocandidates->size()>0)){
               for (unsigned i = 0; i < recocandidates->size(); ++i) {
                  double pt=recocandidates->ptrAt(i)->pt();
                  if(pt>maxpt)
                  {
                      track = dynamic_cast<const reco::Track*>(recocandidates->ptrAt(i)->bestTrack());
                      maxpt=pt;
                  }
               }
           }
       }
   }
   
   if (useTriggerFilterElectrons_)
   {
       edm::Handle<trigger::TriggerFilterObjectWithRefs> triggerfilter;
       event.getByLabel(triggerFilterElectronsSrc_, triggerfilter);
       std::vector<reco::ElectronRef> recocandidates;
       triggerfilter->getObjects(trigger::TriggerElectron,recocandidates);
       if ((recocandidates.size()>0)){
           for (unsigned i = 0; i < recocandidates.size(); ++i) {
              double pt=recocandidates.at(i)->pt();
              if(pt>maxpt)
              {
        	  track = dynamic_cast<const reco::Track*>(recocandidates.at(i)->bestTrack());
        	  maxpt=pt;
              }
           }
       }
   }
   
   if (useTriggerFilterMuons_)
   {
       edm::Handle<trigger::TriggerFilterObjectWithRefs> triggerfilter;
       event.getByLabel(triggerFilterMuonsSrc_, triggerfilter);
       std::vector<reco::RecoChargedCandidateRef> recocandidates;
       triggerfilter->getObjects(trigger::TriggerMuon,recocandidates);
       if ((recocandidates.size()>0)){
           for (unsigned i = 0; i < recocandidates.size(); ++i) {
              double pt=recocandidates.at(i)->pt();
              if(pt>maxpt)
              {
        	  track = dynamic_cast<const reco::Track*>(recocandidates.at(i)->bestTrack());
        	  maxpt=pt;
              }
           }
       }
   }
   
   reco::PFTauCollection* selTaus = new reco::PFTauCollection;
   edm::Handle<edm::View<reco::PFTau> > taus;
   event.getByLabel(tauSrc_, taus);
   for( edm::View<reco::PFTau>::const_iterator pfTau = taus->begin(); pfTau != taus->end(); ++ pfTau ) {
       // if no leading track assigned skip
       if ((!pfTau->leadPFChargedHadrCand().isNonnull())||
           (!pfTau->leadPFChargedHadrCand()->trackRef().isNonnull()))
          continue;

       if(vertexAvailable)
       {
           // select by z position of leading track at vertex
           if ((useLeadingTrack_)||(useLeadingRecoCandidate_)||(useTriggerFilterElectrons_)||(useTriggerFilterMuons_))
           {
               if((track)&&(fabs(pfTau->leadPFChargedHadrCand()->trackRef()->dz(vertexPoint) - track->dz(vertexPoint)) < dZ_))
                   selTaus->push_back(*pfTau);
           }
           // select by z position of leading vertex
           else
           {
               if (fabs(pfTau->leadPFChargedHadrCand()->trackRef()->dz(vertexPoint))<dZ_)
                   selTaus->push_back(*pfTau);
           }
       }
       else
       {
           // select by z position of leading track at (0,0,0)
           if ((useLeadingTrack_)||(useLeadingRecoCandidate_)||(useTriggerFilterElectrons_)||(useTriggerFilterMuons_))
           {
               if((track)&&(fabs(pfTau->leadPFChargedHadrCand()->trackRef()->dz() - track->dz()) < dZ_))
                   selTaus->push_back(*pfTau);
           }
       }
   }
   unsigned filterTaus=selTaus->size();
   std::auto_ptr<reco::PFTauCollection> selectedTaus(selTaus);
   event.put(selectedTaus);
   
   return (filterTaus>=filterOnNTaus_);
}
