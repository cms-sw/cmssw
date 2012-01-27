#include "RecoTauTag/HLTProducers/interface/PFTauVertexSelector.h"
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

bool PFTauVertexSelector::filter(edm::Event& event, const edm::EventSetup& eventSetup) {
   edm::Handle<edm::View<reco::Vertex> > vertices;
   event.getByLabel(vertexSrc_, vertices);
   
   // if no vertex in vertex collection return
   if((!vertices.isValid())||
      (vertices->size()==0))
      return (0>=filterOnNTaus_);
      
   const reco::Track* track=0;
   if (useLeadingTrack_)
   {
       edm::Handle<edm::View<reco::Track> > tracks;
       event.getByLabel(trackSrc_, tracks);
       if ((tracks.isValid())&&(tracks->size()>0)){
       double maxpt=0.;
       unsigned i_maxpt=0;
       for (unsigned i = 0; i < tracks->size(); ++i) {
   	 double pt=tracks->ptrAt(i)->pt();
   	 if(pt>maxpt)
   	 {
   	   i_maxpt=i;
   	   maxpt=pt;
   	 }
       }
       track = &*tracks->ptrAt(i_maxpt);
       }
   }
   else if (useLeadingRecoCandidate_)
   {
       edm::Handle<edm::View<reco::RecoCandidate> > recocandidates;
       event.getByLabel(recoCandidateSrc_, recocandidates);
       if ((recocandidates.isValid())&&(recocandidates->size()>0)){
       double maxpt=0.;
       unsigned i_maxpt=0;
       for (unsigned i = 0; i < recocandidates->size(); ++i) {
         double pt=recocandidates->ptrAt(i)->pt();
         if(pt>maxpt)
         {
           i_maxpt=i;
           maxpt=pt;
         }
       }
       track = dynamic_cast<const reco::Track*>(recocandidates->ptrAt(i_maxpt)->bestTrack());
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

       // select by z position of leading track at vertex
       if ((useLeadingTrack_)||(useLeadingRecoCandidate_))
       {
           if((track)&&(fabs(pfTau->leadPFChargedHadrCand()->trackRef()->dz(vertices->at(0).position()) - track->dz(vertices->at(0).position()) < dZ_)))
               selTaus->push_back(*pfTau);
       }
       // select by z position of leading vertex
       else
       {
           if (fabs(pfTau->leadPFChargedHadrCand()->trackRef()->dz(vertices->at(0).position()))<dZ_)
               selTaus->push_back(*pfTau);
       }
   }
   unsigned filterTaus=selTaus->size();
   std::auto_ptr<reco::PFTauCollection> selectedTaus(selTaus);
   event.put(selectedTaus);
   
   return (filterTaus>=filterOnNTaus_);
}
