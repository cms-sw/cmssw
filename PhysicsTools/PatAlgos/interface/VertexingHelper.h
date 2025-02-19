#ifndef PhysicsTools_PatAlgos_interface_VertexingHelper_h
#define PhysicsTools_PatAlgos_interface_VertexingHelper_h
/**
  \class    pat::helper::VertexingHelper VertexingHelper.h "PhysicsTools/PatAlgos/interface/VertexingHelper.h"
  \brief    Produces and/or checks pat::VertexAssociation's

   The VertexingHelper produces pat::VertexAssociation, or reads them from the event,
   and can use them to select if a candidate is good or not. 

  \author   Giovanni Petrucciani
  \version  $Id: VertexingHelper.h,v 1.1 2008/07/22 12:47:01 gpetrucc Exp $
*/


#include "DataFormats/PatCandidates/interface/Vertexing.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "PhysicsTools/PatUtils/interface/VertexAssociationSelector.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
namespace reco {
    namespace modules {
        /// Helper struct to convert from ParameterSet to ElectronSelection
        template<> 
        struct ParameterAdapter<pat::VertexAssociationSelector> { 
            static pat::VertexAssociationSelector make(const edm::ParameterSet & iConfig) {
                pat::VertexAssociationSelector::Config assoconf;
                if (iConfig.existsAs<double>("deltaZ"))  assoconf.dZ = iConfig.getParameter<double>("deltaZ");
                if (iConfig.existsAs<double>("deltaR"))  assoconf.dR = iConfig.getParameter<double>("deltaR");
                if (iConfig.existsAs<double>("sigmasZ")) assoconf.sigmasZ = iConfig.getParameter<double>("sigmasZ");
                if (iConfig.existsAs<double>("sigmasR")) assoconf.sigmasR = iConfig.getParameter<double>("sigmasR");
                return pat::VertexAssociationSelector(assoconf);
            }
        };
    }
}

namespace pat { namespace helper {
    class VertexingHelper {
        public:
            VertexingHelper() : enabled_(false) {}
            VertexingHelper(const edm::ParameterSet &iConfig) ;
    
            /// returns true if this was given a non dummy configuration
            bool enabled() const {  return enabled_; }

            /// To be called for each new event, reads in the vertex collection
            void newEvent(const edm::Event &event) ;

            /// To be called for each new event, reads in the vertex collection and the tracking info
            /// You need this if 'useTrack' is true
            void newEvent(const edm::Event &event, const edm::EventSetup & setup) ;

            /// Return true if this candidate is associated to a valid vertex
            /// AnyCandRef should be a Ref<>, RefToBase<> or Ptr to a Candidate object
            template<typename AnyCandRef>
            pat::VertexAssociation  operator()(const AnyCandRef &) const ;

        private: 
            /// true if it has non null configuration
            bool enabled_;
        
            /// true if it's just reading the associations from the event
            bool playback_;
       
            /// selector of associations 
            pat::VertexAssociationSelector assoSelector_;

            //-------- Tools for production of vertex associations -------
            edm::InputTag vertices_;
            edm::Handle<reco::VertexCollection > vertexHandle_;
            /// use tracks inside candidates
            bool useTracks_;
            edm::ESHandle<TransientTrackBuilder> ttBuilder_;
          
            //--------- Tools for reading vertex associations (playback mode) ----- 
            edm::InputTag vertexAssociations_;
            edm::Handle<edm::ValueMap<pat::VertexAssociation> > vertexAssoMap_;

            /// Get out the track from the Candidate / RecoCandidate / PFCandidate
            reco::TrackBaseRef  getTrack_(const reco::Candidate &c) const ;
            
            /// Try to associated this candidate to a vertex. 
            /// If no association is found passing all cuts, return a null association
            pat::VertexAssociation associate(const reco::Candidate &) const ;

    }; // class

    template<typename AnyCandRef>
    pat::VertexAssociation
    pat::helper::VertexingHelper::operator()(const AnyCandRef &cand) const 
    {
        if (playback_) {
            const pat::VertexAssociation &assoc = (*vertexAssoMap_)[cand];
            return assoSelector_(assoc) ? assoc : pat::VertexAssociation();
        } else {
            return associate( *cand );
        }

    }

} }




#endif
