#ifndef PhysicsTools_PatUtils_interface_VertexAssociationSelector_h
#define PhysicsTools_PatUtils_interface_VertexAssociationSelector_h

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/Vertexing.h"

namespace pat {
    class VertexAssociationSelector {
        public:
            /// Structure to pack config for the constructor.
            /// Note that even if no cuts are selected (all values set to 0),
            /// this will still return 'false' for a null pat::VertexAssociation
            struct Config {
                Config() : dZ(0), dR(0), sigmasZ(0), sigmasR(0) {} //, sigmas3d(0) {}
                /// cuts on Z and transverse distance from the vertex, absolute values or significances
                float dZ, dR, sigmasZ, sigmasR; //, sigmas3d; 
            };
#if 0            
            /// Candidate selector made by a VertexAssociationSelector and a single vertex
            class SelectCandidates {
                public:
                    SelectCandidates(const VertexAssociationSelector &asso, const reco::Vertex &vtx) : vtx_(vtx), asso_(asso) { }
                    bool operator()(const reco::Candidate &c) const { return asso_(c, vtx_); }
                private:
                    const reco::Vertex              & vtx_;
                    const VertexAssociationSelector & asso_;
            };
            /// Vertex selector made by a VertexAssociationSelector and a single candidate
            class SelectVertices {
                public:
                    SelectVertices(const VertexAssociationSelector &asso, const reco::Candidate &cand) : cand_(cand), asso_(asso) { }
                    bool operator()(const reco::Vertex &vtx) const { return asso_(cand_, vtx); }
                private:
                    const reco::Candidate           & cand_;
                    const VertexAssociationSelector & asso_;
            };
#endif

            /// an empty constructor is sometimes needed
            VertexAssociationSelector() {} 
            /// constructor from a configuration
            VertexAssociationSelector(const Config & conf) ;

            /// check if this VertexAssociation is ok
            bool operator()(const pat::VertexAssociation & vass) const ;

            /// check if this candidate and this vertex are compatible
            /// this will just use the basic candidate vertex position,
            /// without any fancy track extrapolation.
            bool operator()(const reco::Candidate &c, const reco::Vertex &) const ;

            pat::VertexAssociation simpleAssociation(const reco::Candidate &c, const reco::VertexRef & vtx) const ;
#if 0
            /// Make a Candidate selector from this association selector and a vertex
            SelectCandidates selectCandidates(const reco::Vertex    & vtx ) const { return SelectCandidates(*this, vtx  ); }

            /// Make a Vertex selector from this association selector and a candidate
            SelectVertices   selectVertices(  const reco::Candidate & cand) const { return SelectVertices(  *this, cand ); }
#endif
        private:
            Config conf_;

    }; // class
} // namespace

#endif
