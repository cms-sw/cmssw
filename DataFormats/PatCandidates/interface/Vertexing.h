#ifndef DataFormats_PatCandidates_interface_Vertexing_h
#define DataFormats_PatCandidates_interface_Vertexing_h

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1DFloat.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

/**
  \class    pat::VertexAssociation VertexAssociation.h "DataFormats/PatCandidates/interface/Vertexing.h"
  \brief    Analysis-level structure for vertex-related information

  pat::VertexAssociation holds a reference to a vertex and extended information like the distance between the object and the vertex.

  For convenience, pat::VertexAssociation behaves like a VertexRef, that is (*assoc) is a reco::Vertex.

  The original proposal is at https://hypernews.cern.ch/HyperNews/CMS/get/physTools/587.html 

  \author   Giovanni Petrucciani
  \version  $Id: Vertexing.h,v 1.1 2008/07/22 12:44:19 gpetrucc Exp $
*/


namespace pat {
    class VertexAssociation {
        public:
            //! Create a null vertx association
            VertexAssociation() {}
            //! Create a vertex association given a ref to a vertex
            VertexAssociation(const reco::VertexRef &vertex) : vertex_(vertex) {}
            //! Create a vertex association given a ref to a vertex and a reference to a track object
            /// Note: you also have to set the distances, they can be computed by the VertexAssociation itself
            /// because it requires access to the magnetic field and other condition data.
            VertexAssociation(const reco::VertexRef &vertex, const reco::TrackBaseRef &tk) : vertex_(vertex), track_(tk) {}
            // --- Methods to mimick VertexRef
            //! Return 'true' if this is a null association (that is, no vertex)
            bool  isNull() const      { return vertex_.isNull(); }
            //! Return 'true' unless this is a null association (that is, no vertex)
            bool  isNonnull() const   { return vertex_.isNonnull(); }
            //! Return 'true' if the reco::Vertex is available in the file, false if it has been dropped.
            bool  isAvailable() const { return vertex_.isAvailable(); }
            //! Return the vertex (that is, you can do "const reco::Vertex &vtx = *assoc")
            const reco::Vertex & operator*()  const { return * operator->(); }
            //! Allows VertexAssociation to behave like a vertex ref  (e.g. to do "assoc->position()")
            const reco::Vertex * operator->() const { return vertex_.isNull() ? 0 : vertex_.get(); }
            // --- Methods to get the Vertex and track
            //! Returns the reference to the vertex (can be a null reference)
            const reco::VertexRef & vertexRef() const  { return vertex_; }
            //! Returns a pointer to the vertex, or a null pointer if there is no vertex (null association)
            const reco::Vertex    * vertex()    const  { return vertex_.isNull() ? 0 : vertex_.get(); }
            //! Returns 'true' if a reference to a track was stored in this VertexAssociation
            bool                       hasTrack() const { return !track_.isNull(); }
            //! Returns a reference to the track stored in this vertex (can be null)
            const reco::TrackBaseRef & trackRef() const { return track_; }
            //! Returns a C++ pointer to the track stored in this vertex (can be a null pointer)
            const reco::Track        * track()    const { return hasTrack() ? track_.get() : 0; }
            // --- Methods to return distances
            //! Distance between the object and the vertex along the Z axis, and it's error.
            /// Note 1: if the BeamSpot was used as Vertex, the error includes the BeamSpot spread!
            const Measurement1DFloat & dz() const { return dz_; }
            //! Distance between the object and the vertex in the transverse plane, and it's error.
            const Measurement1DFloat & dr() const { return dr_; }
            //! True if the transverse distance was computed for this VertexAssociation
            bool  hasTransverseIP() const { return (dr_.value() != 0); }
            //! True if the errors on dr and dz have been set, false if they're nulls
            bool  hasErrors()       const { return (dz_.error() != 0) && ( !hasTransverseIP() || dr_.error() != 0 ); }
            // ---- Methods to set distances 
            void setDz(const Measurement1DFloat & dz) { dz_ = dz; }
            void setDr(const Measurement1DFloat & dr) { dr_ = dr; }
            void setDz(const Measurement1D & dz)      { dz_ = Measurement1DFloat(dz.value(), dz.error()); }
            void setDr(const Measurement1D & dr)      { dr_ = Measurement1DFloat(dr.value(), dr.error()); }
            //! Set dz and dr given the distance and the 3x3 total covariance matrix of the distance
            void setDistances(const AlgebraicVector3 & dist, const AlgebraicSymMatrix33 &err) ;
            //! Set dz and dr given the two points (object and vertex) and the 3x3 total covariance matrix of the distance
            /// The covariance matrix must already include the covariance matrix of the vertex
            /// 'p1', 'p2' can be anything that has methods .x(), .y(), .z() (e.g. GlobalPoint, Vertex, ...)
            template<typename T1, typename T2>
            void setDistances(const T1 & p1, const T2 &p2, const AlgebraicSymMatrix33 &err) {
                AlgebraicVector3 dist(p1.x() - p2.x(), p1.y() - p2.y(), p1.z() - p2.z());
                setDistances(dist, err);
            }
            // ---- 3D significance of the impact parameter
            // float signif3d()           const { return signif3d_; }
            // bool  hasSignif3d()        const { return signif3d_ != 0; }
            // void setSignif3d(float signif3d) { signif3d_ = signif3d; }
        private:
            // basic information
            reco::VertexRef vertex_;
            Measurement1DFloat dz_, dr_;
            // float signif3d_;
            // extended info
            reco::TrackBaseRef track_;
            // fixme: add refitted momentum
    };
}

#endif
