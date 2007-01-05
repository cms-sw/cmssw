#ifndef BTauReco_BJetTagCombinedInfo_h
#define BTauReco_BJetTagCombinedInfo_h

#include <vector>
#include <map>

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BTauReco/interface/CombinedBTagInfoFwd.h"
#include "DataFormats/BTauReco/interface/CombinedBTagEnums.h"
#include "DataFormats/BTauReco/interface/CombinedBTagTrack.h"
#include "DataFormats/BTauReco/interface/CombinedBTagVertex.h"
#include "DataFormats/BTauReco/interface/MinMeanMax.h"

namespace reco {
  class CombinedBTagInfo  {
  public:
    typedef edm::AssociationMap < edm::OneToValue<std::vector<reco::Track>,
      reco::CombinedBTagTrack, unsigned short> > TrackDataAssociation;

    CombinedBTagInfo();
    virtual ~CombinedBTagInfo();

    void reset(); //< clear all information

    reco::Vertex primaryVertex() const;
    std::vector<reco::Vertex> secVertices() const;
    std::vector<reco::TrackRef> tracksAboveCharm() const;
    std::vector<reco::TrackRef> tracksAtSecondaryVertex() const;

    double jetPt () const;
    double jetEta() const;
    int    nSecVertices() const;
    reco::CombinedBTagEnums::VertexType vertexType() const;
    double vertexMass() const;
    int    vertexMultiplicity() const;
    double eSVXOverE() const;
    GlobalVector pAll() const;
    GlobalVector pB() const;
    double pBLong() const;
    double pBPt() const;
    double meanTrackRapidity() const;
    double angleGeomKinJet() const;
    double angleGeomKinVertex() const;

    MinMeanMax flightDistance2D() const;
    MinMeanMax flightDistanceSignificance2D() const;
    MinMeanMax flightDistance3D() const;
    MinMeanMax flightDistanceSignificance3D() const;

    // possibly revisit this if calculation of lifetime-signed 2d IP
    // is avaialable via Track itself
    double first2DSignedIPSigniAboveCut() const;

    void setJetPt (double pt);
    void setJetEta(double eta);

    void setPrimaryVertex( const reco::Vertex & pv );
    void addSecondaryVertex( const reco::Vertex & sv );
    void addTrackAtSecondaryVertex(reco::TrackRef trackRef);
    void setVertexType( reco::CombinedBTagEnums::VertexType type);
    void setVertexMass( double mass);
    void setVertexMultiplicity(int mult);
    void setESVXOverE( double e);

    void setEnergyBTracks(double energy);
    void setEnergyAllTracks(double energy);

    void setPAll( const GlobalVector & p );
    void setPB( const GlobalVector & p);
    void setBPLong(double pLong);
    void setBPt(double pt);
    void setMeanTrackRapidity(double meanY);

    void setAngleGeomKinJet(double angle);
    void setAngleGeomKinVertex(double angle);
    void addTrackAboveCharm(reco::TrackRef trackRef);

    void setFlightDistance2D ( const MinMeanMax & );
    void setFlightDistanceSignificance2D ( const MinMeanMax & );
    void setFlightDistance3D ( const MinMeanMax & );
    void setFlightDistanceSignificance3D ( const MinMeanMax & );

    void setFirst2DSignedIPSigniAboveCut(double ipSignificance);

    // map to access track map information
    // maybe possible to use map tools here?
    bool existTrackData( const reco::TrackRef & );
    void flushTrackData();
    void storeTrackData ( const reco::TrackRef &,
             const reco::CombinedBTagTrack & );
    void printTrackData() const;
    int  sizeTrackData() const;
    const reco::CombinedBTagTrack * getTrackData( const reco::TrackRef & ) const;

    // is this the "best" way to do it?
    bool existVertexData(std::vector<reco::Vertex>::const_iterator vertexRef);
    void flushVertexData();
    void storeVertexData(std::vector<reco::Vertex>::const_iterator vertexRef,
              const reco::CombinedBTagVertex & vertexData);
    int  sizeVertexData() const;
    reco::CombinedBTagVertex * getVertexData(std::vector<reco::Vertex>::const_iterator vertexRef) const;
    std::string getVertexTypeName() const;

  private:

    // jet information
    double      jetPt_;
    double      jetEta_;

    // vertex information
    reco::Vertex primaryVertex_;  // reference? something like
                                  // edm::Ref<std::vector<reco::Vertex> >  ?

    std::vector<reco::Vertex> secondaryVertices_;
                                  // how to store best as this one is created
                                  // as part of the combined b-tag alg?

    /**
     * Type of vertex which is found in this jet:
     * if at least one secondary vertex has been found, jet has type "RecoVertex", otherwise
     * "PseudoVertex" or "NoVertex"
     */
    reco::CombinedBTagEnums::VertexType  vertexType_;


    /**
     * reference of all tracks at all secondary vertices
     */
    std::vector<reco::TrackRef> tracksAtSecondaryVertex_;

    /**
     * Determine (lifetime-singed 2D) impact parameter of first track
     * above given mass threshold (reco::CombinedBTagAlg::vertexCharmCut_)
     * Idea: if the secondary vertex is due to a charmed hadron,
     *       there will be a distinct gap in the distribution of the
     *       impact parameters:
     *       see e.g. http://www-ekp.physik.uni-karlsruhe.de/~weiser/thesis/P108.html
     */
    std::vector<reco::TrackRef> tracksAboveCharm_;

    /**
     * 2D life-time signed impact parameter of first track
     * to exceed mass threshold
     */
    double first2DSignedIPSigniAboveCut_;


    /** Store for easier access also
     *  min, max, mean of
     *  flightDistance{2D,3D} and significance
     *  at present, the combined b-tagging alg.
     *  uses the min flight distance (2D)
     *  These quantities are computed per
     *  reconstucted secondary vertex and hence
     *  only different if there are more than one.
     */

    MinMeanMax flightDistance2D_;
    MinMeanMax flightDistanceSignificance2D_;
    MinMeanMax flightDistance3D_;
    MinMeanMax flightDistanceSignificance3D_;

    /** angle between vector connecting primary and secondary vertex
     *  track selection
     *  for Jet   : all tracks in jet
     *  for Vertex: all tracks used at all secondary vertices
     */
    double      angleGeomKinJet_;
    double      angleGeomKinVertex_;


    /** The following quantities are omputed from all tracks at all
     *  secondary vertices (in case there are several for type RecoVertex)
     *  see comment at beginning of this header file.
     */

    GlobalVector  pB_;                 /** computed from all tracks all all
          *  secondary vertices,
          *  pX = Sum(pX), etc
          */

    GlobalVector  pAll_;               /** same as above but computed from
           *  all tracks in jet
           */

    double      bPLong_;               /** longitudinal component of B momentum vector
          *  pBLong =  pAll*pB
          *           ---------
          *             |pAll|
          */

    double      bPt_;                  /**  transverse component of B momentum vector
          *   pt     = sqrt(|pB|*|pB| - pBLong*pBLong)
          */

    double      vertexMass_;           /** all tracks are assumed to be Pions,
          *  m = sqrt(E**2 - p**2)
          */

    double      energyBTracks_;        /** energy calculated from all tracks
          *  used at secondary vertices.
          *  tracks are assumed to be Pions
          */

    double      energyAllTracks_;      /** energy calculated from all tracks
          *  tracks associated to jet
          *  Tracks are assumed to be Pions.
          */

    double      eSVXOverE_;            /** energy of all tracks at all secondary
          *  vertices divieded by energy of all tracks
          *  tracks associated to jet,
          *  all tracks are assumed to be Pions
          *  I.e.vertexEnergyCharged_/jetEnergyAll_
          */


    int         vertexMultiplicity_;   /** number of all tracks at all
          *  secondary vertices
          */

    double      meanTrackY_;           /** mean track rapidity
          *  Track rapidities are calculated w.r.t.
          *  vector of all tracks at all secondary vertices
          *  given by vector = (SumPx, SumPy, SumPz)
          *  default value for rapidity: 5
          *  mean is given by arithmentic mean
          */

    // maps for detailed track and vertex information
    TrackDataAssociation trackDataMap_;
    mutable std::map < std::vector<reco::Vertex>::const_iterator, 
                       reco::CombinedBTagVertex> vertexDataMap_;
  }; // class

} // namespace reco

#endif
