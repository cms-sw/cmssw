#ifndef BTauReco_BJetTagCombinedInfo_h
#define BTauReco_BJetTagCombinedInfo_h

#include <vector>
#include <map>

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BTauReco/interface/CombinedBTagInfoFwd.h"

namespace reco {
  class CombinedBTagInfo  {
  public:
    /** Type of secondary vertex found in jet:
     *  - RecoVertex   : a secondary vertex has been fitted from
     *                   a selection of tracks
     *  - PseudoVertex : no RecoVertex has been found but tracks
     *                   with significant impact parameter could be
     *                   combined to a "pseudo" vertex
     *  - NoVertex     : neither of the above attemps were successfull
     *  - NotDefined   : if anything went wrong, set to this value
     */
    enum VertexType {RecoVertex, PseudoVertex, NoVertex, NotDefined};

    /** Type of parton from which the jet originated
     */
    enum PartonType {B, C, UDSG};

    /** list of all variables used to construct the
     *  combined b-tagging discriminator
     */
    enum TaggingVariable{Category,
       VertexMass,
       VertexMultiplicity,
       FlightDistance2DSignificance,
       ESVXOverE,
       TrackRapidity,
       TrackIP2DSignificance,
       TrackIP2DSignificanceAboveCharm};

    /**
     * store all information regarding individual tracks
     * used for tagging and additionally a reference
     * to the track used.
     */
    class TrackData {
      public:
        TrackData();

        TrackData( const reco::TrackRef & ref, bool usedInSVX, double pt, double rapidity, 
                   double eta, double d0, double d0Sign, double d0Error, double jetDistance,
                   int nHitsTotal, int nHitsPixel, bool firstHitPixel, double chi2,
                   double ip2D, double ip2Derror, double ip2DSignificance, double ip3D,
                   double ip3DError, double ip3DSignificance, bool aboveCharmMass );

        reco::TrackRef trackRef;
        bool     usedInSVX;    // part of a secondary vertex?
        double   pt;
        double   rapidity;
        double   eta;
        double   d0;           // 2D impact parameter as given by track
        double   d0Sign;       // same, but lifetime signed
        double   d0Error;
        double   jetDistance;
        int      nHitsTotal;
        int      nHitsPixel;
        bool     firstHitPixel; // true if a valid hit is found in the first pixel barrel layer
        double   chi2;
        double   ip2D;          // lifetime-siged 2D impact parameter
        double   ip2DError;
        double   ip2DSignificance;
        double   ip3D;          // lifetime-siged 3D impact parameter
        double   ip3DError;
        double   ip3DSignificance;
        bool     aboveCharmMass;  /**
           * tracks are sorted by lifetime-signed 2D impact
           * parameter significance. Starting from the
           * highest significance, the invariant mass
           * of the tracks is calculated (using Pion mass
           * hypothesis). If the mass exceeds a threshold,
           * this flag is set to true.
           */
        bool isValid;

        void init();
        void print() const;
    };

    /**
     * Store all information regarding secondary vertices
     * found in current jet
     * N.B. in case of "RecoVertex" the inclusive
     *      vertex finder may find more than one secondary vertex
     */
    class VertexData {
      public:
      VertexData(); 
      reco::Vertex vertex;
      double       chi2;
      double       ndof;
      int          nTracks;      /** number of tracks associated
                                  *  with this vertex.
          */
      GlobalVector trackVector;  // sum of all tracks at this vertex
      double       mass;        /** mass computed from all charged tracks at this
         *  vertex assuming Pion mass hypothesis.
         *  For now, loop over all tracks and
         *  compute m^2 = Sum(E^2) - Sum(p^2)
         */
      bool         isV0;        // has been tagged as V0 (true) or not (false);
      double       fracPV;      // fraction of tracks also used to build primary vertex
      double       flightDistance2D;
      double       flightDistance2DError;
      double       flightDistance2DSignificance;
      double       flightDistance3D;
      double       flightDistance3DError;
      double       flightDistance3DSignificance;

      void init();
      void print() const;
    }; // struct

    typedef edm::AssociationMap < edm::OneToValue<std::vector<reco::Track>,
      reco::CombinedBTagInfo::TrackData, unsigned short> > TrackDataAssociation;

    CombinedBTagInfo();
    virtual ~CombinedBTagInfo();

    void reset(); //< clear all information

    double jetPt () const;
    double jetEta() const;
    reco::Vertex primaryVertex() const;
    std::vector<reco::Vertex> secVertices() const;
    std::vector<reco::TrackRef> tracksAboveCharm() const;
    std::vector<reco::TrackRef> tracksAtSecondaryVertex() const;
    int          nSecVertices() const;
    VertexType   vertexType() const;
    double       vertexMass() const;
    int          vertexMultiplicity() const;
    double       eSVXOverE() const;
    GlobalVector pAll() const;
    GlobalVector pB() const;
    double       pBLong() const;
    double       pBPt() const;
    double       meanTrackRapidity() const;
    double       angleGeomKinJet() const;
    double       angleGeomKinVertex() const;

    double       flightDistance2DMin() const;
    double       flightDistanceSignificance2DMin() const;
    double       flightDistance3DMin() const;
    double       flightDistanceSignificance3DMin() const;
    double       flightDistance2DMax() const;
    double       flightDistanceSignificance2DMax() const;
    double       flightDistance3DMax () const;
    double       flightDistanceSignificance3DMax() const;
    double       flightDistance2DMean() const;
    double       flightDistanceSignificance2DMean() const;
    double       flightDistance3DMean() const;
    double       flightDistanceSignificance3DMean() const;

    // possibly revisit this if calculation of lifetime-signed 2d IP
    // is avaialable via Track itself
    double                      first2DSignedIPSigniAboveCut() const;

    //
    // setters
    //
    void setJetPt (double pt);
    void setJetEta(double eta);

    void setPrimaryVertex( const reco::Vertex & pv );
    void addSecondaryVertex( const reco::Vertex & sv );
    void addTrackAtSecondaryVertex(reco::TrackRef trackRef);
    void setVertexType( VertexType type);
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

    void setFlightDistance2DMin(double value);
    void setFlightDistanceSignificance2DMin (double value);
    void setFlightDistance3DMin(double value);
    void setFlightDistanceSignificance3DMin(double value);

    void setFlightDistance2DMax(double value);
    void setFlightDistanceSignificance2DMax(double value);
    void setFlightDistance3DMax (double value);
    void setFlightDistanceSignificance3DMax(double value);

    void setFlightDistance2DMean(double value);
    void setFlightDistanceSignificance2DMean(double value);
    void setFlightDistance3DMean(double value);
    void setFlightDistanceSignificance3DMean (double value);

    void setFirst2DSignedIPSigniAboveCut(double ipSignificance);

    //
    // map to access track map information
    //
    // maybe possible to use map tools here?
    bool              existTrackData( const reco::TrackRef & trackRef );
    void              flushTrackData();
    void              storeTrackData(reco::TrackRef trackRef,
             const CombinedBTagInfo::TrackData& trackData);
    void              printTrackData();
    int               sizeTrackData();
    const TrackData*  getTrackData(reco::TrackRef trackRef);


    // is this the "best" way to do it?
    bool              existVertexData(std::vector<reco::Vertex>::const_iterator vertexRef);
    void              flushVertexData();
    void storeVertexData(std::vector<reco::Vertex>::const_iterator vertexRef,
              const CombinedBTagInfo::VertexData& vertexData);
    int               sizeVertexData() const;
    VertexData*       getVertexData(std::vector<reco::Vertex>::const_iterator vertexRef) const;

    std::string       getTaggingVarName( reco::CombinedBTagInfo::TaggingVariable t ) const;
    std::string       getVertexTypeName() const;

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
    VertexType  vertexType_;


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

    double flightDistance2DMin_;
    double flightDistanceSignificance2DMin_;
    double flightDistance3DMin_;
    double flightDistanceSignificance3DMin_;

    double flightDistance2DMax_;
    double flightDistanceSignificance2DMax_;
    double flightDistance3DMax_;
    double flightDistanceSignificance3DMax_;

    double flightDistance2DMean_;
    double flightDistanceSignificance2DMean_;
    double flightDistance3DMean_;
    double flightDistanceSignificance3DMean_;

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

    //
    // maps for detailed track and vertex information
    //

    TrackDataAssociation                                                                 trackDataMap_;
    mutable std::map <std::vector<reco::Vertex>::const_iterator, CombinedBTagInfo::VertexData> vertexDataMap_;
    mutable std::map <reco::CombinedBTagInfo::TaggingVariable, std::string>                   taggingVarName_;


  }; // class

} // namespace reco

#endif
