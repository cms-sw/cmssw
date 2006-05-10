#ifndef BTauReco_BJetTagCombinedInfo_h
#define BTauReco_BJetTagCombinedInfo_h


#include <vector> 
#include <map>

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "CLHEP/Vector/ThreeVector.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/BTauReco/interface/CombinedBTagInfoFwd.h"

namespace reco {

  class CombinedBTagInfo  {

  public:
    ////////////////////////////////////////////////////
    //
    // typedef
    //
    ////////////////////////////////////////////////////

    ////////////////////////////////////////////////////
    //
    // definitions
    //
    ////////////////////////////////////////////////////

    /** Type of secondary vertex found in jet:
     *  - RecoVertex   : a secondary vertex has been fitted from
     *                   a selection of tracks
     *  - PseudoVertex : no RecoVertex has been found but tracks
     *                   with significant impact parameter could be 
     *                   combined to a "pseudo" vertex
     *  - NoVertex     : neither of the above attemps were successfull
     *  - NotDefined   : if anything went wrong, set to this value
     */

    enum VertexType {RecoVertex,
		     PseudoVertex,
		     NoVertex,
                     NotDefined};

    
    /**
     * store all information regarding individual tracks
     * used for tagging and additionally a reference
     * to the track used.
     */
    struct TrackData {
      TrackRef trackRef;     // reference to the track used
                             // or we don't need it here as the TrackRef is
                             // used as access key for the map?
      bool     usedInSVX;    // part of a secondary vertex?
      double   pt;
      double   rapidity;
      double   eta;
      double   d0;           // 2D impact parameter as given by track
      int      nHitsTotal;
      int      nHitsPixel;
      double   chi2;
      double   ip2D;          // lifetime-siged 2D impact parameter
      double   ipSigni2D;     // lifetime-siged 2D impact parameter significance
      double   ip3D;          // lifetime-siged 3D impact parameter
      double   ipSigni3D;     // lifetime-siged 3D impact parameter significance     

      void init() {
	usedInSVX  = false;
	pt         = -999;
	rapidity   = -999;
	eta        = -999;
	d0         = -999;
	nHitsTotal = -999; 
	nHitsPixel = -999;
	chi2       = -999;
	ip2D       = -999;
	ipSigni2D  = -999;
	ip3D       = -999;
	ipSigni3D  = -999;
      } //init
    }; // struct   

    /**
     * Store all information regarding secondary vertices
     * found in current jet
     * N.B. in case of "RecoVertex" the inclusive
     *      vertex finder may find more than one secondary vertex
     */
    struct VertexData {
      reco::Vertex vertex;
      double       chi2;
      int          ndof;
      int          nTracks; // number of tracks associated 
                            // with this vertex.
      double       sumPx;   // sum of x-component of momentum of all charged tracks at vertex
      double       sumPy;   //        y-
      double       sumPz;   //        z-
      double       mass;    /** mass computed from all charged tracks at this
	                     *  vertex assuming Pion mass hypothesis.
	                     *  For now, loop over all tracks and
	                     *  compute m^2 = Sum(E^2) - Sum(p^2)
	                     */
      bool         isV0;     // has been tagged as V0 (true) or not (false);
      int          fracPV;   // fraction of tracks also used to build primary vertex
      double       flightDistance2D;
      double       flightDistanceSignificance2D;
      double       flightDistance3D;
      double       flightDistanceSignificance3D;

      void init() {
      chi2                         = -999;
      ndof                         = -999;
      nTracks                      = -999; 
      sumPx                        = -999;  
      sumPy                        = -999;  
      sumPz                        = -999;  
      mass                         = -999;   
      isV0                         = -999;     
      fracPV                       = -999;    
      flightDistance2D             = -999;
      flightDistanceSignificance2D = -999;
      flightDistance3D             = -999;
      flightDistanceSignificance3D = -999;	
	
      } //init
    }; // struct

    ////////////////////////////////////////////////////
    //
    // public
    //
    ////////////////////////////////////////////////////


    //
    // constructor and destructor
    //

    CombinedBTagInfo();
    virtual ~CombinedBTagInfo(); 

    //
    // accessors
    //

    // also need to store
    // - primary vertex
    // - list of secondary vertices

    // members of this class				     
    double                    jetPt ()                    {return jetPt_;}
    double                    jetEta()                    {return jetEta_;}
    			
    // edm::ref to primary vertex ?
    reco::Vertex              primaryVertex()             {return primaryVertex_;}
    std::vector<reco::Vertex> secVertices()               {return secondaryVertices_;}
    int                       nSecVertices()              {return secondaryVertices_.size();}
    VertexType                vertexType()                {return vertexType_;}
    double                    vertexMass()                {return vertexMass_;}
    int                       vertexMultiplicity()        {return vertexMultiplicity_;}
    double                    eSVXOverE()                 {return eSVXOverE_;}
	                          		               
    Hep3Vector                pAll()                      {return pAll_;}
    Hep3Vector                pB()                        {return pB_;}
    double                    pBLong()                    {return bPLong_;}
    double                    pBPt()                      {return bPt_;}
	                          		               
    double                    meanTrackRapidity()         {return meanTrackY_;}
    		             		                  
    double                    angleGeomKinJet()           {return angleGeomKinJet_;}
    double                    angleGeomKinVertex()        {return angleGeomKinVertex_;}
				                             
    //
    // setters
    //
    void        setJetPt (double pt)                         { jetPt_                 = pt;}
    void        setJetEta(double eta)                        { jetEta_                = eta;}
    	
    // pass (ref to?) primary vertex
    void        setPrimaryVertex(reco::Vertex pv)            { primaryVertex_         = pv;}
    void        addSecondaryVertex(reco::Vertex sv)          { secondaryVertices_.push_back(sv);}
    void        setVertexType( VertexType type)              { vertexType_            = type;}
    void        setVertexMass( double mass)                  { vertexMass_            = mass;}
    void        setVertexMultiplicity(int mult)              { vertexMultiplicity_    = mult;}
    void        setESVXOverE( double e)                      { eSVXOverE_             = e;}
	             		                             			      
    void        setPAll(Hep3Vector p)                        { pAll_                  = p;}
    void        setPB(Hep3Vector p)                          { pB_                    = p;}
    void        setBPLong(double pLong)                      { bPLong_                = pLong;}
    void        setBPt(double pt)                            { bPt_                   = pt;}
    void        setMeanTrackRapidity(double meanY)           { meanTrackY_            = meanY;}
    				                        
    void        setAngleGeomKinJet(double angle)             {angleGeomKinJet_        = angle;}
    void        setAngleGeomKinVertex(double angle)          {angleGeomKinVertex_     = angle;}			                        




    //
    // map to access track map information
    //
    // maybe possible to use map tools here?
    bool              existTrackData(TrackRef trackRef);
    void              flushTrackData();
    void              storeTrackData(TrackRef trackRef,
				     const CombinedBTagInfo::TrackData& trackData);
    int               sizeTrackData();
    const TrackData*  getTrackData(TrackRef trackRef);


    // is this the "best" way to do it?
    bool              existVertexData(std::vector<reco::Vertex>::const_iterator vertexRef);
    void              flushVertexData();
    void              storeVertexData(std::vector<reco::Vertex>::const_iterator vertexRef,
				      const CombinedBTagInfo::VertexData& vertexData);
    int               sizeVertexData();
    const VertexData* getVertexData(std::vector<reco::Vertex>::const_iterator vertexRef);


    ////////////////////////////////////////////////////
    //
    // private
    //
    ////////////////////////////////////////////////////
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

    VertexType  vertexType_;      /** if at least one secondary vertex has been found,
				   *  jet has type "RecoVertex", otherwise 
				   *  "PseudoVertex" or "NoVertex"
				   */
	        
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

    Hep3Vector  pB_;                   /** computed from all tracks all all
					*  secondary vertices,
					*  pX = Sum(pX), etc
					*/

    Hep3Vector  pAll_;                  /** same as above but computed from 
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

    double      vertexEnergyCharged_;  /** energy calculated from all tracks
					*  used at secondary vertices.
					*  tracks are assumed to be Pions
					*/

    double      jetEnergyAll_;          /** energy calculated from all tracks
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

    // maybe easier/better to have templated class to handle the maps?
    std::map <TrackRef,                                  CombinedBTagInfo::TrackData>  trackDataMap_;
    std::map <std::vector<reco::Vertex>::const_iterator, CombinedBTagInfo::VertexData> vertexDataMap_;

  }; // class
 
} // namespace reco

#endif
