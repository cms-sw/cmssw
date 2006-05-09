#ifndef BTauReco_BJetTagCombinedInfo_h
#define BTauReco_BJetTagCombinedInfo_h


#include <vector> 
#include <map>

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "CLHEP/Vector/ThreeVector.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/BTauReco/interface/CombinedBTagInfoFwd.h"

// N.B. use SECVERTEXREF as placeholder for edm::Ref<ReferenceToTrack> or so
//      - have to figure out how to do it properly yet.


namespace reco {

  class CombinedBTagInfo  {

  public:
    ////////////////////////////////////////////////////
    //
    // typedef
    //
    ////////////////////////////////////////////////////

    typedef int SECVERTEXREF;  //just for now to compile

    /* mail from Chris Jones how TRACKREF could be done:
     *       
     *    The ProductID is a unique identifier only within one
     *    edm::Event and only refers the the 'top level' object that has been
     *    placed within the edm::Event. So if one placed a
     *    std::vector<reco::Track> into the event, the std::vector<...> would
     *    have a ProductID but the individual reco::Tracks within the
     *    std::vector would not.  To uniquely (within one edm::Event) identify
     *    an object within a container the framework provides an
     *    edm::Ref<...>.  So it is possible to use an
     *    edm::Ref<std::vector<reco::Track> > to refer to one particular
     *    reco::Track within the std::vector<...>.  It is then possible to
     *    embed an edm::Ref<...> as a member data into another object and then
     *    store that other object into the edm::Event and the framework will
     *    guarantee that the edm::Ref<...> 'points to' the proper object even
     *    when read back from a file in a different job.
     *
     */


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
      // refenence to vertex object
      //  or need real reco::Vertex as
      //  secondary vertices found within B-tagging
      //  are not "produced" and written to event record?
      // reference to all tracks used at this vertex
      //  or will this be automatically there via
      //  reference to vertex?
      double x;  // vertex position
      double y;
      double z;
      double chi2;
      int    ndof;
      int    nTracks; // number of tracks associated 
                      // with this vertex.
      double sumPx;   // sum of x-component of momentum of all charged tracks at vertex
      double sumPy;   //        y-
      double sumPz;   //        z-
      double mass;    /** mass computed from all charged tracks at this
		       *  vertex assuming Pion mass hypothesis.
		       *  For now, loop over all tracks and
		       *  compute m^2 = Sum(E^2) - Sum(p^2)
		       */
      bool   isV0;     // has been tagged as V0 (true) or not (false);
      int    fracPV;   // fraction of tracks also used to build primary vertex
      double flightDistance2D;
      double flightDistanceSignificance2D;
      double flightDistance3D;
      double flightDistanceSignificance3D;

      void init() {
      x                            = -999;
      y                            = -999;
      z                            = -999;
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
    double      getJetPt ()                                  {return jetPt_;}
    double      getJetEta()                                  {return jetEta_;}
    			
    // get (ref to?) primary vertex
    // get vector of secondary vertices
    int         getNumSecVertex()                            {return secondaryVertices_.size();}
    VertexType  getVertexType()                              {return vertexType_;}
    double      getVertexMass()                              {return vertexMass_;}
    int         getVertexMultiplicity()                      {return vertexMultiplicity_;}
    double      getESVXOverE()                               {return eSVXOverE_;}
	             		                             
    Hep3Vector  getPAll()                                    {return pAll_;}
    Hep3Vector  getPB()                                      {return pB_;}
    double      getBPLong()                                  {return bPLong_;}
    double      getBPt()                                     {return bPt_;}
	             		                             
    double      getMeanTrackRapidity()                       {return meanTrackY_;}
    				                             
    double      getAngleGeomKinJet()                         {return angleGeomKinJet_;}
    double      getAngleGeomKinVertex()                      {return angleGeomKinVertex_;}
				                             
    //
    // setters
    //
    void        setJetPt (double pt)                         { jetPt_                 = pt;}
    void        setJetEta(double eta)                        { jetEta_                = eta;}
    	
    // pass (ref to?) primary vertex
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
    bool             existTrackData(TrackRef trackRef);
    void             flushTrackData();
    void             storeTrackData(TrackRef trackRef,
				    const CombinedBTagInfo::TrackData& trackData);
    int              sizeTrackData();
    const TrackData* getTrackData(TrackRef);


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
    std::map <TrackRef,     CombinedBTagInfo::TrackData>  trackDataMap_;
    std::map <SECVERTEXREF, CombinedBTagInfo::VertexData> vertexDataMap_;

  }; // class
 
} // namespace reco

#endif
