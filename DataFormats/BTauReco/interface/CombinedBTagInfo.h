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

    /** Type of parton from which the jet originated
     */
    enum PartonType {B,
		     C,
		     UDSG};

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
    struct TrackData {
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
 

      void init() {
	usedInSVX        = false;
	aboveCharmMass   = false;
	pt               = -999;
	rapidity         = -999;
	eta              = -999;
	d0               = -999;
	d0Sign           = -999;
	d0Error          = -999;
	nHitsTotal       = -999; 
	nHitsPixel       = -999;
	firstHitPixel    = false;
	chi2             = -999;
	ip2D             = -999;
	ip2DError        = -999;
	ip2DSignificance = -999;
	ip3D             = -999;
	ip3DError        = -999;
	ip3DSignificance = -999;
      } //init

      void print() const {
	std::cout << "*** printing trackData for combined b-tag info " << std::endl;
	std::cout << "    usedInSVX        " << usedInSVX        << std::endl;
	std::cout << "    aboveCharmMass   " << aboveCharmMass   << std::endl;
	std::cout << "    pt               " << pt               << std::endl;
	std::cout << "    rapidity         " << rapidity         << std::endl;
	std::cout << "    d0               " << d0               << std::endl;
	std::cout << "    d0Sign           " << d0Sign           << std::endl;
	std::cout << "    d0Error          " << d0Error          << std::endl;
	std::cout << "    nHitsTotal       " << nHitsTotal       << std::endl;
	std::cout << "    nHitsPixel       " << nHitsPixel       << std::endl;
	std::cout << "    firstHitPixel    " << firstHitPixel    << std::endl;
	std::cout << "    chi2             " << chi2             << std::endl;
	std::cout << "    ip2D             " << ip2D             << std::endl;
	std::cout << "    ip2DError        " << ip2DError        << std::endl;
	std::cout << "    ip2DSignificance " << ip2DSignificance << std::endl;
	std::cout << "    ip3D             " << ip3D             << std::endl;
	std::cout << "    ip3DError        " << ip3DError        << std::endl;
	std::cout << "    ip3DSignificance " << ip3DSignificance << std::endl;


      }//print

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

      void init() {
      chi2                         = -999;
      ndof                         = -999;
      nTracks                      = -999; 
      mass                         = -999;   
      isV0                         = -999;     
      fracPV                       = -999;    
      flightDistance2D             = -999;
      flightDistance2DError        = -999;
      flightDistance2DSignificance = -999;
      flightDistance3D             = -999;
      flightDistance3DError        = -999;
      flightDistance3DSignificance = -999;	
	
      } //init
    }; // struct


    ////////////////////////////////////////////////////
    //
    // typedef
    //
    ////////////////////////////////////////////////////
    typedef edm::AssociationMap < edm::OneToValue<std::vector<reco::Track>, 
      reco::CombinedBTagInfo::TrackData, unsigned short> > TrackDataAssociation;

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
    // clear all information
    //
    void reset();

    //
    // accessors
    //
    double                                         jetPt ()                           {return jetPt_;}
    double                                         jetEta()                           {return jetEta_;}
    									         
    // edm::ref to primary vertex ?					         
    reco::Vertex                                    primaryVertex()                    {return primaryVertex_;}
    std::vector<reco::Vertex>                       secVertices()                      {return secondaryVertices_;}
    std::vector<reco::Vertex>::const_iterator       secVerticesBegin()                 {return secondaryVertices_.begin();}
    std::vector<reco::Vertex>::const_iterator       secVerticesEnd()                   {return secondaryVertices_.end();}
  									         
    std::vector<reco::TrackRef>                     tracksAboveCharm()                 {return tracksAboveCharm_;}
    std::vector<reco::TrackRef>::const_iterator     tracksAboveCharmBegin()            {return tracksAboveCharm_.begin();}
    std::vector<reco::TrackRef>::const_iterator     tracksAboveCharmEnd()              {return tracksAboveCharm_.end();}
									         
    std::vector<reco::TrackRef>                     tracksAtSecondaryVertex()          {return tracksAtSecondaryVertex_;}
    std::vector<reco::TrackRef>::const_iterator     tracksAtSecondaryVertexBegin()     {return tracksAtSecondaryVertex_.begin();}
    std::vector<reco::TrackRef>::const_iterator     tracksAtSecondaryVertexEnd()       {return tracksAtSecondaryVertex_.end();}
    									         
    int                                             nSecVertices()                     {return secondaryVertices_.size();}
    VertexType                                      vertexType()                       {return vertexType_;}
    double                                          vertexMass()                       {return vertexMass_;}
    int                                             vertexMultiplicity()               {return vertexMultiplicity_;}
    double                                          eSVXOverE()                        {return eSVXOverE_;}
	                                                		                      
    GlobalVector                                    pAll()                             {return pAll_;}
    GlobalVector                                    pB()                               {return pB_;}
    double                                          pBLong()                           {return bPLong_;}
    double                                          pBPt()                             {return bPt_;}
	                                                		                      
    double                                          meanTrackRapidity()                {return meanTrackY_;}
    		                                      	                         
    double                                          angleGeomKinJet()                  {return angleGeomKinJet_;}
    double                                          angleGeomKinVertex()               {return angleGeomKinVertex_;}
					      
					      
    double                                          flightDistance2DMin()              {return flightDistance2DMin_              ;}
    double                                          flightDistanceSignificance2DMin () {return flightDistanceSignificance2DMin_  ;}
    double                                          flightDistance3DMin()              {return flightDistance3DMin_              ;}
    double                                          flightDistanceSignificance3DMin()  {return flightDistanceSignificance3DMin_  ;}
	                                                 				
    double                                          flightDistance2DMax()              {return flightDistance2DMax_              ;}
    double                                          flightDistanceSignificance2DMax()  {return flightDistanceSignificance2DMax_  ;}
    double                                          flightDistance3DMax ()             {return flightDistance3DMax_              ;}
    double                                          flightDistanceSignificance3DMax()  {return flightDistanceSignificance3DMax_  ;}
	                                                 				
    double                                          flightDistance2DMean()             {return flightDistance2DMean_             ;}
    double                                          flightDistanceSignificance2DMean() {return flightDistanceSignificance2DMean_ ;}
    double                                          flightDistance3DMean()             {return flightDistance3DMean_             ;}
    double                                          flightDistanceSignificance3DMean (){return flightDistanceSignificance3DMean_ ;}

    // possibly revisit this if calculation of lifetime-signed 2d IP
    // is avaialable via Track itself
    double                                          first2DSignedIPSigniAboveCut()     {return first2DSignedIPSigniAboveCut_;}
				                             
    //
    // setters
    //
    void setJetPt (double pt)                                   {jetPt_                           = pt;}
    void setJetEta(double eta)                                  {jetEta_                          = eta;}
    						                
    // pass (ref to?) primary vertex		                
    void setPrimaryVertex(reco::Vertex pv)                      {primaryVertex_                   = pv;}
    void addSecondaryVertex(reco::Vertex sv)                    {secondaryVertices_.push_back(sv);}
    void addTrackAtSecondaryVertex(reco::TrackRef trackRef)     {tracksAtSecondaryVertex_.push_back(trackRef);}
    
    void setVertexType( VertexType type)                        {vertexType_                      = type;}
    void setVertexMass( double mass)                            {vertexMass_                      = mass;}
    void setVertexMultiplicity(int mult)                        {vertexMultiplicity_              = mult;}
    void setESVXOverE( double e)                                {eSVXOverE_                       = e;}
	 					                  		               
    void setEnergyBTracks(double energy)                        {energyBTracks_                   = energy;}
    void setEnergyAllTracks(double energy)                      {energyAllTracks_                 = energy;}
	      		                                          		               
    void setPAll(GlobalVector p)                                {pAll_                            = p;}
    void setPB(GlobalVector p)                                  {pB_                              = p;}
    void setBPLong(double pLong)                                {bPLong_                          = pLong;}
    void setBPt(double pt)                                      {bPt_                             = pt;}
    void setMeanTrackRapidity(double meanY)                     {meanTrackY_                      = meanY;}
    	 		                        			               
    void setAngleGeomKinJet(double angle)                       {angleGeomKinJet_                  = angle;}
    void setAngleGeomKinVertex(double angle)                    {angleGeomKinVertex_               = angle;}	

    void addTrackAboveCharm(reco::TrackRef trackRef)            {tracksAboveCharm_.push_back(trackRef);}

    void setFlightDistance2DMin(double value)                   {flightDistance2DMin_              = value;}
    void setFlightDistanceSignificance2DMin (double value)      {flightDistanceSignificance2DMin_  = value;}
    void setFlightDistance3DMin(double value)                   {flightDistance3DMin_              = value;}
    void setFlightDistanceSignificance3DMin(double value)       {flightDistanceSignificance3DMin_  = value;}

    void setFlightDistance2DMax(double value)                   {flightDistance2DMax_              = value;}
    void setFlightDistanceSignificance2DMax(double value)       {flightDistanceSignificance2DMax_  = value;}
    void setFlightDistance3DMax (double value)                  {flightDistance3DMax_              = value;}
    void setFlightDistanceSignificance3DMax(double value)       {flightDistanceSignificance3DMax_  = value;}

    void setFlightDistance2DMean(double value)                  {flightDistance2DMean_             = value;}
    void setFlightDistanceSignificance2DMean(double value)      {flightDistanceSignificance2DMean_ = value;}
    void setFlightDistance3DMean(double value)                  {flightDistance3DMean_             = value;}
    void setFlightDistanceSignificance3DMean (double value)     {flightDistanceSignificance3DMean_ = value;}

    void setFirst2DSignedIPSigniAboveCut(double ipSignificance) {first2DSignedIPSigniAboveCut_ = ipSignificance;}

    //
    // map to access track map information
    //
    // maybe possible to use map tools here?
    bool              existTrackData(reco::TrackRef trackRef);
    void              flushTrackData();
    void              storeTrackData(reco::TrackRef trackRef,
				     const CombinedBTagInfo::TrackData& trackData);
    void              printTrackData();
    int               sizeTrackData();
    const TrackData*  getTrackData(reco::TrackRef trackRef);


    // is this the "best" way to do it?
    bool              existVertexData(std::vector<reco::Vertex>::const_iterator vertexRef);
    void              flushVertexData();
    void              storeVertexData(std::vector<reco::Vertex>::const_iterator vertexRef,
				      const CombinedBTagInfo::VertexData& vertexData);
    int               sizeVertexData();
    VertexData*       getVertexData(std::vector<reco::Vertex>::const_iterator vertexRef);


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

    // maybe easier/better to have templated class to handle the maps?
    TrackDataAssociation                                                                 trackDataMap_;
    std::map <std::vector<reco::Vertex>::const_iterator, CombinedBTagInfo::VertexData>   vertexDataMap_;

  }; // class
 
} // namespace reco

#endif
