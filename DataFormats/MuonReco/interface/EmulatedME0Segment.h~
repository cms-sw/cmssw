#ifndef ME0RecHit_ME0Segment_h
#define ME0RecHit_ME0Segment_h

/** \class ME0Segment
 *  Describes a simulated track segment in a z-plane modeling an ME0 chamber.
 *  It is modeled after CSCSegment, so it is a 4-dimensional object ( origin (x,y) , direction (x,y) )
 *  Formally it must be defined as a LocalPoint and LocalError but we actually use the global coordinate system.
 *  
 *  $Date: 2013/04/22 22:41:32 $
 *  
 *  \author David Nash
 */

#include <DataFormats/TrackingRecHit/interface/RecSegment.h>

#include <iosfwd>

class ME0Segment : public RecSegment {

public:

    /// Default constructor
 ME0Segment() : theOrigin(0,0,0), theLocalDirection(0,0,0), theCovMatrix(4,0),theChi2(0.) {}
	
    /// Constructor
    ME0Segment(LocalPoint origin, LocalVector direction, AlgebraicSymMatrix errors, double chi2);
  
    /// Destructor
    virtual ~ME0Segment();

    //--- Base class interface
    ME0Segment* clone() const { return new ME0Segment(*this); }

    LocalPoint localPosition() const { return theOrigin; }
    LocalError localPositionError() const ;
	
    LocalVector localDirection() const { return theLocalDirection; }
    LocalError localDirectionError() const ;

    /// Parameters of the segment, for the track fit in the order (dx/dz, dy/dz, x, y )
    AlgebraicVector parameters() const;

    /// Covariance matrix of parameters()
    AlgebraicSymMatrix parametersError() const { return theCovMatrix; }

    /// The projection matrix relates the trajectory state parameters to the segment parameters().
    virtual AlgebraicMatrix projectionMatrix() const;

    virtual std::vector<const TrackingRecHit*> recHits() const {return std::vector<const TrackingRecHit*> (); };

    virtual std::vector<TrackingRecHit*> recHits() {return std::vector<TrackingRecHit*>();};

    double chi2() const { return theChi2; };

    virtual int dimension() const { return 4; }

    virtual int degreesOfFreedom() const { return -1;}	 //Maybe  change later?

    //--- Extension of the interface
        
    int nRecHits() const { return 0;}  //theME0RecHits.size(); }        

    void print() const;		
    
 private:
    
    //std::vector<ME0RecHit2D> theME0RecHits;
    //  CAVEAT: these "Local" paramaters will in fact be filled by global coordinates
    LocalPoint theOrigin;   // in chamber frame - the GeomDet local coordinate system
    LocalVector theLocalDirection; // in chamber frame - the GeomDet local coordinate system
    AlgebraicSymMatrix theCovMatrix; // the covariance matrix
    double theChi2;
};

std::ostream& operator<<(std::ostream& os, const ME0Segment& seg);

#endif // ME0RecHit_ME0Segment_h
