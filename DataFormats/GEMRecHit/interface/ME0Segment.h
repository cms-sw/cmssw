#ifndef GEMRecHit_ME0Segment_h
#define GEMRecHit_ME0Segment_h

/** \class ME0Segment derived by the CSC segment
 *  Describes a reconstructed track segment in the 6 layers of the ME0 system.
 *  This is 4-dimensional since it has an origin (x,y) and a direction (x,y)
 *  in the local coordinate system of the chamber.
 *
 *  $Date: 2014/02/04 12:41:32 $
 *  \author Marcello Maggi
 */

#include <DataFormats/TrackingRecHit/interface/RecSegment.h>
#include <DataFormats/GEMRecHit/interface/ME0RecHitCollection.h>

#include <iosfwd>

class ME0DetId;

class ME0Segment GCC11_FINAL : public RecSegment {

public:

    /// Default constructor
    ME0Segment() : theChi2(0.){}
	
    /// Constructor
    ME0Segment(const std::vector<const ME0RecHit*>& proto_segment, LocalPoint origin, 
        	LocalVector direction, AlgebraicSymMatrix errors, double chi2);
  
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

    virtual std::vector<const TrackingRecHit*> recHits() const;

    virtual std::vector<TrackingRecHit*> recHits();

    double chi2() const { return theChi2; };

    virtual int dimension() const { return 4; }

    virtual int degreesOfFreedom() const { return 2*nRecHits() - 4;}	 

    //--- Extension of the interface
        
    const std::vector<ME0RecHit>& specificRecHits() const { return theME0RecHits; }

    int nRecHits() const { return theME0RecHits.size(); }        

    ME0DetId me0DetId() const { return  geographicalId(); }

    float time() const;
    
    void print() const;		
    
 private:
    
    std::vector<ME0RecHit> theME0RecHits;
    LocalPoint theOrigin;   // in chamber frame - the GeomDet local coordinate system
    LocalVector theLocalDirection; // in chamber frame - the GeomDet local coordinate system
    AlgebraicSymMatrix theCovMatrix; // the covariance matrix
    double theChi2;
};

std::ostream& operator<<(std::ostream& os, const ME0Segment& seg);

#endif 
