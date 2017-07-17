#ifndef GEMRecHit_GEMSegment_h
#define GEMRecHit_GEMSegment_h

/** \class GEMSegment derived by the CSC segment
 *  Describes a reconstructed track segment in the 2+ layers of a GEM chamber.
 *  This is 4-dimensional since it has an origin (x,y) and a direction (x,y)
 *  in the local coordinate system of the chamber.
 *
 *  \author Piet Verwilligen
 */

#include "DataFormats/TrackingRecHit/interface/RecSegment.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"

#include <iosfwd>

class GEMDetId;

class GEMSegment GCC11_FINAL : public RecSegment {

public:

    /// Default constructor
    GEMSegment() : theChi2(0.){}
	
    /// Constructor
    GEMSegment(const std::vector<const GEMRecHit*>& proto_segment, const LocalPoint& origin, 
	       const LocalVector& direction, const AlgebraicSymMatrix& errors, double chi2);

    GEMSegment(const std::vector<const GEMRecHit*>& proto_segment, const LocalPoint& origin, 
	       const LocalVector& direction, const AlgebraicSymMatrix& errors, double chi2, float bx);

    GEMSegment(const std::vector<const GEMRecHit*>& proto_segment, const LocalPoint& origin, 
	       const LocalVector& direction, const AlgebraicSymMatrix& errors, double chi2, double time, double timeErr);
  
    /// Destructor
    virtual ~GEMSegment();

    //--- Base class interface
    GEMSegment* clone() const { return new GEMSegment(*this); }

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
        
    const std::vector<GEMRecHit>& specificRecHits() const { return theGEMRecHits; }

    int nRecHits() const { return theGEMRecHits.size(); }        

    GEMDetId gemDetId() const { return  geographicalId(); }

    float time()    const { return theTimeValue; }
    float timeErr() const { return theTimeUncrt; }
    float bunchX()  const { return theBX; }
    void  print()   const;		
    
 private:
    
    std::vector<GEMRecHit> theGEMRecHits;
    LocalPoint  theOrigin;           // in chamber frame - the GeomDet local coordinate system
    LocalVector theLocalDirection;   // in chamber frame - the GeomDet local coordinate system
    AlgebraicSymMatrix theCovMatrix; // the covariance matrix
    double theChi2;                  // the Chi squared of the segment fit
    double theTimeValue;             // the best time estimate of the segment
    double theTimeUncrt;             // the uncertainty on the time estimation
    float  theBX;                    // the bunch crossing
};

std::ostream& operator<<(std::ostream& os, const GEMSegment& seg);

#endif 
