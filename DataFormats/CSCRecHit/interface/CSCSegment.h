#ifndef CSCRecHit_CSCSegment_h
#define CSCRecHit_CSCSegment_h

/** \class CSCSegment
 *  Describes a reconstructed track segment in the 6 layers of a CSC chamber. 
 *  This is 4-dimensional since it has an origin (x,y) and a direction (x,y)
 *  in the local coordinate system of the chamber.
 *
 *  $Date: 2006/07/03 15:13:01 $
 *  \author Matteo Sani
 *  \author Rick Wilkinson
 *  \author Tim Cox
 */

#include <DataFormats/TrackingRecHit/interface/RecSegment.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>

#include <iosfwd>

class CSCDetId;

class CSCSegment : public RecSegment {

public:

    /// Default constructor
    CSCSegment() {};
	
    /// Constructor
    CSCSegment(std::vector<const CSCRecHit2D*> proto_segment, LocalPoint origin, 
        	LocalVector direction, AlgebraicSymMatrix errors, double chi2);
  
    /// Destructor
    virtual ~CSCSegment();

    //--- Base class interface
    CSCSegment* clone() const { return new CSCSegment(*this); }

    LocalPoint localPosition() const { return theOrigin; }
    LocalError localPositionError() const ;
	
    LocalVector localDirection() const { return theLocalDirection; }
    LocalError localDirectionError() const ;

    /// Parameters of the segment, for the track fit: (x,y,dx/dy,dy/dz)
    AlgebraicVector parameters() const;

    /// Covariance matrix fo parameters()
    AlgebraicSymMatrix parametersError() const { return theCovMatrix; }

    /// The projection matrix relates the trajectory state parameters to the segment parameters().
    virtual AlgebraicMatrix projectionMatrix() const;

    virtual std::vector<const TrackingRecHit*> recHits() const;

    virtual std::vector<TrackingRecHit*> recHits();

    double chi2() const { return theChi2; };

    virtual int dimension() const { return 4; }

    virtual int degreesOfFreedom() const { return 2*nRecHits() - 4;}	 

    virtual DetId geographicalId() const { return theDetId; }  // Slice off the CSC part :)

    //--- Extension of the interface
        
    const std::vector<CSCRecHit2D>& specificRecHits() const { return theCSCRecHits; }

    int nRecHits() const { return theCSCRecHits.size(); }        

    CSCDetId cscDetId() const { return theDetId; }
		
    void print() const;		
   
private:
    
    CSCDetId theDetId;
    std::vector<CSCRecHit2D> theCSCRecHits;
    LocalPoint theOrigin;   // in chamber frame - the GeomDet local coordinate system
    LocalVector theLocalDirection; // in chamber frame - the GeomDet local coordinate system
    AlgebraicSymMatrix theCovMatrix; // the covariance matrix
    double theChi2;
};

std::ostream& operator<<(std::ostream& os, const CSCSegment& seg);

#endif // CSCRecHit_CSCSegment_h
