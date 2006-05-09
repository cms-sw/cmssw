#ifndef CSCRecHit_CSCSegment_h
#define CSCRecHit_CSCSegment_h

/** \class CSCSegment
 *  Describes a 4-dim reconstructed segment in a CSC chamber. 
 *
 *  $Date: 2006/05/09 08:38:42 $
 *  \author Matteo Sani
 */

#include <DataFormats/TrackingRecHit/interface/RecSegment4D.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>

#include <iosfwd>

class CSCDetId;

class CSCSegment : public RecSegment4D {

public:

    /// Default constructor
    CSCSegment() {};
	
    /// Constructor	
    CSCSegment(std::vector<CSCRecHit2D> proto_segment, LocalPoint origin, 
        	LocalVector direction, AlgebraicSymMatrix errors, double chi2);
  
    /// Destructor
    virtual ~CSCSegment();

    /* Operations */ 
    const std::vector<CSCRecHit2D>& specificRecHits() const { return theCSCRecHits; }  
	virtual std::vector<const TrackingRecHit*> recHits() const;
	virtual std::vector<TrackingRecHit*> recHits();
        
	int nRecHits() const { return theCSCRecHits.size(); }
        
    CSCDetId cscDetId() const { return theDetId; }
		
	void print() const;
		
	AlgebraicVector parameters() const;
    AlgebraicSymMatrix parametersError() const;
		
	LocalPoint localPosition() const { return theOrigin; }
	LocalError localPositionError() const ;
	
	LocalVector localDirection() const { return theLocalDirection; }
    LocalError localDirectionError() const ;
	
    double chi2() const { return theChi2; };
		
	CSCSegment* clone() const { return new CSCSegment(*this); }
	virtual DetId geographicalId() const { return theDetId; }  // Slice off the CSC part :)
	virtual int degreesOfFreedom() const { return 2*nRecHits() - 4;}	 
		
   
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
