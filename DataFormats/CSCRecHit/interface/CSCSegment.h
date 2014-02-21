#ifndef CSCRecHit_CSCSegment_h
#define CSCRecHit_CSCSegment_h

/** \class CSCSegment
 *  Describes a reconstructed track segment in the 6 layers of a CSC chamber. 
 *  This is 4-dimensional since it has an origin (x,y) and a direction (x,y)
 *  in the local coordinate system of the chamber.
 *
 *  $Date: 2013/04/22 22:41:32 $
 *  \author Matteo Sani
 *  \author Rick Wilkinson
 *  \author Tim Cox
 */

#include <DataFormats/TrackingRecHit/interface/RecSegment.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>

#include <iosfwd>

class CSCDetId;

class CSCSegment GCC11_FINAL : public RecSegment {

public:

    /// Default constructor
    CSCSegment() : theChi2(0.), aME11a_duplicate(false) {}
	
    /// Constructor
    CSCSegment(const std::vector<const CSCRecHit2D*>& proto_segment, LocalPoint origin, 
        	LocalVector direction, AlgebraicSymMatrix errors, double chi2);
  
    /// Destructor
    virtual ~CSCSegment();

    //--- Base class interface
    CSCSegment* clone() const { return new CSCSegment(*this); }

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
        
    const std::vector<CSCRecHit2D>& specificRecHits() const { return theCSCRecHits; }

    int nRecHits() const { return theCSCRecHits.size(); }        

    CSCDetId cscDetId() const { return  geographicalId(); }

    void setDuplicateSegments(std::vector<CSCSegment*>& duplicates);

    bool isME11a_duplicate() const { return (theDuplicateSegments.size() > 0 ? true : false); }
    // a copy of the duplicated segments (ME1/1a only) 
    const std::vector< CSCSegment> & duplicateSegments() const { return theDuplicateSegments; } 
    
    bool testSharesAllInSpecificRecHits( const std::vector<CSCRecHit2D>& specificRecHits_1,
					 const std::vector<CSCRecHit2D>& specificRecHits_2,
					 CSCRecHit2D::SharedInputType) const;
    
    //bool sharesRecHits(CSCSegment  & anotherSegment, CSCRecHit2D::SharedInputType);
    // checks if ALL the rechits share the specific input (allWires, allStrips or all)
    bool sharesRecHits(const CSCSegment  & anotherSegment, CSCRecHit2D::SharedInputType sharesInput) const;
    // checks if ALL the rechits share SOME wire AND SOME strip input
    bool sharesRecHits(const CSCSegment  & anotherSegment) const;

    float time() const;
    
    void print() const;		
    
 private:
    
    std::vector<CSCRecHit2D> theCSCRecHits;
    LocalPoint theOrigin;   // in chamber frame - the GeomDet local coordinate system
    LocalVector theLocalDirection; // in chamber frame - the GeomDet local coordinate system
    AlgebraicSymMatrix theCovMatrix; // the covariance matrix
    double theChi2;
    bool aME11a_duplicate;
    std::vector<CSCSegment> theDuplicateSegments;// ME1/1a only
};

std::ostream& operator<<(std::ostream& os, const CSCSegment& seg);

#endif // CSCRecHit_CSCSegment_h
