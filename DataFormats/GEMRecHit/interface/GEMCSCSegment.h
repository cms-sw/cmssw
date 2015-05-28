#ifndef GEMRecHit_GEMCSCSegment_h
#define GEMRecHit_GEMCSCSegment_h

/** \class GEMCSCSegment
 *
 *  Based on CSCSegment class
 *  Describes a reconstructed track segment in the GEM + CSC chambers. 
 *  This is 4-dimensional since it has an origin (x,y) and a direction (x,y)
 *  in the local coordinate system of the (csc) chamber.
 *
 *  \author R. Radogna
 */

#include <DataFormats/TrackingRecHit/interface/RecSegment.h>

#include <DataFormats/GEMRecHit/interface/GEMRecHitCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iosfwd>

class CSCDetId;

class GEMCSCSegment GCC11_FINAL : public RecSegment {

public:

    /// Default constructor
    GEMCSCSegment() : theChi2(0.) {}
	
    /// Constructor
    GEMCSCSegment(const CSCSegment* csc_segment, const std::vector<const GEMRecHit*> gem_rhs, LocalPoint origin, LocalVector direction, AlgebraicSymMatrix errors, double chi2);
      
    /// Destructor
    virtual ~GEMCSCSegment();

    //--- Base class interface
    GEMCSCSegment* clone() const { return new GEMCSCSegment(*this); }

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

    double chi2() const { return theChi2; };

    virtual int dimension() const { return 4; }

    virtual int degreesOfFreedom() const { return 2*nRecHits() - 4;}

    int nRecHits() const 
    { 
      return (theGEMRecHits.size() + theCSCSegment.specificRecHits().size()); 
      edm::LogVerbatim("GEMCSCSegment")<< "[GEMCSCSegment :: nRecHits] CSC RecHits: " <<theCSCSegment.specificRecHits().size()
				       << " + GEM RecHits: " << theGEMRecHits.size() << " = " <<(theGEMRecHits.size() + theCSCSegment.specificRecHits().size());
    }
    

    //--- Return the constituents in different ways
    const CSCSegment                        cscSegment() const { return theCSCSegment; }
    const std::vector<GEMRecHit>&           gemRecHits() const { return theGEMRecHits; }
    const std::vector<CSCRecHit2D>&         cscRecHits() const { return theCSCSegment.specificRecHits(); }
    virtual std::vector<const TrackingRecHit*> recHits() const;
    virtual std::vector<TrackingRecHit*>       recHits();

    CSCDetId cscDetId() const { return  geographicalId(); }
    
    void print() const;		
    
 private:
    std::vector<GEMRecHit> theGEMRecHits;      // store GEM Rechits
    CSCSegment theCSCSegment;                  // store CSC RecHits and store CSC Segment 
                                               // eventually we have to disentangle if later on we decide 
                                               // not to have a one-to-one relationship anymore
                                               // i.e. if we allow the GEMCSC segment to modify the
                                               // (selection of the rechits of the) CSC segment
    LocalPoint theOrigin;                      // in chamber frame - the GeomDet local coordinate system
    LocalVector theLocalDirection;             // in chamber frame - the GeomDet local coordinate system
    AlgebraicSymMatrix theCovMatrix;           // the covariance matrix
    double theChi2;

};

std::ostream& operator<<(std::ostream& os, const GEMCSCSegment& seg);

#endif 
