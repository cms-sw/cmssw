#ifndef DTRecHit_DTRecSegment2D_h
#define DTRecHit_DTRecSegment2D_h

/** \class DTRecSegment2D
 *
 * 2D Segments for the muon barrel system.
 * 2D means that this segment has information about position and direction in
 * one projection (r-phi or r-theta/zeta).
 *  
 * $Date: 22/02/2006 15:52:43 CET $
 * $Revision: 1.0 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 *
 */

/* Base Class Headers */
#include "DataFormats/TrackingRecHit/interface/RecSegment2D.h"

/* Collaborating Class Declarations */

/* C++ Headers */
#include <iosfwd>

/* ====================================================================== */

/* Class DTRecSegment2D Interface */

class DTRecSegment2D : public RecSegment2D {

  public:

/// Constructor
    DTRecSegment2D() ;

/// Destructor
    virtual ~DTRecSegment2D() ;

/* Operations */ 
    /// local position in SL frame
    virtual LocalPoint localPosition() const {return thePosition; }

    /// local position error in SL frame
    virtual LocalError localPositionError() const ;

    /// the local direction in SL frame
    virtual LocalVector localDirection() const { return theDirection; }

    /// the local direction error (xx,xy,yy) in SL frame: only xx is not 0.
    virtual LocalError localDirectionError() const;

    /// the chi2 of the fit
    virtual double chi2() const { return theChi2; }
    
    /// return the DOF of the segment 
    virtual int degreesOfFreedom() const ;

  private:
    LocalPoint  thePosition;  // in SL frame
    LocalVector theDirection; // in SL frame
    double theChi2;           // chi2 of the fit
    //vector<RecHit> theHits; // the hits with defined R/L

    /// mat[0][0]=sigma (dx/dz)
    /// mat[1][1]=sigma (x)
    /// mat[0][1]=cov(dx/dz,x)
    AlgebraicSymMatrix theCovMatrix; // the covariance matrix

};

std::ostream& operator<<(std::ostream& os, const DTRecSegment2D& seg);

#endif // DTRecHit_DTRecSegment2D_h

