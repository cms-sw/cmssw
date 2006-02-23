#ifndef DTRecHit_DTRecSegment4D_h
#define DTRecHit_DTRecSegment4D_h

/** \class DTRecSegment4D
 *
 * 4 parameters RecHits for MuonBarrel DT
 *
 * $Date: 22/02/2006 15:08:00 CET $
 * $Revision: 1.0 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 *
 */

/* Base Class Headers */
#include "DataFormats/TrackingRecHit/interface/RecSegment4D.h"

/* Collaborating Class Declarations */
class DTRecSegment2DPhi;
class DTRecSegment2D;

/* C++ Headers */
#include <iosfwd>

/* ====================================================================== */

/* Class DTRecSegment4D Interface */

class DTRecSegment4D : public RecSegment4D{

  public:

/// Constructor
    DTRecSegment4D(DTRecSegment2DPhi* phiSeg, DTRecSegment2D* zedSeg) ;
    DTRecSegment4D(DTRecSegment2DPhi* phiSeg) ;
    DTRecSegment4D(DTRecSegment2D* zedSeg) ;

/// Destructor
    ~DTRecSegment4D() ;

/* Operations */ 
    AlgebraicVector parameters() const ;

    AlgebraicSymMatrix parametersError() const ;

    LocalPoint localPosition() const { return thePosition;}

    LocalError localPositionError() const ;

    LocalVector localDirection() const { return theDirection;}

    LocalError localDirectionError() const ;

    double chi2() const ;

    int degreesOfFreedom() const ;

  private:
    LocalPoint thePosition;   // in chamber frame
    LocalVector theDirection; // in chamber frame

    AlgebraicSymMatrix theCovMatrix; // the covariance matrix

    DTRecSegment2DPhi* thePhiSeg;
    DTRecSegment2D* theZedSeg;

    AlgebraicMatrix theProjMatrix;  // the projection matrix
    int theDimension; // the dimension of this rechit

};

std::ostream& operator<<(std::ostream& os, const DTRecSegment4D& seg);

#endif // DTRecHit_DTRecSegment4D_h

