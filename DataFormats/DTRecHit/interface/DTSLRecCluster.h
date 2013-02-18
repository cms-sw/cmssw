#ifndef DTSLRECCLUSTER_H
#define DTSLRECCLUSTER_H

/** \class DTSLRecCluster
 *
 * Description:
 *  
 *  detailed description
 *
 * \author : Stefano Lacaprara - INFN LNL <stefano.lacaprara@pd.infn.it>
 * $date   : 17/04/2008 15:17:03 CEST $
 *
 * Modification:
 *
 */

/* Base Class Headers */
#include "DataFormats/TrackingRecHit/interface/RecHit1D.h"

/* Collaborating Class Declarations */
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"

/* C++ Headers */
#include <iosfwd>

/* ====================================================================== */

/* Class DTSLRecCluster Interface */

class DTSLRecCluster : public RecHit1D {

  public:

    /* Constructor */ 
    DTSLRecCluster() {}

    /// c'tor from hits
    DTSLRecCluster(const DTSuperLayerId id, const std::vector<DTRecHit1DPair>& pair) ;

    /// complete constructor
    DTSLRecCluster(const DTSuperLayerId id,
                   const LocalPoint&,
                   const LocalError&,
                   const std::vector<DTRecHit1DPair>& pair) ;

    /* Destructor */ 
    virtual ~DTSLRecCluster() {}

    /* Operations */ 
    /// The clone method needed by the clone policy
    virtual DTSLRecCluster* clone() const { return new DTSLRecCluster(*this); }

    /// The id of the superlayer on which reside the segment
    DTSuperLayerId superLayerId() const { return theSlid; }

    /// the vector of parameters (dx/dz,x)
    virtual AlgebraicVector parameters() const {
      return param( localPosition());
    }

    // The parameter error matrix 
    virtual AlgebraicSymMatrix parametersError() const {
      return parError( localPositionError());
    }

    /** return the projection matrix, which must project a parameter vector,
     * whose components are (q/p, dx/dz, dy/dz, x, y), into the vector returned
     * by parameters() */
    virtual AlgebraicMatrix projectionMatrix() const {
      return theProjectionMatrix;
    }

    /// return 2. The dimension of the matrix
    virtual int dimension() const { return 2;}
    virtual LocalPoint localPosition() const { return thePos; }
    virtual LocalError localPositionError() const { return thePosError; }

    /// return the hits
    virtual std::vector<const TrackingRecHit*> recHits() const ;

    virtual std::vector<TrackingRecHit*> recHits() ;

    /// Access to specific components
    std::vector<DTRecHit1DPair> specificRecHits() const { return thePairs; }

    int nHits() const { return thePairs.size() ; }

  private:
    DTSuperLayerId theSlid;

    LocalPoint thePos;
    LocalError thePosError;

    std::vector<DTRecHit1DPair> thePairs;

  private:
    static const AlgebraicMatrix theProjectionMatrix;

    AlgebraicVector param( const LocalPoint& lp) const {
      AlgebraicVector result(1);
      result[1]=lp.x();
      return result;
    }

    AlgebraicSymMatrix parError( const LocalError& le) const {
      AlgebraicSymMatrix m(1);
      m[0][0]=le.xx();
      return m;
    }

  protected:

};
std::ostream& operator<<(std::ostream& os, const DTSLRecCluster& seg);
#endif // DTSLRECCLUSTER_H

