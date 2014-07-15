#ifndef DTSegment_DTSegmentCand_h
#define DTSegment_DTSegmentCand_h

/** \class DTSegmentCand
 *
 * A Candidate for a DT segment. It's used by the algorithm to build segments
 * and store relative information. It must be transformed into a DTSegment
 * for further use.
 *
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

/* Base Class Headers */

/* Collaborating Class Declarations */
#include "RecoLocalMuon/DTSegment/src/DTHitPairForFit.h"

/* C++ Headers */
#include <vector>
#include <set>
#include <iostream>

/* ====================================================================== */

/* Class DTSegmentCand Interface */

class DTSLRecSegment2D;
class DTChamberRecSegment2D; 
class DTChamber;
class DTSuperLayer;


class DTSegmentCand{

  public:
    struct AssPointLessZ ;
    typedef std::pair<std::shared_ptr<DTHitPairForFit>, DTEnums::DTCellSide> AssPoint;
    typedef std::set<AssPoint, AssPointLessZ> AssPointCont;

/// Constructor
    DTSegmentCand(AssPointCont& hits,
                  const DTSuperLayer* sl) ;

    DTSegmentCand(const AssPointCont& hits,
                  LocalPoint& position,
                  LocalVector& direction,
                  double chi2,
                  const AlgebraicSymMatrix& covMat,
                  const DTSuperLayer* sl);

/// Destructor
    virtual ~DTSegmentCand() ;

/* Operations */ 
    virtual bool good() const ;

    virtual bool hitsShareLayer() const;

    virtual unsigned int nHits() const { return theHits.size(); }

    /// the chi2 (NOT chi2/NDOF) of the fit
    virtual double chi2() const { return theChi2; }

    /// the chi2/NDOF of the fit
    virtual double chi2ndof() const { return theChi2/(nHits()-2.); }

    /// the t0 of the segment
    virtual double t0() const { return thet0; }

    /// equality operator based on position, direction, chi2 and nHits
    virtual bool operator==(const DTSegmentCand& seg);

    /// less operator based on nHits and chi2
    virtual bool operator<(const DTSegmentCand& seg);

    /// the super layer on which relies 
    const DTSuperLayer* superLayer() const {return theSL;}

    // in SL frame
    virtual LocalPoint  position() const { return thePosition; }

    // in SL frame
    virtual LocalVector direction() const { return theDirection;}

    /// the covariance matrix
    virtual AlgebraicSymMatrix covMatrix() const {return theCovMatrix; }

    virtual unsigned int NDOF() const { return nHits()-2; }

    ///set position
    virtual void setPosition(LocalPoint& pos) { thePosition=pos; }

    /// set direction
    virtual void setDirection(LocalVector& dir) { theDirection = dir; }

    /// add hits to the hit list.
    virtual void add(std::shared_ptr<DTHitPairForFit> hit, DTEnums::DTCellSide code) ;

    /// remove hit from the candidate
    virtual void removeHit(AssPoint hit) ;

    /// set chi2
    virtual void setChi2(double& chi2) { theChi2 = chi2; }

    /// set t0
    virtual void sett0(double& t0) { thet0 = t0; }

    /// number of shared hit pair with other segment candidate
    virtual int nSharedHitPairs(const DTSegmentCand& seg) const;

    /** return the hits shared with other segment and with confliction L/R
     * assignment */
    virtual AssPointCont conflictingHitPairs(const DTSegmentCand& seg) const;

    /// set the cov matrix
    virtual void setCovMatrix(AlgebraicSymMatrix& cov) { theCovMatrix = cov; }

    /// number of different layers with hits
    virtual int nLayers() const ;
    
    /// the used hits
    virtual AssPointCont hits() const { return theHits;}

    /// convert this DTSegmentCand into a DTRecSegment2D
    //  DTSLRecSegment2D* convert() const;
    operator DTSLRecSegment2D*() const;


    /// convert this DTSegmentCand into a DTChamberRecSegment2D
    operator DTChamberRecSegment2D*() const;

    struct AssPointLessZ : 
      public std::binary_function<const AssPoint&, const AssPoint&, bool> {
        public:
          bool operator()(const AssPoint& pt1, 
                          const AssPoint& pt2) const ; 
      };

  private:
    const DTSuperLayer* theSL; // the SL
    LocalPoint  thePosition;  // in SL frame
    LocalVector theDirection; // in SL frame
    double theChi2;           // chi2 of the fit
    double thet0;             // the t0 offset

    /// mat[1][1]=sigma (dx/dz)
    /// mat[2][2]=sigma (x)
    /// mat[1][2]=cov(dx/dz,x)
    AlgebraicSymMatrix theCovMatrix; // the covariance matrix

    AssPointCont theHits; // the used hits

  protected:
    static const double chi2max; // to be tuned!!
    static const unsigned int nHitsMin; // to be tuned!!
};

std::ostream& operator<<(std::ostream& out, const DTSegmentCand& seg) ;
std::ostream& operator<<(std::ostream& out, const DTSegmentCand::AssPoint& hit) ;
#endif // DTSegment_DTSegmentCand_h
