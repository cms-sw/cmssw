#ifndef DTMatchPt_h
#define DTMatchPt_h

/*! \class DTBtiTrigger
 *  \author Ignazio Lazzizzera
 *  \brief container to store momentum information
 *  \date 2010, Apr 2
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <string>

/// Class implementation
class DTMatchPt
{
  public :

    /// Default trivial constructor
    DTMatchPt()
    {
      theRB = NAN;
      theRInvB = NAN;
      thePt = NAN;
      thePtInv = NAN;
      theAlpha0 = NAN;
      theD = NAN;
    }

    /// Copy constructor
    DTMatchPt( const DTMatchPt& aDTMPt )
    {
      theRB = aDTMPt.getRB();
      theRInvB = aDTMPt.getRInvB();
      thePt = aDTMPt.getPt();
      thePtInv = aDTMPt.getPtInv();
      theAlpha0 = aDTMPt.getAlpha0();
      theD = aDTMPt.getD();
    }

    /// Assignment operator
    DTMatchPt& operator = ( const DTMatchPt& aDTMPt )
    {
      if ( this == &aDTMPt ) /// Same object?
      {
        return *this;
      }

      theRB = aDTMPt.getRB();
      theRInvB = aDTMPt.getRInvB();
      thePt = aDTMPt.getPt();
      thePtInv = aDTMPt.getPtInv();
      theAlpha0 = aDTMPt.getAlpha0();
      theD = aDTMPt.getD();

      return *this;
    }

    /// Destructor
    ~DTMatchPt(){}

    /// Methods to retrieve the data members
    inline float const getRB() const     { return theRB; }
    inline float const getRInvB() const  { return theRInvB; }
    inline float const getPt() const     { return thePt; }
    inline float const getPtInv() const  { return thePtInv; }
    inline float const getAlpha0() const { return theAlpha0; }
    inline float const getD() const      { return theD; }

    /// Methods to find bending radius and Pt
    void findCurvatureRadius( float aMinRInvB, float aMaxRInvB,
                              std::vector< GlobalPoint > aPosVec );

    void findPt( float aMinRInvB, float aMaxRInvB,
                 std::vector< GlobalPoint > aPosVec,
                 float const aCorr );

    void findPtAndParameters( float aMinRInvB, float aMaxRInvB,
                              std::vector< GlobalPoint > aPosVec,
                              float const aCorr );

  private :

    /// Data members
    float theRB, theRInvB; /// Bending radius
    float thePt, thePtInv;
    float theAlpha0, theD;
};

#endif

