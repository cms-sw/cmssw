/*! \class DTMatchBase
 *  \author Ignazio Lazzizzera
 *  \author Nicola Pozzobon
 *  \brief DT local triggers matched together, base class
 *         Used to store detector-related information and
 *         matched tracker objects
 *  \date 2010, Apr 10
 */

#include "DataFormats/L1DTPlusTrackTrigger/interface/DTMatchBase.h"

/// Define constant data members
const double DTMatchBase::theInnerCoilR = 315.0;
const double DTMatchBase::theOuterCoilR = 340.0;
const double DTMatchBase::theCoilRTilde =
  ( theInnerCoilR*theInnerCoilR + theInnerCoilR*theOuterCoilR + theOuterCoilR*theOuterCoilR ) / (3. * theOuterCoilR );

/// Constructor
DTMatchBase::DTMatchBase()
 : DTMatchBasePtMethods()
{
  theDTWheel   = -9999999;
  theDTStation = -9999999;
  theDTSector  = -9999999;
  theDTBX      = -9999999;
  theDTCode    = -9999999;

  theTSPhi    = -9999999;
  theTSPhiB   = -9999999;
  theTSTheta  = -9999999;

  theInnerBti = DTBtiId();
  theOuterBti = DTBtiId();
  theMatchedBti = DTBtiId();

  theDTPosition  = GlobalPoint();
  theDTDirection = GlobalVector();

  theFlagBXOK  = false;

  theAlphaDT = NAN;

  theMatchedStubRef = std::map< unsigned int, edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > >();
  theMatchedStubPos = std::map< unsigned int, GlobalPoint >();
  theTrackPtrInWindow = std::vector< edm::Ptr< TTTrack< Ref_PixelDigi_ > > >();
  theMatchedTrackPtr = edm::Ptr< TTTrack< Ref_PixelDigi_ > >();
}

/// Constructor
DTMatchBase::DTMatchBase( int aDTWheel, int aDTStation, int aDTSector,
                          int aDTBX, int aDTCode, int aTSPhi, int aTSPhiB, int aTSTheta,
                          bool aFlagBXOK )
 : DTMatchBasePtMethods()
{
  theDTWheel   = aDTWheel;
  theDTStation = aDTStation;
  theDTSector  = aDTSector;
  theDTBX      = aDTBX;
  theDTCode    = aDTCode;

  theTSPhi    = aTSPhi;
  theTSPhiB   = aTSPhiB;
  theTSTheta  = aTSTheta;

  theInnerBti = DTBtiId();
  theOuterBti = DTBtiId();
  theMatchedBti = DTBtiId();

  theDTPosition  = GlobalPoint();
  theDTDirection = GlobalVector();

  theFlagBXOK  = aFlagBXOK;

  theAlphaDT = NAN;

  theMatchedStubRef = std::map< unsigned int, edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > >();
  theMatchedStubPos = std::map< unsigned int, GlobalPoint >();
  theTrackPtrInWindow = std::vector< edm::Ptr< TTTrack< Ref_PixelDigi_ > > >();
  theMatchedTrackPtr = edm::Ptr< TTTrack< Ref_PixelDigi_ > >();
}

/// Constructor
DTMatchBase::DTMatchBase( int aDTWheel, int aDTStation, int aDTSector,
                          int aDTBX, int aDTCode, int aTSPhi, int aTSPhiB, int aTSTheta,
                          GlobalPoint aDTPosition, GlobalVector aDTDirection,
                          bool aFlagBXOK )
 : DTMatchBasePtMethods()
{
  theDTWheel   = aDTWheel;
  theDTStation = aDTStation;
  theDTSector  = aDTSector;
  theDTBX      = aDTBX;
  theDTCode    = aDTCode;

  theTSPhi    = aTSPhi;
  theTSPhiB   = aTSPhiB;
  theTSTheta  = aTSTheta;

  theInnerBti = DTBtiId();
  theOuterBti = DTBtiId();
  theMatchedBti = DTBtiId();

  theDTPosition  = aDTPosition;
  theDTDirection = aDTDirection;

  theFlagBXOK  = aFlagBXOK;

  theAlphaDT = NAN;

  theMatchedStubRef = std::map< unsigned int, edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > >();
  theMatchedStubPos = std::map< unsigned int, GlobalPoint >();
  theTrackPtrInWindow = std::vector< edm::Ptr< TTTrack< Ref_PixelDigi_ > > >();
  theMatchedTrackPtr = edm::Ptr< TTTrack< Ref_PixelDigi_ > >();
}

/// Copy constructor
DTMatchBase::DTMatchBase( const DTMatchBase& aDTMB )
 : DTMatchBasePtMethods( aDTMB )
{
  theDTWheel   = aDTMB.getDTWheel();
  theDTStation = aDTMB.getDTStation();
  theDTSector  = aDTMB.getDTSector();
  theDTBX      = aDTMB.getDTBX();
  theDTCode    = aDTMB.getDTCode();

  theTSPhi    = aDTMB.getDTTSPhi();
  theTSPhiB   = aDTMB.getDTTSPhiB();
  theTSTheta  = aDTMB.getDTTSTheta();

  theInnerBti = aDTMB.getInnerBtiId();
  theOuterBti = aDTMB.getOuterBtiId();
  theMatchedBti = aDTMB.getMatchedBtiId();

  theDTPosition  = aDTMB.getDTPosition();
  theDTDirection = aDTMB.getDTDirection();

  theFlagBXOK  = aDTMB.getFlagBXOK();

  theAlphaDT = aDTMB.getAlphaDT();

  theMatchedStubRef = aDTMB.getMatchedStubRefs();
  theMatchedStubPos = aDTMB.getMatchedStubPositions();
  theTrackPtrInWindow = aDTMB.getInWindowTrackPtrs();
  theMatchedTrackPtr = aDTMB.getPtMatchedTrackPtr();
}

/// Assignment operator
DTMatchBase& DTMatchBase::operator = ( const DTMatchBase& aDTMB )
{
  if ( this == &aDTMB ) /// Same object?
    return *this;       /// Yes, so skip assignment, and just return *this.

  this->DTMatchBase::operator = (aDTMB);

  theDTWheel   = aDTMB.getDTWheel();
  theDTStation = aDTMB.getDTStation();
  theDTSector  = aDTMB.getDTSector();
  theDTBX      = aDTMB.getDTBX();
  theDTCode    = aDTMB.getDTCode();

  theTSPhi    = aDTMB.getDTTSPhi();
  theTSPhiB   = aDTMB.getDTTSPhiB();
  theTSTheta  = aDTMB.getDTTSTheta();

  theInnerBti = aDTMB.getInnerBtiId();
  theOuterBti = aDTMB.getOuterBtiId();
  theMatchedBti = aDTMB.getMatchedBtiId();

  theDTPosition  = aDTMB.getDTPosition();
  theDTDirection = aDTMB.getDTDirection();

  theFlagBXOK  = aDTMB.getFlagBXOK();

  theAlphaDT = aDTMB.getAlphaDT();

  theMatchedStubRef = aDTMB.getMatchedStubRefs();
  theMatchedStubPos = aDTMB.getMatchedStubPositions();
  theTrackPtrInWindow = aDTMB.getInWindowTrackPtrs();
  theMatchedTrackPtr = aDTMB.getPtMatchedTrackPtr();

  return *this;
}

/// Method that sets all the possible Pt methods envisaged
void DTMatchBase::setPtMethods( float station2Correction, bool thirdMethodAccurate,
                                float aMinRInvB, float aMaxRInvB )
{
  /// First define the direction of assumed straight line trajectory
  /// of the muon inside DT chambers
  float bendingFromDT = this->getGlobalTSPhiB();

  if ( theDTStation == 2 )
  {
    bendingFromDT = bendingFromDT * station2Correction;
  }

  float globalTSPhi = this->getGlobalTSPhi();

  theAlphaDT = globalTSPhi + bendingFromDT;

  if ( theAlphaDT <= 0. )
  {
    theAlphaDT += 2. * M_PI;
  }
  if ( theAlphaDT > 0. )
  {
    theAlphaDT -= 2. * M_PI;
  }

  /// Then get intercept of that straight line trajectory to the ideal cylindrical
  /// boundary surface of the CMS magnetic field: this will be used as a point
  /// belonging to the muon trajectory inside the magnetic field toghether with
  /// already matched tracker stubs.
  /// The idea is trying to avoid use of the outermost tracker layers,
  /// stubs, however keeping comparable precision.

  /// Step 1
  float thisDTRho = static_cast< float >( theDTPosition.perp() );

  float discriminator  = 1. - thisDTRho * thisDTRho * sin( bendingFromDT ) * sin( bendingFromDT ) / ( theCoilRTilde * theCoilRTilde );

  if( discriminator < 0.)
  {
    return;
  }

  float sqrtDiscriminator = sqrt( discriminator );

  float thisXR = theCoilRTilde * cos( theAlphaDT ) * sqrtDiscriminator + thisDTRho * sin( bendingFromDT ) * sin( theAlphaDT );
  float thisYR = theCoilRTilde * sin( theAlphaDT ) * sqrtDiscriminator - thisDTRho * sin( bendingFromDT ) * cos( theAlphaDT );

  float thisPhiR = ( thisYR > 0. ) ? acos( thisXR / theCoilRTilde ) : ( 2. * M_PI - acos( thisXR / theCoilRTilde ) );

  if ( thisPhiR <= 0. )
  {
    thisPhiR += 2. * M_PI;
  }
  if ( thisPhiR > 2. * M_PI )
  {
    thisPhiR -= 2. * M_PI;
  }

//  float deltaPhiR = thisPhiR - globalTSPhi;

  /// Step 2, modify PhiR calculation to account for
  /// systematics effects close to Rtilde

  float anotherXR = theCoilRTilde * cos( theAlphaDT ) + thisDTRho * sin( theAlphaDT ) * bendingFromDT;
  float anotherYR = theCoilRTilde * sin( theAlphaDT ) - thisDTRho * cos( theAlphaDT ) * bendingFromDT;

  float thisPhiR1 = ( anotherYR > 0. ) ? acos( anotherXR / theCoilRTilde ) : ( 2. * M_PI - acos( anotherXR / theCoilRTilde ) );
  float thisPhiR2 = ( anotherXR > 0. ) ? abs( asin( anotherYR / theCoilRTilde ) ) : ( M_PI - abs( asin( anotherYR / theCoilRTilde ) ) );

  if ( thisPhiR1 <= 0. )
  {
    thisPhiR1 += 2. * M_PI;
  }
  if ( thisPhiR1 > 2. * M_PI )
  {
    thisPhiR1 -= 2. * M_PI;
  }

  if ( anotherYR < 0 )
  {
    thisPhiR2 = 2. * M_PI - thisPhiR2;
  }

  thisPhiR = ( thisPhiR1 + thisPhiR2 ) / 2;

  /// Step 3
  double auxVar = bendingFromDT * thisDTRho / theCoilRTilde;
  float thisPhiRI = thirdMethodAccurate ? ( theAlphaDT  - auxVar - 1./6. * auxVar * auxVar * auxVar ) : ( theAlphaDT  - auxVar );

  if ( thisPhiRI <= 0. )
  {
    thisPhiRI += 2. * M_PI;
  }
  if ( thisPhiRI > 2. * M_PI )
  {
    thisPhiRI -= 2. * M_PI;
  }

  /// Now, all the possible methods for the pt
  std::string methodName;
  std::vector< GlobalPoint > positionVector;

  /// Extrapolation to virtual Tracker layer
  GlobalPoint tempDTPosition( anotherXR, anotherYR, 0.0 );

  /// Same as tempDTPosition, but with linear extrapolation
  GlobalPoint lineDTPosition( theCoilRTilde*cos(thisPhiRI), theCoilRTilde*sin(thisPhiRI), 0.0 );

  /// Same as tempDTPosition, but using the proper discriminator
  GlobalPoint discDTPosition( thisXR, thisYR, 0.0 );

  /// Set the relative correction for the Pt calculation
  float thisCorrection = 0.;

  /// First bunch, stubs only
  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(4) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Stubs_6_4_1"); /// Stubs from L1, 4, and 6
    positionVector.push_back( theMatchedStubPos[1] );
    positionVector.push_back( theMatchedStubPos[4] );
    positionVector.push_back( theMatchedStubPos[6] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(2) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Stubs_6_2_1"); /// Stubs from L1, 2, and 6
    positionVector.push_back( theMatchedStubPos[1] );
    positionVector.push_back( theMatchedStubPos[2] );
    positionVector.push_back( theMatchedStubPos[6] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(3) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(4) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Stubs_4_3_1"); /// Stubs from L1, 3, and 4
    positionVector.push_back( theMatchedStubPos[1] );
    positionVector.push_back( theMatchedStubPos[3] );
    positionVector.push_back( theMatchedStubPos[4] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(2) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(4) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Stubs_4_2_1"); /// Stubs from L1, 2, and 4
    positionVector.push_back( theMatchedStubPos[1] );
    positionVector.push_back( theMatchedStubPos[2] );
    positionVector.push_back( theMatchedStubPos[4] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  /// Second bunch, two stubs and (0,0)
  if ( theMatchedStubPos.find(4) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Stubs_6_4_V"); /// Stubs from L4, and 6
    positionVector.push_back( GlobalPoint(0,0,0) );
    positionVector.push_back( theMatchedStubPos[4] );
    positionVector.push_back( theMatchedStubPos[6] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(2) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Stubs_6_2_V"); /// Stubs from L2, and 6
    positionVector.push_back( GlobalPoint(0,0,0) );
    positionVector.push_back( theMatchedStubPos[2] );
    positionVector.push_back( theMatchedStubPos[6] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Stubs_6_1_V"); /// Stubs from L1, and 6
    positionVector.push_back( GlobalPoint(0,0,0) );
    positionVector.push_back( theMatchedStubPos[1] );
    positionVector.push_back( theMatchedStubPos[6] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(2) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(4) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Stubs_4_2_V"); /// Stubs from L2, and 4
    positionVector.push_back( GlobalPoint(0,0,0) );
    positionVector.push_back( theMatchedStubPos[2] );
    positionVector.push_back( theMatchedStubPos[4] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(4) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Stubs_4_1_V"); /// Stubs from L1, and 4
    positionVector.push_back( GlobalPoint(0,0,0) );
    positionVector.push_back( theMatchedStubPos[1] );
    positionVector.push_back( theMatchedStubPos[4] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  /// Third bunch, DT and two stubs
  if ( theMatchedStubPos.find(5) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_6_5"); /// Stubs from L5, and 6
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[6] );
    positionVector.push_back( theMatchedStubPos[5] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(4) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_6_4"); /// Stubs from L4, and 6
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[6] );
    positionVector.push_back( theMatchedStubPos[4] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(3) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_6_3"); /// Stubs from L3, and 6
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[6] );
    positionVector.push_back( theMatchedStubPos[3] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(2) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_6_2"); /// Stubs from L2, and 6
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[6] );
    positionVector.push_back( theMatchedStubPos[2] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_6_1"); /// Stubs from L1, and 6
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[6] );
    positionVector.push_back( theMatchedStubPos[1] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(4) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(5) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_5_4"); /// Stubs from L4, and 5
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[5] );
    positionVector.push_back( theMatchedStubPos[4] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(3) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(5) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_5_3"); /// Stubs from L3, and 5
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[5] );
    positionVector.push_back( theMatchedStubPos[3] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(2) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(5) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_5_2"); /// Stubs from L2, and 5
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[5] );
    positionVector.push_back( theMatchedStubPos[2] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(5) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_5_1"); /// Stubs from L1, and 5
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[5] );
    positionVector.push_back( theMatchedStubPos[1] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(3) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(4) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_4_3"); /// Stubs from L3, and 4
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[4] );
    positionVector.push_back( theMatchedStubPos[3] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(2) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(4) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_4_2"); /// Stubs from L2, and 4
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[4] );
    positionVector.push_back( theMatchedStubPos[2] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(4) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_4_1"); /// Stubs from L1, and 4
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[4] );
    positionVector.push_back( theMatchedStubPos[1] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(2) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(3) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_3_2"); /// Stubs from L2, and 3
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[3] );
    positionVector.push_back( theMatchedStubPos[2] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(3) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_3_1"); /// Stubs from L1, and 3
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[3] );
    positionVector.push_back( theMatchedStubPos[1] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(2) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_2_1"); /// Stubs from L1, and 2
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[2] );
    positionVector.push_back( theMatchedStubPos[1] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  /// Fourth bunch, DT, one stub, and (0,0)
  if ( theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_6_V"); /// Stub from L6
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[6] );
    positionVector.push_back( GlobalPoint(0,0,0) );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(5) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_5_V"); /// Stub from L5
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[5] );
    positionVector.push_back( GlobalPoint(0,0,0) );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(4) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_4_V"); /// Stub from L4
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[4] );
    positionVector.push_back( GlobalPoint(0,0,0) );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(3) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_3_V"); /// Stub from L3
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[3] );
    positionVector.push_back( GlobalPoint(0,0,0) );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(2) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_2_V"); /// Stub from L2
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[2] );
    positionVector.push_back( GlobalPoint(0,0,0) );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("Mu_1_V"); /// Stub from L1
    positionVector.push_back( tempDTPosition );
    positionVector.push_back( theMatchedStubPos[1] );
    positionVector.push_back( GlobalPoint(0,0,0) );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  /// Fifth bunch, DT and two stubs (linear extrapolation to virtual Tracker layer)
  if ( theMatchedStubPos.find(5) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_6_5"); /// Stubs from L5, and 6
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[6] );
    positionVector.push_back( theMatchedStubPos[5] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(4) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_6_4"); /// Stubs from L4, and 6
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[6] );
    positionVector.push_back( theMatchedStubPos[4] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(3) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_6_3"); /// Stubs from L3, and 6
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[6] );
    positionVector.push_back( theMatchedStubPos[3] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(2) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_6_2"); /// Stubs from L2, and 6
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[6] );
    positionVector.push_back( theMatchedStubPos[2] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_6_1"); /// Stubs from L1, and 6
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[6] );
    positionVector.push_back( theMatchedStubPos[1] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(4) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(5) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_5_4"); /// Stubs from L4, and 5
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[5] );
    positionVector.push_back( theMatchedStubPos[4] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(3) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(5) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_5_3"); /// Stubs from L3, and 5
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[5] );
    positionVector.push_back( theMatchedStubPos[3] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(2) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(5) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_5_2"); /// Stubs from L2, and 5
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[5] );
    positionVector.push_back( theMatchedStubPos[2] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(5) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_5_1"); /// Stubs from L1, and 5
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[5] );
    positionVector.push_back( theMatchedStubPos[1] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(3) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(4) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_4_3"); /// Stubs from L3, and 4
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[4] );
    positionVector.push_back( theMatchedStubPos[3] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(2) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(4) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_4_2"); /// Stubs from L2, and 4
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[4] );
    positionVector.push_back( theMatchedStubPos[2] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(4) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_4_1"); /// Stubs from L1, and 4
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[4] );
    positionVector.push_back( theMatchedStubPos[1] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(2) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(3) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_3_2"); /// Stubs from L2, and 3
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[3] );
    positionVector.push_back( theMatchedStubPos[2] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(3) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_3_1"); /// Stubs from L1, and 3
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[3] );
    positionVector.push_back( theMatchedStubPos[1] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(2) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_2_1"); /// Stubs from L1, and 2
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[2] );
    positionVector.push_back( theMatchedStubPos[1] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  /// Sixth bunch, DT, one stub, and (0,0)s (linear extrapolation to virtual Tracker layer)
  if ( theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_6_V"); /// Stub from L6
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[6] );
    positionVector.push_back( GlobalPoint(0,0,0) );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(5) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_5_V"); /// Stub from L5
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[5] );
    positionVector.push_back( GlobalPoint(0,0,0) );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(4) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_4_V"); /// Stub from L4
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[4] );
    positionVector.push_back( GlobalPoint(0,0,0) );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(3) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_3_V"); /// Stub from L3
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[3] );
    positionVector.push_back( GlobalPoint(0,0,0) );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(2) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_2_V"); /// Stub from L2
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[2] );
    positionVector.push_back( GlobalPoint(0,0,0) );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("IMu_1_V"); /// Stub from L1
    positionVector.push_back( lineDTPosition );
    positionVector.push_back( theMatchedStubPos[1] );
    positionVector.push_back( GlobalPoint(0,0,0) );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  /// Seventh bunch, DT and two stubs (proper discriminator used)
  if ( theMatchedStubPos.find(5) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_6_5"); /// Stubs from L5, and 6
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[6] );
    positionVector.push_back( theMatchedStubPos[5] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(4) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_6_4"); /// Stubs from L4, and 6
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[6] );
    positionVector.push_back( theMatchedStubPos[4] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(3) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_6_3"); /// Stubs from L3, and 6
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[6] );
    positionVector.push_back( theMatchedStubPos[3] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(2) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_6_2"); /// Stubs from L2, and 6
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[6] );
    positionVector.push_back( theMatchedStubPos[2] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_6_1"); /// Stubs from L1, and 6
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[6] );
    positionVector.push_back( theMatchedStubPos[1] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(4) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(5) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_5_4"); /// Stubs from L4, and 5
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[5] );
    positionVector.push_back( theMatchedStubPos[4] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(3) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(5) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_5_3"); /// Stubs from L3, and 5
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[5] );
    positionVector.push_back( theMatchedStubPos[3] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(2) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(5) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_5_2"); /// Stubs from L2, and 5
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[5] );
    positionVector.push_back( theMatchedStubPos[2] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(5) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_5_1"); /// Stubs from L1, and 5
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[5] );
    positionVector.push_back( theMatchedStubPos[1] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(3) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(4) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_4_3"); /// Stubs from L3, and 4
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[4] );
    positionVector.push_back( theMatchedStubPos[3] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(2) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(4) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_4_2"); /// Stubs from L2, and 4
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[4] );
    positionVector.push_back( theMatchedStubPos[2] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(4) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_4_1"); /// Stubs from L1, and 4
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[4] );
    positionVector.push_back( theMatchedStubPos[1] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(2) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(3) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_3_2"); /// Stubs from L2, and 3
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[3] );
    positionVector.push_back( theMatchedStubPos[2] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(3) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_3_1"); /// Stubs from L1, and 3
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[3] );
    positionVector.push_back( theMatchedStubPos[1] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(2) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_2_1"); /// Stubs from L1, and 2
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[2] );
    positionVector.push_back( theMatchedStubPos[1] );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  /// Eight bunch, DT, one stub, and (0,0)s (proper discriminator used)
  if ( theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_6_V"); /// Stub from L6
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[6] );
    positionVector.push_back( GlobalPoint(0,0,0) );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(5) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_5_V"); /// Stub from L5
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[5] );
    positionVector.push_back( GlobalPoint(0,0,0) );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(4) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_4_V"); /// Stub from L4
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[4] );
    positionVector.push_back( GlobalPoint(0,0,0) );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(3) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_3_V"); /// Stub from L3
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[3] );
    positionVector.push_back( GlobalPoint(0,0,0) );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(2) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_2_V"); /// Stub from L2
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[2] );
    positionVector.push_back( GlobalPoint(0,0,0) );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("mu_1_V"); /// Stub from L1
    positionVector.push_back( discDTPosition );
    positionVector.push_back( theMatchedStubPos[1] );
    positionVector.push_back( GlobalPoint(0,0,0) );
    thisDTMPt->findPt( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  /// Ninth bunch, stubs only but finding all parameters
  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(4) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("LinStubs_6_4_1"); /// Stubs from L1, 4, and 6
    positionVector.push_back( theMatchedStubPos[1] );
    positionVector.push_back( theMatchedStubPos[4] );
    positionVector.push_back( theMatchedStubPos[6] );
    thisDTMPt->findPtAndParameters( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(2) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(6) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("LinStubs_6_2_1"); /// Stubs from L1, 2, and 6
    positionVector.push_back( theMatchedStubPos[1] );
    positionVector.push_back( theMatchedStubPos[2] );
    positionVector.push_back( theMatchedStubPos[6] );
    thisDTMPt->findPtAndParameters( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }


  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(3) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(4) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("LinStubs_4_3_1"); /// Stubs from L1, 3, and 4
    positionVector.push_back( theMatchedStubPos[1] );
    positionVector.push_back( theMatchedStubPos[3] );
    positionVector.push_back( theMatchedStubPos[4] );
    thisDTMPt->findPtAndParameters( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

  if ( theMatchedStubPos.find(1) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(2) != theMatchedStubPos.end() &&
       theMatchedStubPos.find(4) != theMatchedStubPos.end() )
  {
    positionVector.clear();
    DTMatchPt* thisDTMPt = new DTMatchPt();
    methodName = std::string("LinStubs_4_2_1"); /// Stubs from L1, 2, and 4
    positionVector.push_back( theMatchedStubPos[1] );
    positionVector.push_back( theMatchedStubPos[2] );
    positionVector.push_back( theMatchedStubPos[4] );
    thisDTMPt->findPtAndParameters( aMinRInvB, aMaxRInvB, positionVector, thisCorrection );
    this->addPtMethod( methodName, thisDTMPt );
  }

/*

  /// Tenth bunch, stubs with linear fit dephi vs invPt

std::string("LinFit_3_1")
std::string("LinFit_3_2")
std::string("LinFit_4_1")
std::string("LinFit_4_2")
std::string("LinFit_5_1")
std::string("LinFit_5_2")
std::string("LinFit_5_3")
std::string("LinFit_5_4")
std::string("LinFit_6_1")
std::string("LinFit_6_2")
std::string("LinFit_6_3")
std::string("LinFit_6_4")


    else if(theMethodLabels.at(s).find(string("LinFit")) != string::npos)  {
      // using linear fit of stub dephi vs invPt
      const int I = tracker_lay_Id_to_our( atoi( &((theMethodLabels.at(s))[7]) ) );
      const int J = tracker_lay_Id_to_our( atoi( &((theMethodLabels.at(s))[9]) ) );
      const float dephi = fabs(stubstub_dephi(I, J));
      const float slope = slope_linearfit(I, J);
      const float dephi_zero = y_intercept_linearfit(I, J);
      aPt = new DTMatchPt(theMethodLabels.at(s), slope, dephi_zero,
                  I, J, dephi, pSet, _flagMatch);



DTMatchPt::DTMatchPt(std::string const s, 
		     const float slope, const float dephi_zero,
		     const int I, const int J, const float dephi, 
		     const edm::ParameterSet& pSet, 
		     bool const flagMatch[]):
  _label(s) { 
  _Rb = _invRb = NAN;
  _Pt = _invPt = NAN;
  _alpha0 = _d = NAN;
  if( isnan(dephi) || isnan(dephi_zero) ) //|| dephi < 0.0015 ) 
    return;
  if( flagMatch[I] && flagMatch[J] && !isnan(slope) ) {
    _invPt = (dephi - dephi_zero)/slope;
    _Pt = slope/(dephi - dephi_zero);
  }
  return;
}

*/

}

