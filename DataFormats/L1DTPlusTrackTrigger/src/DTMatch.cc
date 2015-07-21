/*! \class DTMatch
 *  \author Ignazio Lazzizzera
 *  \author Sara Vanini
 *  \author Pierluigi Zotto
 *  \author Nicola Pozzobon
 *  \brief DT local triggers matched together.
 *         Objects of this class do correspond to DT muons that are then extrapolated
 *         to the stubs on the tracker layers, including a virtual one just enclosing
 *         the magnetic field volume. The main methods do aim at getting a tracker
 *         precision Pt.
 *         The matching stubs, mapped by tracker layer id, and tracker tracks are
 *         set as data members of the virtual base class DTMatchBase.
 *         Several objects of class DTMatchPt are built by methods defined in the
 *         base class DTMatchBasePtMethods.
 *  \date 2009, Feb 2
 */

#include "DataFormats/L1DTPlusTrackTrigger/interface/DTMatch.h"

/// Initialization method used in constructors
void DTMatch::init()
{
  /// DT Trigger order
  theDTTrigOrder = 9999999; /// unsigned int

  /// Flag if theta missing and BTI is used instead
  theThetaFlag = true;
  theDeltaTheta = -555555555; /// int

  /// Flag if redundant and to be rejected
  theRejectionFlag = false;

  /// Predicted positions
  /// NOTE: using arrays, overdimensioned to account for
  /// DetId layer index ranging from 1 to numberOfTriggerLayers
  /// Element [ 0 ] used to store prediction at vertex
  for ( unsigned int iLayer = 0; iLayer <= numberOfTriggerLayers; iLayer++ )
  {
    thePredPhi[ iLayer ] = -555555555; /// int
    thePredSigmaPhi[ iLayer ] = -555555555;
    thePredTheta[ iLayer ] = -555555555;
    thePredSigmaTheta[ iLayer ] = -555555555;
  }

  /// Predicted error on bending angle inside the tracker
  thePredSigmaPhiB = NAN; /// float

  /// Pt information
  thePtPriority = NAN; /// float
  thePtAverage = NAN;
  thePtPriorityFlag = false;
  thePtAverageFlag = false;
  thePtPriorityBin = NAN;
  thePtAverageBin = NAN;
  thePtTTTrackBin = NAN;
  thePtMajorityFullTkBin = NAN;
  thePtMajorityBin = NAN;
  thePtMixedModeBin = NAN;
}

/// Constructor
DTMatch::DTMatch()
 : DTMatchBase()
{
  init();
}

/// Constructor
DTMatch::DTMatch( int aDTWheel, int aDTStation, int aDTSector,
                  int aDTBX, int aDTCode, int aTSPhi, int aTSPhiB, int aTSTheta,
                  bool aFlagBXOK )
 : DTMatchBase( aDTWheel, aDTStation, aDTSector,
                aDTBX, aDTCode, aTSPhi, aTSPhiB, aTSTheta,
                aFlagBXOK )
{
  init();
}

/// Constructor
DTMatch::DTMatch( int aDTWheel, int aDTStation, int aDTSector,
                  int aDTBX, int aDTCode, int aTSPhi, int aTSPhiB, int aTSTheta,
                  GlobalPoint aDTPosition, GlobalVector aDTDirection,
                  bool aFlagBXOK )
 : DTMatchBase( aDTWheel, aDTStation, aDTSector,
                aDTBX, aDTCode, aTSPhi, aTSPhiB, aTSTheta,
                aDTPosition, aDTDirection,
                aFlagBXOK )
{
  init();
}

// Copy constructor
DTMatch::DTMatch( const DTMatch& aDTM )
 : DTMatchBase( aDTM )
{
  theDTTrigOrder = aDTM.getDTTTrigOrder();

  theThetaFlag = aDTM.getThetaFlag();
  theDeltaTheta = aDTM.getDeltaTheta();

  theRejectionFlag = aDTM.getRejectionFlag();

  thePredPhi[ 0 ] = aDTM.getPredVtxPhi();
  thePredSigmaPhi[ 0 ] = aDTM.getPredVtxSigmaPhi();
  thePredTheta[ 0 ] = aDTM.getPredVtxTheta();
  thePredSigmaTheta[ 0 ] = aDTM.getPredVtxSigmaTheta();

  for ( unsigned int iLayer = 1; iLayer <= numberOfTriggerLayers; iLayer++ )
  {
    thePredPhi[ iLayer ] = aDTM.getPredStubPhi( iLayer );
    thePredSigmaPhi[ iLayer ] = aDTM.getPredStubSigmaPhi( iLayer );
    thePredTheta[ iLayer ] = aDTM.getPredStubTheta( iLayer );
    thePredSigmaTheta[ iLayer ] = aDTM.getPredStubSigmaTheta( iLayer );
  }

  thePredSigmaPhiB = aDTM.getPredSigmaPhiB();

  thePtPriority = aDTM.getPtPriority();
  thePtAverage = aDTM.getPtAverage();
  thePtPriorityFlag = aDTM.getPtPriorityFlag();
  thePtAverageFlag = aDTM.getPtAverageFlag();

  thePtPriorityBin = aDTM.getPtPriorityBin();
  thePtAverageBin = aDTM.getPtAverageBin();
  thePtTTTrackBin = aDTM.getPtTTTrackBin();
  thePtMajorityFullTkBin = aDTM.getPtMajorityFullTkBin();
  thePtMajorityBin = aDTM.getPtMajorityBin();
  thePtMixedModeBin = aDTM.getPtMixedModeBin();
}

/// Assignment operator
DTMatch& DTMatch::operator = ( const DTMatch& aDTM )
{
  if ( this == &aDTM ) /// Same object?
    return *this;      /// Yes, so skip assignment, and just return *this.

  this->DTMatchBase::operator = (aDTM);

  theDTTrigOrder = aDTM.getDTTTrigOrder();

  theThetaFlag = aDTM.getThetaFlag();
  theDeltaTheta = aDTM.getDeltaTheta();

  theRejectionFlag = aDTM.getRejectionFlag();

  thePredPhi[ 0 ] = aDTM.getPredVtxPhi();
  thePredSigmaPhi[ 0 ] = aDTM.getPredVtxSigmaPhi();
  thePredTheta[ 0 ] = aDTM.getPredVtxTheta();
  thePredSigmaTheta[ 0 ] = aDTM.getPredVtxSigmaTheta();

  for ( unsigned int iLayer = 1; iLayer <= numberOfTriggerLayers; iLayer++ )
  {
    thePredPhi[ iLayer ] = aDTM.getPredStubPhi( iLayer );
    thePredSigmaPhi[ iLayer ] = aDTM.getPredStubSigmaPhi( iLayer );
    thePredTheta[ iLayer ] = aDTM.getPredStubTheta( iLayer );
    thePredSigmaTheta[ iLayer ] = aDTM.getPredStubSigmaTheta( iLayer );
  }

  thePredSigmaPhiB = aDTM.getPredSigmaPhiB();

  thePtPriority = aDTM.getPtPriority();
  thePtAverage = aDTM.getPtAverage();
  thePtPriorityFlag = aDTM.getPtPriorityFlag();
  thePtAverageFlag = aDTM.getPtAverageFlag();

  thePtPriorityBin = aDTM.getPtPriorityBin();
  thePtAverageBin = aDTM.getPtAverageBin();
  thePtTTTrackBin = aDTM.getPtTTTrackBin();
  thePtMajorityFullTkBin = aDTM.getPtMajorityFullTkBin();
  thePtMajorityBin = aDTM.getPtMajorityBin();
  thePtMixedModeBin = aDTM.getPtMixedModeBin();

  return *this;
}

/*** DT TRIGGER MOMENTUM PARAMETERISATION ***/
/// Return function for the parameterised Pt
int DTMatch::getDTPt() const
{
  /// For high quality seeds, it comes from a fit to PhiB [int] as a function of Pt [GeV/c]
  /// PhiB [int] = A + B / Pt [GeV/c]
  /// Pt [GeV/c] = B * 1/( PhiB [int] - A ) ~ B/PhiB [int]
  /// Low quality seeds after correction: assumed to be the same as if it were a high
  /// quality seed, as resolution is low enough to assume they are compatible

  int iSt = this->getDTStation() - 1;
  int iWh = this->getDTWheel() + 2;
  float thisPhiB = fabs( static_cast< float >(this->getDTTSPhiB()) );

  /// By Station + 2*Wheel
  float B_Pt[10] = {-668.1, -433.6, -757.8, -524.7, -751.8, -539.5, -757.5, -525.6, -667.0, -435.0};

  if ( thisPhiB > 0.0 )
  {
    return static_cast< int >( fabs(B_Pt[iSt+2*iWh]) / thisPhiB );
  }

  return 1000;
}

/// Return function for the minimum allowed parameterised Pt
int DTMatch::getDTPtMin( float nSigmas ) const
{
  /// For high quality seeds, it comes from a fit to PhiB [int] as a function of Pt [GeV/c]
  /// PhiB [int] = A + B / Pt [GeV/c]
  /// Pt [GeV/c] = B * 1/( PhiB [int] - A ) ~ B/PhiB [int]
  /// Low quality seeds after correction: assumed to be the same as if it were a high
  /// quality seed, as resolution is low enough to assume they are compatible

  int iSt = this->getDTStation() - 1;
  int iWh = this->getDTWheel() + 2;
  float thisPhiB = fabs( static_cast< float >(this->getDTTSPhiB()) );
  float thisSigmaPhiB = this->getPredSigmaPhiB();

  /// By Station + 2*Wheel
  float B_Pt[10] = {-668.1, -433.6, -757.8, -524.7, -751.8, -539.5, -757.5, -525.6, -667.0, -435.0};

  if ( thisPhiB > 0.0 )
  {
    float thisPhiBMax = thisPhiB + nSigmas * thisSigmaPhiB;
    return static_cast< int >( fabs(B_Pt[iSt+2*iWh]) / thisPhiBMax - 1);
  }

  return 0;
}

/// Return function for the maximum allowed parameterised Pt
int DTMatch::getDTPtMax( float nSigmas ) const
{
  /// For high quality seeds, it comes from a fit to PhiB [int] as a function of Pt [GeV/c]
  /// PhiB [int] = A + B / Pt [GeV/c]
  /// Pt [GeV/c] = B * 1/( PhiB [int] - A ) ~ B/PhiB [int]
  /// Low quality seeds after correction: assumed to be the same as if it were a high
  /// quality seed, as resolution is low enough to assume they are compatible

  int iSt = this->getDTStation() - 1;
  int iWh = this->getDTWheel() + 2;
  float thisPhiB = fabs( static_cast< float >(this->getDTTSPhiB()) );
  float thisSigmaPhiB = this->getPredSigmaPhiB();
  float thisPhiBMin = thisPhiB - nSigmas * thisSigmaPhiB;

  /// By Station + 2*Wheel
  float B_Pt[10] = {-668.1, -433.6, -757.8, -524.7, -751.8, -539.5, -757.5, -525.6, -667.0, -435.0};

  if ( thisPhiBMin > 0.0 )
  {
    return static_cast< int >( fabs(B_Pt[iSt+2*iWh]) / thisPhiBMin + 1);
  }

  return 1000;
}

/*** CHECK THE MATCHES ***/
/// Check the match with a stub in phi
bool DTMatch::checkStubPhiMatch( int anotherPhi, unsigned int aLayer, float nSigmas ) const
{
  int deltaPhi = this->findStubDeltaPhi( anotherPhi, aLayer );
  if ( deltaPhi < nSigmas * this->getPredStubSigmaPhi( aLayer ) )
  {
    return true;
  }
  return false;
}

/// Check the match with a stub in theta
bool DTMatch::checkStubThetaMatch( int anotherTheta, unsigned int aLayer, float nSigmas ) const
{
  int deltaTheta = abs( this->getPredStubTheta( aLayer ) - anotherTheta );
  int dtStubSigmaTheta = this->getPredStubSigmaTheta( aLayer );

  /// Check the theta flag (if false, rough theta is used,
  /// and a wire-based correction has been set)
  if ( !this->getThetaFlag() )
  {
    dtStubSigmaTheta += this->getDeltaTheta();
  }

  if ( deltaTheta < nSigmas * dtStubSigmaTheta )
  {
    return true;
  }
  return false;
}

/// Find the phi difference between the projected position and the stub position
int DTMatch::findStubDeltaPhi( int anotherPhi, unsigned int aLayer ) const
{
  int IMPI = static_cast< int >( M_PI * 4096. );
  int tempPhi1 = this->getPredStubPhi( aLayer );
  int tempPhi2 = anotherPhi;
  if ( tempPhi1 < 0 ) tempPhi1 += 2 * IMPI;
  if ( tempPhi2 < 0 ) tempPhi2 += 2 * IMPI;
  if ( tempPhi1 >= 2 * IMPI ) tempPhi1 -= 2 * IMPI;
  if ( tempPhi2 >= 2 * IMPI ) tempPhi2 -= 2 * IMPI;
  int tempDeltaPhi = abs( tempPhi1 - tempPhi2 );
  if ( tempDeltaPhi > IMPI )
  {
    tempDeltaPhi = 2 * IMPI - tempDeltaPhi;
  }
  return tempDeltaPhi;
}

/// Check the match with a track in phi
bool DTMatch::checkVtxPhiMatch( int anotherPhi, float nSigmas ) const
{
  int deltaPhi = this->findVtxDeltaPhi( anotherPhi );
  if ( deltaPhi < nSigmas * this->getPredVtxSigmaPhi() )
  {
    return true;
  }
  return false;
}

/// Check the match with a track in theta
bool DTMatch::checkVtxThetaMatch( int anotherTheta, float nSigmas ) const
{
  int deltaTheta = abs( this->getPredVtxTheta() - anotherTheta );
  int dtVtxSigmaTheta = this->getPredVtxSigmaTheta();

  /// Check the theta flag (if false, rough theta is used,
  /// and a wire-based correction has been set)
  if ( !this->getThetaFlag() )
  {
    dtVtxSigmaTheta += this->getDeltaTheta();
  }

  if ( deltaTheta < nSigmas * dtVtxSigmaTheta )
  {
    return true;
  }
  return false;
}

/// Find the phi difference between the projected direction
/// and the track direction at vertex
int DTMatch::findVtxDeltaPhi( int anotherPhi ) const
{
  int IMPI = static_cast< int >( M_PI * 4096. );
  int tempPhi1 = this->getPredVtxPhi();
  int tempPhi2 = anotherPhi;
  if ( tempPhi1 < 0 ) tempPhi1 += 2 * IMPI;
  if ( tempPhi2 < 0 ) tempPhi2 += 2 * IMPI;
  if ( tempPhi1 >= 2 * IMPI ) tempPhi1 -= 2 * IMPI;
  if ( tempPhi2 >= 2 * IMPI ) tempPhi2 -= 2 * IMPI;
  int tempDeltaPhi = abs( tempPhi1 - tempPhi2 );
  if ( tempDeltaPhi > IMPI )
  {
    tempDeltaPhi = 2 * IMPI - tempDeltaPhi;
  }
  return tempDeltaPhi;
}

/*
int DTMatch::corrPhiBend1ToCh2(int phib2) {
  // compatibility between primitives in different stations: used for ghost reduction
  // correlation function parameters for each wheel
  float a[5] = {5.E-5,                 1.E-5,                 2.E-5,                 2.E-5,                 5.E-5};
  float b[5] = {-0.0002,        6.E-5,                 -0.0001,        -7.E-05,        2.E-6};
  float c[5] = {1.4886,         1.4084,         1.3694,         1.4039,         1.4871};
  float d[5] = {0.7017,         0.3776,         0.6627,         0.623,                 0.5025};

  // find phib in station 1 correlated with phib given for station 2
  int phib1 = 0;
  int iwh = this->getDTWheel()+2;

  if(this->getDTStation()==1)
    phib1 = static_cast< int >(a[iwh]*phib2*phib2*phib2 +
                             b[iwh]*phib2*phib2 + c[iwh]*phib2 + d[iwh]);
  else //no correlation, return the value in input
    phib1 = phib2;

  return phib1;
}


int DTMatch::corrSigmaPhiBend1ToCh2(int phib2, int sigma_phib2) {
  // compatibility between primitives in different stations: used for ghost reduction
  // correlation function parameters for each wheel
  float a[5] = {5.E-5,                 1.E-5,                 2.E-5,                 2.E-5,                 5.E-5};
  float b[5] = {-0.0002,        6.E-5,                 -0.0001,        -7.E-05,        2.E-6};
  float c[5] = {1.4886,         1.4084,         1.3694,         1.4039,         1.4871};
  //float d[5] = {0.7017,         0.3776,         0.6627,         0.623,                 0.5025};

  // find phib error in station 1 correlated with phib given for station 2
  int sigma_phib1 = 0;
  int iwh = this->getDTWheel()+2;

  if(this->getDTStation()==1)
    sigma_phib1 = static_cast< int >(fabs((3*a[iwh]*phib2*phib2 +
                                         2*b[iwh]*phib2 + c[iwh]*phib2) * sigma_phib2));
  else //no correlation, return the value in input
    sigma_phib1 = sigma_phib2;

  return sigma_phib1;
}

*/


/// Method to set the priority encoding of the Pt
void DTMatch::findPtPriority()
{
  if( !this->getFlagBXOK() ||
      this->getRejectionFlag() )
  {
    return;
  }

  /// Prepare a vector to store the distance between the predicted
  /// phi and the stub phi in units of sigmas
  float nSigmaPhiDist[ numberOfTriggerLayers + 1 ] = { 9999. }; /// Size is made larger by 1
                                                                /// to make the use of indices easier

  /// Compute phi distance between stubs and predicted position as number of
  /// sigmas for each tracker layer and store it in nSigmaPhiDist[]
  for ( unsigned int iLayer = 1; iLayer <= numberOfTriggerLayers; iLayer++ )
  {
    /// Do this only if there is a matched stub!
    if ( this->getMatchedStubRefs().find(iLayer) == this->getMatchedStubRefs().end() )
    {
      continue;
    }

    /// Here we do have the stub
    int stubPhi = static_cast< int >( 4096. * this->getMatchedStubPositions().find(iLayer)->second.phi() );
    int dtStubDeltaPhi = this->findStubDeltaPhi( stubPhi, iLayer );
    int dtSigmaPhi = this->getPredStubSigmaPhi(iLayer);
    nSigmaPhiDist[ iLayer ] = static_cast< float >( dtStubDeltaPhi )/static_cast< float >( dtSigmaPhi );

  } /// End of loop over tracker layers

  /// Priority encoding: choose the closest stub and choose combination according to
  /// some kind of priority rule as explained in the following "if-else" statements
  float thisPtInv = 99999.;

  /// Availability of stubs is checked using Pt method strings from DTMatchBasePtMethods
  std::map< std::string, DTMatchPt* > thisPtMethodMap = this->getPtMethodsMap();

  if ( thisPtMethodMap.find(std::string("Mu_4_3")) != thisPtMethodMap.end() )
  {
    /// Both L3 and L4 are matched
    if ( nSigmaPhiDist[4] <= nSigmaPhiDist[3] )
    {
      /// L4 is closer than L3
      if ( thisPtMethodMap.find(std::string("Mu_2_1")) != thisPtMethodMap.end() )
      {
        /// Both L1 and L2 are matched
        if ( nSigmaPhiDist[1] <= nSigmaPhiDist[2] )
        {
          thisPtInv = 1./this->getPt(std::string("Mu_4_1"));
        }
        else
        {
          thisPtInv = 1./this->getPt(std::string("Mu_4_2"));
        }
      }
      else
      {
        /// Only one out of L1 and L2 is matched (or none)
        if ( thisPtMethodMap.find(std::string("Mu_2_V")) != thisPtMethodMap.end() )
        {
          /// L2 is matched
          thisPtInv = 1./this->getPt(std::string("Mu_4_2"));
        }
        else if ( thisPtMethodMap.find(std::string("Mu_1_V")) != thisPtMethodMap.end() )
        {
          /// L1 is matched
          thisPtInv = 1./this->getPt(std::string("Mu_4_1"));
        }
        else
        {
          /// None from L1 and L2 is matched
          thisPtInv = 1./this->getPt(std::string("Mu_4_3"));
        }
      }
    }
    else
    {
      /// L3 is closer than L4
      if ( thisPtMethodMap.find(std::string("Mu_2_1")) != thisPtMethodMap.end() )
      {
        /// Both L1 and L2 are matched
        if ( nSigmaPhiDist[1] <= nSigmaPhiDist[2] )
        {
          thisPtInv = 1./this->getPt(std::string("Mu_3_1"));
        }
        else
        {
          thisPtInv = 1./this->getPt(std::string("Mu_3_2"));
        }
      }
      else
      {
        /// Only one out of L1 and L2 is matched (or none)
        if ( thisPtMethodMap.find(std::string("Mu_2_V")) != thisPtMethodMap.end() )
        {
          /// L2 is matched
          thisPtInv = 1./this->getPt(std::string("Mu_3_2"));
        }
        else if ( thisPtMethodMap.find(std::string("Mu_1_V")) != thisPtMethodMap.end() )
        {
          /// L1 is matched
          thisPtInv = 1./this->getPt(std::string("Mu_3_1"));
        }
        else
        {
          /// None from L1 and L2 is matched
          thisPtInv = 1./this->getPt(std::string("Mu_4_3"));
        }
      }
    }
  }
  else
  {
    /// Only one out of L3 and L4 is matched (or none)
    if ( thisPtMethodMap.find(std::string("Mu_4_V")) != thisPtMethodMap.end() )
    {
      /// L4 is matched
      if ( thisPtMethodMap.find(std::string("Mu_2_1")) != thisPtMethodMap.end() )
      {
        /// Both L1 and L2 are matched
        if ( nSigmaPhiDist[1] <= nSigmaPhiDist[2] )
        {
          thisPtInv = 1./this->getPt(std::string("Mu_4_1"));
        }
        else
        {
          thisPtInv = 1./this->getPt(std::string("Mu_4_2"));
        }
      }
      else
      {
        /// Only one out of L1 and L2 is matched (or none)
        if ( thisPtMethodMap.find(std::string("Mu_2_V")) != thisPtMethodMap.end() )
        {
          /// L2 is matched
          thisPtInv = 1./this->getPt(std::string("Mu_4_2"));
        }
        else if ( thisPtMethodMap.find(std::string("Mu_1_V")) != thisPtMethodMap.end() )
        {
          /// L1 is matched
          thisPtInv = 1./this->getPt(std::string("Mu_4_1"));
        }
      }
    }
    else if ( thisPtMethodMap.find(std::string("Mu_3_V")) != thisPtMethodMap.end() )
    {
      /// L3 is matched
      if ( thisPtMethodMap.find(std::string("Mu_2_1")) != thisPtMethodMap.end() )
      {
        /// Both L1 and L2 are matched
        if ( nSigmaPhiDist[1] <= nSigmaPhiDist[2] )
        {
          thisPtInv = 1./this->getPt(std::string("Mu_3_1"));
        }
        else
        {
          thisPtInv = 1./this->getPt(std::string("Mu_3_2"));
        }
      }
      else
      {
        /// Only one out of L1 and L2 is matched (or none)
        if ( thisPtMethodMap.find(std::string("Mu_2_V")) != thisPtMethodMap.end() )
        {
          /// L2 is matched
          thisPtInv = 1./this->getPt(std::string("Mu_3_2"));
        }
        else if ( thisPtMethodMap.find(std::string("Mu_1_V")) != thisPtMethodMap.end() )
        {
          /// L1 is matched
          thisPtInv = 1./this->getPt(std::string("Mu_3_1"));
        }
      }
    }
    else
    {
      /// None from L3 and L4 is matched
      if ( thisPtMethodMap.find(std::string("Mu_2_1")) != thisPtMethodMap.end() )
      {
        /// Both L1 and L2 are matched
        thisPtInv = 1./this->getPt(std::string("Mu_2_1"));
      }
    }
  }

  /// If we had a valid combination of stubs in Layers 1, 2, 3, 4
  /// then some value has been set and we can store it
  if ( thisPtInv < 99999. )
  {
    this->setPtPriority(thisPtInv);
  }
  return;
}


/// Method to set the average encoding of the Pt
void DTMatch::findPtAverage()
{
  if( !this->getFlagBXOK() ||
      this->getRejectionFlag() )
  {
    return;
  }

  float tempInvPt = 0.;
  float thisInvPt = 0.;
  unsigned int nInvPt = 0;

  /// Availability of stubs is checked using Pt method strings from DTMatchBasePtMethods
  std::map< std::string, DTMatchPt* > thisPtMethodMap = this->getPtMethodsMap();

  if ( thisPtMethodMap.find(std::string("Mu_4_3")) != thisPtMethodMap.end() )
  {
    tempInvPt = 1./this->getPt(std::string("Mu_4_3"));
    if ( tempInvPt < 0.25 )
    {
      thisInvPt += tempInvPt;
      nInvPt++;
    }
  }
  if ( thisPtMethodMap.find(std::string("Mu_4_2")) != thisPtMethodMap.end() )
  {
    tempInvPt = 1./this->getPt(std::string("Mu_4_2"));
    if ( tempInvPt < 0.25 )
    {
      thisInvPt += tempInvPt;
      nInvPt++;
    }
  }
  if ( thisPtMethodMap.find(std::string("Mu_4_1")) != thisPtMethodMap.end() )
  {
    tempInvPt = 1./this->getPt(std::string("Mu_4_1"));
    if ( tempInvPt < 0.25 )
    {
      thisInvPt += tempInvPt;
      nInvPt++;
    }
  }
  if ( thisPtMethodMap.find(std::string("Mu_3_2")) != thisPtMethodMap.end() )
  {
    tempInvPt = 1./this->getPt(std::string("Mu_3_2"));
    if ( tempInvPt < 0.25 )
    {
      thisInvPt += tempInvPt;
      nInvPt++;
    }
  }
  if ( thisPtMethodMap.find(std::string("Mu_3_1")) != thisPtMethodMap.end() )
  {
    tempInvPt = 1./this->getPt(std::string("Mu_3_1"));
    if ( tempInvPt < 0.25 )
    {
      thisInvPt += tempInvPt;
      nInvPt++;
    }
  }
  if ( thisPtMethodMap.find(std::string("Mu_2_1")) != thisPtMethodMap.end() )
  {
    tempInvPt = 1./this->getPt(std::string("Mu_2_1"));
    if ( tempInvPt < 0.25 )
    {
      thisInvPt += tempInvPt;
      nInvPt++;
    }
  }

  if ( nInvPt > 0 )
  {
    thisInvPt = thisInvPt / static_cast< float >(nInvPt);
  }

  if ( fabs(thisInvPt) > 0 ) /// Check it is a number and not a NAN
  {
    this->setPtAverage(thisInvPt);
  }
  return;
}

/// Method to assign the priority encoding of the Pt bin
void DTMatch::findPtPriorityBin()
{
  /// Start by setting 0 as default value
  this->setPtPriorityBin(binPt[0]);

  /// Is the DT seed ok? Is the priority encoding Pt available?
  if ( this->getFlagBXOK() &&
       !this->getRejectionFlag() &&
       this->getPtPriorityFlag() )
  {
    /// Set a flag to escape once done
    bool escapeFlag = false;

    /// Get the muon station and the Pt (curvature)
    int stat = this->getDTStation();
    float thisPtInv = 1./this->getPtPriority();

    /// Loop over all the Pt bins
    /// Example of the loop:
    /// iBin = 0: if the curvature is larger than the one of the 4 GeV muon,
    ///           this means Pt < 4 GeV, so the corresponding bin is the 0 GeV bin
    /// if not, then lower the curvature threshold and go to iBin = 1
    /// iBin = 1: if the curvature is larger than the one of the 5 GeV muon but smaller
    ///           than the one of the 4 GeV muon, this means 4 <= Pt < 5 GeV, so
    ///           the corresponding bin is the 4 GeV bin
    /// if not, then lower the curvature threshold and go to iBin = 2
    /// ...
    /// iBin = 23: if the curvature is larger than the one of the 140 GeV muon this means
    ///            Pt < 140 but we already know Pt >= 120, so the 23rd Pt bins (120 GeV) is set
    /// if none of the previous is set, then it is for sure larger than 140 GeV
    for ( unsigned int iBin = 0; iBin < 24 && !escapeFlag ; iBin++ )
    {
      if ( stat == 1 && thisPtInv > cutPtInvMB1[0][iBin] ) /// Use the same thresholds as as Mu_2_1
      {
        this->setPtPriorityBin(binPt[iBin]);
        escapeFlag = true;
      }
      else if ( stat == 2 && thisPtInv > cutPtInvMB2[0][iBin] ) /// Use the same thresholds as as Mu_2_1
      {
        this->setPtPriorityBin(binPt[iBin]);
        escapeFlag = true;
      }
    }

    /// Here's the Pt > 140 GeV case
    if ( escapeFlag == false )
    {
      this->setPtPriorityBin(binPt[24]);
    }
  }
  return;
}

/// Method to assign the average encoding of the Pt bin
void DTMatch::findPtAverageBin()
{
  /// Start by setting 0 as default value
  this->setPtAverageBin(binPt[0]);

  /// Is the DT seed ok? Is the priority encoding Pt available?
  if ( this->getFlagBXOK() &&
       !this->getRejectionFlag() &&
       this->getPtAverageFlag() )
  {
    /// Set a flag to escape once done
    bool escapeFlag = false;

    /// Get the muon station and the Pt (curvature)
    int stat = this->getDTStation();
    float thisPtInv = 1./this->getPtAverage();

    /// Loop over all the Pt bins
    /// Example of the loop:
    /// iBin = 0: if the curvature is larger than the one of the 4 GeV muon,
    ///           this means Pt < 4 GeV, so the corresponding bin is the 0 GeV bin
    /// if not, then lower the curvature threshold and go to iBin = 1
    /// iBin = 1: if the curvature is larger than the one of the 5 GeV muon but smaller
    ///           than the one of the 4 GeV muon, this means 4 <= Pt < 5 GeV, so
    ///           the corresponding bin is the 4 GeV bin
    /// if not, then lower the curvature threshold and go to iBin = 2
    /// ...
    /// iBin = 23: if the curvature is larger than the one of the 140 GeV muon this means
    ///            Pt < 140 but we already know Pt >= 120, so the 23rd Pt bins (120 GeV) is set
    /// if none of the previous is set, then it is for sure larger than 140 GeV
    for ( unsigned int iBin = 0; iBin < 24 && !escapeFlag ; iBin++ )
    {
      if ( stat == 1 && thisPtInv > cutPtInvMB1[0][iBin] ) /// Use the same thresholds as as Mu_2_1
      {
        this->setPtAverageBin(binPt[iBin]);
        escapeFlag = true;
      }
      else if ( stat == 2 && thisPtInv > cutPtInvMB2[0][iBin] ) /// Use the same thresholds as as Mu_2_1
      {
        this->setPtAverageBin(binPt[iBin]);
        escapeFlag = true;
      }
    }

    /// Here's the Pt > 140 GeV case
    if ( escapeFlag == false )
    {
      this->setPtAverageBin(binPt[24]);
    }
  }
  return;
}

/// Method to find a generic Pt bin in the table
unsigned int DTMatch::findPtBin( float aPtInv, unsigned int aMethod )
{
  /// Prepare the output
  unsigned int thisPtBin = 0;

  /// Is the DT seed ok? Is the priority encoding Pt available?
  if ( this->getFlagBXOK() &&
       !this->getRejectionFlag() )
  {
    /// Set a flag to escape once done
    bool escapeFlag = false;

    /// Get the muon station and the Pt (curvature)
    int stat = this->getDTStation();

    /// Loop over all the Pt bins
    /// Example of the loop:
    /// iBin = 0: if the curvature is larger than the one of the 4 GeV muon,
    ///           this means Pt < 4 GeV, so the corresponding bin is the 0 GeV bin
    /// if not, then lower the curvature threshold and go to iBin = 1
    /// iBin = 1: if the curvature is larger than the one of the 5 GeV muon but smaller
    ///           than the one of the 4 GeV muon, this means 4 <= Pt < 5 GeV, so
    ///           the corresponding bin is the 4 GeV bin
    /// if not, then lower the curvature threshold and go to iBin = 2
    /// ...
    /// iBin = 23: if the curvature is larger than the one of the 140 GeV muon this means
    ///            Pt < 140 but we already know Pt >= 120, so the 23rd Pt bins (120 GeV) is set
    /// if none of the previous is set, then it is for sure larger than 140 GeV
    for ( unsigned int iBin = 0; iBin < 24 && !escapeFlag ; iBin++ )
    {
      if ( stat == 1 && aPtInv > cutPtInvMB1[aMethod][iBin] ) /// aMethod gives the line in the table
                                                              /// One line for each Mu_X_Y
      {
        thisPtBin = iBin;
        escapeFlag = true;
      }
      else if ( stat == 2 && aPtInv > cutPtInvMB2[aMethod][iBin] ) /// aMethod gives the line in the table
                                                                   /// One line for each Mu_X_Y
      {
        thisPtBin = iBin;
        escapeFlag = true;
      }
    }

    /// Here's the Pt > 140 GeV case
    if ( escapeFlag == false )
    {
      thisPtBin = 24;
    }
  }

  return thisPtBin;
}

/// Method to assign the Pt bin from a L1 Track
void DTMatch::findPtTTTrackBin()
{
  /// Here's the curvature table
/*
  float cutPtInvTK[24] = { 0.2533,0.2026,0.1689,0.1448,0.1266,0.1014,
                           0.0845,0.0724,0.0634,0.0564,0.0508,0.0406,0.0339,
                           0.0291,0.0255,0.0227,0.0204,0.0171,0.0147,0.0129,
                           0.0115,0.0104,0.0087,0.0075 };
*/

  float const cutPtInvTK[24] = {0.2525, 0.2019, 0.1682, 0.1441, 0.1261, 0.1008, 0.0840, 0.0720, 0.0630, 0.0560, 0.0504, 0.0403, 0.0336, 0.0288, 0.0252, 0.0224, 0.0202, 0.0168, 0.0144, 0.0127, 0.0113, 0.0101, 0.0085, 0.0073};

  /// Start by setting 0 as default value
  this->setPtTTTrackBin(binPt[0]);

  /// Is the DT seed ok? Is the priority encoding Pt available?
  if ( this->getFlagBXOK() &&
       !this->getRejectionFlag() &&
       this->getPtMatchedTrackPtr().isNull() == false )
  {
    /// Set a flag to escape once done
    bool escapeFlag = false;

    /// Get the Pt (curvature)
    float thisPtInv = 1./( this->getPtMatchedTrackPtr()->getMomentum().perp() );

    /// Loop over all the Pt bins
    /// Example of the loop:
    /// iBin = 0: if the curvature is larger than the one of the 4 GeV muon,
    ///           this means Pt < 4 GeV, so the corresponding bin is the 0 GeV bin
    /// if not, then lower the curvature threshold and go to iBin = 1
    /// iBin = 1: if the curvature is larger than the one of the 5 GeV muon but smaller
    ///           than the one of the 4 GeV muon, this means 4 <= Pt < 5 GeV, so
    ///           the corresponding bin is the 4 GeV bin
    /// if not, then lower the curvature threshold and go to iBin = 2
    /// ...
    /// iBin = 23: if the curvature is larger than the one of the 140 GeV muon this means
    ///            Pt < 140 but we already know Pt >= 120, so the 23rd Pt bins (120 GeV) is set
    /// if none of the previous is set, then it is for sure larger than 140 GeV
    for ( unsigned int iBin = 0; iBin < 24 && !escapeFlag ; iBin++ )
    {
      if ( thisPtInv > cutPtInvTK[iBin] )
      {
        this->setPtTTTrackBin(binPt[iBin]);
        escapeFlag = true;
      }
    }

    /// Here's the Pt > 140 GeV case
    if ( escapeFlag == false )
    {
      this->setPtTTTrackBin(binPt[24]);
    }
  }
  return;
}

/// Method to assign the majority encoding Pt bin
void DTMatch::findPtMajorityFullTkBin()
{
  /// Start by setting 0 as default value
  unsigned int majorityPtBin = 0;

  /// Is the DT seed ok?
  if ( this->getFlagBXOK() &&
       !this->getRejectionFlag() )
  {
    /// Prepare the counters
    unsigned int ptCounter[25] = { 0 }; /// One for each Pt bin

    /// Get the Pt range from DT at 3 sigma
    int dtPtMin = this->getDTPtMin(3.);
    int dtPtMax = this->getDTPtMax(3.);

    /// Availability of stubs is checked using Pt method strings from DTMatchBasePtMethods
    std::map< std::string, DTMatchPt* > thisPtMethodMap = this->getPtMethodsMap();

    /// Prepare also the string array
    std::string theMethods[15] = {
      std::string("Mu_2_1"),
      std::string("Mu_3_1"), std::string("Mu_3_2"),
      std::string("Mu_4_1"), std::string("Mu_4_2"), std::string("Mu_4_3"),
      std::string("Mu_5_1"), std::string("Mu_5_2"), std::string("Mu_5_3"), std::string("Mu_5_4"),
      std::string("Mu_6_1"), std::string("Mu_6_2"), std::string("Mu_6_3"), std::string("Mu_6_4"), std::string("Mu_6_5") };

    /// Loop over the methods
    for ( unsigned int iMethod = 0; iMethod < 15; iMethod++ )
    {
      /// Check if the method is available
      if ( thisPtMethodMap.find( theMethods[iMethod] ) != thisPtMethodMap.end() )
      {
        float thisPt = this->getPt( theMethods[iMethod] );
        float thisPtInv = 1./thisPt; /// Sure it is not a NAN as the method has been found
                                     /// hence all the needed points are available

        int thisPtInt = static_cast< int >(thisPt);

        /// Check if the Pt is in the allowed range
        if ( thisPtInt >= dtPtMin &&
             thisPtInt <= dtPtMax )
        {
          /// Find the Pt bin and increment its counter
          ptCounter[ this->findPtBin( thisPtInv, iMethod ) ]++;
        } 
      } /// End of availability of the Pt method
    } /// End of loop over the Pt methods

    /// Loop over the bins
    for ( unsigned int iBin = 0; iBin < 25; iBin++ )
    {
      /// Is this a non-zero bin?
      if ( ptCounter[iBin] == 0 )
      {
        continue;
      }

      /// Is this the bin with the highest population?      
      if ( ptCounter[iBin] >= ptCounter[majorityPtBin] )
      {
        majorityPtBin = iBin;
      }
    }
  }

  this->setPtMajorityFullTkBin(binPt[majorityPtBin]);

  return;
}

/// Method to assign the majority encoding Pt bin (inner layers only)
void DTMatch::findPtMajorityBin()
{
  /// Start by setting 0 as default value
  unsigned int majorityPtBin = 0;

  /// Is the DT seed ok?
  if ( this->getFlagBXOK() &&
       !this->getRejectionFlag() )
  {
    /// Prepare the counters
    unsigned int ptCounter[25] = { 0 }; /// One for each Pt bin

    /// Get the Pt range from DT at 3 sigma
    int dtPtMin = this->getDTPtMin(3.);
    int dtPtMax = this->getDTPtMax(3.);

    /// Availability of stubs is checked using Pt method strings from DTMatchBasePtMethods
    std::map< std::string, DTMatchPt* > thisPtMethodMap = this->getPtMethodsMap();

    /// Prepare also the string array
    std::string theMethods[6] = {
      std::string("Mu_2_1"),
      std::string("Mu_3_1"), std::string("Mu_3_2"),
      std::string("Mu_4_1"), std::string("Mu_4_2"), std::string("Mu_4_3") };

    /// Loop over the methods
    for ( unsigned int iMethod = 0; iMethod < 6; iMethod++ )
    {
      /// Check if the method is available
      if ( thisPtMethodMap.find( theMethods[iMethod] ) != thisPtMethodMap.end() )
      {
        float thisPt = this->getPt( theMethods[iMethod] );
        float thisPtInv = 1./thisPt; /// Sure it is not a NAN as the method has been found
                                     /// hence all the needed points are available

        int thisPtInt = static_cast< int >(thisPt);

        /// Check if the Pt is in the allowed range
        if ( thisPtInt >= dtPtMin &&
             thisPtInt <= dtPtMax )
        {
          /// Find the Pt bin and increment its counter
          ptCounter[ this->findPtBin( thisPtInv, iMethod ) ]++;
        } 
      } /// End of availability of the Pt method
    } /// End of loop over the Pt methods

    /// Loop over the bins
    for ( unsigned int iBin = 0; iBin < 25; iBin++ )
    {
      /// Is this a non-zero bin?
      if ( ptCounter[iBin] == 0 )
      {
        continue;
      }

      /// Is this the bin with the highest population?
      if ( ptCounter[iBin] >= ptCounter[majorityPtBin] )
      {
        majorityPtBin = iBin;
      }
    }
  }

  this->setPtMajorityBin(binPt[majorityPtBin]);

  return;
}

/// Method to assign the mixed mode encoding Pt bin
void DTMatch::findPtMixedModeBin()
{
  /// This is based on findPtMajorityBin() but with a backup solution
  /// in case the max count for the majority gives 1
  /// The backup is to average on the Pt bins: i.e. if bins 1, 3, 4, and 5 have count == 1
  /// and none else is != 0, then the average is (1+3+4+5)/4 = 13/4 = 3 (all integer operations)

  /// Start by setting 0 as default value
  unsigned int majorityPtBin = 0;

  /// Use this vector to store the bins
  std::vector< unsigned int > firedBins;

  /// Is the DT seed ok?
  if ( this->getFlagBXOK() &&
       !this->getRejectionFlag() )
  {
    /// Prepare the counters
    unsigned int ptCounter[25] = { 0 }; /// One for each Pt bin

    /// Get the Pt range from DT at 3 sigma
    int dtPtMin = this->getDTPtMin(3.);
    int dtPtMax = this->getDTPtMax(3.);

    /// Availability of stubs is checked using Pt method strings from DTMatchBasePtMethods
    std::map< std::string, DTMatchPt* > thisPtMethodMap = this->getPtMethodsMap();

    /// Prepare also the string array
    std::string theMethods[6] = {
      std::string("Mu_2_1"),
      std::string("Mu_3_1"), std::string("Mu_3_2"),
      std::string("Mu_4_1"), std::string("Mu_4_2"), std::string("Mu_4_3") };

    /// Loop over the methods
    for ( unsigned int iMethod = 0; iMethod < 6; iMethod++ )
    {
      /// Check if the method is available
      if ( thisPtMethodMap.find( theMethods[iMethod] ) != thisPtMethodMap.end() )
      {
        float thisPt = this->getPt( theMethods[iMethod] );

        /// Set a cut at 4 GeV
        if ( thisPt < 4. )
        {
          continue;
        }
        
        float thisPtInv = 1./thisPt; /// Sure it is not a NAN as the method has been found
                                     /// hence all the needed points are available

        int thisPtInt = static_cast< int >(thisPt);

        /// Check if the Pt is in the allowed range
        if ( thisPtInt >= dtPtMin &&
             thisPtInt <= dtPtMax )
        {
          /// Find the Pt bin and increment its counter
          ptCounter[ this->findPtBin( thisPtInv, iMethod ) ]++;
        }        
      } /// End of availability of the Pt method
    } /// End of loop over the Pt methods

    /// Loop over the bins
    for ( unsigned int iBin = 0; iBin < 25; iBin++ )
    {
      /// Is this a non-zero bin?
      if ( ptCounter[iBin] == 0 )
      {
        continue;
      }

      /// Is this the bin with the highest population?
      if ( ptCounter[iBin] >= ptCounter[majorityPtBin] )
      {
        majorityPtBin = iBin;
        if ( iBin != 0 )
        {
          firedBins.push_back( iBin );
        }
      }
    }

    if ( ptCounter[majorityPtBin] == 1 )
    {
      /// Backup solution!
      unsigned int totBinCount = 0;
      for ( unsigned int jBin = 0; jBin < firedBins.size(); jBin++ )
      {
        totBinCount += firedBins.at(jBin);
      }
      majorityPtBin = static_cast< unsigned int >(static_cast< float >(totBinCount) / static_cast< float >(firedBins.size()));
    }
  }

  this->setPtMixedModeBin(binPt[majorityPtBin]);

  return;
}

