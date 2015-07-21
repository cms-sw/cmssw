/*! \class DTUtilities
 *  \author Ignazio Lazzizzera
 *  \author Sara Vanini
 *  \author Nicola Pozzobon
 *  \brief Utilities of L1 DT + Track Trigger for the HL-LHC
 *  \date 2008, Dec 25
 */

#include "L1Trigger/DTPlusTrackTrigger/interface/DTUtilities.h"

/// Method to get rid of redundancies (setting a rejection flag)
void DTUtilities::removeRedundantDTTriggers()
{
  /// Redundant DTMatch cancellation
  /// choose one layer for extrapolation (central layer for the time being)
  int testLayer = 3;
  int numSigmaCut = 3;

  /// Combine together the two vector of pointers
  /// Then you can change the objects as you are working with pointers
  std::vector< DTMatch* > tempVector = theDTMatchContainer->at(1);
  tempVector.insert( tempVector.end(),
                     theDTMatchContainer->at(2).begin(), theDTMatchContainer->at(2).end() );

  /// Check the size is correct
  unsigned int sumSizes = theDTMatchContainer->at(1).size() + theDTMatchContainer->at(2).size();
  unsigned int mergedSize = tempVector.size();

  if ( sumSizes != mergedSize )
  {
    exit(0);
  }

  /// Find II tracks in SAME station SAME sector SAME bx and remove single L in any case
  for ( unsigned int iDt = 0; iDt < mergedSize; iDt++ )
  {
    if ( tempVector.at(iDt)->getRejectionFlag() == false )
    {
      /// Record MB I track station, sector and bx
      int station = tempVector.at(iDt)->getDTStation();
      int bx = tempVector.at(iDt)->getDTBX();
      int sector = tempVector.at(iDt)->getDTSector();

      for ( unsigned int jDt = (iDt + 1); jDt < mergedSize; jDt++ )
      {
        if ( tempVector.at(jDt)->getDTStation() == station &&
             tempVector.at(jDt)->getDTBX() == bx &&
             tempVector.at(jDt)->getDTSector() == sector &&
             tempVector.at(jDt)->getRejectionFlag() == false &&
             tempVector.at(jDt)->getDTCode() <= 7 )
        {
          tempVector.at(jDt)->setRejectionFlag(true);
        }
      }
    }
  } /// End L II track rejection

  /// Collect MB1 and MB2 DTMatch at same bx in same sector and compare phi, phib
  for ( unsigned int iDt = 0; iDt < theDTMatchContainer->at(1).size(); iDt++ )
  {
    /// Loop over station == 1
    DTMatch* mb1Match = theDTMatchContainer->at(1).at(iDt);

    if ( mb1Match->getDTStation() != 1 )
    {
      exit(0);
    }

    if ( mb1Match->getRejectionFlag() == false )
    {
      /// Record MB1 track sector and bx
      int bx1 = mb1Match->getDTBX();
      int sector1 = mb1Match->getDTSector();

      /// Get quantities to compare
      int phi1 = mb1Match->getPredStubPhi(testLayer);
      int theta1 = mb1Match->getPredStubTheta(testLayer);
      float phib1 = static_cast< float >( mb1Match->getDTTSPhiB() );

      /// Needing a small correction in phi predicted (extrapolation precision?)
      /// Correction in phib due to field between ST1 and ST2
      int dphicor = static_cast< int >( -0.0097 * phib1 * phib1 + 1.0769 * phib1 + 4.2324 );
      int dphibcor = static_cast< int >( 0.3442 * phib1 );

      /// Tolerances parameterization
      int sigmaPhi =static_cast< int >( 0.006 * phib1 * phib1 + 0.4821 * phib1 + 37.64 );
      int sigmaPhib =static_cast< int >( 0.0005 * phib1 * phib1 + 0.01211 * phib1 + 3.4125 );
      int sigmaTheta = 100;

      /// Find tracks in MB2: SAME sector SAME bx
      for ( unsigned int jDt = 0; jDt < theDTMatchContainer->at(2).size(); jDt++ )
      {
        /// Loop over station == 2
        DTMatch* mb2Match = theDTMatchContainer->at(2).at(jDt);

        if ( mb2Match->getDTStation() != 2 )
        {
          exit(0);
        }

        if ( mb2Match->getRejectionFlag() == false &&
             mb2Match->getDTBX() == bx1 &&
             mb2Match->getDTSector() == sector1 )
        {
          /// Get quantities to compare
          int phi2 = mb2Match->getPredStubPhi(testLayer);
          int theta2 = mb2Match->getPredStubTheta(testLayer);
          float phib2 = static_cast< float >( mb2Match->getDTTSPhiB() );

          int dphi = abs( phi1 - phi2 ) - dphicor;
          int dtheta = abs( theta1 - theta2 );
          int dphib = static_cast< int >(fabs( phib1 - phib2 )) - dphibcor;

          /// Remove redundant DTMatch
          /// For the moment keep the one with higher quality
          /// Remove if inside all tolerances
          if ( dphi < (numSigmaCut*sigmaPhi)  &&
               dphib < (numSigmaCut*sigmaPhib) &&
               dtheta < (numSigmaCut*sigmaTheta) )
          {
            if( mb2Match->getDTCode() <= mb1Match->getDTCode() )
            {
              mb2Match->setRejectionFlag(true);
            }
            else
            {
              mb1Match->setRejectionFlag(true);
            }
          }
        }
      } /// End MB2 loop
    }
  } /// End MB1 loop

  return;
}

