/*! \class DTUtilities
 *  \author Ignazio Lazzizzera
 *  \author Sara Vanini
 *  \author Nicola Pozzobon
 *  \brief Utilities of L1 DT + Track Trigger for the HL-LHC
 *  \date 2008, Dec 25
 */

#include "L1Trigger/DTPlusTrackTrigger/interface/DTUtilities.h"

/// Method to give ordering number to the DTTriggers
void DTUtilities::orderDTTriggers()
{
  /// If no DTTrigger is found, skip
  if ( theDTMatchContainer->at(1).size() == 0 &&
       theDTMatchContainer->at(2).size() == 0 )
  {
    return;
  }

  /// Order DT phi-eta matches: first by higher code, then by lower phib

  /// Utility variables
  /// 24 bx, 5 wheels, 12 sectors, 10 max matches
  unsigned int sortDTTIMatch[24][5][12][10];
  unsigned int indexDTTIMatch[24][5][12];

  /// Initialize and store matches per bx, wheel, sector
  unsigned int maxMatches = 10;

  for ( int ibx = 8; ibx < 25; ibx++ )
  {
    for ( int iwh = -2; iwh < 3; iwh++ )
    {
      for ( int isec = 1; isec < 13; isec++ )
      {
        indexDTTIMatch[ibx-8][iwh+2][isec-1] = 0;
      }
    }
  }

  unsigned int numDTMatchesSt1 = theDTMatchContainer->at(1).size();
  unsigned int numDTMatches = theDTMatchContainer->at(1).size() + theDTMatchContainer->at(2).size();

  for ( int ibx = 8; ibx < 25; ibx++ )
  {
    for ( int iwh = -2; iwh < 3; iwh++ )
    {
      for ( int isec = 1; isec < 13; isec++ )
      {
        for ( unsigned int dm = 0; dm < numDTMatchesSt1; dm++ )
        {
          DTMatch* thisMatch = theDTMatchContainer->at(1).at(dm);

          if ( thisMatch->getDTBX() == ibx &&
               thisMatch->getDTWheel() == iwh &&
               thisMatch->getDTSector() == isec )
          {
            if ( indexDTTIMatch[ibx-8][iwh+2][isec-1] < maxMatches )
            {
              sortDTTIMatch[ibx-8][iwh+2][isec-1][indexDTTIMatch[ibx-8][iwh+2][isec-1]] = dm;
            }

            indexDTTIMatch[ibx-8][iwh+2][isec-1]++;
          }
        }

        for ( unsigned int dm = numDTMatchesSt1; dm < numDTMatches; dm++ )
        {
          DTMatch* thisMatch = theDTMatchContainer->at(2).at( dm - numDTMatchesSt1 );

          if ( thisMatch->getDTBX() == ibx &&
               thisMatch->getDTWheel() == iwh &&
               thisMatch->getDTSector() == isec )
          {
            if ( indexDTTIMatch[ibx-8][iwh+2][isec-1] < maxMatches )
            {
              sortDTTIMatch[ibx-8][iwh+2][isec-1][ indexDTTIMatch[ibx-8][iwh+2][isec-1] ] = dm;
            }

            indexDTTIMatch[ibx-8][iwh+2][isec-1]++;
          }
        }
      }
    }
  }

  /// Order matches
  /// Loop over matches for every bx, wheel, sector
  for ( int ibx = 8; ibx < 25; ibx++ )
  {
    for ( int iwh = -2; iwh < 3; iwh++ )
    {
      for ( int isec = 1; isec < 13; isec++ )
      {
        unsigned int im[10];
        unsigned int is[10];

        for ( unsigned int i = 0; i < maxMatches; i++ )
        {
          im[i] = 0;
          is[i] = 0;
        }

        int ntrig = indexDTTIMatch[ibx-8][iwh+2][isec-1];

        /// First compare quality code; when these are equal then compare by
        /// bending angle: better grade for higher bending.
        for ( int itrig = 0; itrig < ntrig; itrig++ )
        {
          unsigned int testNumber = sortDTTIMatch[ibx-8][iwh+2][isec-1][itrig];

          if ( testNumber < numDTMatchesSt1 )
          {
            is[itrig] = 1;
            im[itrig] = testNumber;
          }
          else
          {
            is[itrig] = 2;
            im[itrig] = testNumber - numDTMatchesSt1;
          }

          if ( itrig == 0 )
          {
            /// This is the first match; set as first:
            int triggerOrder = 1;

            theDTMatchContainer->at(is[itrig]).at(im[itrig])->setDTTrigOrder(triggerOrder);
          }
          else if ( itrig > 0 )
          {
            /// These are the remaining matches,
            /// check against already ordered ones:
            int currentCode = theDTMatchContainer->at(is[itrig]).at(im[itrig])->getDTCode();
            int currentPhiB = theDTMatchContainer->at(is[itrig]).at(im[itrig])->getDTTSPhiB();

            /// Stop loop when the current trigger is higher grade
            /// than already ordered ones
            int istop = -1;
            for ( int i = 0; i < itrig; i++ )
            {
              int checkCode = theDTMatchContainer->at(is[i]).at(im[i])->getDTCode();
              int checkPhiB = theDTMatchContainer->at(is[i]).at(im[i])->getDTTSPhiB();

              /// First case: current code is higher than stored code
              if ( checkCode < currentCode )
              {
                istop = i;
                break;
              }

              /// Second case: code is the same: order by lower bending angle
              else if ( currentCode == checkCode )
              {
                if ( checkPhiB > currentPhiB )
                {
                  istop = i;
                  break;
                }
              }
            } /// End loop over already ordered matches

            /// Now set ordinal number
            /// First case: current match is lowest rank
            if ( istop < 0 )
            {
              theDTMatchContainer->at(is[itrig]).at(im[itrig])->setDTTrigOrder(itrig+1);
            }
            // Second case: current match goes somewhere in the existing list
            else
            {
              int triggerOrder = theDTMatchContainer->at(is[istop]).at(im[istop])->getDTTTrigOrder();
              theDTMatchContainer->at(is[itrig]).at(im[itrig])->setDTTrigOrder(triggerOrder);

              for ( int i = 0; i < itrig; i++ )
              {
                if ( theDTMatchContainer->at(is[i]).at(im[i])->getDTTTrigOrder() >= triggerOrder )
                {
                  theDTMatchContainer->at(is[i]).at(im[i])->setDTTrigOrder( theDTMatchContainer->at(is[i]).at(im[i])->getDTTTrigOrder() + 1 );
                }
              }
            }
          } /// End of setting of order position of the current match
        } /// End loop for ordering all matches
      } /// End loop over sect
    } /// End loop over wh
  } /// End loop over bx

  return;
}

