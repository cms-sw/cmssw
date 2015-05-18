/*! \class DTUtilities
 *  \author Ignazio Lazzizzera
 *  \author Sara Vanini
 *  \author Nicola Pozzobon
 *  \brief Utilities of L1 DT + Track Trigger for the HL-LHC
 *  \date 2008, Dec 25
 */

#include "L1Trigger/DTPlusTrackTrigger/interface/DTUtilities.h"

/// Class constructor
DTUtilities::DTUtilities( DTTrig* aDTTrigger,
                          BtiTrigsCollection* someBtiTrigsToStore,
                          TSPhiTrigsCollection* someTSPhiTrigsToStore,
                          TSThetaTrigsCollection* someTSThetaTrigsToStore,
                          bool useTS, bool useRough,
                          edm::ESHandle< DTGeometry > aMuonDTGeometryHandle,
                          std::map< unsigned int, std::vector< DTMatch* > >* aDTMatchContainer )
{
  theDTTrigger = aDTTrigger;
  theBtiTrigsToStore = someBtiTrigsToStore;
  theTSPhiTrigsToStore = someTSPhiTrigsToStore;
  theTSThetaTrigsToStore = someTSThetaTrigsToStore;
  useTSTheta = useTS;
  useRoughTheta = useRough;
  theMuonDTGeometryHandle = aMuonDTGeometryHandle;
  theDTMatchContainer = aDTMatchContainer;
}

/// Class destructor
DTUtilities::~DTUtilities()
{
  delete theDTTrigger;
  delete theBtiTrigsToStore;
  delete theTSPhiTrigsToStore;
  delete theTSThetaTrigsToStore;
  delete theDTMatchContainer;
}

/// Extrapolation of DT Triggers to Tracker and Vtx
void DTUtilities::extrapolateDTTriggers()
{
  for ( unsigned int i = 1; i <= 2; i++ )
  {
    for ( unsigned int j = 0; j < theDTMatchContainer->at(i).size(); j++ )
    {
      DTMatch* thisDTMatch = theDTMatchContainer->at(i).at(j);

      /// Extrapolate to each Tracker layer
      for ( unsigned int lay = 1; lay <= 6; lay++ )
      {
        thisDTMatch->extrapolateToTrackerLayer(lay);
      }

      /// Extrapolate also to the Vtx
      thisDTMatch->extrapolateToVertex();
    }
  }
}

