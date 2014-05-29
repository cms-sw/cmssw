//#define npDEBUG

/*! \class DTUtilities
 *  \author Ignazio Lazzizzera
 *  \author Sara Vanini
 *  \author Nicola Pozzobon
 *  \brief Utilities of L1 DT + Track Trigger for the HL-LHC
 *  \date 2008, Dec 25
 */

#ifndef DTUtilities_h
#define DTUtilities_h

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include "L1Trigger/DTTrigger/interface/DTTrig.h"
#include "DataFormats/L1DTPlusTrackTrigger/interface/DTBtiTrigger.h"
#include "DataFormats/L1DTPlusTrackTrigger/interface/DTTSPhiTrigger.h"
#include "DataFormats/L1DTPlusTrackTrigger/interface/DTTSThetaTrigger.h"
#include "DataFormats/L1DTPlusTrackTrigger/interface/DTMatch.h"

/// Class implementation
class DTUtilities
{
  public :

    /// Constructor and destructor
    DTUtilities( DTTrig* aDTTrigger,
                 BtiTrigsCollection* someBtiTrigsToStore,
                 TSPhiTrigsCollection* someTSPhiTrigsToStore,
                 TSThetaTrigsCollection* someTSThetaTrigsToStore,
                 bool useTS, bool useRough,
                 edm::ESHandle< DTGeometry > aMuonDTGeometryHandle,
                 std::map< unsigned int, std::vector< DTMatch* > >* aDTMatchContainer );

    ~DTUtilities();

    /// Main methods
    void getDTTrigger();
    void orderDTTriggers();
    void extrapolateDTTriggers();
    void removeRedundantDTTriggers();

    /// Method to match the BTI with the Phi segment
    bool match( DTBtiTrigger const bti, DTChambPhSegm const tsphi )
    {
      return ( tsphi.wheel() == bti.wheel() &&
               tsphi.station() == bti.station() &&
               tsphi.sector() == bti.sector() &&
               tsphi.step() == bti.step() &&
               2 == bti.btiSL() );
    }

    /// Method to match the Phi and Theta track segments
    bool match( DTChambThSegm const tstheta, DTChambPhSegm const tsphi )
    {
      return ( tsphi.wheel() == tstheta.ChamberId().wheel() &&
               tsphi.station() == tstheta.ChamberId().station() &&
               tsphi.sector() == tstheta.ChamberId().sector() &&
               tsphi.step() == tstheta.step() );
    }

  private :

    DTTrig* theDTTrigger;
    BtiTrigsCollection* theBtiTrigsToStore;
    TSPhiTrigsCollection* theTSPhiTrigsToStore;
    TSThetaTrigsCollection* theTSThetaTrigsToStore;
    bool useTSTheta;
    bool useRoughTheta;
    edm::ESHandle< DTGeometry > theMuonDTGeometryHandle;
    std::map< unsigned int, std::vector< DTMatch* > >* theDTMatchContainer;

}; /// Close class

#endif

