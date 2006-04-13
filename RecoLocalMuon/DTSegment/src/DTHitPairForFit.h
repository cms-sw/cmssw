#ifndef DTSegment_DTHitPairForFit_h
#define DTSegment_DTHitPairForFit_h

/** \class DTHitPairForFit
 *
 * Hit pair used for the segments fit
 *
 * This class is useful for segment fitting, which is done in SL or Chamber
 * reference frame, while the DT hits live on the layer.
 *
 * $Date: 2006/04/12 15:15:48 $
 * $Revision: 1.2 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

/* Base Class Headers */

/* Collaborating Class Declarations */
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"

/* C++ Headers */

/* ====================================================================== */

/* Class DTHitPairForFit Interface */

class DTHitPairForFit{

  public:

/// Constructor
    DTHitPairForFit(const DTRecHit1DPair& pair,
                    const DTSuperLayer& sl,
                    const edm::ESHandle<DTGeometry>& dtGeom) ;

/// Destructor
    ~DTHitPairForFit() ;

/* Operations */ 
    LocalPoint localPosition(DTEnums::DTCellSide s) const;
    GlobalPoint globalPosition(DTEnums::DTCellSide s) const;
    LocalPoint leftPos() const { return theLeftPos; }
    LocalPoint rightPos() const { return theRightPos; }
    LocalError localPositionError() const { return theError; }

    DTWireId id() const { return theWireId; }
    float digiTime() const {return theDigiTime;}

    /** check for compatibility of the hit pair with a given position and direction: 
     * the first bool of the returned pair is for the left hit, the second for
     * the right one */
    std::pair<bool,bool> isCompatible(const LocalPoint& posIni, const LocalVector &dirIni) const;

    /// define the order by increasing z
    bool operator<(const DTHitPairForFit& hit) const ; 

    bool operator==(const DTHitPairForFit& hit) const ; 

    // /// return the layer number, 1 to 4
    // int layerNumber() const { return theLayerNumber;}

  protected:

  private:
    LocalPoint theLeftPos ;         // left hit pos in SL ref frame
    LocalPoint theRightPos ;        // right hit pos in SL ref frame
    LocalError theError;            // it's the same for left and right
    DTWireId  theWireId;
    float theDigiTime;             // the time of the corresp. digi
//    int theLayerNumber;             // the layer number

    bool isCompatible(const LocalPoint& posIni, 
                      const LocalVector& dirIni,
                      DTEnums::DTCellSide code) const;
};

std::ostream& operator<<(std::ostream& out, const DTHitPairForFit& hit) ;
#endif // DTSegment_DTHitPairForFit_h
