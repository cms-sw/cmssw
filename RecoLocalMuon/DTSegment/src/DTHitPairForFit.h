#ifndef DTSegment_DTHitPairForFit_h
#define DTSegment_DTHitPairForFit_h

/** \class DTHitPairForFit
 *
 * Hit pair used for the segments fit
 *
 * This class is useful for segment fitting, which is done in SL or Chamber
 * reference frame, while the DT hits live on the layer.
 *
 * $Date: 2009/11/27 11:59:48 $
 * $Revision: 1.7 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

/* Base Class Headers */

/* Collaborating Class Declarations */
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
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
    
    /// Returns the local position in the layer
    LocalPoint localPosition(DTEnums::DTCellSide s) const;

    /// Returns the position in the layer r.f. of the left rechit
    LocalPoint leftPos() const { return theLeftPos; }

    /// Returns the position in the layer r.f. of the right rechit
    LocalPoint rightPos() const { return theRightPos; }

    // Returns the LocalError
    LocalError localPositionError() const { return theError; }

    /// Returns the Id of the wire on which the rechit rely
    const DTWireId & id() const { return theWireId; }

    /// Returns the time of the corresponding digi
    float digiTime() const {return theDigiTime;}

    /** check for compatibility of the hit pair with a given position and direction: 
     * the first bool of the returned pair is for the left hit, the second for
     * the right one */
    std::pair<bool,bool> isCompatible(const LocalPoint& posIni, const LocalVector &dirIni) const;
    
    /// define the order by increasing z
    bool operator<(const DTHitPairForFit& hit) const ; 

    bool operator==(const DTHitPairForFit& hit) const ; 

  protected:

  private:
    LocalPoint theLeftPos ;         // left hit pos in SL ref frame
    LocalPoint theRightPos ;        // right hit pos in SL ref frame
    LocalError theError;            // it's the same for left and right
    DTWireId  theWireId;
    float theDigiTime;             // the time of the corresp. digi
//    int theLayerNumber;             // the layer number

};

std::ostream& operator<<(std::ostream& out, const DTHitPairForFit& hit) ;
#endif // DTSegment_DTHitPairForFit_h
