//-------------------------------------------------
//
/**  \class DTTSTheta
 *    Implementation of TS Theta L1Trigger algorithm
 *
 *   $Date: 2003/10/17 08:22:23 $
 *   $Revision: 1.9 $
 *
 *   \author C.Grandi
 */
//
//--------------------------------------------------
#ifndef DT_TS_THETA_H
#define DT_TS_THETA_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
//class DTChambThSegm;
class DTBtiCard;
class DTBtiTrigData;
class DTTrigGeom;

//----------------------
// Base Class Headers --
//----------------------

// #include "CARF/Reco/interface/RecDet.h"
// #include "CARF/G3Event/interface/G3EventProxy.h"
#include "L1Trigger/DTUtilities/interface/DTGeomSupplier.h"
#include "L1Trigger/DTUtilities/interface/BitArray.h"
#include "L1Trigger/DTTriggerServerTheta/interface/DTChambThSegm.h"
#include "L1Trigger/DTUtilities/interface/DTConfig.h"
#include "L1Trigger/DTUtilities/interface/DTCache.h"
//---------------
// C++ Headers --
//---------------
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

//typedef RecDet < DTChambThSegm,G3EventProxy*, std::vector<DTChambThSegm> > DTTSThetaManager;

typedef DTCache < DTChambThSegm, std::vector<DTChambThSegm> > DTTSThetaManager;

class DTTSTheta : public DTTSThetaManager, public DTGeomSupplier {

// class DTTSTheta : public DTGeomSupplier {

  public:

    ///  Constructor
    DTTSTheta(DTTrigGeom*, DTBtiCard*);
  
    ///  Destructor 
    ~DTTSTheta();

    /// Return number of TStheta segments (just 1)
    int nSegm(int step);

    /// Return the requested DTTSTheta segment (only the first)
    const DTChambThSegm* segment(int step, unsigned n);

    /// Return number of DTBtiChip fired (used by DTTracoChip)
    int nTrig(int step);

    /// Return number of DTBtiChip fired with a HTRIG (used by DTTracoChip)
    int nHTrig(int step);

    /// Local position in chamber of a L1Trigger-data object
    LocalPoint localPosition(const DTTrigData*) const;

    /// Local direction in chamber of a L1Trigger-data object
    LocalVector localDirection(const DTTrigData*) const;

    /// Print a L1Trigger-data object with also local and global position/direction
    void print(const DTTrigData* trig) const;

    /// Load BTIs triggers and run TSTheta algoritm
    void reconstruct() { clearCache(); loadDTTSTheta(); runDTTSTheta(); }

  private:

    /// store DTBtiChip L1Triggers in the TST
    void loadDTTSTheta();

    /// run DTTSTheta algorithm (build the mask)
    void runDTTSTheta();

    /// Add a DTBtiChip L1Trigger to the DTTSTheta
    void add_btiT(int step, const DTBtiTrigData* btitrig);

    /// Clear
    void localClear();

    /// Return the BitArray of DTBtiChip fired
    BitArray<DTConfig::NCELLTH>* btiMask(int step) const;

    /// Return the BitArray of DTBtiChip fired with a HTRIG
    BitArray<DTConfig::NCELLTH>* btiQual(int step) const;

  private:

    DTBtiCard* _bticard;

    // Input data
    BitArray<DTConfig::NCELLTH> _trig[DTConfig::NSTEPL-DTConfig::NSTEPF+1];
    BitArray<DTConfig::NCELLTH> _Htrig[DTConfig::NSTEPL-DTConfig::NSTEPF+1];
    int _ntrig[DTConfig::NSTEPL-DTConfig::NSTEPF+1];
    int _nHtrig[DTConfig::NSTEPL-DTConfig::NSTEPF+1];

};

#endif
