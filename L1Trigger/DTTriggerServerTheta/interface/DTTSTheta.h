//-------------------------------------------------
//
/**  \class DTTSTheta
 *    Implementation of TS Theta L1Trigger algorithm
 *
 *   $Date: 2008/09/05 15:57:29 $
 *   $Revision: 1.7 $
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
class DTBtiCard;
class DTBtiTrigData;
class DTTrigGeom;

//----------------------
// Base Class Headers --
//----------------------
#include "L1Trigger/DTUtilities/interface/DTGeomSupplier.h"
#include "L1TriggerConfig/DTTPGConfig/interface/BitArray.h"
#include "L1Trigger/DTTriggerServerTheta/interface/DTChambThSegm.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTSTheta.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"
#include "L1Trigger/DTUtilities/interface/DTCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//---------------
// C++ Headers --
//---------------
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

typedef std::vector<DTChambThSegm> DTChambThVector;
typedef DTCache < DTChambThSegm, DTChambThVector > DTTSThetaManager;

class DTTSTheta : public DTTSThetaManager, public DTGeomSupplier {

  public:

    ///  Constructor
    //DTTSTheta(DTTrigGeom*, DTBtiCard*, edm::ParameterSet&);
    DTTSTheta(DTTrigGeom*, DTBtiCard*);

    ///  Destructor 
    ~DTTSTheta();

    /// Return configuration
    inline DTConfigTSTheta* config() const { return _config; }

    /// Set configuration
    void setConfig(const DTConfigManager *conf);

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
    void reconstruct() { loadDTTSTheta(); runDTTSTheta(); }

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
    BitArray<DTConfigTSTheta::NCELLTH>* btiMask(int step) const;

    /// Return the BitArray of DTBtiChip fired with a HTRIG
    BitArray<DTConfigTSTheta::NCELLTH>* btiQual(int step) const;

  private:

    DTBtiCard* _bticard;

    DTConfigTSTheta* _config;

    // Input data
    BitArray<DTConfigTSTheta::NCELLTH> _trig[DTConfigTSTheta::NSTEPL-DTConfigTSTheta::NSTEPF+1];
    BitArray<DTConfigTSTheta::NCELLTH> _Htrig[DTConfigTSTheta::NSTEPL-DTConfigTSTheta::NSTEPF+1];
    int _ntrig[DTConfigTSTheta::NSTEPL-DTConfigTSTheta::NSTEPF+1];
    int _nHtrig[DTConfigTSTheta::NSTEPL-DTConfigTSTheta::NSTEPF+1];

};

#endif
