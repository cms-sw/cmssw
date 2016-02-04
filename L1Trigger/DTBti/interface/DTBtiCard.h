//-------------------------------------------------
//
/**  \class DTBtiCard
 *     Contains active DTBtiChips
 *
 *
 *   $Date: 2010/11/11 16:26:54 $
 *   $Revision: 1.12 $
 *
 *   \author C. Grandi, S. Vanini
 *
 *   Modifications:
 *   V/05 S.Vanini : modified to run with new geometry
 *   III/07 : SV configuration with DTConfigManager 
 */
//
//--------------------------------------------------
#ifndef DT_BTI_CARD_H
#define DT_BTI_CARD_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTBtiChip;
class DTBtiTrig;
class DTTrigGeom;
class DTTTrigBaseSync;

//----------------------
// Base Class Headers --
//----------------------
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "L1Trigger/DTUtilities/interface/DTGeomSupplier.h"
#include "DataFormats/MuonDetId/interface/DTBtiId.h"
#include "L1Trigger/DTBti/interface/DTBtiTrigData.h"
#include "L1Trigger/DTUtilities/interface/DTCache.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigBti.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"


//---------------
// C++ Headers --
//---------------
#include <vector>
#include <map>

namespace edm {class ParameterSet; class Event; class EventSetup;}

//              ---------------------
//              -- Class Interface --
//              ---------------------

typedef std::map< int,DTBtiChip*,std::less<int> >  BTIContainer;
typedef BTIContainer::const_iterator BTI_const_iter;
typedef BTIContainer::iterator BTI_iter;

typedef std::map<DTBtiId,DTConfigBti> ConfBtiMap;

typedef DTCache<DTBtiTrigData,std::vector<DTBtiTrigData> > BTICache;

class DTBtiCard : public BTICache, public DTGeomSupplier {

  public:

    /// Constructor
    DTBtiCard(DTTrigGeom *);

    /// Destructor 
    ~DTBtiCard();

    /// Clear all BTI stuff (map & cache)
    void clearCache();

    /// Set configuration
    void setConfig(const DTConfigManager *conf);

    /// Return TU debug flag
    inline bool debug() const {return _debug;}

    /// Returns the required BTI. Return 0 if it doesn't exist
    DTBtiChip* getBTI(int sl, int n) const; 

    /// Returns the required BTI. Return 0 if it doesn't exist
    DTBtiChip* getBTI(const DTBtiId& btiid) const {
      return getBTI(btiid.superlayer(),btiid.bti());
    }

    /// NEWGEO Local position in chamber of a trigger-data object
    LocalPoint localPosition(const DTTrigData*) const;
    /// NEWGEO Local direction in chamber of a trigger-data object
    LocalVector localDirection(const DTTrigData*) const;

    /// Returns the active BTI list in a given superlayer
    std::vector<DTBtiChip*> btiList(int);

    /**
     * Returns a DTBtiTrig corresponding to a DTBtiTrigData.
     * Creates the corresponding BTI chip if needed and stores the trigger
     */
    DTBtiTrig* storeTrigger(DTBtiTrigData);

    // run the trigger algorithm
    virtual void reconstruct(const DTDigiCollection dtDigis) { clearCache();loadBTI(dtDigis); runBTI(); }
 
    /// Return bti chip configuration
    DTConfigBti* config_bti(DTBtiId& btiid) const;

   /// Return acceptance flag
   inline bool useAcceptParamFlag() { return _flag_acc; } 
 
 private:

    /// store digi's in DTBtiChip's
    void loadBTI(const DTDigiCollection dtDigis);

    /// run DTBtiChip algorithm
    void runBTI();

    /// Returns the required DTBtiChip. Create it if it doesn't exist
    DTBtiChip* activeGetBTI(int sl, int n);

    /// Returns the required DTBtiChip. Create it if it doesn't exist
    DTBtiChip* activeGetBTI(const DTBtiId& btiid) {
      return activeGetBTI(btiid.superlayer(),btiid.bti());
    }

    /// clear the BTI maps
    void localClear();

  private:

    BTIContainer _btimap[3];
    ConfBtiMap _conf_bti_map;	//bti configuration map for this chamber

    std::vector<DTDigi*> _digis; 

    bool _debug;
    DTConfigPedestals* _pedestals;

    bool _flag_acc;
};

#endif
