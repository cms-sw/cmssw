//-------------------------------------------------
//
/**  \class DTBtiCard
 *     Contains active L1MuDTBtiChips
 *
 *
 *   $Date: 2007/02/09 11:20:06 $
 *   $Revision: 1.2 $
 *
 *   \author C. Grandi, S. Vanini
 *
 *   Modifications:
 *   V/05 S.Vanini : modified to run with new geometry
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

//----------------------
// Base Class Headers --
//----------------------
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "L1Trigger/DTUtilities/interface/DTGeomSupplier.h"
#include "L1Trigger/DTUtilities/interface/DTBtiId.h"
#include "L1Trigger/DTBti/interface/DTBtiTrigData.h"
#include "L1Trigger/DTUtilities/interface/DTCache.h"

#include "L1Trigger/DTBti/interface/DTConfigBti.h"


//---------------
// C++ Headers --
//---------------
#include <vector>
#include <map>

namespace edm {class ParameterSet; class Event; class EventSetup;}

//              ---------------------
//              -- Class Interface --
//              ---------------------

typedef DTCache<DTBtiTrigData,std::vector<DTBtiTrigData> > BTICache;

typedef std::map< int,DTBtiChip*,std::less<int> >  BTIContainer;
typedef BTIContainer::const_iterator BTI_const_iter;
typedef BTIContainer::iterator BTI_iter;

class DTBtiCard : public BTICache, public DTGeomSupplier {

  public:

    /// Constructor
    DTBtiCard(DTTrigGeom*,edm::ParameterSet&);

    /// Destructor 
    ~DTBtiCard();

    /// Returns configuration
    inline DTConfigBti* config() const { return _configBti; }

    /// Clear all BTI stuff (map & cache)
    void clearCache();

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
    std::vector<DTDigi*> _digis; 
    DTConfigBti * _configBti; 

};

#endif
