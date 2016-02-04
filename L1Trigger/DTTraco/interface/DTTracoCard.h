//-------------------------------------------------
//
/**  \class DTTracoCard
 *   Contains active DTTracoChips
 *
 *
 *   $Date: 2009/11/12 14:33:22 $
 *   $Revision: 1.10 $
 *
 *   \author C. Grandi, S. Vanini 
 *
 *    Modifications:
 *   III/07 : SV configuration with DTConfigManager 
*/
//
//--------------------------------------------------
#ifndef DT_TRACO_CARD_H
#define DT_TRACO_CARD_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTTracoChip;
class DTTracoTrig;
class DTBtiCard;
class DTTSTheta;
class DTTrigGeom;

//----------------------
// Base Class Headers --
//----------------------
#include "L1Trigger/DTUtilities/interface/DTGeomSupplier.h"
#include "DataFormats/MuonDetId/interface/DTTracoId.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfig.h"
#include "L1Trigger/DTTraco/interface/DTTracoTrigData.h"
#include "L1Trigger/DTUtilities/interface/DTCache.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTraco.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigLUTs.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"

//---------------
// C++ Headers --
//---------------
#include <vector>
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

typedef std::map< int,DTTracoChip*,std::less<int> >  TRACOContainer;
typedef TRACOContainer::const_iterator TRACO_const_iter;
typedef TRACOContainer::iterator TRACO_iter;

typedef std::map<DTTracoId,DTConfigTraco> ConfTracoMap;

typedef DTCache<DTTracoTrigData,std::vector<DTTracoTrigData> > TRACOCache;
  
class DTTracoCard : public TRACOCache, public DTGeomSupplier {

  public:

    /// Constructor
    //DTTracoCard(DTTrigGeom*, DTBtiCard*, DTTSTheta*,edm::ParameterSet&);
    DTTracoCard(DTTrigGeom*, DTBtiCard*, DTTSTheta*);

    /// Destructor 
    ~DTTracoCard();

    /// Clear all traco stuff (cache & map)
    void clearCache();

    /// Set configuration
    void setConfig(const DTConfigManager *conf);

    /// Return TU debug flag
    inline bool debug() {return _debug;}

    /// Return TSTheta
    inline DTTSTheta* TSTh() const { return _tstheta; }

    /// Returns the required DTTracoChip. Return 0 if it doesn't exist
    DTTracoChip* getTRACO(int n) const;

    /// Returns the required DTTracoChip. Return 0 if it doesn't exist
    DTTracoChip* getTRACO(const DTTracoId& tracoid) const {
      return getTRACO(tracoid.traco());
    }

    /// Returns the active TRACO list
    std::vector<DTTracoChip*> tracoList();

    /**
     * Returns a DTTracoTrig corresponding to a DTTracoTrigData.
     * Creates the corresponding TRACO chip if needed and stores the trigger
     */
    DTTracoTrig* storeTrigger(DTTracoTrigData);

    /// NEWGEO Local position in chamber of a trigger-data object
    LocalPoint localPosition(const DTTrigData*) const;

    /// NEWGEO Local direction in chamber of a trigger-data object
    LocalVector localDirection(const DTTrigData*) const;
    
    /// Load BTIs triggers and run TRACOs algorithm
    virtual void reconstruct() { clearCache(); loadTRACO(); runTRACO(); }

    /// Return LUTS config for this chamber (=minicrate)
    DTConfigLUTs* config_luts() const { return _conf_luts; } 

   /// Return acceptance flag
   inline bool useAcceptParamFlag() { return _flag_acc; } 

   /// Return lut computation option (DB/geometry)
   inline bool lutFromDBFlag() { return _lut_from_db; }

  private:

    /// store BTI triggers in TRACO's
    void loadTRACO();

    /// run TRACO algorithm
    void runTRACO();

    /// Returns the required DTTracoChip. Create it if it doesn't exist
    DTTracoChip* activeGetTRACO(int);

    /// Returns the required DTTracoChip. Create it if it doesn't exist
    DTTracoChip* activeGetTRACO(const DTTracoId& tracoid) {
      return activeGetTRACO(tracoid.traco());
    }

    /// clear the TRACO map
    void localClear();

    /// Return single TRACO config
    DTConfigTraco* config_traco(const DTTracoId& tracoid) const; 

  private:

    DTBtiCard* _bticard;
    DTTSTheta* _tstheta;

    TRACOContainer _tracomap;
    ConfTracoMap _conf_traco_map;	//bti configuration map for this chamber

    DTConfigLUTs* _conf_luts;
    
    bool _debug;

    bool _flag_acc;
    bool _lut_from_db;
};

#endif
