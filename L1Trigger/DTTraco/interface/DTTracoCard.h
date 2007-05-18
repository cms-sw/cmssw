//-------------------------------------------------
//
/**  \class DTTracoCard
 *   Contains active DTTracoChips
 *
 *
 *   $Date: 2007/02/09 11:20:49 $
 *   $Revision: 1.2 $
 *
 *   \author C. Grandi, S. Vanini 
 *
 *    Modifications:
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
#include "L1Trigger/DTUtilities/interface/DTTracoId.h"
#include "L1Trigger/DTUtilities/interface/DTConfig.h"
#include "L1Trigger/DTTraco/interface/DTTracoTrigData.h"
#include "L1Trigger/DTUtilities/interface/DTCache.h"
#include "L1Trigger/DTTraco/interface/DTConfigTraco.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//---------------
// C++ Headers --
//---------------
#include <map>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

typedef DTCache< DTTracoTrigData,std::vector<DTTracoTrigData> > TRACOCache;

typedef std::map< int,DTTracoChip*,std::less<int> >  TRACOContainer;
typedef TRACOContainer::const_iterator TRACO_const_iter;
typedef TRACOContainer::iterator TRACO_iter;
  
class DTTracoCard : public TRACOCache, public DTGeomSupplier {

  public:

    /// Constructor
    DTTracoCard(DTTrigGeom*, DTBtiCard*, DTTSTheta*,edm::ParameterSet&);

    /// Destructor 
    ~DTTracoCard();

    /// Clear all traco stuff (cache & map)
    void clearCache();

    /// Return config
    inline DTConfigTraco* config() const { return _configTraco; } 

    /// Return TSTheta
    inline DTTSTheta* TSTh() const { return _tstheta; }

    /// Returns the required DTTracoChip. Return 0 if it doesn't exist
    DTTracoChip* getTRACO(int n) const;

    /// Returns the required DTTracoChip. Return 0 if it doesn't exist
    DTTracoChip* getTRACO(const DTTracoId& tracoid) const {
      return getTRACO(tracoid.traco());
    }

    /// minimum angle accepted by the TRACO as function of the BTI position
    inline int psimin(int pos) const { return _PSIMIN[pos-1]; }

    /// maximum angle accepted by the TRACO as function of the BTI position
    inline int psimax(int pos) const { return _PSIMAX[pos-1]; }

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

  private:

    DTBtiCard* _bticard;
    DTTSTheta* _tstheta;

    TRACOContainer _tracomap;

    // psi acceptance of correlator MT ports
    int _PSIMIN[4*DTConfig::NBTITC];
    int _PSIMAX[4*DTConfig::NBTITC];

    DTConfigTraco * _configTraco;
};

#endif
