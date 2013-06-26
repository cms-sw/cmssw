//-------------------------------------------------
//
/**  \class DTTSPhi
 *    Implementation of TS Phi trigger algorithm
 *
 *
 *   $Date: 2008/09/05 15:59:57 $
 *   $Revision: 1.7 $
 *
 *   \author C. Grandi, D. Bonacorsi, S. Marcellini
 */
//
//--------------------------------------------------
#ifndef DT_TS_PHI_H
#define DT_TS_PHI_H

//-------------------
// Constants file  --
//-------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTTracoCard;
class DTTracoTrigData;
class DTTSS;
class DTTSM;
class DTSectColl;
class DTTSCand;
class DTTrigGeom;

//----------------------
// Base Class Headers --
//----------------------
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/DTUtilities/interface/DTCache.h"
#include "L1Trigger/DTUtilities/interface/DTGeomSupplier.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTSPhi.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"

#include "L1Trigger/DTTriggerServerPhi/interface/DTChambPhSegm.h"

//---------------
// C++ Headers --
//---------------
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

typedef std::vector<DTChambPhSegm> DTChambPhVector;
typedef DTCache < DTChambPhSegm, DTChambPhVector > DTTSPhiManager;

class DTTSPhi : public DTTSPhiManager, public DTGeomSupplier {
  
 public:
  
  /// Constructor
  DTTSPhi(DTTrigGeom*, DTTracoCard*);

  /// Destructor 
  ~DTTSPhi();

  /// Return the configuration class
  inline DTConfigTSPhi* config() const {return _config; }

  /// Set configuration
  void setConfig(const DTConfigManager *conf);
  
  /// Return number of DTTSPhi segments  
  int nSegm(int step);
  
  /// Return the requested DTTSPhi segment
  const DTChambPhSegm* segment(int step, unsigned n);
  
  /// Local position in chamber of a trigger-data object
  LocalPoint localPosition(const DTTrigData*) const;
  
  /// Local direction in chamber of a trigger-data object
  LocalVector localDirection(const DTTrigData*) const;
  
  /// Load TRACO triggers and run TSPhi algorithm
  virtual void reconstruct() { loadTSPhi(); runTSPhi(); }

 private:
  
  /// store DTTracoChip triggers in the DTTSS's
  void loadTSPhi();
  
  /// run DTTSPhi algorithm (DTTSS+DTTSM)
  void runTSPhi();
  
  /// Add a DTTracoChip trigger to the DTTSPhi, ifs is track number (first or second)
  void addTracoT(int step, const DTTracoTrigData* tracotrig, int ifs);
  
  /// Set a flag to ignore second tracks (if first track at following BX)
  void ignoreSecondTrack(int step, int tracon);
  
  /// Clear
  void localClear();
  
  // Return a DTTSS
  DTTSS* getDTTSS(int step, unsigned n) const;
  
  // SM double TSM
  // Return a DTTSM
  DTTSM* getDTTSM(int step, unsigned n) const;
  

  
 private:
  
  DTTracoCard* _tracocard;

  DTConfigTSPhi* _config;
  
  // Components
  std::vector<DTTSS*> _tss[DTConfigTSPhi::NSTEPL-DTConfigTSPhi::NSTEPF+1];
  // DBSM-doubleTSM
  std::vector<DTTSM*> _tsm[DTConfigTSPhi::NSTEPL-DTConfigTSPhi::NSTEPF+1];
  
  // Input data
  std::vector<DTTSCand*> _tctrig[DTConfigTSPhi::NSTEPL-DTConfigTSPhi::NSTEPF+1];
  
};

#endif
