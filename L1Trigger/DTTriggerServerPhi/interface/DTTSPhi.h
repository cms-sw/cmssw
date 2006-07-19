//-------------------------------------------------
//
/**  \class DTTSPhi
 *    Implementation of TS Phi trigger algorithm
 *
 *
 *   $Date: 2004/03/18 09:23:02 $
 *   $Revision: 1.12 $
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

//#include "CARF/Reco/interface/RecDet.h"
//#include "CARF/G3Event/interface/G3EventProxy.h"
#include "L1Trigger/DTUtilities/interface/DTCache.h"
#include "L1Trigger/DTUtilities/interface/DTGeomSupplier.h"
#include "L1Trigger/DTUtilities/interface/DTConfig.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTChambPhSegm.h"

//---------------
// C++ Headers --
//---------------
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

//typedef RecDet < DTChambPhSegm,G3EventProxy*, std::vector<DTChambPhSegm> > DTTSPhiManager;
typedef DTCache < DTChambPhSegm, std::vector<DTChambPhSegm> > DTTSPhiManager;

class DTTSPhi : public DTTSPhiManager, public DTGeomSupplier {
  
 public:
  
  /// Constructor
  DTTSPhi(DTTrigGeom*, DTTracoCard*);
  
  /// Destructor 
  ~DTTSPhi();
  
  /// Return number of DTTSPhi segments  
  int nSegm(int step);
  
  /// Return the requested DTTSPhi segment
  const DTChambPhSegm* segment(int step, unsigned n);
  
  /// Local position in chamber of a trigger-data object
  LocalPoint localPosition(const DTTrigData*) const;
  
  /// Local direction in chamber of a trigger-data object
  LocalVector localDirection(const DTTrigData*) const;
  
  /// Load TRACO triggers and run TSPhi algorithm
  virtual void reconstruct() { clearCache();  loadTSPhi(); runTSPhi(); }

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
  
  // Components
  std::vector<DTTSS*> _tss[DTConfig::NSTEPL-DTConfig::NSTEPF+1];
  // DBSM-doubleTSM
  //  DTTSM* _tsm[DTConfig::NSTEPL-DTConfig::NSTEPF+1];
  std::vector<DTTSM*> _tsm[DTConfig::NSTEPL-DTConfig::NSTEPF+1];
  
  // Input data
  std::vector<DTTSCand*> _tctrig[DTConfig::NSTEPL-DTConfig::NSTEPF+1];
  
};

#endif
