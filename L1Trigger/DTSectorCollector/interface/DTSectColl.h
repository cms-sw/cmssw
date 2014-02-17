//-------------------------------------------------
//
/**   \class DTSectColl.h
 *    Implementation of Sector Collector trigger algorithm
 *
 *
 *
 *    $Date: 2008/09/05 16:03:44 $
 *
 *    \author D. Bonacorsi, S. Marcellini
 */
//--------------------------------------------------
#ifndef DT_SECT_COLL_H   
#define DT_SECT_COLL_H   

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTTracoTrigData;
class DTSectCollCand;
class DTConfigSectColl;
class DTTrigGeom;
class DTChambPhSegm;
class DTChambThSegm;
class DTSectCollSegm;
class DTSC;
class DTTSPhi;
class DTTSTheta;
class DTSCTrigUnit;

//----------------------
// Base Class Headers --
//----------------------
#include "L1Trigger/DTUtilities/interface/DTCache.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollPhCand.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollThCand.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollPhSegm.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollThSegm.h"
#include "DataFormats/MuonDetId/interface/DTSectCollId.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//---------------
// C++ Headers --
//---------------
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

typedef std::vector<DTSectCollPhSegm> DTSectCollPhVector;
typedef std::vector<DTSectCollThSegm> DTSectCollThVector;
typedef DTCache< DTSectCollPhSegm, DTSectCollPhVector > DTSCPhCache;
typedef DTCache< DTSectCollThSegm, DTSectCollThVector > DTSCThCache;

class DTSectColl : public DTSCPhCache, public DTSCThCache  {

 public:

  //!  Constructor
  //DTSectColl(edm::ParameterSet& sc_pset);
  DTSectColl(DTSectCollId id);
  
  //!  Destructor 
  ~DTSectColl();

  /// Set configuration
  void setConfig(const DTConfigManager *conf);

  //! Return TSPhi
  inline DTTSPhi* getTSPhi(int istat) const { return _tsphi[istat-1]; }
  
  //! Return TSTheta
  inline DTTSTheta* getTSTheta(int istat) const { return _tstheta[istat-1]; }

  //!Configuration
  inline DTConfigSectColl* config() const { return _config; }

  // non-const methods

/*   //! Add a TSM candidate to the Sect Coll, ifs is first/second track flag  */
/*   void addCandPh(DTSectCollPhCand* cand);  */

/*   //! Add a TS Theta candidate to the Sect Coll, ifs is first/second track flag  */
/*   void addCandTh(DTSectCollThCand* cand); */

/*    //! Set a flag to skip sort2 */
/*    void ignoreSecondTrack() { _ignoreSecondTrack=1; } */

  //! Local Clear
  void localClear();

  //! Load a Sector Collector
  void loadSectColl();

  //! Add a TSM candidate (step is TSM step not SC one)
  void addTSPhi(int step, const DTChambPhSegm* tsmsegm, int ifs, int istat);
  
  //! Add a TS Theta candidate (step is TSTheta step not sc one)
  void addTSTheta(int step, const DTChambThSegm* tsmsegm, int istat);

  //! Add a Trigger Unit to the Sector Collector
  void addTU(DTSCTrigUnit* tru);

  //! Get a Sector Collector (step is TS one)
  DTSC* getDTSC(int step, int istat) const;

  //! Run Sector Collector
  void runSectColl();

  //! Get a Phi Candidate for Sector Collector
  DTSectCollPhCand* getDTSectCollPhCand(int ifs, unsigned n) const;

  //! Get a Candidate for Sector Collector
  DTSectCollThCand* getDTSectCollThCand(unsigned n) const;

  // const methods

  //! Return the requested Phi track
  DTSectCollPhCand* getTrackPh(int n) const ; 
  
  //! Return the requested Theta track
  DTSectCollThCand* getTrackTh(int n) const ;

  //! Return the number of Phi input tracks (first/second)
  unsigned nCandPh(int ifs) const;  
  
  //! Return the number of Theta input tracks
  unsigned nCandTh() const;

  //! Return number of DTSectCollPhi segments (SC step) 
  int nSegmPh(int step);
  
  //! Return number of DTSectCollTheta segments (SC step)  
  int nSegmTh(int step);

  //! Return the number of output Phi tracks
  inline int nTracksPh() const { return _outcand_ph.size(); }

  //! Return the number of output Theta tracks
  inline int nTracksTh() const { return _outcand_th.size(); }

  //! Return requested Sector Collector Phi Segment 1st/2nd
  const DTSectCollPhSegm* SectCollPhSegment(int step, unsigned n);

  //! Return requested Sector Collector Theta Segment
  const DTSectCollThSegm* SectCollThSegment(int step);

  //! Return iterator to the beghin of Phi cache
  std::vector<DTSectCollPhSegm>::const_iterator beginPh() { return  DTSCPhCache::_cache.begin(); }

  //! Return Phi cache size 
  int sizePh() { return DTSCPhCache::_cache.size(); } 

  //! Return iterator to the end of Phi cache
  std::vector<DTSectCollPhSegm>::const_iterator endPh() { return DTSCPhCache::_cache.end(); } 
  
  //! Return iterator to the begni of Theta cache
  std::vector<DTSectCollThSegm>::const_iterator beginTh() { return DTSCThCache::_cache.begin(); }
  
  //! Return Theta cache size 
  int sizeTh() { return DTSCThCache::_cache.size(); }

  //! Return iterator to the end of Theta cache
  std::vector<DTSectCollThSegm>::const_iterator endTh() { return DTSCThCache::_cache.end(); }
  
  //! Local position in chamber of a trigger-data object
  //  LocalPoint LocalPosition(const DTTrigData*) const;

  //! Local direction in chamber of a trigger-data object
  //  LocalVector LocalDirection(const DTTrigData*) const;

  //! Clear both (Phi and Theta) caches
  void clearCache() { DTSCPhCache::clearCache();  DTSCThCache::clearCache(); }
  
  //! Load Trigger Units triggers and run Sector Collector algorithm
  virtual void reconstruct() { loadSectColl(); runSectColl(); }

  //! Return the Sector Collector Id
  DTSectCollId SectCollId() { return _sectcollid; }
 
 private:

  // Configuration
  DTConfigSectColl* _config;

  // SC Id
  DTSectCollId _sectcollid;

  DTTSPhi* _tsphi[DTConfigSectColl::NTSPSC];
  DTTSTheta* _tstheta[DTConfigSectColl::NTSTSC];

  // SM: new  sector collector 
  DTSC* _tsc[DTConfigSectColl::NSTEPL-DTConfigSectColl::NSTEPF+1][DTConfigSectColl::NDTSC];

  // input data Phi
  std::vector<DTSectCollPhCand*> _incand_ph[2]; 

  // output data Phi
  std::vector<DTSectCollPhCand*> _outcand_ph; 
  
  // input data Theta
  std::vector<DTSectCollThCand*> _incand_th;  

  // output data Theta
  std::vector<DTSectCollThCand*> _outcand_th; 

};
#endif
