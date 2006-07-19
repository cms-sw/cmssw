//-------------------------------------------------
//
/**   \class L1MuDTSectColl.h
 *    Implementation of Sector Collector trigger algorithm
 *
 *
 *
 *    $Date: 2004/08/17 14:17:06 $
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
class DTConfig;
class DTTrigGeom;
class DTChambPhSegm;
class DTSectCollSegm;
class DTSC;
class DTTSPhi;
class DTSCTrigUnit;

//----------------------
// Base Class Headers --
//----------------------


//#include "CARF/Reco/interface/RecDet.h"
//#include "CARF/G3Event/interface/G3EventProxy.h"
#include "L1Trigger/DTUtilities/interface/DTCache.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollCand.h"

//---------------
// C++ Headers --
//---------------
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

//typedef RecDet< DTChambPhSegm,G3EventProxy*, std::vector<DTChambPhSegm> > DTSCManager;
typedef DTCache< DTChambPhSegm, std::vector<DTChambPhSegm> > DTSCManager;

class DTSectColl : public DTSCManager {

 public:

  //!  Constructor
  DTSectColl(DTConfig*);

  //!  Destructor 
  ~DTSectColl();

  //! Return TSPhi
  inline DTTSPhi* TSPhi() const { return _tsphi1; }

  //!Configuration
  inline DTConfig* config() const { return _config; }

  // non-const methods
  //! Add a TSM candidate to the Sect Coll, ifs is first/second track flag
  void addCand(DTSectCollCand* cand);

/*    //! Set a flag to skip sort2 */
/*    void ignoreSecondTrack() { _ignoreSecondTrack=1; } */

  //! localClear
  void localClear();

  //! load a Sector Collector
  void loadSectColl();

  //! add a TSM candidate
  void addTSPhi(int step, const DTChambPhSegm* tsmsegm, int ifs);

  //! add a Trigger Unit to the Sector Collector
  void addTU(DTSCTrigUnit* tru, int flag);

  //! get a Sector Collector
  DTSC* getDTSC(int step) const;

  //! Run Sector Collector
  void runSectColl();

  //! get a Candidate for Sector Collector
  DTSectCollCand* getDTSectCollCand(int ifs, unsigned n) const;

  // const methods

  //! Configuration set
  //   inline DTConfig* config() const { return _config; }

  //! Return requested TRACO trigger
  const DTTracoTrigData* getTracoT(int ifs, unsigned n) const;
 
  //! Return the requested track
  DTSectCollCand* getTrack(int n) const ;

  //! Return the number of input tracks (first/second)
  unsigned nCand(int ifs) const;

  //! Return number of DTTSPhi segments  
  int nSegm(int step);

  //! Return the requested DTTSPhi segment
  const DTChambPhSegm* segment(int step, unsigned n);

  //! Return the number of sorted tracks
  inline int nTracks() const { return _outcand.size(); }

  const DTChambPhSegm* SectCollSegment(int step, int n) {return segment(step,n); }

  //! Local position in chamber of a trigger-data object
  //  LocalPoint LocalPosition(const DTTrigData*) const;

  //! Local direction in chamber of a trigger-data object
  //  LocalVector LocalDirection(const DTTrigData*) const;

  /// Load TRUnits triggers and run Sector Collector algorithm
  virtual void reconstruct() { clearCache(); loadSectColl(); runSectColl(); }
 private:

  // Configuration
  DTConfig* _config;

  DTTSPhi* _tsphi1;
  DTTSPhi* _tsphi2;
  //DTTSPhi* _tsphi;
  //DTTSPhi* _tsphis[2];

  // SM: new  sector collector 
  DTSC* _tsc[DTConfig::NSTEPL-DTConfig::NSTEPF+1];

  std::vector<DTSectCollCand*> _tstrig[DTConfig::NSTEPL-DTConfig::NSTEPF+1];

  // input data
  std::vector<DTSectCollCand*> _incand[2];

  // output data
  std::vector<DTSectCollCand*> _outcand;

};
#endif
