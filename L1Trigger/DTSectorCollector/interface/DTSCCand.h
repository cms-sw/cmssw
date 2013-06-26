//-------------------------------------------------
//
//   \class: DTSC.h
/**
 *   Implementation of Sector Collector trigger algorithm
 *
 *
 *   $Date: 2007/02/09 11:24:32 $
 *
 *   \author D. Bonacorsi, S. Marcellini
 */
//
//--------------------------------------------------
#ifndef DT_SC_H  
#define DT_SC_H   

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTTracoTrigData;
class DTTSCand;
class DTConfigSectColl;
// added DBSM
class DTTrigGeom;

//----------------------
// Base Class Headers --
//----------------------
// added DBSM
#include "L1Trigger/DTUtilities/interface/DTGeomSupplier.h"

//---------------
// C++ Headers --
//---------------
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------


class DTSC{

 public:

  //!  Constructor 
  DTSC(DTConfigSectColl*);

  //!  Destructor 
  ~DTSC();

  // non-const methods

  //! Add a TSM candidate to the Sect Coll, ifs is first/second track flag
  void addCand(DTTSCand* cand);

  //! Set a flag to skip sort2
  void ignoreSecondTrack() { _ignoreSecondTrack=1; }

  //! Run the Sector Collector algorithm
  void run();

  //! Sort 1
  DTTSCand* DTSectCollsort1();
  
  //! Sort 2
  DTTSCand* DTSectCollsort2();

  //! Clear
  void clear();

  // const methods

  //! Configuration set
  inline DTConfigSectColl* config() const { return _config; }

  //! Return the number of input tracks (first/second)
  unsigned nCand(int ifs) const;

  //! Return the number of input first tracks
  inline int nFirstT() const { return _incand[0].size(); }

  //! Return the number of input second tracks
  inline int nSecondT() const { return _incand[1].size(); }

  //! Return requested TSS candidate
  DTTSCand* getDTTSCand(int ifs, unsigned n) const;

  //! Return requested TRACO trigger
  const DTTracoTrigData* getTracoT(int ifs, unsigned n) const;

  //! Return the number of sorted tracks
  inline int nTracks() const { return _outcand.size(); }

  //! Return the requested track
  DTTSCand* getTrack(int n) const ;


 private:

  // Configuration
  DTConfigSectColl* _config;

  // input data
  std::vector<DTTSCand*> _incand[2];

  // output data
  std::vector<DTTSCand*> _outcand;

  // internal use variables
  int _ignoreSecondTrack;


};
#endif









