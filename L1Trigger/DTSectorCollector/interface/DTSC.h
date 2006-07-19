//-------------------------------------------------
//
/**   \class: L1MuDTSC.h
 *
 *
 *   $Date: 2004/03/24 14:23:07 $
 *
 *   Implementation of Sector Collector trigger algorithm
 *
 *   \Author S. Marcellini
 */
//
//
//--------------------------------------------------
#ifndef DT_SC_H  
#define DT_SC_H   

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTTracoTrigData;
class DTChambPhSegm;
class DTSectCollCand;
class DTConfig;
class DTTrigGeom;

//----------------------
// Base Class Headers --
//----------------------
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
  DTSC(DTConfig*);
 
  //!  Destructor 
  ~DTSC();

  // non-const methods

  //! Add a TSM candidate to the Sect Coll, ifs is first/second track flag
  void addCand(DTSectCollCand* cand);

  //! Set a flag to skip sort2
  void ignoreSecondTrack() { _ignoreSecondTrack=1; }

  //! Run the Sector Collector algorithm
  void run();

  //! Sort 1
  DTSectCollCand* DTSectCollsort1();
  
  //! Sort 2
  DTSectCollCand* DTSectCollsort2();

  //! Clear
  void clear();

  // const methods

  //! Configuration set
  inline DTConfig* config() const { return _config; }

  //! Return the number of input tracks (first/second)
  unsigned nCand(int ifs) const;

  //! Return the number of input first tracks
  inline int nFirstT() const { return _incand[0].size(); }

  //! Return the number of input second tracks
  inline int nSecondT() const { return _incand[1].size(); }

  //! add a Sector Collector
  void addDTSectCollCand(DTSectCollCand* cand);

  //! Return requested TSS candidate
  DTSectCollCand* getDTSectCollCand(int ifs, unsigned n) const;

  //! Return requested TRACO trigger
  const DTTracoTrigData* getTracoT(int ifs, unsigned n) const;

  //! Return the number of sorted tracks
  inline int nTracks() const { return _outcand.size(); }

  //! Return the requested track
  //  DTSectCollCand* getTrack(int n) const ;
  DTSectCollCand* getTrack(int n) const ;

 private:

  // Configuration
  DTConfig* _config;

  // input data
  std::vector<DTSectCollCand*> _incand[2];

  // output data
  std::vector<DTSectCollCand*> _outcand;

  // internal use variables
  int _ignoreSecondTrack;


};

#endif
