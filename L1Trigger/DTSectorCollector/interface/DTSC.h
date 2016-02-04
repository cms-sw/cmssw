//-------------------------------------------------
//
/**   \class: DTSC.h
 *
 *
 *   $Date: 2008/09/05 16:03:44 $
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
class DTSectCollPhCand;
class DTSectCollThCand;
class DTConfigSectColl;
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
  DTSC(int istat);
 
  //!  Destructor 
  ~DTSC();

  // non-const methods

    /// Set configuration
  void setConfig(DTConfigSectColl *conf) { _config=conf; };

  //! Add a TSM candidate to the Sect Coll, ifs is first/second track flag
  void addPhCand(DTSectCollPhCand* cand);

  //! Add a Theta candidate to sect coll
  void addThCand(DTSectCollThCand* cand);

  // CB CONTROLLA LA DIFFERENZA TRA QUESTO E addPhCand!!!!!!!!!!
  //! Add a Sector Collector  
  void addDTSectCollPhCand(DTSectCollPhCand* cand);


  //! Set a flag to skip sort2
  void ignoreSecondTrack() { _ignoreSecondTrack=1; }

  //! Run the Sector Collector algorithm
  void run();

  //! Phi Sort 1
  DTSectCollPhCand* DTSectCollsort1();
  
  //! Phi Sort 2
  DTSectCollPhCand* DTSectCollsort2();

  //! Clear
  void clear();

  // const methods

  //! Configuration set
  inline DTConfigSectColl* config() const { return _config; }

  //! Return the number of Phi input tracks (first/second)
  unsigned nCandPh (int ifs) const;

  //! Return the number of Theta input tracks
  unsigned nCandTh () const;

  //! Return the number of input first tracks
  inline int nFirstTPh() const { return _incand_ph[0].size(); }

  //! Return the number of input second tracks
  inline int nSecondTPh() const { return _incand_ph[1].size(); }

  //! Return requested TSS candidate
  DTSectCollPhCand* getDTSectCollPhCand(int ifs, unsigned n) const;

  //! Return requested Theta candidate
  DTSectCollThCand* getDTSectCollThCand(unsigned n) const;

  //! Return the number of output Phi tracks
  inline int nTracksPh() const { return _outcand_ph.size(); }

  //! Return the number of output Theta tracks
  inline int nTracksTh() const { return _cand_th.size(); }

  //! Return the requested Phi track
  DTSectCollPhCand* getTrackPh(int n) const ; 
  
  //! Return the requested Theta track
  DTSectCollThCand* getTrackTh(int n) const ;

 private:

  // Configuration
  DTConfigSectColl* _config;

  // input phi data
  std::vector<DTSectCollPhCand*> _incand_ph[2];

  // output phi data
  std::vector<DTSectCollPhCand*> _outcand_ph;

  // theta data 
  std::vector<DTSectCollThCand*> _cand_th;

  // internal use variables
  int _ignoreSecondTrack;

  // station number [1-5]
  int _stat;

};

#endif
