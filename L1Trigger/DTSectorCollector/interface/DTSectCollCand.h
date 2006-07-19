//-------------------------------------------------
//
/**   \Class: L1MuDTSectCollCand.h
 *    A Trigger Server Candidate
 *
 *   $Date: 2004/03/24 14:39:07 $
 *  
 *
 *   \author D. Bonacorsi, S. Marcellini
 */
//
//--------------------------------------------------
#ifndef DT_SECT_COLL_CAND_H
#define DT_SECT_COLL_CAND_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTConfig;

//----------------------
// Base Class Headers --
//----------------------
#include "L1Trigger/DTSectorCollector/interface/DTSC.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTChambPhSegm.h"
#include "L1Trigger/DTTraco/interface/DTTracoTrigData.h"
#include "L1Trigger/DTUtilities/interface/BitArray.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTSectCollCand {

 public:
  
  DTSectCollCand(DTSC*, const DTChambPhSegm*, int);

  //!  Constructor
  DTSectCollCand();

  //!  Constructor
  DTSectCollCand(const DTSectCollCand& tsccand);
  
  //! Assignment operator
  DTSectCollCand& operator=(const DTSectCollCand& tsccand);

  //!  Destructor 
  ~DTSectCollCand();

  // Non-const methods

  //! Set the  bits for DTTSM analysis
  void setBitsSectColl();

  //! Clear (set to 1) the quality bits (but first/second track bit)
  void clearBits(){ _dataword.assign(5,3,0x7); }

  //! Clear (set to 1) the quality bits for Sector Collector
  void clearBitsSectColl(){ _dataword.assign(0,13,0x1fff); }

  //! Set the first track bit to second track (used for carry)
  void setSecondTrack() { _dataword.set(14); _isCarry=1; }
    
  //! Reset the carry bit
  void resetCarry() { _isCarry=0; }

  //! clear the trigger
  inline void clear();

  // Const methods

  //! Configuration set
  inline DTConfig* config() const { return _tsc->config(); }

  //! Return the DTTSS
  inline DTSC* tsc() const { return _tsc; }

  inline int isFirst() const { return _dataword.element(14)==0; }

  //! Return associated TSPhi trigger
  inline const DTTracoTrigData* tracoTr() const { return _tctrig; }
  inline const DTChambPhSegm* tsTr() const { return _tsmsegm; }

  //! Return an uint16 with the content of the data word (for debugging)
  inline unsigned dataword() const { return _dataword.dataWord(0)&0x1ff; }

  //! Operator < used for sorting
  bool operator < (const DTSectCollCand& c) const { return _dataword<c._dataword; }

  //! Operator <= used for sorting
  bool operator <= (const DTSectCollCand& c) const { return _dataword<=c._dataword; }

  //! Print the trigger
  void print() const; 


 private:

  DTSC* _tsc;
  const DTChambPhSegm* _tsmsegm;
  const DTTracoTrigData* _tctrig;

  // BitArray<9> _dataword;   // the word on which sorting is done
  BitArray<15> _dataword;   // the word on which sorting is done. reserve space enough for Preview and full data

  int _isCarry;            

};
#endif
