//-------------------------------------------------
//
/**   \Class DTSectCollPhCand.h
 *    A Trigger Server Phi Candidate
 *
 *   $Date: 2008/06/30 13:44:28 $
 *  
 *
 *   \Authors D. Bonacorsi, S. Marcellini
 */
//
//--------------------------------------------------
#ifndef DT_SECT_COLL_PH_CAND_H
#define DT_SECT_COLL_PH_CAND_H


//----------------------
// Base Class Headers --
//----------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigSectColl.h"
#include "L1Trigger/DTSectorCollector/interface/DTSC.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTChambPhSegm.h"
#include "L1Trigger/DTTraco/interface/DTTracoTrigData.h"
#include "L1TriggerConfig/DTTPGConfig/interface/BitArray.h"
//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTSectCollPhCand {

 public:
  
  //!  Constructor
  DTSectCollPhCand(DTSC* , const DTChambPhSegm*, int);

  //!  Constructor
  DTSectCollPhCand();

  //!  Constructor
  DTSectCollPhCand(const DTSectCollPhCand& tsccand);
  
  //! Assignment operator
  DTSectCollPhCand& operator=(const DTSectCollPhCand& tsccand);

  //!  Destructor 
  ~DTSectCollPhCand();

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

  //! Set the SC Candidate step
  
  
  //! Clear the trigger
  inline void clear();

  // Const methods

  //! Configuration set
  inline DTConfigSectColl* config() const { return _tsc->config(); }

  //! Return the DTTSS
  inline DTSC* tsc() const { return _tsc; }

  //! Return first/second track bit value
  inline int isFirst() const { return _dataword.element(14)==0; }

   //! Return associated TSPhi trigger 
   inline const DTChambPhSegm* tsTr() const { return _tsmsegm; } 

  //! Return an uint16 with the content of the data word (for debugging)
  inline unsigned dataword() const { return _dataword.dataWord(0)&0x1ff; }

  //! Operator < used for sorting
  bool operator < (const DTSectCollPhCand& c) const { return _dataword<c._dataword; }

  //! Operator <= used for sorting
  bool operator <= (const DTSectCollPhCand& c) const { return _dataword<=c._dataword; }

  //! Print the trigger
  void print() const; 

  //! Return the Coarse Sync Parameter
  int CoarseSync() const;


 private:

  DTSC* _tsc;
  const DTChambPhSegm* _tsmsegm;

  // BitArray<9> _dataword;   // the word on which sorting is done
  BitArray<15> _dataword;   // the word on which sorting is done. reserve space enough for Preview and full data

  int _isCarry;

};
#endif
