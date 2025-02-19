//-------------------------------------------------
//
/**   \Class DTSectCollThCand.h
 *    A Trigger Server Theta Candidate
 *
 *   $Date: 2008/06/30 13:44:28 $
 *  
 *
 *   \Author C. Battilana
 */
//
//--------------------------------------------------
#ifndef DT_SECT_COLL_TH_CAND_H
#define DT_SECT_COLL_TH_CAND_H

//----------------------
// Base Class Headers --
//----------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigSectColl.h"
#include "L1Trigger/DTSectorCollector/interface/DTSC.h"
#include "L1Trigger/DTTriggerServerTheta/interface/DTChambThSegm.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTSectCollThCand {

 public:
  
  DTSectCollThCand(DTSC*, const DTChambThSegm*);

  //!  Constructor
  DTSectCollThCand();

  //!  Constructor
  DTSectCollThCand(const DTSectCollThCand& tsccand);
  
  //! Assignment operator
  DTSectCollThCand& operator=(const DTSectCollThCand& tsccand);

  //!  Destructor 
  ~DTSectCollThCand();

  // Non-const methods

  //! Clear the trigger
  inline void clear();

  // Const methods

  //! Configuration set
  inline DTConfigSectColl* config() const { return _tsc->config(); }

  //! Return the DTTSS
  inline DTSC* tsc() const { return _tsc; }

  //! Return associated TSTheta trigger
  inline const DTChambThSegm* tsTr() const { return _tstsegm; }

  //! Print the trigger
  void print() const; 

  //! Return the Coarse Sync Parameter
  int CoarseSync() const;

 private:

  DTSC* _tsc;
  const DTChambThSegm* _tstsegm;

};
#endif
