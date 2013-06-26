//-------------------------------------------------
//
/**   \class DTSectCollPhSegm
 *
 *    Muon Sector Collector Trigger Phi candidate 
 *
 *    $Date: 2007/04/27 08:45:51 $
 *    
 *
 *   \author S. Marcellini, D. Bonacorsi
 * 
 */
//
//--------------------------------------------------
#ifndef DT_SECT_COLL_PH_SEGM_H
#define DT_SECT_COLL_PH_SEGM_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//----------------------
// Base Class Headers --
//----------------------
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSectCollId.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTChambPhSegm.h"
#include "L1Trigger/DTUtilities/interface/DTTrigData.h"

//---------------
// C++ Headers --
//---------------
#include <vector>
//              ---------------------
//              -- Class Interface --
//              ---------------------

typedef unsigned char myint8;

class DTSectCollPhSegm : public DTTrigData {

 public:

  //!  Constructor
  DTSectCollPhSegm(DTSectCollId scId, int step);
  
  //!  Constructor
  DTSectCollPhSegm(DTSectCollId scId, int step, 
		    const DTChambPhSegm* tsPhiTrig, int isFirst);
  //!  Constructor
  DTSectCollPhSegm(const DTSectCollPhSegm&);
  
  //!  Destructor 
  ~DTSectCollPhSegm();

  // Non-const methods
  
  //! Assignment operator
  DTSectCollPhSegm& operator=(const DTSectCollPhSegm& seg);

  //! Associate a Trigger Server Phi trigger
  inline void setTsPhiTrig(const DTChambPhSegm* tsphitrig, int isFirst) {
    m_tsphitrig=tsphitrig; 
    m_isFirst=isFirst;
  }

  //! Clear
  void clear();

  // Const methods

  // //! Return associated TRACO trigger
  // inline const DTTracoTrigData* tracoTrig() const { return m_tsphitrig->TracoTrigData(); }
  
  //! Return associated Trigger Server Phi trigger
  inline const DTChambPhSegm* tsPhiTrig() const { return m_tsphitrig; }
  
  //! Return step number
  inline int step() const { return m_step; }

  //! Return SC identifier
  DTSectCollId SCId() const { return m_sectcollid; }

  //! Return chamber identifier 
  DTChamberId ChamberId() const { return m_tsphitrig->ChamberId(); } 

  //! Print
  void print() const;

  //! Return parent TRACO number
  inline int tracoNumber() const { return m_tsphitrig->tracoNumber(); }

  //! Return if it is a first track
  inline int isFirst() const { return m_isFirst==1; }

  //! Return trigger code (MTTF input format [0,7])
  int code() const { return m_tsphitrig->code(); }

  //! Return trigger code (10*inner_code+outer_code; X_code=1,2,3,4,8)
  inline int oldCode() const { return m_tsphitrig->oldCode(); }

  //! Return trigger K parameter
  inline float K() const { return m_tsphitrig->K(); }

  //! Return trigger X parameter
  inline float X() const { return m_tsphitrig->X(); }

  //! Return trigger K parameter converted to angle (bit pattern)
  inline int psi() const { return m_tsphitrig->psi(); }

  //! Return trigger X parameter converted to angle (bit pattern)
  inline int psiR() const { return m_tsphitrig->psiR(); }

  //! Return trigger X parameter converted to angle (bit pattern)
  inline int phi() const { return m_tsphitrig->psiR(); }

  //! Return bending angle (bit pattern)
  inline int DeltaPsiR() const { return m_tsphitrig->DeltaPsiR(); }

  //! Return bending angle (bit pattern)
  inline int phiB() const { return m_tsphitrig->DeltaPsiR(); }

  //! Return correlator output code (position of segments)
  inline int posMask() const { return m_tsphitrig->posMask(); }
  
  //! Return the preview code (10*inner_code or outer_code; X_code=1,2,3,4,8)
  inline int pvCode() const { return m_tsphitrig->pvCode(); }

  //! Return the preview K
  inline int pvK() const { return m_tsphitrig->pvK(); }

 private:
  // parent sectcoll
  DTSectCollId m_sectcollid;

  // step number
  myint8 m_step;

  // first or second track
  myint8 m_isFirst;

  // the corresponding traco trigger
  const DTChambPhSegm* m_tsphitrig;

};
#endif
