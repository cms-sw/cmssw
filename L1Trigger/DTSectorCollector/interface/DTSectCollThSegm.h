//-------------------------------------------------
//
/**  \class DTSectCollThSegm
 *
 *   Muon Sector Collector Trigger Theta candidate 
 *
 *
 *   $Date: 2007/04/27 08:45:51 $
 *   $Revision: 1.3 $
 * 
 *   \author C. Battilana
 */
//
//--------------------------------------------------
#ifndef DT_SECTCOLL_TH_SEGM_H
#define DT_SECTCOLL_TH_SEGM_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//----------------------
// Base Class Headers --
//----------------------
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSectCollId.h"
#include "L1Trigger/DTTriggerServerTheta/interface/DTChambThSegm.h"
#include "L1Trigger/DTUtilities/interface/DTTrigData.h"

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

typedef unsigned char myint8;

class DTSectCollThSegm : public DTTrigData {

  public:

    /// Constructor
    DTSectCollThSegm(DTSectCollId, int, const DTChambThSegm*);

    /// Constructor
    DTSectCollThSegm(const DTSectCollThSegm& seg);
  
    /// Destructor 
    ~DTSectCollThSegm();

    /// Assignment operator
    DTSectCollThSegm& operator=(const DTSectCollThSegm& seg);

    /// Clear
    void clear();

    /// Return step number
    inline int step() const { return m_step; }

    /// Identifier of the associated chamber
    DTSectCollId SCId() const { return m_sectcollid; }

    /// Identifier of the associated chamber
    DTChamberId ChamberId() const { return m_tsthetatrig->ChamberId(); }

    /// Print
    void print() const;

    /// Return the code for a given set of 7 BTI
    inline int code(const int i) const { return m_tsthetatrig->code(i); } ;

    /// Return the position for a given set of 7 BTI
    inline int position(const int i) const { return m_tsthetatrig->position(i); } ;

    /// Return the quality for a given set of 7 BTI
    inline int quality(const int i) const { return m_tsthetatrig->quality(i); };

  private:

    /// parent sector collector
    DTSectCollId m_sectcollid;

    /// step number
    myint8 m_step;

    /// the corresponding TS theta trigger
    const DTChambThSegm* m_tsthetatrig;
};

#endif
