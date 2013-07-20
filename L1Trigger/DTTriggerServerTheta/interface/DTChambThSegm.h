//-------------------------------------------------
//
/**  \class DTChambThSegm
 *
 *   Muon Chamber Trigger Theta candidate 
 *
 *
 *   $Date: 2008/06/30 13:42:53 $
 *   $Revision: 1.5 $
 * 
 *   \author C. Grandi
 */
//--------------------------------------------------
#ifndef DT_CHAMB_TH_SEGM_H
#define DT_CHAMB_TH_SEGM_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//----------------------
// Base Class Headers --
//----------------------
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "L1TriggerConfig/DTTPGConfig/interface/BitArray.h"
#include "L1Trigger/DTUtilities/interface/DTTrigData.h"

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

typedef unsigned char myint8;

class DTChambThSegm : public DTTrigData {

  public:

    /// Constructor
    DTChambThSegm(DTChamberId,int,int*,int*);

    /// Constructor
    DTChambThSegm(const DTChambThSegm& seg);
  
    /// Destructor 
    ~DTChambThSegm();

    /// Assignment operator
    DTChambThSegm& operator=(const DTChambThSegm& seg);

    /// Clear
    void clear();

    /// Return step number
    inline int step() const { return m_step; }

    /// Identifier of the associated chamber
    DTChamberId ChamberId() const { return m_chamberid; }

    /// Print
    void print() const;

    /// Return the code for a given set of 7 BTI
    int code(const int i) const;

    /// Return the position for a given set of 7 BTI
    int position(const int i) const;

    /// Return the quality for a given set of 7 BTI
    int quality(const int i) const;

  private:

    /// parent chamber
    DTChamberId m_chamberid;

    /// step number
    int m_step;

    /// output code
    myint8 m_outPos[7];
    myint8 m_outQual[7];

};

#endif
