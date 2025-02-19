//-------------------------------------------------
//
/**  \class DTChambPhSegm
 *
 *    Muon Chamber Trigger Phi candidate 
 *
 *
 *   $Date: 2007/03/09 15:17:44 $
 *   $Revision: 1.3 $
 *
 *   \author C. Grandi S. Marcellini. D. Bonacorsi
 */
//--------------------------------------------------
#ifndef DT_CHAMB_PH_SEGM_H
#define DT_CHAMB_PH_SEGM_H

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
#include "L1Trigger/DTTraco/interface/DTTracoTrigData.h"
#include "L1Trigger/DTUtilities/interface/DTTrigData.h"

//---------------
// C++ Headers --
//---------------
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

typedef unsigned char myint8;

class DTChambPhSegm : public DTTrigData {

  public:

    /// Constructor
    DTChambPhSegm(DTChamberId, int);

    /// Constructor
/*sm   DTChambPhSegm(MuBarChamberId, int, */
/*sm 		    const DTTracoTrigData* tracotrig, int); */
    DTChambPhSegm(DTChamberId, int,
		    const DTTracoTrigData*, int);
    /// Constructor */
/*sm   DTChambPhSegm(const DTChambPhSegm& seg); */
    DTChambPhSegm(const DTChambPhSegm&); 

    /// Destructor 
    ~DTChambPhSegm();

  
    /// Assignment operator
    /*sm    DTChambPhSegm& operator=(const DTChambPhSegm& seg); */
    DTChambPhSegm& operator=(const DTChambPhSegm&);
    /// Associate a TRACO trigger
    inline void setTracoTrig(const DTTracoTrigData* tracotrig, int isFirst) {

       m_tracotrig = tracotrig; 
       m_isFirst = isFirst;
    }

    /// Clear
    void clear();

    /// Return associated TRACO trigger
    inline const DTTracoTrigData* tracoTrig() const { return m_tracotrig; }

    /// Return step number
    inline int step() const { return m_step; }

    /// Return chamber identifier
    DTChamberId ChamberId() const { return m_chamberid; }

    /// Print
    void print() const;

    /// Return parent TRACO number
    inline int tracoNumber() const { return m_tracotrig->tracoNumber(); }

    /// Return if it is a first track
    inline int isFirst() const { return m_isFirst == 1; }

    /// Return trigger code (MTTF input format [0,7])
    int code() const { return m_tracotrig->qdec(); }

    /// Return trigger code (10*inner_code+outer_code; X_code=1,2,3,4,8)
    inline int oldCode() const { return m_tracotrig->code(); }

    /// Return trigger K parameter
    inline float K() const { return m_tracotrig->K(); }

    /// Return trigger X parameter
    inline float X() const { return m_tracotrig->X(); }

    /// Return trigger K parameter converted to angle (bit pattern)
    int psi() const { return m_tracotrig->psi(); }

    /// Return trigger X parameter converted to angle (bit pattern)
    int psiR() const { return m_tracotrig->psiR(); }

    /// Return trigger X parameter converted to angle (bit pattern)
    int phi() const { return m_tracotrig->psiR(); }

    /// Return bending angle (bit pattern)
    inline int DeltaPsiR() const { return m_tracotrig->DeltaPsiR(); }

    /// Return bending angle (bit pattern)
    inline int phiB() const { return m_tracotrig->DeltaPsiR(); }

    /// Return correlator output code (position of segments)
    inline int posMask() const { return m_tracotrig->posMask(); }
  
    /// Return the preview code (10*inner_code or outer_code; X_code=1,2,3,4,8)
    inline int pvCode() const { return m_tracotrig->pvCode(); }

    /// Return the preview K
    inline int pvK() const { return m_tracotrig->pvK(); }

  private:

    /// parent chamber
    DTChamberId m_chamberid;

    /// step number
    myint8 m_step;

    /// first or second track
    myint8 m_isFirst;

    /// the corresponding traco trigger
    const DTTracoTrigData* m_tracotrig;

};

#endif
