//-------------------------------------------------
//
/**  \class DTTSCand
 *    A Trigger Server Candidate
 *
 *   $Date: 2008/06/30 13:43:31 $
 *   $Revision: 1.5 $
 *
 *   \author C. Grandi, D. Bonacorsi, S. Marcellini
 */
//
//--------------------------------------------------
#ifndef DT_TS_CAND_H
#define DT_TS_CAND_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTConfigTSPhi;

//----------------------
// Base Class Headers --
//----------------------
#include "L1Trigger/DTTriggerServerPhi/interface/DTTSS.h"
#include "L1Trigger/DTTraco/interface/DTTracoTrigData.h"
#include "L1TriggerConfig/DTTPGConfig/interface/BitArray.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTTSCand {

  public:

    /// Constructor
    DTTSCand(DTTSS*, const DTTracoTrigData*, int, int);

    /// Constructor
    DTTSCand();

    /// Constructor
    DTTSCand(const DTTSCand& tscand);
  
    /// Assignment operator
    DTTSCand& operator=(const DTTSCand& tscand);

    /// Destructor 
    ~DTTSCand();

    /// Set the quality bits for DTTSS analysis
    void setBitsTss();

    /// Set the bits for TSM back-up mode
    void setBitsBkmod();

    /// Set the quality bits for DTTSM analysis
    void setBitsTsm();

    /// Set the first track bit to second track (used for carry)
    // SM sector collector: it becomes bit 14
    void setSecondTrack() { _dataword.set(14); _isCarry=1; }

    /// Reset the carry bit
    void resetCarry() { _isCarry=0; }

    /// Clear (set to 1) the quality bits (but first/second track bit)
    void clearBits(){ _dataword.assign(5,3,0x7); }

    /// Clear (set to 1) all the bits (back-up mode)
    void clearBitsBkmod(){ _dataword.assign(0,7,0xff); }

    /// Clear the trigger
    inline void clear();

    /// Configuration set
    inline DTConfigTSPhi* config() const { return _tss->config(); }

    /// Return associated TRACO trigger
    inline const DTTracoTrigData* tracoTr() const { return _tctrig; }

    /// Retrun the TRACO position inside the TSS
    inline int TcPos() const { return _tcPos; }

    /// Return the DTTSS
    inline DTTSS* tss() const { return _tss; }

    /// Return the DTTSS number
    inline int tssNumber() const { return _tss->number(); }

    /// Return the TRACO number
    inline int tracoNumber() const { return _tctrig->tracoNumber(); }

    /// Return the first/second track bit
    inline int isFirst() const { return _dataword.element(14)==0; }

    /// Return HTRIG/LTRIG bit
    inline int isHtrig() const { return _tctrig->pvCode()==8 || 
                                        _tctrig->pvCode()==80; }
    /// Return Inner/Outer bit
    inline int isInner() const { return _tctrig->pvCode()>8; }

    /// Return correlation bit
    inline int isCorr() const { return _tctrig->pvCorr(); }

    /// Return the carry bit
    inline int isCarry() const { return _isCarry; }

    /// Return if HH or HL
    inline int isHHorHL() const { return _tctrig->pvCorr() && _tctrig->pvCode()==80; }
 
    /// Return if LH
    inline int isLH() const { return _tctrig->pvCorr() && _tctrig->pvCode()==8; }
 
    /// Return if LL
    inline int isLL() const { return _tctrig->pvCorr() && !_tctrig->pvCode()==8 &&
                                                          !_tctrig->pvCode()==80; }  
    /// Return if H inner
    inline int isH0() const { return !_tctrig->pvCorr() && _tctrig->pvCode()==80; }
 
    /// Return if H outer
    inline int is0H() const { return !_tctrig->pvCorr() && _tctrig->pvCode()==8; }
 
    /// Return if L inner
    inline int isL0() const {  return !_tctrig->pvCorr() && _tctrig->pvCode()<80 &&
                                                            _tctrig->pvCode()>8; }
    /// Return if L outer
    inline int is0L() const {  return !_tctrig->pvCorr() && _tctrig->pvCode()<8; }

    /// Return an uint16 with the content of the data word (for debugging)
    inline unsigned dataword() const { return _dataword.dataWord(0)&0x1ff; }

    /// Operator < used for sorting
    bool operator < (const DTTSCand& c) const { return _dataword<c._dataword; }

    /// Operator <= used for sorting
    bool operator <= (const DTTSCand& c) const { return _dataword<=c._dataword; }

    // Operator <<= used for sorting in TSM back-up mode
    // SM double TSM  bool operator <<= (const DTTSCand& c) const { return _datawordbk<c._datawordbk; }

    /// Print the trigger
    void print() const;

  private:
  DTTSS* _tss;
  const DTTracoTrigData* _tctrig;
    BitArray<15> _dataword;   // the word on which sorting is done. reserve space enough for Preview and full data
  // SM double TSM  BitArray<9> _datawordbk; // the word on which sorting is done (back-up mode)
  int _tcPos;            // TRACO position in TSS
  // SM double TSM   int _bkmod;            // TSM back-up mode flag
  int _isCarry;          // info for TSM

};

#endif
