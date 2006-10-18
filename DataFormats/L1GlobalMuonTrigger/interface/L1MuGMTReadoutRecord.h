//-------------------------------------------------
//
/** \class L1MuGMTReadoutRecord
 *
 *  L1 Global Muon Trigger Readout Buffer
 *
 *  Contains the data that the GMT sends to readout
 *  for one bunch crossing.
 *
 *  Only simple data members are used so that it is easier
 *  to make the data persistent or use it in an other context.
*/
//
//   $Date $
//   $Revision $
//
//   Author :
//   H. Sakulin                  HEPHY Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------
#ifndef DataFormatsL1GlobalMuonTrigger_L1MuGMTReadoutRecord_h 
#define DataFormatsL1GlobalMuonTrigger_L1MuGMTReadoutRecord_h 

//---------------
// C++ Headers --
//---------------
#include <vector>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
//class L1MuGMTExtendedCand;
class L1MuRegionalCand;

//              ---------------------
//              -- Class Interface --
//              ---------------------
using namespace std;

class L1MuGMTReadoutRecord {

  public:
    /// constructors
    L1MuGMTReadoutRecord();
    L1MuGMTReadoutRecord(int bxie);

    /// destructor
    virtual ~L1MuGMTReadoutRecord();

    /// reset the record
    void reset();

    //
    // Getters
    //

    /// get counters
    int getBxNr() const { return (int) m_BxNr; }; 
    int getBxCounter() const { return (int) m_BxInEvent; }; // for backward compatibility, do not use
    int getBxInEvent() const { return (int) m_BxInEvent; };
    int getEvNr() const { return (int) m_EvNr; };
    int getBCERR() const { return (int) m_BCERR; };
    
    /// get GMT candidates vector
    vector<L1MuGMTExtendedCand> getGMTCands() const;

    /// get GMT barrel candidates vector
    vector<L1MuGMTExtendedCand> getGMTBrlCands() const;

    /// get GMT forward candidates vector
    vector<L1MuGMTExtendedCand> getGMTFwdCands() const;

    /// get DT candidates vector
    vector<L1MuRegionalCand> getDTBXCands() const;

    /// get CSC candidates vector
    vector<L1MuRegionalCand> getCSCCands() const;

    /// get barrel RPC candidates vector
    vector<L1MuRegionalCand> getBrlRPCCands() const;

    /// get forward RPC candidates vector
    vector<L1MuRegionalCand> getFwdRPCCands() const;

    //
    // Setters
    //

    /// set counters
    void setBxNr(int bxnr) { m_BxNr = (unsigned) bxnr; }; 
    void setBxCounter(int bxie) { m_BxInEvent = (unsigned) bxie; }; // for backward compatibility, do not use
    void setBxInEvent(int bxie) { m_BxInEvent = (unsigned) bxie; }; 
    void setEvNr(int evnr) { m_EvNr = (unsigned) evnr; }; 
    void setBCERR(int bcerr) { m_BCERR = (unsigned) bcerr; }; 

    /// set GMT barrel candidate
    void setGMTBrlCand(int nr, L1MuGMTExtendedCand const& cand);

    /// set GMT barrel candidate
    void setGMTBrlCand(int nr, unsigned data, unsigned rank);

    /// set GMT forward candidate
    void setGMTFwdCand(int nr, L1MuGMTExtendedCand const& cand);

    /// set GMT forward candidate
    void setGMTFwdCand(int nr, unsigned data, unsigned rank);

    /// set GMT candidate (does not store rank)
    void setGMTCand(int nr, L1MuGMTExtendedCand const& cand);

    /// set GMT candidate (does not store rank)
    void setGMTCand(int nr, unsigned data);

    /// set Input muon
    void setInputCand(int nr, unsigned data) { if (nr>=0 && nr < 16) m_InputCands[nr] = data; };

  private:
    unsigned getBrlRank(int i) const;
    unsigned getFwdRank(int i) const;

    void setBrlRank(int i, unsigned value);
    void setFwdRank(int i, unsigned value);
    
  private:
    unsigned m_BxNr;
    unsigned m_BxInEvent;
    unsigned m_EvNr;

    unsigned m_BCERR;

    unsigned m_InputCands[16];

    unsigned m_BarrelCands[4];
    unsigned m_ForwardCands[4];

    unsigned m_BrlSortRanks;
    unsigned m_FwdSortRanks;

    unsigned m_GMTCands[4];

};

#endif










