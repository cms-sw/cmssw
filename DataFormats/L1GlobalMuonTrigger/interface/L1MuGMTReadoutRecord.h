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
    L1MuGMTReadoutRecord(int bxnr);

    /// destructor
    virtual ~L1MuGMTReadoutRecord();

    /// reset the record
    void reset();

    //
    // Getters
    //

    /// get bx counter
    int getBxCounter() const { return (int) m_BxCounter; }; 
    
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

    /// get MIP bit
    unsigned getMIPbit(int eta, int phi) const;

    /// get Quiet bit
    unsigned getQuietbit(int eta, int phi) const;

    //
    // Setters
    //

    /// set bx counter
    void setBxCounter(int bxnr) { m_BxCounter = (unsigned) bxnr; }; 

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
    /// set MIP bit
    void setMIPbit(int eta, int phi);

    /// set Quiet bit
    void setQuietbit(int eta, int phi);

  private:
    unsigned getBrlRank(int i) const;
    unsigned getFwdRank(int i) const;

    void setBrlRank(int i, unsigned value);
    void setFwdRank(int i, unsigned value);
    
  private:
    unsigned m_BxCounter;

    unsigned m_InputCands[16];

    unsigned m_BarrelCands[4];
    unsigned m_ForwardCands[4];

    unsigned m_BrlSortRanks;
    unsigned m_FwdSortRanks;

    unsigned m_GMTCands[4];

    // mip/iso bits (252 EACH)
    unsigned m_MIPbits[8];
    unsigned m_Quietbits[8];

};

#endif










