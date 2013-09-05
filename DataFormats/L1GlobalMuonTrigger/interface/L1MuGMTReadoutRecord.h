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
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

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
    std::vector<L1MuGMTExtendedCand> getGMTCands() const;

    /// get GMT candidates vector as stored in data (no rank info)
    std::vector<L1MuGMTExtendedCand>& getGMTCandsData();

    /// get GMT barrel candidates vector
    std::vector<L1MuGMTExtendedCand> getGMTBrlCands() const;
    std::vector<L1MuGMTExtendedCand>& getGMTBrlCandsData() {return m_BarrelCands;};

    /// get GMT forward candidates vector
    std::vector<L1MuGMTExtendedCand> getGMTFwdCands() const;

    /// get DT candidates vector
    std::vector<L1MuRegionalCand> getDTBXCands() const;

    /// get CSC candidates vector
    std::vector<L1MuRegionalCand> getCSCCands() const;

    /// get barrel RPC candidates vector
    std::vector<L1MuRegionalCand> getBrlRPCCands() const;

    /// get forward RPC candidates vector
    std::vector<L1MuRegionalCand> getFwdRPCCands() const;

    /// get MIP bit
    unsigned getMIPbit(int eta, int phi) const;

    /// get Quiet bit
    unsigned getQuietbit(int eta, int phi) const;


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
    void setInputCand(int nr, unsigned data);

    /// set Input muon
    void setInputCand(int nr, L1MuRegionalCand const& cand);

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
    unsigned m_BxNr;
    unsigned m_BxInEvent;
    unsigned m_EvNr;

    unsigned m_BCERR;

    std::vector<L1MuRegionalCand> m_InputCands;

    std::vector<L1MuGMTExtendedCand> m_BarrelCands;
    std::vector<L1MuGMTExtendedCand> m_ForwardCands;
    std::vector<L1MuGMTExtendedCand> m_GMTCands;

    // mip/iso bits (252 EACH)
    unsigned m_MIPbits[8];
    unsigned m_Quietbits[8];

};

#endif










