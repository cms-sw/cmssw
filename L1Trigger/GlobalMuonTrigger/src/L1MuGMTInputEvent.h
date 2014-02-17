//-------------------------------------------------
//
//   \class L1MuGMTInputEvent
//
//   Description:	Test file for the GMT standalone program. Contains
//			all the input data the GMT receives (CSC, RPC, DT, GCT)
//
//                
//   $Date: 2007/04/12 13:21:14 $
//   $Revision: 1.3 $
//
//   Author :
//   Tobias Noebauer                 HEPHY Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------
#ifndef L1TriggerGlobalMuonTrigger_L1MuGMTInputEvent_h
#define L1TriggerGlobalMuonTrigger_L1MuGMTInputEvent_h

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>
#include <map>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMatrix.h"
class L1MuRegionalCand;

//---------------------
//-- Class Interface --
//---------------------

class L1MuGMTInputEvent {

  public:

    /// constructor, initializes everything to 0/false, apart from 
    /// the ISO bits, which are initialized to true;
    L1MuGMTInputEvent() ;

    // destructor
    virtual ~L1MuGMTInputEvent();

    /// resets everything to 0/false, apart from
    /// the ISO bits, which are all set to true;
    void reset();

    void setRunNumber(unsigned long runnr) { m_runnr=runnr; };

    void setEventNumber(unsigned long eventnr) { m_evtnr=eventnr; };

    void addInputMuon(std::string chipid, const L1MuRegionalCand& inMu);

    //eta = [0,13], phi = [0,17]
    void setMipBit(unsigned etaIndex, unsigned phiIndex, bool val) { 
      m_mip_bits(etaIndex, phiIndex) = val; };
    
    //eta = [0,13], phi = [0,17]
    void setIsoBit(unsigned etaIndex, unsigned phiIndex, bool val) {
      m_iso_bits(etaIndex, phiIndex) = val; };

    /// get the Run number
    unsigned long getRunNumber() const { return m_runnr; };

    /// get the Event number
    unsigned long getEventNumber() const { return m_evtnr; };

    /// is the event empty?
    bool isEmpty() const { return (m_runnr == 0L) && (m_evtnr == 0L); };

    /// get [index]th input muon in chip [chipid]
    /// @param chipid is the input chip ID (IND, INC, INB, INF)
    /// @param index  is the index of the muon in the input chip (starting with 0)
    /// @return the L1MuRegionalCand specified or 0 if index is out of range
    const L1MuRegionalCand* getInputMuon(std::string chipid, unsigned index) const;

    /// get the MIP / ISO bits
    const L1MuGMTMatrix<bool>& getMipBits() const { return m_mip_bits; };
    
    const L1MuGMTMatrix<bool>& getIsoBits() const { return m_iso_bits; };

    const bool& getMipBit(unsigned etaIndex, unsigned phiIndex) { return m_mip_bits(etaIndex, phiIndex); };
    const bool& getIsoBit(unsigned etaIndex, unsigned phiIndex) { return m_iso_bits(etaIndex, phiIndex); };
    


  private:
    unsigned long m_runnr;
    unsigned long m_evtnr;

    std::map<std::string, std::vector <L1MuRegionalCand> > m_inputmuons;

    L1MuGMTMatrix<bool> m_mip_bits;
    L1MuGMTMatrix<bool> m_iso_bits;


};

#endif


