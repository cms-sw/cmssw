//---------------------------------------------
//
//   \class L1MuGlobalMuonTrigger
//
//   Description: L1 Global Muon Trigger
//
//
//   $Date: 2010/02/12 12:07:37 $
//   $Revision: 1.7 $
//
//   Author :
//   Ivan Mikulec                    HEPHY Vienna
//
//--------------------------------------------------
#ifndef L1TriggerGlobalMuonTrigger_L1MuGlobalMuonTrigger_h
#define L1TriggerGlobalMuonTrigger_L1MuGlobalMuonTrigger_h

//---------------
// C++ Headers --
//---------------

//----------------------
// Base Class Headers --
//----------------------
#include "FWCore/Framework/interface/EDProducer.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutRecord.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

class L1MuGMTConfig;
class L1MuGMTPSB;
class L1MuGMTMatcher;
class L1MuGMTCancelOutUnit;
class L1MuGMTMipIsoAU;
class L1MuGMTMerger;
class L1MuGMTSorter;

class L1MuGMTExtendedCand;

class L1MuGMTDebugBlock;

//---------------------
//-- Class Interface --
//---------------------
class L1MuGlobalMuonTrigger : public edm::EDProducer {

 public:

    explicit L1MuGlobalMuonTrigger(const edm::ParameterSet&);
    ~L1MuGlobalMuonTrigger();
    virtual void beginJob();

    virtual void produce(edm::Event&, const edm::EventSetup&);

    void reset();

    /// return pointer to PSB
    inline const L1MuGMTPSB* Data() const { return m_PSB; }

    /// return pointer to Matcher
    inline const L1MuGMTMatcher* Matcher(int id) const { return m_Matcher[id]; }

    /// return pointer to Cancel Out Unit
    inline const L1MuGMTCancelOutUnit* CancelOutUnit(int id) const { return m_CancelOutUnit[id]; }

    /// return pointer to MIP & ISO bit assignment unit
    inline const L1MuGMTMipIsoAU* MipIsoAU(int id) const { return m_MipIsoAU[id]; }

    /// return pointer to Merger
    inline const L1MuGMTMerger* Merger(int id) const { return m_Merger[id]; }

    /// return pointer to Sorter
    inline const L1MuGMTSorter* Sorter() const { return m_Sorter; }


    /// get the GMT readout data for the triggered bx
    /// readout data contains input and output muons as well as MIP and Quiet bits
    /// for 3 or 5 bx around the triggered bx
    std::auto_ptr<L1MuGMTReadoutCollection> getReadoutCollection();

    /// return a reference to the current record in the ring buffer
    L1MuGMTReadoutRecord* currentReadoutRecord () const { return m_ReadoutRingbuffer.back(); };


    /// for debug: return the debug block (in order to fill it)
    L1MuGMTDebugBlock* DebugBlockForFill() const { return m_db; };

 private:

    L1MuGMTPSB*             m_PSB;
    L1MuGMTMatcher*         m_Matcher[2];
    L1MuGMTCancelOutUnit*   m_CancelOutUnit[4];
    L1MuGMTMipIsoAU*        m_MipIsoAU[2];
    L1MuGMTMerger*          m_Merger[2];
    L1MuGMTSorter*          m_Sorter;

    std::vector<L1MuGMTExtendedCand> m_ExtendedCands;
    std::vector<L1MuGMTReadoutRecord*> m_ReadoutRingbuffer;

    bool m_writeLUTsAndRegs;
    bool m_sendMipIso;

    static L1MuGMTConfig* m_config;

    static L1MuGMTDebugBlock* m_db;
    
    unsigned long long m_L1MuGMTScalesCacheID;
    unsigned long long m_L1MuTriggerScalesCacheID;
    unsigned long long m_L1MuTriggerPtScaleCacheID;
    unsigned long long m_L1MuGMTParametersCacheID;
    unsigned long long m_L1MuGMTChannelMaskCacheID;
    unsigned long long m_L1CaloGeometryCacheID;
};

#endif // L1TriggerGlobalMuonTrigger_L1MuGlobalMuonTrigger_h
