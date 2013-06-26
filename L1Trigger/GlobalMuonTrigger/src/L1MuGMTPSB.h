//-------------------------------------------------
//
/** \class L1MuGMTPSB
 *  L1 Global Muon Trigger Pipelined Synchronising Buffer module. 
 *
 *                the PSB receives muon candidates
 *                from the barrel track finder, the endcap track finder
 *                and from the RPC trigger. In addition it gets
 *                isolation and mip bits from the 
 *                regional Calorimeter Trigger
*/
//
//   $Date: 2007/04/12 13:21:14 $
//   $Revision: 1.3 $
//
//   Author :
//   N. Neumeister            CERN EP 
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------
#ifndef L1TriggerGlobalMuonTrigger_L1MuGMTPSB_h
#define L1TriggerGlobalMuonTrigger_L1MuGMTPSB_h

//---------------
// C++ Headers --
//---------------

#include <vector>

//----------------------
// Base Class Headers --
//----------------------

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMatrix.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

class L1MuGlobalMuonTrigger;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuGMTPSB {

  public:

    /// constructor
    L1MuGMTPSB(const L1MuGlobalMuonTrigger& gmt);

    /// destructor
    virtual ~L1MuGMTPSB();

    /// receive muon candidates
    void receiveData(edm::Event& e, int bx);
    
    /// clear PSB
    void reset();
    
    /// print PSB
    void print() const;
            
    /// get RPC muon
    const L1MuRegionalCand* RPCMuon(int index) const;  
    
    /// get DTBX muon
    const L1MuRegionalCand* DTBXMuon(int index) const;

    /// get CSC muon
    const L1MuRegionalCand* CSCMuon(int index) const;
    
    /// return number of non-empty RPC muons
    int numberRPC() const;
    
    /// return number of non-empty DTBX muons
    int numberDTBX() const;

    /// return number of non-empty CSC muons
    int numberCSC() const;
    
    /// are there any data in the PSB
    bool empty() const;

    /// return isolation bits
    const L1MuGMTMatrix<bool>& isolBits() const { return m_Isol; }
    
    /// return minimum ionizing bits
    const L1MuGMTMatrix<bool>& mipBits() const { return m_Mip; }

  private:

    /// get muons from RPCb Trigger
    void getRPCb(std::vector<L1MuRegionalCand> const* data, int bx);

    /// get muons from RPCf Trigger
    void getRPCf(std::vector<L1MuRegionalCand> const* data, int bx);

    /// get muons from barrel Muon Trigger Track Finder
    void getDTBX(std::vector<L1MuRegionalCand> const* data, int bx);
   
    /// get muons from endcap Muon Trigger Track Finder
    void getCSC(std::vector<L1MuRegionalCand> const* data, int bx);

    /// get Calorimeter Trigger data
    void getCalo(edm::Event& e); 
    
    /// print barrel RPC muons
    void printRPCbarrel() const;

    /// print endcap RPC muons
    void printRPCendcap() const;
    
    /// print DTBX muons
    void printDTBX() const;

    /// print CSC muons
    void printCSC() const;
    
  private:

    const L1MuGlobalMuonTrigger& m_gmt;
    
    std::vector<L1MuRegionalCand> m_RpcMuons;
    std::vector<L1MuRegionalCand> m_DtbxMuons;
    std::vector<L1MuRegionalCand> m_CscMuons;
    
    L1MuGMTMatrix<bool> m_Isol;
    L1MuGMTMatrix<bool> m_Mip;
   
};
  
#endif
