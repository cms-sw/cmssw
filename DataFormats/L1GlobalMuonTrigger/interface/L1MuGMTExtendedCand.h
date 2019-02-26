//-------------------------------------------------
//
/**  \class L1MuGMTExtendedCand
 *
 *   L1 Global Muon Trigger Extended Candidate.
 *
 *   This is a GMT candidate with extended information 
 *   that will be sent to Readout.
 * 
 *   This candidates contains extra information such 
 *   as sort rank and indices of the contributing muons.
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
#ifndef DataFormatsL1GlobalMuonTrigger_L1MuGMTExtendedCand_h
#define DataFormatsL1GlobalMuonTrigger_L1MuGMTExtendedCand_h

//---------------
// C++ Headers --
//---------------

#include <iosfwd>
#include <string>

//----------------------
// Base Class Headers --
//----------------------

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuGMTExtendedCand : public L1MuGMTCand {

  public:
    /// constructor   
    L1MuGMTExtendedCand();
   
    /// constructor   
    L1MuGMTExtendedCand(unsigned data, unsigned rank, int bx=0);
   
    /// copy constructor
    L1MuGMTExtendedCand(const L1MuGMTExtendedCand&);

    /// destructor
     ~L1MuGMTExtendedCand() override;

    /// reset muon candidate
    void reset();

    //
    // Getters
    //
    
    /// get rank
    unsigned int rank() const { return m_rank; }
    
    /// get index of contributing DT/CSC muon
    unsigned getDTCSCIndex() const { 
      return readDataField( IDXDTCSC_START, IDXDTCSC_LENGTH); 
    }

    /// get index of contributing RPC muon
    unsigned getRPCIndex() const { 
      return readDataField( IDXRPC_START, IDXRPC_LENGTH); 
    }

    /// get forward bit (true=forward, false=barrel)
    bool isFwd() const { return readDataField( FWDBIT_START, FWDBIT_LENGTH) == 1; }

    /// get RPC bit (true=RPC, false = DT/CSC or matched)
    bool isRPC() const { return readDataField( ISRPCBIT_START, ISRPCBIT_LENGTH) == 1; }

      /// set rank
    void setRank(unsigned int rank) { m_rank = rank; }

    /// get detector bits
    /// 1=rpc, 2=dtbx, 4=csc, 3=rpc+dtbx, 5=rpc+csc
    /// supported for backward compatibility only
    unsigned int detector() const ;

    //
    // Setters
    // 

    /// set index of contributing DT/CSC muon
    void setDTCSCIndex(unsigned int idxdtcsc) { 
      writeDataField( IDXDTCSC_START, IDXDTCSC_LENGTH, idxdtcsc); 
    }

    /// set index of contributing RPC muon
    void setRPCIndex(unsigned int idxrpc) { writeDataField( IDXRPC_START, IDXRPC_LENGTH, idxrpc); }

    /// set forward bit (1=forward, 0=barrel)
    void setFwdBit(unsigned int fwdbit) { writeDataField( FWDBIT_START, FWDBIT_LENGTH, fwdbit); }

    /// set RPC bit (1=RPC, 0=DT/CSC or matched)
    void setRPCBit(unsigned int rpcbit) { writeDataField( ISRPCBIT_START, ISRPCBIT_LENGTH, rpcbit); }

    /// equal operator
    bool operator==(const L1MuGMTExtendedCand&) const;
    
    /// unequal operator
    bool operator!=(const L1MuGMTExtendedCand&) const;

    /// print parameters of muon candidate
    void print() const;
  
    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const L1MuGMTExtendedCand&);

    /// define a rank for muon candidates
    static bool compareRank( const L1MuGMTExtendedCand* first, const L1MuGMTExtendedCand* second ) {
      unsigned int rank_f = (first) ? first->rank(): 0;
      unsigned int rank_s = (second) ? second->rank() : 0;
      return rank_f > rank_s;
    }

    /// define a rank for muon candidates
    static bool rankRef( const L1MuGMTExtendedCand& first, const L1MuGMTExtendedCand& second ) {
      unsigned int rank_f = first.rank();
      unsigned int rank_s = second.rank();
      return rank_f > rank_s;
    }

  private:
    unsigned int m_rank;
    
    enum { IDXDTCSC_START=26}; enum { IDXDTCSC_LENGTH = 2}; // Bit  26:27 DT/CSC muon index
    enum { IDXRPC_START=28};   enum { IDXRPC_LENGTH = 2};   // Bit  28:29 RPC muon index
    enum { FWDBIT_START=30};   enum { FWDBIT_LENGTH = 1};   // Bit  30    fwd bit
    enum { ISRPCBIT_START=31}; enum { ISRPCBIT_LENGTH = 1}; // Bit  31    isRPC bit
};
  
#endif
