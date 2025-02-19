//-------------------------------------------------
//
/**  \class L1MuGMTCand
 *
 *   L1 Global Muon Trigger Candidate.
 *
 *   This candidate contains only information sent to the GT.
*/
//
//   $Date: 2007/04/02 15:44:06 $
//   $Revision: 1.5 $
//
//   Author :
//   H. Sakulin               HEPHY Vienna
//   N. Neumeister            CERN EP 
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------
#ifndef DataFormatsL1GlobalMuonTrigger_L1MuGMTCand_h
#define DataFormatsL1GlobalMuonTrigger_L1MuGMTCand_h

//---------------
// C++ Headers --
//---------------
#include <string>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuGMTCand {

  public:

    /// constructor   
    L1MuGMTCand();
   
    /// constructor from dataword
    L1MuGMTCand(unsigned data, int bx=0);
   
    /// copy constructor
    L1MuGMTCand(const L1MuGMTCand&);

    /// destructor
    virtual ~L1MuGMTCand();

    /// reset muon candidate
    void reset();

    //
    // Getters
    //

    /// is it an empty  muon candidate?
    bool empty() const { return readDataField( PT_START, PT_LENGTH) == 0; }

    /// get muon data word
    unsigned getDataWord() const { return m_dataWord; }

    /// get name of object
    std::string name() const { return m_name; }

    /// get phi-code
    unsigned int phiIndex() const { return readDataField( PHI_START, PHI_LENGTH); }
    
    /// get pt-code
    unsigned int ptIndex() const { return readDataField( PT_START, PT_LENGTH); }
    
    /// get quality

    /// Quality codes:
    ///
    /// 0 .. no muon 
    /// 1 .. beam halo muon (CSC)
    /// 2 .. very low quality level 1 (e.g. ignore in single and di-muon trigger)
    /// 3 .. very low quality level 2 (e.g. ignore in single muon trigger use in di-muon trigger)
    /// 4 .. very low quality level 3 (e.g. ignore in di-muon trigger, use in single-muon trigger)
    /// 5 .. unmatched RPC 
    /// 6 .. unmatched DT or CSC
    /// 7 .. matched DT-RPC or CSC-RPC
    ///
    /// attention: try not to rely on quality codes in analysis: they may change again
    /// 
    unsigned int quality() const { return readDataField( QUAL_START, QUAL_LENGTH); }

    /// interpretation of quality code: is the candidate to be used in a single muon trigger ?
    bool useInSingleMuonTrigger() const { return quality() >= 4; };
    
    /// interpretation of quality code: is the candidate to be used in a di-muon trigger ?
    bool useInDiMuonTrigger() const { return (quality() >= 3) && (quality() !=4); }; 

    /// interpretation of quality code: is the candidate a matched candidate ?
    bool isMatchedCand() const { return quality() == 7; }

    /// interpretation of quality code: is the candidate a beam halo muon ?
    bool isHaloCand() const { return quality() == 1; }

     /// get eta-code
    unsigned int etaIndex() const { return readDataField( ETA_START, ETA_LENGTH); }
    
    /// get charge/synchronization word (0=POS, 1=NEG, 2=UNDEF, 3=SYNC)
    unsigned sysign() const { return readDataField( SYSIGN_START, SYSIGN_LENGTH); }
    
    /// get isolation
    bool isol() const { return readDataField( ISO_START, ISO_LENGTH) == 1; }

    /// get mip 
    bool mip() const { return readDataField( MIP_START, MIP_LENGTH) == 1; } 
    
    /// get bunch crossing identifier
    int bx() const { return m_bx; }
    
    /// get phi-value of muon candidate in radians (low edge of bin) 
    /// this functionality will be moved to an extra Producer
    float phiValue() const;
    
    /// get eta-value of muon candidate
    /// this functionality will be moved to an extra Producer
    float etaValue() const;
    
    /// get pt-value of muon candidate in GeV
    /// this functionality will be moved to an extra Producer
    float ptValue() const;
    
     /// get charge (+1  -1)
    int charge() const { return (readDataField( SYSIGN_START, SYSIGN_LENGTH) & 1 ) == 0 ? 1: -1; }
        
    /// is the charge valid ?
    bool charge_valid() const { 
      unsigned sysign = readDataField( SYSIGN_START, SYSIGN_LENGTH) ;
      return  (sysign == 0 || sysign == 1 );
    }
    
    /// is the candidate a sync word 
    bool isSyncWord() const { return readDataField( SYSIGN_START, SYSIGN_LENGTH) == 3; }
    
    ///
    /// Setters
    ///
    
    /// set packed phi-code of muon candidate
    void setPhiPacked(unsigned phi) { writeDataField( PHI_START, PHI_LENGTH, phi); }
    
    /// set packed pt-code of muon candidate
    void setPtPacked(unsigned pt) { writeDataField( PT_START, PT_LENGTH, pt); }
    
    /// set quality of muon candidate
    void setQuality(unsigned quality) { writeDataField( QUAL_START, QUAL_LENGTH, quality); }

    /// set packed eta-code of muon candidate
    void setEtaPacked(unsigned eta) { writeDataField( ETA_START, ETA_LENGTH, eta); }

    /// set isolation of muon candidate
    void setIsolation(bool isol) { writeDataField( ISO_START, ISO_LENGTH, isol?1:0); }
    
    /// set min ionizing bit for muon candidate
    void setMIP(bool mip) { writeDataField( MIP_START, MIP_LENGTH, mip?1:0); }    

    /// set packed charge/synchronization word of muon candidate (0=POS, 1=NEG, 2=UNDEF, 3=SYNC)
    void setChargePacked(unsigned ch) { writeDataField( SYSIGN_START, SYSIGN_LENGTH, ch); }    

    /// set bunch crossing identifier
    void setBx(int bx) { m_bx = bx; }

    /// Setters for physical values

    /// Set Phi Value
    void setPhiValue(float phiVal) {m_phiValue = phiVal;}

    /// Set Pt Value
    void setPtValue(float ptVal) {m_ptValue = ptVal;}

    /// Set Eta Value (need to set type, first)
    void setEtaValue(float etaVal) {m_etaValue = etaVal;}

    //
    // Other
    //

    unsigned int linearizedPt(float lsbValue, unsigned maxScale) const { return 0; }

    unsigned int etaRegionIndex() const { return etaIndex(); }

    unsigned int phiRegionIndex() const { return phiIndex(); }

    /// equal operator
    bool operator==(const L1MuGMTCand&) const;
    
    /// unequal operator
    bool operator!=(const L1MuGMTCand&) const;

    /// print parameters of muon candidate
    void print() const;
  
    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const L1MuGMTCand&);

  protected: 

  protected: 
    inline unsigned readDataField(unsigned start, unsigned count) const; 
    inline void writeDataField(unsigned start, unsigned count, unsigned value); 

    std::string       m_name;
    int          m_bx;      // in here only for technical reasons in simulation
    unsigned m_dataWord;                                // muon data word (26 bits) :

    float m_phiValue;
    float m_etaValue;
    float m_ptValue;
    static const float m_invalidValue;


                                                        // definition of the bit fields
    enum { PHI_START=0};     enum { PHI_LENGTH = 8};    // Bits 0:7   phi (8 bits)
    enum { PT_START=8};      enum { PT_LENGTH =  5};    // Bits 8:12  pt  (5 bits)
    enum { QUAL_START=13};   enum { QUAL_LENGTH = 3};   // Bits 13:15 quality (3 bits)
    enum { ETA_START=16};    enum { ETA_LENGTH = 6};    // Bits 16:21 eta (6 bits)
    enum { ISO_START=22};    enum { ISO_LENGTH = 1};    // Bit  22    Isolation
    enum { MIP_START=23};    enum { MIP_LENGTH = 1};    // Bit  23    MIP
    enum { SYSIGN_START=24}; enum { SYSIGN_LENGTH = 2}; // Bit  24:25 Charge/Syncword
};

unsigned L1MuGMTCand::readDataField(unsigned start, unsigned count) const {
  unsigned mask = ( (1 << count) - 1 ) << start;
  return (m_dataWord & mask) >> start;
}

void L1MuGMTCand::writeDataField(unsigned start, unsigned count, unsigned value) {
  unsigned mask = ( (1 << count) - 1 ) << start;
  m_dataWord &= ~mask; // clear
  m_dataWord |= (value << start) & mask ;
}


#endif
