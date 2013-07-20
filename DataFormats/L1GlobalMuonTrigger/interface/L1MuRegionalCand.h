//-------------------------------------------------
//
/** \class L1MuRegionalCand
 *    A regional muon trigger candidate as received by the GMT
*/
//
//   $Date: 2007/10/18 12:19:14 $
//   $Revision: 1.3 $
//
//   Author :
//   H. Sakulin                    HEPHY Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------
#ifndef DataFormatsL1GlobalMuonTrigger_L1MuRegionalCand_h
#define DataFormatsL1GlobalMuonTrigger_L1MuRegionalCand_h

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuRegionalCand {

  public:
  
    /// constructor from data word
    L1MuRegionalCand(unsigned dataword = 0, int bx = 0); 

    /// constructor from packed members
    L1MuRegionalCand(unsigned type_idx, unsigned phi, unsigned eta, unsigned pt, unsigned charge,
		     unsigned ch_valid, unsigned finehalo, unsigned quality, int bx);

    /// destructor
    virtual ~L1MuRegionalCand() {}
    
    ///
    /// Getters - values
    ///

    /// return empty flag
    virtual bool empty() const { 
      return (readDataField( PT_START, PT_LENGTH) == 0) || (readDataField( PHI_START, PHI_LENGTH) == 0xff); 
    }

    /// get phi-value of muon candidate in radians (low edge of bin)
    float phiValue() const;   

    /// get eta-value of muon candidate
    float etaValue() const;    

    /// get pt-value of muon candidate in GeV
    float ptValue() const;
    
    /// get charge
    int chargeValue() const { return readDataField( CHARGE_START, CHARGE_LENGTH) == 0 ? 1: -1; }
        
    /// is the charge valid ?
    bool chargeValid() const { return charge_valid_packed() == 1; }

    /// is it fine (DT) / halo (CSC) ?
    bool isFineHalo() const { return finehalo_packed() == 1; }

    /// return quality
    unsigned int quality() const { return quality_packed(); }    
        
    /// return type: 0 DT, 1 bRPC, 2 CSC, 3 fRPC
    unsigned type_idx() const { return (int) readDataField( TYPE_START, TYPE_LENGTH); };

    /// return bunch crossing identifier
    int bx() const { return  m_bx; }


    ///
    /// Getters - packed format
    ///

    /// return phi packed as in hardware
    unsigned phi_packed() const { return readDataField (PHI_START, PHI_LENGTH); } 

    /// return pt packed as in hardware
    unsigned pt_packed() const { return readDataField (PT_START, PT_LENGTH); } 

    /// return quality packed as in hardware
    unsigned int quality_packed() const { return readDataField( QUAL_START, QUAL_LENGTH); }

    /// return eta packed as in hardware
    unsigned eta_packed() const { return readDataField( ETA_START, ETA_LENGTH); }

    /// return eta-fine (for DT) / halo (for CSC) bit
    unsigned finehalo_packed() const { return readDataField( FINEHALO_START, FINEHALO_LENGTH); }

    /// return charge packed as in hardware (0=pos, 1=neg)
    unsigned charge_packed() const { return readDataField( CHARGE_START, CHARGE_LENGTH); } 

    /// return charge valid packed as in hardware (1=valid, 0=not valid)
    unsigned charge_valid_packed() const { return readDataField( CHVALID_START, CHVALID_LENGTH); } 

    /// return data word
    unsigned getDataWord() const { return m_dataWord; };

    ///
    /// Setters - packed format
    ///

    /// Set Type: 0 DT, 1 bRPC, 2 CSC, 3 fRPC
    void setType(unsigned type) { writeDataField( TYPE_START, TYPE_LENGTH, type); }

    /// Set Bunch Crossing
    void setBx(int bx) { m_bx = bx; }

    /// Set Phi: 0..143
    void setPhiPacked(unsigned phi) { writeDataField (PHI_START, PHI_LENGTH, phi); }

    /// Set Pt: 0..31
    void setPtPacked(unsigned pt) { writeDataField (PT_START, PT_LENGTH, pt); }

    /// Set Quality: 0..7
    void setQualityPacked(unsigned qual) { writeDataField (QUAL_START, QUAL_LENGTH, qual); }

    /// Set Charge (0=pos, 1=neg)
    void setChargePacked(unsigned ch) { writeDataField (CHARGE_START, CHARGE_LENGTH, ch); }

    /// Set Charge Valid
    void setChargeValidPacked(unsigned valid) { writeDataField( CHVALID_START, CHVALID_LENGTH, valid ); }

    /// Set Eta: 6-bit code
    void setEtaPacked(unsigned eta) { writeDataField (ETA_START, ETA_LENGTH, eta); }

    /// Set Fine / Halo
    void setFineHaloPacked(unsigned fh) { writeDataField (FINEHALO_START, FINEHALO_LENGTH, fh); }


    ///
    /// Setters - values
    ///

    /// Set Phi Value
    void setPhiValue(float phiVal) {m_phiValue = phiVal;}

    /// Set Pt Value
    void setPtValue(float ptVal) {m_ptValue = ptVal;}

    /// Set Eta Value (need to set type, first)
    void setEtaValue(float etaVal) {m_etaValue = etaVal;}

    /// Set Charge Value: -1, 1
    void setChargeValue(int charge) { writeDataField (CHARGE_START, CHARGE_LENGTH, charge == 1 ? 0 : 1); }

    /// Set Charge Valid
    void setChargeValid(bool valid) { writeDataField( CHVALID_START, CHVALID_LENGTH, valid ? 1 : 0); }

    /// Set Fine / Halo
    void setFineHalo(bool fh) { writeDataField (FINEHALO_START, FINEHALO_LENGTH, fh ? 1 : 0); }


    /// reset
    virtual void reset();

    /// Set data word
    void setDataWord(unsigned dataword) { m_dataWord = dataword;}

    /// print candidate
    virtual void print() const;

  private:
    unsigned readDataField(unsigned start, unsigned count) const; 
    void writeDataField(unsigned start, unsigned count, unsigned value); 

  private:
    int m_bx;
    unsigned m_dataWord;                                   // muon data word (25 bits) :

    float m_phiValue;
    float m_etaValue;
    float m_ptValue;
    static const float m_invalidValue;


  public:
                                                           // definition of the bit fields
    enum { PHI_START=0};       enum { PHI_LENGTH = 8};     // Bits 0:7   phi (8 bits)
    enum { PT_START=8};        enum { PT_LENGTH =  5};     // Bits 8:12  pt  (5 bits)
    enum { QUAL_START=13};     enum { QUAL_LENGTH = 3};    // Bits 13:15 quality (3 bits)
    enum { ETA_START=16};      enum { ETA_LENGTH = 6};     // Bits 16:21 eta (6 bits)
    enum { FINEHALO_START=22}; enum { FINEHALO_LENGTH = 1};// Bit  22 Eta is fine (DT) / Halo (CSC)
    enum { CHARGE_START=23};   enum { CHARGE_LENGTH = 1};  // Bit  23 Charge: 0 = positive
    enum { CHVALID_START=24};  enum { CHVALID_LENGTH = 1}; // Bit  24 Charge is vaild (1=valid)
                                                           // Bits 26 to 29: Synchronization
 
    enum { TYPE_START=30};     enum { TYPE_LENGTH = 2};    // Bit  30/31 type DT, bRPC, CSC, fRPC
                                                           // these bits are not sent to the GMT in hardware
};
#endif
