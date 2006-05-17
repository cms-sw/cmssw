/**
 * \class CSCTrackStub
 * \author L. Gray
 *
 * A transient data class used to wrap a Correlated LCT
 * and give access to its eta and phi coordinates.
 * This is essentially the merging of a CSCDetId and a CorrelatedLCT
 * into one class.
 *
 * \remark Takes the place of both L1MuCSCCorrelatedLCT and L1MuCSCTrackStub
 *        
 */

#include <L1Trigger/CSCCommonTrigger/interface/CSCTransientDataType.h>
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCTrackStub : public CSCTransientDataType
{
 public:
  CSCTrackStub() {}
  CSCTrackStub(const CSCCorrelatedLCTDigi&, const CSCDetId&, const unsigned& phi, const unsigned& eta);
  CSCTrackStub(const CSCTrackStub&);

  /// return the Eta Value of this stub's position.
  double etaValue() const {return (theEta_*theEtaBinning);}
  /// return the Phi Value of this stub's position.
  double phiValue() const {return (thePhi_*thePhiBinning);}

  /// Return the binned eta for this stub.
  unsigned etaPacked() const {return theEta_;}
  /// Return the binned phi for this stub.
  unsigned phiPacked() const {return thePhi_;}

  /// return valid pattern bit
  bool isValid() const {return theDigi_.isValid();}
  /// return the 4 bit Correlated LCT Quality
  int getQuality() const {return theDigi_.getQuality();}
  /// return the key wire group
  int getKeyWG() const {return theDigi_.getKeyWG();}
  /// return the strip
  int getStrip() const {return theDigi_.getStrip();}
  /// return CLCT pattern number
  int getCLCTPattern() const {return theDigi_.getCLCTPattern();}
  /// return pattern 
  int getPattern() const {return theDigi_.getPattern();}
  /// return strip type
  int getStripType() const {return theDigi_.getStripType();}
  /// return bend
  int getBend() const {return theDigi_.getBend();}
  
  /// Get the digi this stub was made from.
  CSCCorrelatedLCTDigi getDigi() const {return theDigi_;}
  CSCDetId getDetId() const {return theDetId_;}

  /// Time / Space identifiers
  /// See CSCTransientDataType.h for more details.
  int station() const;
  int sector() const;
  int subsector() const;
  int cscid() const;
  int BX() const {return theDigi_.getBX();}


  /// Comparision Operators, used for MPC sorting
  bool operator >  (const CSCTrackStub &) const;
  bool operator <  (const CSCTrackStub &) const;
  bool operator >= (const CSCTrackStub &rhs) const { return !(this->operator<(rhs)); }
  bool operator <= (const CSCTrackStub &rhs) const { return !(this->operator>(rhs)); }
  bool operator == (const CSCTrackStub &rhs) const { return ((theDetId_ == rhs.theDetId_) && (theDigi_ == rhs.theDigi_)); }
  bool operator != (const CSCTrackStub &rhs) const { return !(this->operator==(rhs)); }

 private:
  CSCDetId theDetId_;
  CSCCorrelatedLCTDigi theDigi_;

  unsigned thePhi_, theEta_;

  static double theEtaBinning, thePhiBinning;
};
