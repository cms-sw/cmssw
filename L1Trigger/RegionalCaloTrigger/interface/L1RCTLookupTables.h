#ifndef L1RCTLookupTables_h
#define L1RCTLookupTables_h

class L1RCTParameters;
struct L1RCTChannelMask;
class CaloTPGTranscoder;
class L1CaloEtScale;
class EcalTPGScale;

class L1RCTLookupTables {
 
 public:

  // constructor

  L1RCTLookupTables() : rctParameters_(0), channelMask_(0), transcoder_(0), etScale_(0) {}
  
  // this needs to be refreshed every event -- constructor inits to zero
  // to indicate that it cannot be used -- if this set function is
  // called, lookup after that call will use it.
  void setRCTParameters(const L1RCTParameters* rctParameters)
    {
      rctParameters_ = rctParameters;
    }
  // ditto for channel mask
  void setChannelMask(const L1RCTChannelMask* channelMask)
    {
      channelMask_ = channelMask;
    }
  // ditto for transcoder
  void setTranscoder(const CaloTPGTranscoder* transcoder)
    {
      transcoder_ = transcoder;
    }
  // ditto for caloEtScale
  void setL1CaloEtScale(const L1CaloEtScale* etScale)
    {
      etScale_ = etScale;
    }
  // ditto for ecalTPGScale
  void setEcalTPGScale(EcalTPGScale* ecalScale)
    {
      ecalScale_ = ecalScale;
    }

  const L1RCTParameters* rctParameters() const {return rctParameters_;}
  
  unsigned int lookup(unsigned short ecalInput,
		      unsigned short hcalInput,
		      unsigned short fgbit,
		      unsigned short crtNo,
		      unsigned short crdNo,
		      unsigned short twrNo
		      ) const;

  unsigned int lookup(unsigned short hfInput, 
		      unsigned short crtNo,
		      unsigned short crdNo,
		      unsigned short twrNo
		      ) const;

  unsigned int emRank(unsigned short energy) const;
  unsigned int eGammaETCode(float ecal, float hcal, int iAbsEta) const;
  unsigned int jetMETETCode(float ecal, float hcal, int iAbsEta) const;
  bool hOeFGVetoBit(float ecal, float hcal, bool fgbit) const;
  bool activityBit(float ecal, float hcal) const;

  // need this to be public to determine good ecal tower in phi for
  // ecaldetid in L1RCTLutWriter
  float convertEcal(unsigned short ecal, unsigned short iAbsEta, unsigned short iPhi, short sign) const;
  
 private:

  // helper functions


  float convertHcal(unsigned short hcal, unsigned short iAbsEta) const;
  unsigned long convertToInteger(float et, float lsb, int precision) const;

  const L1RCTParameters* rctParameters_;
  const L1RCTChannelMask* channelMask_;
  const CaloTPGTranscoder* transcoder_;
  const L1CaloEtScale* etScale_;
  EcalTPGScale* ecalScale_;

};
#endif
