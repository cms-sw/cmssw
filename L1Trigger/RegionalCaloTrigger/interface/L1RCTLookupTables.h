#ifndef L1RCTLookupTables_h
#define L1RCTLookupTables_h

class L1RCTParameters;
struct L1RCTChannelMask;
struct L1RCTNoisyChannelMask;
class L1CaloEcalScale;
class L1CaloHcalScale;
class L1CaloEtScale;

class L1RCTLookupTables {
 
 public:

  // constructor

  L1RCTLookupTables() : rctParameters_(0), channelMask_(0), ecalScale_(0), hcalScale_(0), etScale_(0) {}
  
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
  void setNoisyChannelMask(const L1RCTNoisyChannelMask* channelMask)
    {
      noisyChannelMask_ = channelMask;
    }

  // ditto for hcal TPG scale
  void setHcalScale(const L1CaloHcalScale* hcalScale)
    {
      hcalScale_ = hcalScale;
    }
  // ditto for caloEtScale
  void setL1CaloEtScale(const L1CaloEtScale* etScale)
    {
      etScale_ = etScale;
    }
  // ditto for ecal TPG Scale
  void setEcalScale(const L1CaloEcalScale* ecalScale)
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
  bool activityBit(float ecal, float hcal,bool fgbit) const;

 private:

  // helper functions

  float convertEcal(unsigned short ecal, unsigned short iAbsEta, short sign) const;  
  float convertHcal(unsigned short hcal, unsigned short iAbsEta, short sign) const;
  unsigned long convertToInteger(float et, float lsb, int precision) const;

  const L1RCTParameters* rctParameters_;
  const L1RCTChannelMask* channelMask_;
  const L1RCTNoisyChannelMask* noisyChannelMask_;
  const L1CaloEcalScale* ecalScale_;
  const L1CaloHcalScale* hcalScale_;
  const L1CaloEtScale* etScale_;

};
#endif
