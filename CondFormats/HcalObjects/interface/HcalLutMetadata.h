#ifndef HcalLutMetadata_h
#define HcalLutMetadata_h

/*
\class HcalLutMetadata
\author Gena Kukartsev 17 Sep 2009
POOL object to store Hcal trigger LUT channel metadata
*/

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalLutMetadatum.h"

class HcalLutMetadata: public HcalCondObjectContainer<HcalLutMetadatum>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalLutMetadata() : HcalCondObjectContainer<HcalLutMetadatum>(nullptr){}
#endif
  HcalLutMetadata(const HcalTopology* topo) : HcalCondObjectContainer<HcalLutMetadatum>(topo){}
    
  std::string myname() const override {return (std::string)"HcalLutMetadata";}
    
  bool  setRctLsb(float rctlsb);
  float getRctLsb() const {return mNonChannelData.mRctLsb;}
  
  bool  setNominalGain(float gain);
  float getNominalGain() const {return mNonChannelData.mNominalGain;}
  
  class NonChannelData{
    friend class HcalLutMetadata;
  public:
    NonChannelData():
      mRctLsb(0.0),
      mNominalGain(0.0){}
      
  protected:
    float mRctLsb;
    float mNominalGain;
  
  COND_SERIALIZABLE;
};

 protected:
  NonChannelData mNonChannelData;

 COND_SERIALIZABLE;
};

#endif
