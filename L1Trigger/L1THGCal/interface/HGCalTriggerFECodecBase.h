#ifndef __L1Trigger_L1THGCal_HGCalTriggerFECodecBase_h__
#define __L1Trigger_L1THGCal_HGCalTriggerFECodecBase_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "DataFormats/L1THGCal/interface/HGCFETriggerDigi.h"

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

/*******
 *
 * class: HGCalTriggerCodecBase
 * author: L.Gray (FNAL)
 * date: 27 July, 2015
 *
 * Base classes for defining HGCal FE codec classes.
 * The base class defines an abstract interface, which is then specialized
 * by the "Codec<>" class to handle a specific data format.
 * To keep the implementation properly factorized, an implementation class
 * is used to provide the appropriate coding and decoding.
 *
 *******/

class HGCalTriggerFECodecBase { 
 public:  
  HGCalTriggerFECodecBase(const edm::ParameterSet& conf) : 
    geometry_(nullptr),
    name_(conf.getParameter<std::string>("CodecName")),
    codec_idx_(static_cast<unsigned char>(conf.getParameter<uint32_t>("CodecIndex")))
    {}
  virtual ~HGCalTriggerFECodecBase() {}

  const std::string& name() const { return name_; } 
  
  const unsigned char getCodecType() const { return codec_idx_; }
  void setGeometry(const HGCalTriggerGeometryBase* const geom) { geometry_ = geom;}
  
  // give the FECodec the input digis and it sets itself
  // with the approprate data
  virtual void setDataPayload(const HGCEEDigiCollection&,
                              const HGCHEDigiCollection&,
                              const HGCHEDigiCollection& ) = 0;
  virtual void setDataPayload(const l1t::HGCFETriggerDigi&) = 0;
  virtual void unSetDataPayload() = 0;
  // get the set data out for your own enjoyment
  virtual std::vector<bool> getDataPayload() const = 0;

  // abstract interface to manipulating l1t::HGCFETriggerDigis
  // these will yell at you if you haven't set the data in the Codec class
  virtual void encode(l1t::HGCFETriggerDigi&) = 0;
  virtual void decode(const l1t::HGCFETriggerDigi&) = 0;
  virtual void print(const l1t::HGCFETriggerDigi& digi,
                     std::ostream& out = std::cout) const = 0;

 protected:
  const HGCalTriggerGeometryBase* geometry_;

 private:
  const std::string name_;
  unsigned char codec_idx_; // I hope we never come to having 256 FE codecs :-)
};

// ----> all codec classes derive from this <----
// inheritance looks like class MyCodec : public HGCalTriggerFE::Codec<MyCodec,MyData>
namespace HGCalTriggerFE {
  template<typename Impl,typename DATA>
  class Codec : public HGCalTriggerFECodecBase { 
  public:
    Codec(const edm::ParameterSet& conf) :  
    HGCalTriggerFECodecBase(conf),
    dataIsSet_(false) {
    }
    
    // mark these as final since at this level we know 
    // the implementation of the codec
    virtual void encode(l1t::HGCFETriggerDigi& digi) override final {
      if( !dataIsSet_ ) {
        edm::LogWarning("HGCalTriggerFECodec|NoDataPayload")
          << "No data payload was set for HGCTriggerFECodec: "
          << this->name();
      }
      digi.encode(static_cast<const Impl&>(*this),data_);      
    }
    virtual void decode(const l1t::HGCFETriggerDigi& digi) override final {
      if( dataIsSet_ ) {
        edm::LogWarning("HGCalTriggerFECodec|OverwritePayload")
          << "Data payload was already set for HGCTriggerFECodec: "
          << this->name() << " overwriting current data!";
      }
      digi.decode(static_cast<const Impl&>(*this),data_);
      dataIsSet_ = true;
    }  
    
    virtual void setDataPayload(const HGCEEDigiCollection& ee, 
                                const HGCHEDigiCollection& fh,
                                const HGCHEDigiCollection& bh ) override final {
      if( dataIsSet_ ) {
        edm::LogWarning("HGCalTriggerFECodec|OverwritePayload")
          << "Data payload was already set for HGCTriggerFECodec: "
          << this->name() << " overwriting current data!";
      }
      static_cast<Impl&>(*this).setDataPayloadImpl(ee,fh,bh);
      dataIsSet_ = true;
    }

    virtual void setDataPayload(const l1t::HGCFETriggerDigi& digi) override final {
      if( dataIsSet_ ) {
        edm::LogWarning("HGCalTriggerFECodec|OverwritePayload")
          << "Data payload was already set for HGCTriggerFECodec: "
          << this->name() << " overwriting current data!";
      }
      static_cast<Impl&>(*this).setDataPayloadImpl(digi);
      dataIsSet_ = true;
    }

    virtual void unSetDataPayload() override final {
      data_.reset();
      dataIsSet_ = false;
    }
    std::vector<bool> getDataPayload() const override final { 
      return this->encode(data_); 
    }
        
    virtual void print(const l1t::HGCFETriggerDigi& digi,
                       std::ostream& out = std::cout) const override final {
      digi.print(static_cast<const Impl&>(*this),out);
    }

    std::vector<bool> encode(const DATA& data) const {
      return static_cast<const Impl&>(*this).encodeImpl(data);
    }

    DATA decode(const std::vector<bool>& data) const {
      return static_cast<const Impl&>(*this).decodeImpl(data);
    }    

  protected:    
    DATA data_;
  private:
    bool dataIsSet_;
  };
}

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< HGCalTriggerFECodecBase* (const edm::ParameterSet&) > HGCalTriggerFECodecFactory;

#endif
