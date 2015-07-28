#ifndef __L1Trigger_L1THGCal_HGCalTriggerGeometryBase_h__
#define __L1Trigger_L1THGCal_HGCalTriggerGeometryBase_h__

#include <iostream>
#include <unordered_set>
#include <unordered_map>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/FCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"

/*******
 *
 * class: HGCalTriggerCodecBase
 * author: L.Gray (FNAL)
 * date: 27 July, 2015
 *
 * 
 *
 *******/

class HGCalTriggerFECodecBase { 
 public:  
  
  HGCalTriggerFECodecBase(const edm::ParameterSet& conf) : 
    name_(conf.getParameter<std::string>("CodecName")),
    codec_idx_(static_cast<unsigned char>(conf.getParameter<uint32_t>("CodecIndex")))
    {}
  virtual ~HGCalTriggerFECodecBase() {}

  const std::string& name() const { return name_; } 
  
  const unsigned char getCodecType() const { return codec_idx_; }
  
  template<typename DECODER,typename DATA>
  DATA decode(const DECODER& decoder, const std::vector<bool>& stream) const {
    return decoder(stream);
  }

  template<typename ENCODER,typename DATA>
  std::vector<bool> encode(const ENCODER& decoder, const DATA& data) const {
    return encoder(data);
  }

 private:
  const std::string name_;
  unsigned char codec_idx_; // I hope we never come to having 256 FE codecs :-)
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< HGCalTriggerFECodecBase* (const edm::ParameterSet&) > HGCalTriggerFECodecFactory;

#endif
