#ifndef MTDTOPOLOGY_H
#define MTDTOPOLOGY_H

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/MTDDetId.h"

#include <vector>
#include <string>

//knower of all things tracker geometry
//flexible replacement for PXBDetId and friends
//to implement
// endcap pixel


class MTDTopology {

 public:

  struct BTLValues {
    unsigned int sideStartBit_;
    unsigned int layerStartBit_;
    unsigned int trayStartBit_;
    unsigned int moduleStartBit_;
    unsigned int sideMask_;
    unsigned int layerMask_;
    unsigned int trayMask_;
    unsigned int moduleMask_;
  };

  struct ETLValues {
    unsigned int sideStartBit_;
    unsigned int layerStartBit_;
    unsigned int ringStartBit_;
    unsigned int moduleStartBit_;
    unsigned int sideMask_;
    unsigned int layerMask_;
    unsigned int ringMask_;
    unsigned int moduleMask_;
  };
  
  enum DetIdFields {
    BTLModule, BTLTray, BTLLayer, BTLSide,
    ETLModule, ETLRing, ETLLayer, ETLSide, 
    /* TODO: this can be extended for all subdetectors */
    DETID_FIELDS_MAX
  };

  class SameLayerComparator {
  public:
    explicit SameLayerComparator(const MTDTopology *topo): topo_(topo) {}

    bool operator()(DetId i1, DetId i2) const {
      if(i1.det() == i2.det() &&
         i1.subdetId() == i2.subdetId() &&
         topo_->side(i1) == topo_->side(i2) &&
         topo_->layer(i1) == topo_->layer(i2)) {
        return false;
      }
      return i1 < i2;
    }

    bool operator()(uint32_t i1, uint32_t i2) const {
      return operator()(DetId(i1), DetId(i2));
    }
  private:
    const MTDTopology *topo_;
  };

  
  MTDTopology( const int& topologyMode, const BTLValues& btl, const ETLValues& etl);

  int getMTDTopologyMode() const { return mtdTopologyMode_; }

  unsigned int side(const DetId &id) const;
  unsigned int layer(const DetId &id) const;
  unsigned int module(const DetId &id) const;
  unsigned int tray(const DetId& id) const;
  unsigned int ring(const DetId& id) const;
  
  //module
  unsigned int btlModule(const DetId &id) const {
    return ((id.rawId()>>btlVals_.moduleStartBit_)& btlVals_.moduleMask_);
  }
  unsigned int etlModule(const DetId &id) const {
    return int((id.rawId()>>btlVals_.moduleStartBit_) & btlVals_.moduleMask_);
  }   
  
  //tray
  unsigned int btlTray(const DetId &id) const {
    return ((id.rawId()>>btlVals_.trayStartBit_) & btlVals_.trayMask_) ;
  }

  // ring id
  unsigned int etlRing(const DetId &id) const { 
    return ((id.rawId()>>etlVals_.ringStartBit_) & etlVals_.ringMask_) ;
  }
    
  // layer numbers
  unsigned int btlLayer(const DetId &id) const {
    return int((id.rawId()>>btlVals_.layerStartBit_) & btlVals_.layerMask_);
  }
  unsigned int etlLayer(const DetId &id) const {
    return int((id.rawId()>>etlVals_.layerStartBit_) & etlVals_.layerMask_);
  }   

  //side
  unsigned int btlSide(const DetId &id) const {
    return ((id.rawId()>>btlVals_.sideStartBit_)&btlVals_.sideMask_);
  }

  unsigned int etlSide(const DetId &id) const {
    return ((id.rawId()>>etlVals_.sideStartBit_)&etlVals_.sideMask_);
  }
    
  // which disc is this ring on the forward or backward one?
  unsigned int etlDisc(const DetId &id) const {
    return int((id.rawId()>>etlVals_.ringStartBit_) & etlVals_.ringMask_)%2;
  }
  
  //old constructors, now return DetId
  DetId btlDetId(uint32_t side,
		 uint32_t layer,
		 uint32_t tray,
		 uint32_t module) const {
    //uply
    DetId id(DetId::Forward,ForwardSubdetector::FastTime);
    uint32_t rawid=id.rawId();
    rawid |= MTDDetId::BTL           << MTDDetId::kMTDsubdOffset |
      (side& btlVals_.sideMask_)     << btlVals_.sideStartBit_   |
      (layer& btlVals_.layerMask_)   << btlVals_.layerStartBit_  |
      (tray& btlVals_.trayMask_)     << btlVals_.trayStartBit_    |
      (module& btlVals_.moduleMask_) << btlVals_.moduleStartBit_;
    return DetId(rawid);
  }

  DetId etlDetId(uint32_t side,
		 uint32_t layer,
		 uint32_t ring,
		 uint32_t module) const {
    DetId id(DetId::Forward,ForwardSubdetector::FastTime);
    uint32_t rawid=id.rawId();
    rawid |= MTDDetId::ETL           << MTDDetId::kMTDsubdOffset |
      (side& etlVals_.sideMask_)     << etlVals_.sideStartBit_   |
      (layer& etlVals_.layerMask_)   << etlVals_.layerStartBit_  |
      (ring& etlVals_.ringMask_)     << etlVals_.ringStartBit_   |
      (module& etlVals_.moduleMask_) << etlVals_.moduleStartBit_;
    return DetId(rawid);
  }
  
  std::pair<DetId, SameLayerComparator> btlDetIdLayerComparator(uint32_t side, uint32_t layer) const {
    return std::make_pair(btlDetId(side, layer, 1,1), SameLayerComparator(this));
  }

  std::pair<DetId, SameLayerComparator> etlDetIdDiskComparator(uint32_t side, uint32_t layer) const {
    return std::make_pair(etlDetId(side, layer, 1,1), SameLayerComparator(this));
  }
  
  std::string print(DetId detid) const;
  
  int getMTDLayerNumber(const DetId &id)const;
  
  // Extract the raw bit value for a given field type.
  // E.g. getField(id, PBLadder) == pxbLadder(id)
  unsigned int getField(const DetId &id, DetIdFields idx) const {
    return ((id.rawId()>>bits_per_field[idx].startBit)&bits_per_field[idx].mask);
  }
  // checks whether a given field can be extracted from a given DetId.
  // This boils down to checking whether it is the correct subdetector.
  bool hasField(const DetId &id, DetIdFields idx) const {
    return id.subdetId() == bits_per_field[idx].subdet;
  }
 
 private:

  const int mtdTopologyMode_;

  const BTLValues btlVals_;
  const ETLValues etlVals_;
  
  struct BitmaskAndSubdet { 
    unsigned int startBit; 
    unsigned int mask;
    int subdet;
  };
  const BitmaskAndSubdet bits_per_field[DETID_FIELDS_MAX];
};

#endif

