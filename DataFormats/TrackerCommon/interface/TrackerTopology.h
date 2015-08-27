#ifndef TRACKERTOPOLOGY_H
#define TRACKERTOPOLOGY_H

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include <vector>
#include <string>

//knower of all things tracker geometry
//flexible replacement for PXBDetId and friends
//to implement
// endcap pixel


class TrackerTopology {

 public:

  struct PixelBarrelValues {
    unsigned int layerStartBit_;
    unsigned int ladderStartBit_;
    unsigned int moduleStartBit_;
    unsigned int layerMask_;
    unsigned int ladderMask_;
    unsigned int moduleMask_;
  };

  struct PixelEndcapValues {
    unsigned int sideStartBit_;
    unsigned int diskStartBit_;
    unsigned int bladeStartBit_;
    unsigned int panelStartBit_;
    unsigned int moduleStartBit_;
    unsigned int sideMask_;
    unsigned int diskMask_;
    unsigned int bladeMask_;
    unsigned int panelMask_;
    unsigned int moduleMask_;
  };

  struct TECValues {
    unsigned int sideStartBit_;
    unsigned int wheelStartBit_;
    unsigned int petal_fw_bwStartBit_;
    unsigned int petalStartBit_;
    unsigned int ringStartBit_;
    unsigned int moduleStartBit_;
    unsigned int sterStartBit_;
    unsigned int sideMask_;
    unsigned int wheelMask_;
    unsigned int petal_fw_bwMask_;
    unsigned int petalMask_;
    unsigned int ringMask_;
    unsigned int moduleMask_;
    unsigned int sterMask_;
  };

  struct TIBValues {
    unsigned int layerStartBit_;
    unsigned int str_fw_bwStartBit_;
    unsigned int str_int_extStartBit_;
    unsigned int strStartBit_;
    unsigned int moduleStartBit_;
    unsigned int sterStartBit_;

    unsigned int layerMask_;
    unsigned int str_fw_bwMask_;
    unsigned int str_int_extMask_;
    unsigned int strMask_;
    unsigned int moduleMask_;
    unsigned int sterMask_;
  };

  struct TIDValues {
    unsigned int sideStartBit_;
    unsigned int wheelStartBit_;
    unsigned int ringStartBit_;
    unsigned int module_fw_bwStartBit_;
    unsigned int moduleStartBit_;
    unsigned int sterStartBit_;
    unsigned int sideMask_;
    unsigned int wheelMask_;
    unsigned int ringMask_;
    unsigned int module_fw_bwMask_;
    unsigned int moduleMask_;
    unsigned int sterMask_;
  };

  struct TOBValues {
    unsigned int layerStartBit_;
    unsigned int rod_fw_bwStartBit_;
    unsigned int rodStartBit_;
    unsigned int moduleStartBit_;
    unsigned int sterStartBit_;
    unsigned int layerMask_;
    unsigned int rod_fw_bwMask_;
    unsigned int rodMask_;
    unsigned int moduleMask_;
    unsigned int sterMask_;
  };

  class SameLayerComparator {
  public:
    explicit SameLayerComparator(const TrackerTopology *topo): topo_(topo) {}

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
    const TrackerTopology *topo_;
  };

  
  TrackerTopology( const PixelBarrelValues& pxb, const PixelEndcapValues& pxf,
		   const TECValues& tecv, const TIBValues& tibv, 
		   const TIDValues& tidv, const TOBValues& tobv);

  unsigned int side(const DetId &id) const;
  unsigned int layer(const DetId &id) const;
  unsigned int module(const DetId &id) const;

  // layer numbers
  unsigned int pxbLayer(const DetId &id) const {
    return int((id.rawId()>>pbVals_.layerStartBit_) & pbVals_.layerMask_);
  }
  unsigned int tobLayer(const DetId &id) const {
    return int((id.rawId()>>tobVals_.layerStartBit_) & tobVals_.layerMask_);
  }
  unsigned int tibLayer(const DetId &id) const {
    return int((id.rawId()>>tibVals_.layerStartBit_) & tibVals_.layerMask_);
  }


  //ladder
  unsigned int pxbLadder(const DetId &id) const {
    return ((id.rawId()>>pbVals_.ladderStartBit_) & pbVals_.ladderMask_) ;
  }

  //module
  unsigned int pxbModule(const DetId &id) const {
    return ((id.rawId()>>pbVals_.moduleStartBit_)& pbVals_.moduleMask_);
  }
  unsigned int pxfModule(const DetId &id) const {
    return int((id.rawId()>>pfVals_.moduleStartBit_) & pfVals_.moduleMask_);
  }
  unsigned int tobModule(const DetId &id) const {
    return ((id.rawId()>>tobVals_.moduleStartBit_)& tobVals_.moduleMask_);
  }
  unsigned int tecModule(const DetId &id) const { 
    return ((id.rawId()>>tecVals_.moduleStartBit_) & tecVals_.moduleMask_);
  }
  unsigned int tibModule(const DetId &id) const {
    return ((id.rawId()>>tibVals_.moduleStartBit_)& tibVals_.moduleMask_);
  }
  unsigned int tidModule(const DetId &id) const {
    return ((id.rawId()>>tidVals_.moduleStartBit_)& tidVals_.moduleMask_);
  }


  //side
  unsigned int tobSide(const DetId &id) const {
    return ((id.rawId()>>tobVals_.rod_fw_bwStartBit_) & tobVals_.rod_fw_bwMask_);
  }

  unsigned int tecSide(const DetId &id) const {
    return ((id.rawId()>>tecVals_.sideStartBit_)&tecVals_.sideMask_);
  }

  unsigned int tibSide(const DetId &id) const {
    return ((id.rawId()>>tibVals_.str_fw_bwStartBit_) & tibVals_.str_fw_bwMask_);
  }

  unsigned int tidSide(const DetId &id) const {
    return ((id.rawId()>>tidVals_.sideStartBit_)&tidVals_.sideMask_);
  }

  unsigned int pxfSide(const DetId &id) const {
    return ((id.rawId()>>pfVals_.sideStartBit_)&pfVals_.sideMask_);
  }

  //rod
  unsigned int tobRod(const DetId &id) const {
    return  ((id.rawId()>>tobVals_.rodStartBit_) & tobVals_.rodMask_);
  }

  //wheel
  unsigned int tecWheel(const DetId &id) const { 
    return ((id.rawId()>>tecVals_.wheelStartBit_) & tecVals_.wheelMask_) ;
  }
  unsigned int tidWheel(const DetId &id) const { 
    return ((id.rawId()>>tidVals_.wheelStartBit_) & tidVals_.wheelMask_) ;
  }

  //order
  unsigned int tecOrder(const DetId &id) const { 
    return ((id.rawId()>>tecVals_.petal_fw_bwStartBit_) & tecVals_.petal_fw_bwMask_);
  }
  unsigned int tibOrder(const DetId &id) const { 
    return ((id.rawId()>>tibVals_.str_int_extStartBit_) & tibVals_.str_int_extMask_);
  }
  unsigned int tidOrder(const DetId &id) const { 
    return ((id.rawId()>>tidVals_.module_fw_bwStartBit_) & tidVals_.module_fw_bwMask_);
  }


  /// ring id
  unsigned int tecRing(const DetId &id) const { 
    return ((id.rawId()>>tecVals_.ringStartBit_) & tecVals_.ringMask_) ;
  }
  unsigned int tidRing(const DetId &id) const { 
    return ((id.rawId()>>tidVals_.ringStartBit_) & tidVals_.ringMask_) ;
  }


  //petal
  unsigned int tecPetalNumber(const DetId &id) const
  { return ((id.rawId()>>tecVals_.petalStartBit_) & tecVals_.petalMask_);}


  

  //misc tob
  std::vector<unsigned int> tobRodInfo(const DetId &id) const {
    std::vector<unsigned int> num;
    num.push_back( tobSide(id) );
    num.push_back( tobRod(id) );
    return num ;
  }

  bool tobIsDoubleSide(const DetId &id) const { return SiStripDetId(id).glued()==0 && (tobLayer(id)==1 || tobLayer(id)==2);}
  bool tecIsDoubleSide(const DetId &id) const { return SiStripDetId(id).glued()==0 && (tecRing(id)==1 || tecRing(id)==2 || tecRing(id)==5);}
  bool tibIsDoubleSide(const DetId &id) const { return SiStripDetId(id).glued()==0 && (tibLayer(id)==1 || tibLayer(id)==2);}
  bool tidIsDoubleSide(const DetId &id) const { return SiStripDetId(id).glued()==0 && (tidRing(id)==1 || tidRing(id)==2);}

  bool tobIsZPlusSide(const DetId &id) const {return !tobIsZMinusSide(id);}
  bool tobIsZMinusSide(const DetId &id) const { return tobSide(id)==1;}

  bool tibIsZPlusSide(const DetId &id) const {return !tibIsZMinusSide(id);}
  bool tibIsZMinusSide(const DetId &id) const { return tibSide(id)==1;}

  bool tidIsZPlusSide(const DetId &id) const {return !tidIsZMinusSide(id);}
  bool tidIsZMinusSide(const DetId &id) const { return tidSide(id)==1;}

  bool tecIsZPlusSide(const DetId &id) const {return !tecIsZMinusSide(id);}
  bool tecIsZMinusSide(const DetId &id) const { return tecSide(id)==1;}

  //these are from the old TOB/TEC/TID/TIB DetId
  bool tobIsStereo(const DetId &id) const {return tobStereo(id)!=0 && !tobIsDoubleSide(id);}
  bool tecIsStereo(const DetId &id) const {return tecStereo(id)!=0 && !tecIsDoubleSide(id);}
  bool tibIsStereo(const DetId &id) const {return tibStereo(id)!=0 && !tibIsDoubleSide(id);}
  bool tidIsStereo(const DetId &id) const {return tidStereo(id)!=0 && !tidIsDoubleSide(id);}

  //these are clones of the old SiStripDetId
  uint32_t tobStereo(const DetId &id) const {
    return ( ((id.rawId() >>tobVals_.sterStartBit_ ) & tobVals_.sterMask_ ) == 1 ) ? 1 : 0;
  }

  uint32_t tibStereo(const DetId &id) const {
    return ( ((id.rawId() >>tibVals_.sterStartBit_ ) & tibVals_.sterMask_ ) == 1 ) ? 1 : 0;
  }

  uint32_t tidStereo(const DetId &id) const {
    return ( ((id.rawId() >>tidVals_.sterStartBit_ ) & tidVals_.sterMask_ ) == 1 ) ? 1 : 0;
  }

  uint32_t tecStereo(const DetId &id) const {
    return ( ((id.rawId() >>tecVals_.sterStartBit_ ) & tecVals_.sterMask_ ) == 1 ) ? 1 : 0;
  }

  uint32_t tibGlued(const DetId &id) const {
    uint32_t testId = (id.rawId()>>tibVals_.sterStartBit_) & tibVals_.sterMask_;
    return ( testId == 0 ) ? 0 : (id.rawId() - testId);
  }

  uint32_t tecGlued(const DetId &id) const {
    uint32_t testId = (id.rawId()>>tecVals_.sterStartBit_) & tecVals_.sterMask_;
    return ( testId == 0 ) ? 0 : (id.rawId() - testId);
  }

  uint32_t tobGlued(const DetId &id) const {
    uint32_t testId = (id.rawId()>>tobVals_.sterStartBit_) & tobVals_.sterMask_;
    return ( testId == 0 ) ? 0 : (id.rawId() - testId);
  }

  uint32_t tidGlued(const DetId &id) const {
    uint32_t testId = (id.rawId()>>tidVals_.sterStartBit_) & tidVals_.sterMask_;
    return ( testId == 0 ) ? 0 : (id.rawId() - testId);
  }

  bool tobIsRPhi(const DetId &id) const { return SiStripDetId(id).stereo()==0 && !tobIsDoubleSide(id);}
  bool tecIsRPhi(const DetId &id) const { return SiStripDetId(id).stereo()==0 && !tecIsDoubleSide(id);}
  bool tibIsRPhi(const DetId &id) const { return SiStripDetId(id).stereo()==0 && !tibIsDoubleSide(id);}
  bool tidIsRPhi(const DetId &id) const { return SiStripDetId(id).stereo()==0 && !tidIsDoubleSide(id);}


  //misc tec
  std::vector<unsigned int> tecPetalInfo(const DetId &id) const {
    std::vector<unsigned int> num;
    num.push_back(tecOrder(id));
    num.push_back(tecPetalNumber(id));
    return num ;
  }

  bool tecIsBackPetal(const DetId &id) const {
    return (tecOrder(id)==1);
  }

  bool tecIsFrontPetal(const DetId &id) const {return !tecIsBackPetal(id);}

  //misc tib
  unsigned int tibString(const DetId &id) const {
    return (id.rawId()>>tibVals_.strStartBit_)&tibVals_.strMask_;
  }

  std::vector<unsigned int> tibStringInfo(const DetId &id) const
    { std::vector<unsigned int> num;
      num.push_back( tibSide(id) );
      num.push_back( tibOrder(id) );
      num.push_back(tibString(id));
      return num ;
    }

  bool tibIsInternalString(const DetId &id) const {
    return (tibOrder(id)==1);
  }

  bool tibIsExternalString(const DetId &id) const {
    return !tibIsInternalString(id);
  }

  //misc tid
  std::vector<unsigned int> tidModuleInfo(const DetId &id) const {
    std::vector<unsigned int> num;
    num.push_back( tidOrder(id) );
    num.push_back( tidModule(id) );
    return num ;
  }

  bool tidIsBackRing(const DetId &id) const {
    return (tidOrder(id)==1);
  }

  bool tidIsFrontRing(const DetId &id) const {return !tidIsBackRing(id);}


  //misc pf
  unsigned int pxfDisk(const DetId &id) const {
    return int((id.rawId()>>pfVals_.diskStartBit_) & pfVals_.diskMask_);
  }
  unsigned int pxfBlade(const DetId &id) const {
    return int((id.rawId()>>pfVals_.bladeStartBit_) & pfVals_.bladeMask_);
  }
  unsigned int pxfPanel(const DetId &id) const {
    return int((id.rawId()>>pfVals_.panelStartBit_) & pfVals_.panelMask_);
  }

  //old constructors, now return DetId
  DetId pxbDetId(uint32_t layer,
		 uint32_t ladder,
		 uint32_t module) const {
    //uply
    DetId id(DetId::Tracker,PixelSubdetector::PixelBarrel);
    uint32_t rawid=id.rawId();
    rawid |= (layer& pbVals_.layerMask_) << pbVals_.layerStartBit_     |
      (ladder& pbVals_.ladderMask_) << pbVals_.ladderStartBit_  |
      (module& pbVals_.moduleMask_) << pbVals_.moduleStartBit_;
    return DetId(rawid);
  }

  DetId pxfDetId(uint32_t side,
		 uint32_t disk,
		 uint32_t blade,
		 uint32_t panel,
		 uint32_t module) const {
    DetId id(DetId::Tracker,PixelSubdetector::PixelEndcap);
    uint32_t rawid=id.rawId();
    rawid |= (side& pfVals_.sideMask_)  << pfVals_.sideStartBit_   |
      (disk& pfVals_.diskMask_)        << pfVals_.diskStartBit_      |
      (blade& pfVals_.bladeMask_)      << pfVals_.bladeStartBit_     |
      (panel& pfVals_.panelMask_)      << pfVals_.panelStartBit_     |
      (module& pfVals_.moduleMask_)    << pfVals_.moduleStartBit_  ;
    return DetId(rawid);
  }

  DetId tecDetId(uint32_t side, uint32_t wheel,
		 uint32_t petal_fw_bw, uint32_t petal,
		 uint32_t ring, uint32_t module, uint32_t ster) const {  

    DetId id=SiStripDetId(DetId::Tracker,StripSubdetector::TEC);
    uint32_t rawid=id.rawId();

    rawid |= (side& tecVals_.sideMask_)         << tecVals_.sideStartBit_ |
      (wheel& tecVals_.wheelMask_)             << tecVals_.wheelStartBit_ |
      (petal_fw_bw& tecVals_.petal_fw_bwMask_) << tecVals_.petal_fw_bwStartBit_ |
      (petal& tecVals_.petalMask_)             << tecVals_.petalStartBit_ |
      (ring& tecVals_.ringMask_)               << tecVals_.ringStartBit_ |
      (module& tecVals_.moduleMask_)                 << tecVals_.moduleStartBit_ |
      (ster& tecVals_.sterMask_)               << tecVals_.sterStartBit_ ;
    return DetId(rawid);
  }

  DetId tibDetId(uint32_t layer,
		 uint32_t str_fw_bw,
		 uint32_t str_int_ext,
		 uint32_t str,
		 uint32_t module,
		 uint32_t ster) const {
    DetId id=SiStripDetId(DetId::Tracker,StripSubdetector::TIB);
    uint32_t rawid=id.rawId();
    rawid |= (layer& tibVals_.layerMask_) << tibVals_.layerStartBit_ |
      (str_fw_bw& tibVals_.str_fw_bwMask_) << tibVals_.str_fw_bwStartBit_ |
      (str_int_ext& tibVals_.str_int_extMask_) << tibVals_.str_int_extStartBit_ |
      (str& tibVals_.strMask_) << tibVals_.strStartBit_ |
      (module& tibVals_.moduleMask_) << tibVals_.moduleStartBit_ |
      (ster& tibVals_.sterMask_) << tibVals_.sterStartBit_ ;
    return DetId(rawid);
  }

  DetId tidDetId(uint32_t side,
		 uint32_t wheel,
		 uint32_t ring,
		 uint32_t module_fw_bw,
		 uint32_t module,
		 uint32_t ster) const { 
    DetId id=SiStripDetId(DetId::Tracker,StripSubdetector::TID);
    uint32_t rawid=id.rawId();
    rawid |= (side& tidVals_.sideMask_)      << tidVals_.sideStartBit_    |
      (wheel& tidVals_.wheelMask_)          << tidVals_.wheelStartBit_      |
      (ring& tidVals_.ringMask_)            << tidVals_.ringStartBit_       |
      (module_fw_bw& tidVals_.module_fw_bwMask_)  << tidVals_.module_fw_bwStartBit_  |
      (module& tidVals_.moduleMask_)              << tidVals_.moduleStartBit_        |
      (ster& tidVals_.sterMask_)            << tidVals_.sterStartBit_ ;
    return DetId(rawid);
  }

  DetId tobDetId(uint32_t layer,
		 uint32_t rod_fw_bw,
		 uint32_t rod,
		 uint32_t module,
		 uint32_t ster) const {
    DetId id=SiStripDetId(DetId::Tracker,StripSubdetector::TOB);
    uint32_t rawid=id.rawId();
    rawid |= (layer& tobVals_.layerMask_) << tobVals_.layerStartBit_ |
      (rod_fw_bw& tobVals_.rod_fw_bwMask_) << tobVals_.rod_fw_bwStartBit_ |
      (rod& tobVals_.rodMask_) << tobVals_.rodStartBit_ |
      (module& tobVals_.moduleMask_) << tobVals_.moduleStartBit_ |
      (ster& tobVals_.sterMask_) << tobVals_.sterStartBit_ ;
    return DetId(rawid);
  }

  std::pair<DetId, SameLayerComparator> pxbDetIdLayerComparator(uint32_t layer) const {
    return std::make_pair(pxbDetId(layer, 1,1), SameLayerComparator(this));
  }

  std::pair<DetId, SameLayerComparator> pxfDetIdDiskComparator(uint32_t side, uint32_t disk) const {
    return std::make_pair(pxfDetId(side, disk, 1,1,1), SameLayerComparator(this));
  }

  std::pair<DetId, SameLayerComparator> tecDetIdWheelComparator(uint32_t side, uint32_t wheel) const {
    return std::make_pair(tecDetId(side, wheel, 1,1,1,1,1), SameLayerComparator(this));
  }

  std::pair<DetId, SameLayerComparator> tibDetIdLayerComparator(uint32_t layer) const {
    return std::make_pair(tibDetId(layer, 1,1,1,1,1), SameLayerComparator(this));
  }

  std::pair<DetId, SameLayerComparator> tidDetIdWheelComparator(uint32_t side, uint32_t wheel) const {
    return std::make_pair(tidDetId(side, wheel, 1,1,1,1), SameLayerComparator(this));
  }

  std::pair<DetId, SameLayerComparator> tobDetIdLayerComparator(uint32_t layer) const {
    return std::make_pair(tobDetId(layer, 1,1,1,1), SameLayerComparator(this));
  }

  std::string print(DetId detid) const;

  SiStripDetId::ModuleGeometry moduleGeometry(const DetId &id) const; 

 private:

  PixelBarrelValues pbVals_;
  PixelEndcapValues pfVals_;

  TOBValues tobVals_;
  TIBValues tibVals_;
  TIDValues tidVals_;
  TECValues tecVals_;
  
};

#endif

