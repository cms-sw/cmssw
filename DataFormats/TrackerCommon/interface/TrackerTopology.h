#ifndef TRACKERTOPOLOGY_H
#define TRACKERTOPOLOGY_H

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include <vector>

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


  
  TrackerTopology( const PixelBarrelValues pxb, const PixelEndcapValues pxf,
		   const TECValues tecv, const TIBValues tibv, 
		   const TIDValues tidv, const TOBValues tobv);

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

  bool tobIsStereo(const DetId &id) const {return SiStripDetId(id).stereo()!=0 && !tobIsDoubleSide(id);}
  bool tecIsStereo(const DetId &id) const {return SiStripDetId(id).stereo()!=0 && !tecIsDoubleSide(id);}
  bool tibIsStereo(const DetId &id) const {return SiStripDetId(id).stereo()!=0 && !tibIsDoubleSide(id);}
  bool tidIsStereo(const DetId &id) const {return SiStripDetId(id).stereo()!=0 && !tidIsDoubleSide(id);}

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
  unsigned int tibStringNumber(const DetId &id) const {
    return (id.rawId()>>tibVals_.strStartBit_)&tibVals_.strMask_;
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

 private:

  PixelBarrelValues pbVals_;
  PixelEndcapValues pfVals_;

  TOBValues tobVals_;
  TIBValues tibVals_;
  TIDValues tidVals_;
  TECValues tecVals_;
  
};

#endif

