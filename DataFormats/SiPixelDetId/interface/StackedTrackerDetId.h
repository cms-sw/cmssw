/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Andrew W. Rose                       ///
/// 2008                                 ///
///                                      ///
/// Changed by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2012, July                           ///
///                                      ///
/// Added features:                      ///
/// Expanded for Endcap                  ///
/// ////////////////////////////////////////



#ifndef STACKED_TRACKER_ID_H
#define STACKED_TRACKER_ID_H

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"

#include<iostream>

/// DetId class for of staked modules made of PXB+PXB and PXF+PXF
/// NOTE this is complementary to PXB and PXF DetId's

class StackedTrackerDetId : public DetId {
 private:
  static const unsigned int StackedTracker = 7;

 public:
  /// Constructors for compatibility with DetId parent class
  /// Constructor of a null id
  StackedTrackerDetId();
  /// Constructor from a raw value
  StackedTrackerDetId(uint32_t rawid);
  /// Construct from generic DetId
  StackedTrackerDetId(const DetId& id); 

  /// Dedicated constructors
  /// Constructor for Barrel
  StackedTrackerDetId(uint32_t layer, uint32_t rod, uint32_t module);
  /// Constructor for Endcap
  StackedTrackerDetId(uint32_t side, uint32_t disk, uint32_t ring, uint32_t module);


  /// Methods to retrieve positioning index

  /// SubDet ID
  /// This should be equal to StackedTracker = 7
  unsigned int subdet() const
  { return ((id_ >> iSubDetStartBit_) & iSubDetMask_); }

  /// Barrel/Endcap
  bool isBarrel() const {
    unsigned int layoutType = ((id_ >> iSwitchStartBit_) & iSwitchMask_);
    return (layoutType == 1);
  }
  bool isEndcap() const {
    unsigned int layoutType = ((id_ >> iSwitchStartBit_) & iSwitchMask_);
    return (layoutType == 2);
  }

  /// B: Layer
  unsigned int iLayer() const {
    if (this->isBarrel())
      return ((id_ >> iLayerStartBit_B_) & iLayerMask_B_);
    else
      return 999999;
  }

  /// B: Layer redundancy for backward compatibility
  unsigned int layer() const {
    std::cerr << "W A R N I N G! StackedTrackerDetId::layer()" << std::endl;
    std::cerr << "               This method is OBSOLETE, please change it!" << std::endl;
    return this->iLayer();
  }

  /// BE: Phi
  unsigned int iPhi() const {
    if (this->isBarrel())
      return ((id_ >> iPhiStartBit_B_) & iPhiMask_B_);
    else if (this->isEndcap())
      return ((id_ >> iPhiStartBit_E_) & iPhiMask_E_);
    else
      return 999999;
  }

  /// BE: Z
  unsigned int iZ() const {
    if (this->isBarrel())
      return ((id_ >> iZStartBit_B_) & iZMask_B_);
    else if (this->isEndcap())
      return ((id_ >> iZStartBit_E_) & iZMask_E_);
    else
      return 999999;
  }

  /// E: Z Redundancy
  unsigned int iDisk() const {
    if (this->isEndcap())
      return ((id_ >> iZStartBit_E_) & iZMask_E_);
    else
      return 999999;
  }

  /// E: Ring
  unsigned int iRing() const {
    if (this->isEndcap())
      return ((id_ >> iRingStartBit_E_) & iRingMask_E_);
    else
      return 999999;
  }

  /// E: Side
  unsigned int iSide() const {
    if (this->isEndcap())
      return ((id_ >> iSideStartBit_E_) & iSideMask_E_);
    else
      return 999999;
  }

 private:

  // SUBDET AND SWITCH BARREL/ENDCAP
  static const unsigned int iSubDetStartBit_  = 25;
  static const unsigned int iSubDetMask_      = 0x7; // 0-7

  static const unsigned int iSwitchStartBit_  = 23;
  static const unsigned int iSwitchMask_      = 0x3; // 0-3

  // BARREL SCHEME
  // 0001111BB___LLLLPPPPPPPPZZZZZZZZ
  // F   7  3 ___F   FF      FF
  //     25 23   15  7       0
  
  static const unsigned int iZStartBit_B_     = 0;
  static const unsigned int iPhiStartBit_B_   = 8;
  static const unsigned int iLayerStartBit_B_ = 16;

  static const unsigned int iZMask_B_         = 0xFF; // 0-127
  static const unsigned int iPhiMask_B_       = 0xFF; // 0-255
  static const unsigned int iLayerMask_B_     = 0xF;  // 0-15

  // ENDCAP SCHEME
  // 0001111BBSS_____ZZZZRRRRRPPPPPPP
  // F   7  3 3 _____F   1F   7F
  //    25 23 21     12  7    0

  static const unsigned int iPhiStartBit_E_   = 0;
  static const unsigned int iRingStartBit_E_  = 7;
  static const unsigned int iZStartBit_E_     = 12;
  static const unsigned int iSideStartBit_E_  = 21;

  static const unsigned int iPhiMask_E_       = 0x7F; // 0-127
  static const unsigned int iRingMask_E_      = 0x1F; // 0-31
  static const unsigned int iZMask_E_         = 0xF;  // 0-15
  static const unsigned int iSideMask_E_      = 0x3;  // 0-3

};


std::ostream& operator<<(std::ostream& os,const StackedTrackerDetId& id);

#endif

