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

#include "DataFormats/SiPixelDetId/interface/StackedTrackerDetId.h"

/// Constructors inherited from parent classes
StackedTrackerDetId::StackedTrackerDetId() : DetId() {}
StackedTrackerDetId::StackedTrackerDetId( uint32_t rawid ) : DetId(rawid) {}
StackedTrackerDetId::StackedTrackerDetId( const DetId& id ) : DetId(id.rawId()) {}

/// Constructor for Barrel
StackedTrackerDetId::StackedTrackerDetId( uint32_t layer, uint32_t rod, uint32_t module )
  : DetId(DetId::Tracker, StackedTracker)
{
  id_ |= (1 & iSwitchMask_)      << iSwitchStartBit_ |
         (layer & iLayerMask_B_) << iLayerStartBit_B_ |
         (rod & iPhiMask_B_ )    << iPhiStartBit_B_ |
         (module & iZMask_B_ )   << iZStartBit_B_ ;
}

/// Constructor for Endcap
StackedTrackerDetId::StackedTrackerDetId( uint32_t side, uint32_t disk, uint32_t ring, uint32_t module )
  : DetId(DetId::Tracker, StackedTracker) 
{
  id_ |= (2 & iSwitchMask_)      << iSwitchStartBit_ |
         (side & iSideMask_E_)   << iSideStartBit_E_ |
         (disk & iZMask_E_ )     << iZStartBit_E_ |
         (ring & iRingMask_E_ )  << iRingStartBit_E_ |
         (module & iPhiMask_E_ ) << iPhiStartBit_E_ ;
}

std::ostream& operator << ( std::ostream& os, const StackedTrackerDetId& id ) {
  return os << id.rawId()
    << "   subdetector: "  <<  id.subdet()  <<  "\n"
    << "   BARREL: " << id.isBarrel()
    << "\tLAY:  " << id.iLayer() << "\tROD:  " << id.iPhi() << "\tMOD:  " << id.iZ() << "\n"
    << "   ENDCAP: " << id.isEndcap()
    << "\tSIDE: " << id.iSide()  << "\tDISK: " << id.iZ()   << "\tRING: " << id.iRing() << "\tMOD: " << id.iPhi() << "\n";
}

