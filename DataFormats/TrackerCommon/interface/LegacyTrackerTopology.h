#ifndef TRACKER_TOPOLOGY_LEGACY_H
#define TRACKER_TOPOLOGY_LEGACY_H

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/DetId/interface/DetId.h"


// WARNING: this header has been introduced to call a TrackerTopology
// object whenever it is not possible to access it through an EventSetup.
// Do not use it if that is the case.


namespace LegacyTrackerTopology {
  
  inline static std::unique_ptr<TrackerTopology> getTrackerTopology()
  {
        TrackerTopology::PixelBarrelValues pxbVals;
        pxbVals.layerStartBit_ =  16;
        pxbVals.ladderStartBit_ = 8;
        pxbVals.moduleStartBit_ = 2;
        pxbVals.layerMask_ =  0xF;
        pxbVals.ladderMask_ = 0xFF;
        pxbVals.moduleMask_ = 0x3F;

        TrackerTopology::PixelEndcapValues pxfVals;
        pxfVals.sideStartBit_ =   23;
        pxfVals.diskStartBit_ =   16;
        pxfVals.bladeStartBit_ =  10;
        pxfVals.panelStartBit_ =  8;
        pxfVals.moduleStartBit_ = 2;
        pxfVals.sideMask_ =   0x3;
        pxfVals.diskMask_ =   0xF;
        pxfVals.bladeMask_ =  0x3F;
        pxfVals.panelMask_ =  0x3;
        pxfVals.moduleMask_ = 0x3F;

        TrackerTopology::TECValues tecVals;
        tecVals.sideStartBit_ =        18;
        tecVals.wheelStartBit_ =       14;
        tecVals.petal_fw_bwStartBit_ = 12;
        tecVals.petalStartBit_ =       8;
        tecVals.ringStartBit_ =        5;
        tecVals.moduleStartBit_ =      2;
        tecVals.sterStartBit_ =        0;
        tecVals.sideMask_ =        0x3;
        tecVals.wheelMask_ =       0xF;
        tecVals.petal_fw_bwMask_ = 0x3;
        tecVals.petalMask_ =       0xF;
        tecVals.ringMask_ =        0x7;
        tecVals.moduleMask_ =      0x7;
        tecVals.sterMask_ =        0x3;

        TrackerTopology::TIBValues tibVals;
        tibVals.layerStartBit_ =       14;
        tibVals.str_fw_bwStartBit_ =   12;
        tibVals.str_int_extStartBit_ = 10;
        tibVals.strStartBit_ =         4;
        tibVals.moduleStartBit_ =      2;
        tibVals.sterStartBit_ =        0;
        tibVals.layerMask_ =       0x7;
        tibVals.str_fw_bwMask_ =   0x3;
        tibVals.str_int_extMask_ = 0x3;
        tibVals.strMask_ =         0x3F;
        tibVals.moduleMask_ =      0x3;
        tibVals.sterMask_ =        0x3;

        TrackerTopology::TIDValues tidVals;
        tidVals.sideStartBit_ =         13;
        tidVals.wheelStartBit_ =        11;
        tidVals.ringStartBit_ =         9;
        tidVals.module_fw_bwStartBit_ = 7;
        tidVals.moduleStartBit_ =       2;
        tidVals.sterStartBit_ =         0;
        tidVals.sideMask_ =         0x3;
        tidVals.wheelMask_ =        0x3;
        tidVals.ringMask_ =         0x3;
        tidVals.module_fw_bwMask_ = 0x3;
        tidVals.moduleMask_ =       0x1F;
        tidVals.sterMask_ =         0x3;
	
        TrackerTopology::TOBValues tobVals;
        tobVals.layerStartBit_ =     14;
        tobVals.rod_fw_bwStartBit_ = 12;
        tobVals.rodStartBit_ =       5;
        tobVals.moduleStartBit_ =    2;
        tobVals.sterStartBit_ =      0;
        tobVals.layerMask_ =     0x7;
        tobVals.rod_fw_bwMask_ = 0x3;
        tobVals.rodMask_ =       0x7F;
        tobVals.moduleMask_ =    0x7;
        tobVals.sterMask_ =      0x3;

        return std::unique_ptr<TrackerTopology>{new TrackerTopology(pxbVals, pxfVals, tecVals, tibVals, tidVals, tobVals)};
  }
};

#endif //
