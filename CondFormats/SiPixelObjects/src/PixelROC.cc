#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <sstream>
#include <algorithm>
using namespace std;
using namespace sipixelobjects;

// Constructor with transformation frame initilization - NEVER CALLED
PixelROC::PixelROC(uint32_t du, int idDU, int idLk)
  : theDetUnit(du), theIdDU(idDU), theIdLk(idLk) {
  initFrameConversion();
}

// // for testing, uses topology fot det id, it works but I cannot pass topology here
// // not used  
// void PixelROC::initFrameConversion(const TrackerTopology *tt, bool phase1) {
//   const bool TEST = false;
//   if(phase1) { // phase1
//     bool isBarrel = PixelModuleName::isBarrel(theDetUnit);
//     int side = 0;
//     if(isBarrel) {
//       // Barrel Z-index=1,8
//       if((tt->pxbModule(theDetUnit))<5) side=-1;
//       else side=1;
//       if(TEST) {
// 	// phase0 code 
// 	PXBDetId det(theDetUnit);
// 	unsigned  int module = bpixSidePhase0(theDetUnit);
// 	if(!phase1 && (tt->pxbModule(theDetUnit) != module) ) 
// 	// phase1 code
// 	unsigned  int module1 = bpixSidePhase1(theDetUnit);
//       }
//     } else {
//       // Endcaps, use the panel to find the direction 
//       if((tt->pxfPanel(theDetUnit))==1) side=-1; // panel 1
//       else side =1; // panel 2
//       if(TEST) {
// 	// code -phase0
// 	PXFDetId det(theDetUnit);      
// 	unsigned int module = fpixSidePhase0(theDetUnit);
// 	// phase1 code
// 	unsigned int module1 = fpixSidePhase1(theDetUnit);
//       }
//     }
//     theFrameConverter = FrameConversion(isBarrel,side, theIdDU);
//   } else { // phase0
//     initFrameConversion();  // old code for phase0
//   }
// }

// works for phase 1, find det side from the local method
// Frame conversion compatible with CMSSW_9_0_X Monte Carlo samples
void PixelROC::initFrameConversionPhase1_CMSSW_9_0_X() {
  int side = 0;
  bool isBarrel = PixelModuleName::isBarrel(theDetUnit);
  if(isBarrel) {
    side = bpixSidePhase1(theDetUnit); // find the side for phase1
  } else {
    side = fpixSidePhase1(theDetUnit);
  }

  theFrameConverter = FrameConversion(isBarrel,side, theIdDU);

}

// works for phase 1, find det side from the local method
void PixelROC::initFrameConversionPhase1() {
  int side = 0;
  int layer = 0;
  bool isBarrel = PixelModuleName::isBarrel(theDetUnit);
  if(isBarrel) {
    side = bpixSidePhase1(theDetUnit); // find the side for phase1
    layer = bpixLayerPhase1(theDetUnit);
  } else {
    side = fpixSidePhase1(theDetUnit);
  }

  theFrameConverter = FrameConversion(isBarrel, side, layer, theIdDU);

}

// Works only for phase0, uses the fixed pixel id
void PixelROC::initFrameConversion() {

    if ( PixelModuleName::isBarrel(theDetUnit) ) {
      PixelBarrelName barrelName(theDetUnit);
      theFrameConverter = FrameConversion(barrelName, theIdDU);
    } else {
      PixelEndcapName endcapName(theDetUnit);
      theFrameConverter =  FrameConversion(endcapName, theIdDU); 
    }

}

// These are methods to find the module side.
// The are hardwired for phase0 and phase1
// Will not work for phase2 or when the detid coding changes.
int PixelROC::bpixSidePhase0(uint32_t rawId) const {
  int side = 1;
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  //const unsigned int layerStartBit_=   16;
  //const unsigned int ladderStartBit_=   8;
  const unsigned int moduleStartBit_=   2;
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  //const unsigned int layerMask_=       0xF;
  //const unsigned int ladderMask_=      0xFF;
  const unsigned int moduleMask_=      0x3F;

  /// layer id
  //unsigned int layer = (rawId>>layerStartBit_) & layerMask_;
  /// ladder  id
  //unsigned int ladder = (rawId>>ladderStartBit_) & ladderMask_;
  /// det id
  unsigned int module = (rawId>>moduleStartBit_)& moduleMask_;

  if(module<5) side=-1; // modules 1-4 are on -z
  return side;
}
int PixelROC::bpixSidePhase1(uint32_t rawId) const {
  int side = 1;

  /// two bits would be enough, but  we could use the number "0" as a wildcard
  //const unsigned int layerStartBit_=   20;
  //const unsigned int ladderStartBit_=  12;
  const unsigned int moduleStartBit_=   2;
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  //const unsigned int layerMask_=       0xF;
  //const unsigned int ladderMask_=      0xFF;
  const unsigned int moduleMask_=      0x3FF;

  /// layer id
  //unsigned int layer = (rawId>>layerStartBit_) & layerMask_;
  /// ladder  id
  //unsigned int ladder = (rawId>>ladderStartBit_) & ladderMask_;
  /// det id
  unsigned int module = (rawId>>moduleStartBit_)& moduleMask_;

  if(module<5) side=-1; // modules 1-4 are on -z
  return side;
}
int PixelROC::bpixLayerPhase1(uint32_t rawId) {
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  const unsigned int layerStartBit_=   20;
  //const unsigned int ladderStartBit_=  12;
  //const unsigned int moduleStartBit_=   2;
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  const unsigned int layerMask_=       0xF;
  //const unsigned int ladderMask_=      0xFF;
  //const unsigned int moduleMask_=      0x3FF;

  /// layer id
  unsigned int layer = (rawId>>layerStartBit_) & layerMask_;
  /// ladder  id
  //unsigned int ladder = (rawId>>ladderStartBit_) & ladderMask_;
  /// det id
  //unsigned int module = (rawId>>moduleStartBit_)& moduleMask_;

  //if(module<5) side=-1; // modules 1-4 are on -z
  return layer;
}

int PixelROC::fpixSidePhase0(uint32_t rawId) const {
  int side = 1;

  /// two bits would be enough, but  we could use the number "0" as a wildcard
  //const unsigned int sideStartBit_=   23;
  //const unsigned int diskStartBit_=   16;
  //const unsigned int bladeStartBit_=  10;
  const unsigned int panelStartBit_=  8;
  //const unsigned int moduleStartBit_= 2;
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  
  //const unsigned int sideMask_=     0x3;
  //const unsigned int diskMask_=     0xF;
  //const unsigned int bladeMask_=    0x3F;
  const unsigned int panelMask_=    0x3;
  //const unsigned int moduleMask_=   0x3F;

  /// positive or negative id
  //unsigned int sides = int((rawId>>sideStartBit_) & sideMask_);
  /// disk id
  //unsigned int disk = int((rawId>>diskStartBit_) & diskMask_);
  /// blade id
  //unsigned int blade = ((rawId>>bladeStartBit_) & bladeMask_);
  /// panel id
  unsigned int panel = ((rawId>>panelStartBit_) & panelMask_);
  /// det id
  //unsigned int module = ((rawId>>moduleStartBit_) & moduleMask_);

  if(panel==1) side=-1; // panel 1 faces -z (is this true for all disks?)
  return side;
}
int PixelROC::fpixSidePhase1(uint32_t rawId) const {
  int side = 1;

  /// two bits would be enough, but  we could use the number "0" as a wildcard
  //const unsigned int sideStartBit_=   23;
  //const unsigned int diskStartBit_=   18;
  //const unsigned int bladeStartBit_=  12;
  const unsigned int panelStartBit_=  10;
  //const unsigned int moduleStartBit_= 2;
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  
  //const unsigned int sideMask_=     0x3;
  //const unsigned int diskMask_=     0xF;
  //const unsigned int bladeMask_=    0x3F;
  const unsigned int panelMask_=    0x3;
  //const unsigned int moduleMask_=   0xFF;

  /// positive or negative id
  //unsigned int sides = int((rawId>>sideStartBit_) & sideMask_);
  /// disk id
  //unsigned int disk = int((rawId>>diskStartBit_) & diskMask_);
  
  /// blade id
  //unsigned int blade = ((rawId>>bladeStartBit_) & bladeMask_);
  
 /// panel id 1 or 2
  unsigned int panel = ((rawId>>panelStartBit_) & panelMask_);

  /// det id
  //unsigned int module = ((rawId>>moduleStartBit_) & moduleMask_);

  if(panel==1) side=-1; // panel 1 faces -z (is this true for all disks?)
  return side;
}


string PixelROC::print(int depth) const {

  ostringstream out;
  bool barrel = PixelModuleName::isBarrel(theDetUnit);
  DetId detId(theDetUnit);
  if (depth-- >=0 ) {
    out <<"======== PixelROC ";
    //out <<" unit: ";
    //if (barrel) out << PixelBarrelName(detId).name();
    //else        out << PixelEndcapName(detId).name(); 
    if (barrel) out << " barrel ";
    else        out << " endcap "; 
    out <<" ("<<theDetUnit<<")"
        <<" idInDU: "<<theIdDU
        <<" idInLk: "<<theIdLk
//        <<" frame: "<<theRowOffset<<","<<theRowSlopeSign<<","<<theColOffset<<","<<theColSlopeSign
//        <<" frame: "<<*theFrameConverter
        <<endl;
  }
  return out.str();
}

