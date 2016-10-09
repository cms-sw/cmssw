#ifndef CTPPSDigi_HPTDCErrorFlags_h
#define CTPPSDigi_HPTDCErrorFlags_h

#include <string>

/** \class HPTDCErrorFlags
 *
 * HPTDC error flags container
 *  
 * \author Seyed Mohsen Etesami
 * \author Laurent Forthomme
 * July 2016
 */

// namespace HPTDCErrors { 
struct HPTDCErrorFlags {
HPTDCErrorFlags(unsigned short flags=0): error_flags(flags) {;}

  bool ErrorId(unsigned short id) const {
    switch (id) {
    case 0: return InternalFatalChipError();
    case 1: return EventLost();
    case 2: return HitRejectedByEventSizeLimit();
    case 3: return HitErrorGroup3();
    case 4: return HitLostL1OverflowGroup3();
    case 5: return HitLostROFifoOverflowGroup3();
    case 6: return HitErrorGroup2();
    case 7: return HitLostL1OverflowGroup2();
    case 8: return HitLostROFifoOverflowGroup2();
    case 9: return HitErrorGroup1();
    case 10: return HitLostL1OverflowGroup1();
    case 11: return HitLostROFifoOverflowGroup1();
    case 12: return HitErrorGroup0();
    case 13: return HitLostL1OverflowGroup0();
    case 14: return HitLostROFifoOverflowGroup0();
    default: return true;
    }
  }

  bool InternalFatalChipError() const      { return error_flags&0x1; }
  bool EventLost() const                   { return (error_flags>> 1)&0x1; }
  bool HitRejectedByEventSizeLimit() const { return (error_flags>> 2)&0x1; }
  bool HitErrorGroup3() const              { return (error_flags>> 3)&0x1; }
  bool HitLostL1OverflowGroup3() const     { return (error_flags>> 4)&0x1; }
  bool HitLostROFifoOverflowGroup3() const { return (error_flags>> 5)&0x1; }
  bool HitErrorGroup2() const              { return (error_flags>> 6)&0x1; }
  bool HitLostL1OverflowGroup2() const     { return (error_flags>> 7)&0x1; }
  bool HitLostROFifoOverflowGroup2() const { return (error_flags>> 8)&0x1; }
  bool HitErrorGroup1() const              { return (error_flags>> 9)&0x1; }
  bool HitLostL1OverflowGroup1() const     { return (error_flags>>10)&0x1; }
  bool HitLostROFifoOverflowGroup1() const { return (error_flags>>11)&0x1; }
  bool HitErrorGroup0() const              { return (error_flags>>12)&0x1; }
  bool HitLostL1OverflowGroup0() const     { return (error_flags>>13)&0x1; }
  bool HitLostROFifoOverflowGroup0() const { return (error_flags>>14)&0x1; }

  unsigned short error_flags;
  
  
  static std::string HPTDCErrorName(const unsigned short id) {
    std::string error_name;
    switch (id) {
    case 0: error_name = "InternalFatalChipError";
      break;
    case 1: error_name = "EventLost";
      break;
    case 2: error_name = "HitRejectedByEventSizeLimit";
      break;
    case 3: error_name = "HitErrorGroup3";
      break;
    case 4: error_name = "HitLostL1OverflowGroup3";
      break;
    case 5: error_name = "HitLostROFifoOverflowGroup3";
      break;
    case 6: error_name = "HitErrorGroup2";
      break;
    case 7: error_name = "HitLostL1OverflowGroup2";
      break;
    case 8: error_name = "HitLostROFifoOverflowGroup2";
      break;
    case 9: error_name = "HitErrorGroup1";
      break;
    case 10: error_name = "HitLostL1OverflowGroup1";
      break;
    case 11: error_name = "HitLostROFifoOverflowGroup1";
      break;
    case 12: error_name = "HitErrorGroup0";
      break;
    case 13: error_name = "HitLostL1OverflowGroup0";
      break;
    case 14: error_name = "HitLostROFifoOverflowGroup0";
      break;
    default: error_name = "NONE";
    }
    return error_name;
  }
      
};
//     }
//}

#endif

