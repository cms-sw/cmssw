#ifndef DataFormats_CTPPSDigi_HPTDCErrorFlags_h
#define DataFormats_CTPPSDigi_HPTDCErrorFlags_h

#include <string>

/** \class HPTDCErrorFlags
 *
 * HPTDC error flags container
 *
 * \author Seyed Mohsen Etesami
 * \author Laurent Forthomme
 * July 2016
 */

class HPTDCErrorFlags {
public:
  HPTDCErrorFlags(unsigned short flags = 0) : error_flags(flags) {}

  bool errorId(unsigned short id) const {
    switch (id) {
      case 0:
        return internalFatalChipError();
      case 1:
        return eventLost();
      case 2:
        return hitRejectedByEventSizeLimit();
      case 3:
        return hitErrorGroup3();
      case 4:
        return hitLostL1OverflowGroup3();
      case 5:
        return hitLostROFifoOverflowGroup3();
      case 6:
        return hitErrorGroup2();
      case 7:
        return hitLostL1OverflowGroup2();
      case 8:
        return hitLostROFifoOverflowGroup2();
      case 9:
        return hitErrorGroup1();
      case 10:
        return hitLostL1OverflowGroup1();
      case 11:
        return hitLostROFifoOverflowGroup1();
      case 12:
        return hitErrorGroup0();
      case 13:
        return hitLostL1OverflowGroup0();
      case 14:
        return hitLostROFifoOverflowGroup0();
      default:
        return true;
    }
  }

  bool internalFatalChipError() const { return error_flags & 0x1; }
  bool eventLost() const { return (error_flags >> 1) & 0x1; }
  bool hitRejectedByEventSizeLimit() const { return (error_flags >> 2) & 0x1; }
  bool hitErrorGroup3() const { return (error_flags >> 3) & 0x1; }
  bool hitLostL1OverflowGroup3() const { return (error_flags >> 4) & 0x1; }
  bool hitLostROFifoOverflowGroup3() const { return (error_flags >> 5) & 0x1; }
  bool hitErrorGroup2() const { return (error_flags >> 6) & 0x1; }
  bool hitLostL1OverflowGroup2() const { return (error_flags >> 7) & 0x1; }
  bool hitLostROFifoOverflowGroup2() const { return (error_flags >> 8) & 0x1; }
  bool hitErrorGroup1() const { return (error_flags >> 9) & 0x1; }
  bool hitLostL1OverflowGroup1() const { return (error_flags >> 10) & 0x1; }
  bool hitLostROFifoOverflowGroup1() const { return (error_flags >> 11) & 0x1; }
  bool hitErrorGroup0() const { return (error_flags >> 12) & 0x1; }
  bool hitLostL1OverflowGroup0() const { return (error_flags >> 13) & 0x1; }
  bool hitLostROFifoOverflowGroup0() const { return (error_flags >> 14) & 0x1; }

  inline unsigned short errorFlag() const { return error_flags; }

  static std::string hptdcErrorName(const unsigned short id) {
    switch (id) {
      case 0:
        return "InternalFatalChipError";
      case 1:
        return "EventLost";
      case 2:
        return "HitRejectedByEventSizeLimit";
      case 3:
        return "HitErrorGroup3";
      case 4:
        return "HitLostL1OverflowGroup3";
      case 5:
        return "HitLostROFifoOverflowGroup3";
      case 6:
        return "HitErrorGroup2";
      case 7:
        return "HitLostL1OverflowGroup2";
      case 8:
        return "HitLostROFifoOverflowGroup2";
      case 9:
        return "HitErrorGroup1";
      case 10:
        return "HitLostL1OverflowGroup1";
      case 11:
        return "HitLostROFifoOverflowGroup1";
      case 12:
        return "HitErrorGroup0";
      case 13:
        return "HitLostL1OverflowGroup0";
      case 14:
        return "HitLostROFifoOverflowGroup0";
      default:
        return "NONE";
    }
  }

private:
  unsigned short error_flags;
};

#endif
