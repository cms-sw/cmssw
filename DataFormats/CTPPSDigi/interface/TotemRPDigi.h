/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Hubert Niewiadomski
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef DataFormats_CTPPSDigi_TotemRPDigi
#define DataFormats_CTPPSDigi_TotemRPDigi

/**
 * Digi structure for TOTEM RP silicon strip sensors.
**/
class TotemRPDigi {
public:
  TotemRPDigi(unsigned short strip_no = 0) : strip_no_(strip_no){};

  unsigned short stripNumber() const { return strip_no_; }

private:
  /// index of the activated strip
  unsigned short strip_no_;
};

inline bool operator<(const TotemRPDigi& one, const TotemRPDigi& other) {
  return one.stripNumber() < other.stripNumber();
}

#endif
