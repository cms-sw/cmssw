/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef DataFormats_CTPPSDigi_TotemFEDInfo
#define DataFormats_CTPPSDigi_TotemFEDInfo

#include <cstdint>

/**
 * \brief OptoRx headers and footers.
 **/
class TotemFEDInfo {
public:
  TotemFEDInfo(int id = 0) : fedId_(id), header(0), orbitCounter_(0), footer(0) {}

  void setFEDId(int f) { fedId_ = f; }
  int fedId() const { return fedId_; }

  void setHeader(uint64_t _h) { header = _h; }
  uint8_t boe() const { return (header >> 60) & 0xF; }
  uint32_t lv1() const { return (header >> 32) & 0xFFFFFF; }
  uint16_t bx() const { return (header >> 20) & 0xFFF; }
  uint16_t optoRxId() const { return (header >> 8) & 0xFFF; }
  uint8_t fov() const { return (header >> 4) & 0xF; }
  uint8_t h0() const { return (header >> 0) & 0xF; }

  void setOrbitCounter(uint32_t oc) { orbitCounter_ = oc; }
  uint32_t orbitCounter() const { return orbitCounter_; }

  void setFooter(uint64_t _f) { footer = _f; }
  uint8_t eoe() const { return (footer >> 60) & 0xF; }
  uint16_t fSize() const { return (footer >> 32) & 0x3FF; }
  uint8_t f0() const { return (footer >> 0) & 0xF; }

private:
  /// Id from FEDRawDataCollection.
  int fedId_;

  /// Data from OptoRx headers and footer.
  uint64_t header;
  uint32_t orbitCounter_;
  uint64_t footer;
};

#endif
