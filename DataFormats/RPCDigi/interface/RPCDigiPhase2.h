#ifndef RPCDigi_RPCDigiPhase2_h
#define RPCDigi_RPCDigiPhase2_h

/** \class RPCDigiPhase2
 *
 * Digi for Resisitive Plate Chamber, after Phase2 upgrade 
 *  
 * \author Borislav Pavlov - University of Sofia
*/

#include <cstdint>
#include <iosfwd>

class RPCDigiPhase2 {
public:
  explicit RPCDigiPhase2(int strip, int bx, int sbx);
  RPCDigiPhase2();

  bool operator==(const RPCDigiPhase2& digi) const;
  bool operator<(const RPCDigiPhase2& digi) const;
  void print() const;
  int strip() const { return strip_; }
  int bx() const { return bx_; }
  int sbx() const { return sbx_; }

private:
  uint16_t strip_;
  int32_t bx_;  // for BX
  int8_t sbx_;  // for sub-BX
};

std::ostream& operator<<(std::ostream& o, const RPCDigiPhase2& digi);

#endif
