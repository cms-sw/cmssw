#ifndef IRPCDigi_IRPCDigi_h
#define IRPCDigi_IRPCDigi_h

/** \class IRPCDigi
 *
 * Digi for Improved Resisitive Plate Chamber (IRPC)
 *  
 * \author Borislav Pavlov - University of Sofia
*/

#include <cstdint>
#include <iosfwd>

class IRPCDigi {
public:
  explicit IRPCDigi(int strip, int bxLR, int bxHR, int sbxLR, int sbxHR, int fineLR, int fineHR);
  IRPCDigi();

  bool operator==(const IRPCDigi& digi) const;
  bool operator<(const IRPCDigi& digi) const;
  void print() const;
  int strip() const { return strip_; }
  int bx() const { return bxLR_; }
  int sbx() const { return sbxLR_; }
  int bxLR() const { return bxLR_; }
  int bxHR() const { return bxHR_; }
  int sbxLR() const { return sbxLR_; }
  int sbxHR() const { return sbxHR_; }
  int fineLR() const { return fineLR_; }
  int fineHR() const { return fineHR_; }

private:
  uint16_t strip_;
  int32_t bxLR_;   //BX from low radius FEB
  int32_t bxHR_;   //BX from high radius FEB
  int8_t sbxLR_;   //sub-BX from low radius FEB
  int8_t sbxHR_;   //sub-BX from high radius FEB
  int8_t fineLR_;  //high resolution time rom low radius FEB
  int8_t fineHR_;  //high resolution time rom high radius FEB
};

std::ostream& operator<<(std::ostream& o, const IRPCDigi& digi);

#endif
