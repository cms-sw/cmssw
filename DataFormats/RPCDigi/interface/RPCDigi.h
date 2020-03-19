#ifndef RPCDigi_RPCDigi_h
#define RPCDigi_RPCDigi_h

/** \class RPCDigi
 *
 * Digi for Rsisitive Plate Chamber
 *  
 *
 * \author I. Segoni -- CERN & M. Maggi -- INFN Bari
 * 
 * modified by Borislav Pavlov - University of Sofia
 * modification to be used for upgrade and for "pseudodigi"
 * 
*/

#include <cstdint>
#include <iosfwd>

class RPCDigi {
public:
  explicit RPCDigi(int strip, int bx);
  RPCDigi();

  bool operator==(const RPCDigi& digi) const;
  bool operator<(const RPCDigi& digi) const;
  void print() const;
  int strip() const { return strip_; }
  int bx() const { return bx_; }
  double time() const { return time_; }
  double coordinateX() const { return coordinateX_; }
  double coordinateY() const { return coordinateY_; }
  bool hasTime() const { return hasTime_; }
  bool hasX() const { return hasX_; }
  bool hasY() const { return hasY_; }
  void hasTime(bool has) { hasTime_ = has; }
  void hasX(bool has) { hasX_ = has; }
  void hasY(bool has) { hasY_ = has; }
  double deltaTime() const { return deltaTime_; }
  double deltaX() const { return deltaX_; }
  double deltaY() const { return deltaY_; }
  void setTime(double time) { time_ = time; }
  void setDeltaTime(double dt) { deltaTime_ = dt; }
  void setX(double x) { coordinateX_ = x; }
  void setY(double y) { coordinateY_ = y; }
  void setDeltaX(double dx) { deltaX_ = dx; }
  void setDeltaY(double dy) { deltaY_ = dy; }
  bool isPseudoDigi() const { return hasX_ || hasY_; }

private:
  uint16_t strip_;
  int32_t bx_;
  double time_;
  double coordinateX_;
  double coordinateY_;
  double deltaTime_;
  double deltaX_;
  double deltaY_;
  bool hasTime_;
  bool hasX_;
  bool hasY_;
};

std::ostream& operator<<(std::ostream& o, const RPCDigi& digi);

#endif
