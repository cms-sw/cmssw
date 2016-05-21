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

#include <boost/cstdint.hpp>
#include <iosfwd>

class RPCDigi{

public:
  explicit RPCDigi (int strip, int bx);
  RPCDigi ();

  bool operator==(const RPCDigi& digi) const;
  bool operator<(const RPCDigi& digi) const;

  int strip() const ;
  int bx() const;
  void print() const;
  double time() const;
  double deltaTime() const;
  double coordinateX() const;
  double deltaX() const;
  double coordinateY() const;
  double deltaY() const;
  void setTime(double);
  void setDeltaTime(double);
  void setX(double);
  void setY(double);
  void setDeltaX(double);
  void setDeltaY(double);
  bool hasTime() const;
  bool hasX() const;
  bool hasY() const;
  void hasTime(bool);
  void hasX(bool);
  void hasY(bool);
  bool isPseudoDigi() const;

private:
  uint16_t strip_;
  int32_t  bx_;
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

std::ostream & operator<<(std::ostream & o, const RPCDigi& digi);

#endif

