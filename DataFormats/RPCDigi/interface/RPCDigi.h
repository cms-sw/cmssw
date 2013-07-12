#ifndef RPCDigi_RPCDigi_h
#define RPCDigi_RPCDigi_h

/** \class RPCDigi
 *
 * Digi for Rsisitive Plate Chamber
 *  
 *  $Date: 2006/05/16 14:36:29 $
 *  $Revision: 1.8 $
 *
 * \author I. Segoni -- CERN & M. Maggi -- INFN Bari
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

private:
  uint16_t strip_;
  int32_t  bx_; 
};

std::ostream & operator<<(std::ostream & o, const RPCDigi& digi);

#endif

