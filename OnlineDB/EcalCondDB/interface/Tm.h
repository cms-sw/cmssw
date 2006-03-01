// $Id: Tm.h,v 1.4 2005/12/30 22:54:27 egeland Exp $

#ifndef TM_HH
#define TM_HH

#include <stdexcept>
#include <string>
#include <iostream>
#include <time.h>
#include <stdint.h>

// Wrapper class for time.h tm struct
class Tm {
  static const uint64_t NEG_INF_MICROS = 0;
  static const uint64_t PLUS_INF_MICROS = (uint64_t)-1;

 public:
  // Default constructor makes a null Tm
  Tm();

  // Initialized constructor
  Tm(struct tm * initTm);

  Tm(uint64_t micros);

  // Destructor
  virtual ~Tm();

  /*
   *  Return a pointer to the tm data structure.
   */
  struct tm c_tm() const;

  /*
   *  Returns true if this Tm is null
   */
  int isNull() const;

  /*
   *  Sets this Tm to null
   */
  void setNull();


  static Tm plusInfinity()
    {
      return Tm(PLUS_INF_MICROS);
    };


  static Tm negInfinity()
    {
      return Tm(NEG_INF_MICROS);
    };

  /*
   *  String representation of Tm in YYYY-MM-DD hh:mm:ss format
   */
  std::string str() const;

  /*
   *  return number of microseconds since Jan 1 1970
   */
  uint64_t microsTime() const;

  /*
   *  Set self to current time
   */
  void setToCurrentLocalTime();
  void setToCurrentGMTime();

  /*
   *  Set using time_t
   */
  void setToLocalTime( time_t t);
  void setToGMTime( time_t t );

  /*
   *  Set using microseconds
   */
  void setToMicrosTime(uint64_t micros);

  /*
   *  Set to string of format YYYY-MM-DD HH:MM:SS
   */
  void setToString(const std::string s) throw(std::runtime_error);


  void dumpTm();

 private:
  struct tm m_tm;
};

#endif
