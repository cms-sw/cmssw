#ifndef Geom_TrivialError_H
#define Geom_TrivialError_H

/* struct to disambiguate constructors
 *
 */

struct InvalidError {};
struct ZeroError {};
// diagonal error with all value equal
struct TrivialError {
  double m_value;
  TrivialError(double v) : m_value(v) {}
  double value() const { return m_value; }
  operator double() const { return value(); }
};

#endif
