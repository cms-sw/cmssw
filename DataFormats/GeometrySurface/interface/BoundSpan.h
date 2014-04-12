#ifndef Geom_BoundSpan_H
#define Geom_BoundSpan_H

/**
 *  compute the span of a bound surface in the global space
 *
 *
 */

#include <utility>
class Surface;

class BoundSpan {
public:
  void compute(Surface const & plane);

  BoundSpan() :
    m_phiSpan( 0., 0.),
    m_zSpan(   0., 0.),
    m_rSpan(   0., 0.){}
  
  std::pair<float,float> const & phiSpan() const { return m_phiSpan; }
  std::pair<float,float> const & zSpan()   const { return m_zSpan; }
  std::pair<float,float> const & rSpan()   const { return m_rSpan; }

private:
  std::pair<float,float> m_phiSpan;
  std::pair<float,float> m_zSpan;
  std::pair<float,float> m_rSpan;

};

#endif
