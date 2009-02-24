#ifndef CommonTools_Utils_PtComparator_h
#define CommonTools_Utils_PtComparator_h
/** \class PtComparator
 *
 * compare by pt
 * 
 * \author Luca Lista, INFN
 *         numeric safe implementation by Fedor Ratnikov, FNAL
 *
 * \version $Revision: 1.4 $
 *
 * $Id: PtComparator.h,v 1.4 2007/04/23 20:54:26 llista Exp $
 *
 */

template<typename T>
struct LessByPt {
  typedef T first_argument_type;
  typedef T second_argument_type;
  bool operator()( const T & t1, const T & t2 ) const {
    return t1.pt() < t2.pt();
  }
};

template<typename T>
struct GreaterByPt {
  typedef T first_argument_type;
  typedef T second_argument_type;
  bool operator()( const T & t1, const T & t2 ) const {
    return t1.pt() > t2.pt();
  }
};

#include<limits>
#include <cmath>

template <class T>
struct NumericSafeLessByPt {
  typedef T first_argument_type;
  typedef T second_argument_type;
  bool operator()(const T& a1, const T& a2) {
    return
      fabs (a1.pt()-a2.pt()) > std::numeric_limits<double>::epsilon() ? a1.pt() < a2.pt() :
      fabs (a1.px()-a2.px()) > std::numeric_limits<double>::epsilon() ? a1.px() < a2.px() :
      a1.pz() < a2.pz();
  }
};

template <class T>
struct NumericSafeGreaterByPt {
  typedef T first_argument_type;
  typedef T second_argument_type;
  bool operator()(const T& a1, const T& a2) {
    return
      fabs (a1.pt()-a2.pt()) > std::numeric_limits<double>::epsilon() ? a1.pt() > a2.pt() :
      fabs (a1.px()-a2.px()) > std::numeric_limits<double>::epsilon() ? a1.px() > a2.px() :
      a1.pz() > a2.pz();
  }
};

#endif
