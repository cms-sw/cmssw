#ifndef CommonTools_Utils_EtComparator_h
#define CommonTools_Utils_EtComparator_h
/** \class EtComparator
 *
 * compare by Et
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: EtComparator.h,v 1.2 2007/04/23 20:41:01 llista Exp $
 *
 */

template <typename T>
struct LessByEt {
  typedef T first_argument_type;
  typedef T second_argument_type;
  bool operator()(const T& t1, const T& t2) const { return t1.et() < t2.et(); }
};

template <typename T>
struct GreaterByEt {
  typedef T first_argument_type;
  typedef T second_argument_type;
  bool operator()(const T& t1, const T& t2) const { return t1.et() > t2.et(); }
};

#include <limits>
#include <cmath>

template <class T>
struct NumericSafeLessByEt {
  typedef T first_argument_type;
  typedef T second_argument_type;
  bool operator()(const T& a1, const T& a2) {
    return fabs(a1.et() - a2.et()) > std::numeric_limits<double>::epsilon()   ? a1.et() < a2.et()
           : fabs(a1.px() - a2.px()) > std::numeric_limits<double>::epsilon() ? a1.px() < a2.px()
                                                                              : a1.pz() < a2.pz();
  }
};

template <class T>
struct NumericSafeGreaterByEt {
  typedef T first_argument_type;
  typedef T second_argument_type;
  bool operator()(const T& a1, const T& a2) {
    return fabs(a1.et() - a2.et()) > std::numeric_limits<double>::epsilon()   ? a1.et() > a2.et()
           : fabs(a1.px() - a2.px()) > std::numeric_limits<double>::epsilon() ? a1.px() > a2.px()
                                                                              : a1.pz() > a2.pz();
  }
};

#endif
