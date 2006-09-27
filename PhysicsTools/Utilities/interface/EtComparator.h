#ifndef RecoAlgos_EtComparator_h
#define RecoAlgos_EtComparator_h
/** \class EtComparator
 *
 * compare by Et
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: PtComparator.h,v 1.1 2006/07/25 09:02:56 llista Exp $
 *
 */

template<typename T>
struct EtComparator {
  bool operator()( const T & t1, const T & t2 ) const {
    return t1.et() < t2.et();
  }
};

template<typename T>
struct EtInverseComparator {
  bool operator()( const T & t1, const T & t2 ) const {
    return t1.et() > t2.et();
  }
};

#endif
