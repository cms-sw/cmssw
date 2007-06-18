#ifndef RecoAlgos_PtMinSelector_h
#define RecoAlgos_PtMinSelector_h
/* \class PtMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: PtMinSelector.h,v 1.5 2007/01/31 14:42:59 llista Exp $
 */

struct PtMinSelector {
  PtMinSelector( double ptMin ) : ptMin_( ptMin ) { }
  template<typename T>
  bool operator()( const T & t ) const { return t.pt() >= ptMin_; }

private:
  double ptMin_;
};

#endif
