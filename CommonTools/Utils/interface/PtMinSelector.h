#ifndef RecoAlgos_PtMinSelector_h
#define RecoAlgos_PtMinSelector_h
/* \class PtMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: PtMinSelector.h,v 1.1 2009/02/24 14:40:26 llista Exp $
 */

struct PtMinSelector {
  PtMinSelector( double ptMin ) : ptMin_( ptMin ) { }
  template<typename T>
  bool operator()( const T & t ) const { return t.pt() >= ptMin_; }

private:
  double ptMin_;
};

#endif
