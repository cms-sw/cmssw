#ifndef RecoAlgos_PtMinSelector_h
#define RecoAlgos_PtMinSelector_h
/* \class PtMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: PtMinSelector.h,v 1.6 2007/06/18 18:33:54 llista Exp $
 */

struct PtMinSelector {
  PtMinSelector( double ptMin ) : ptMin_( ptMin ) { }
  template<typename T>
  bool operator()( const T & t ) const { return t.pt() >= ptMin_; }

private:
  double ptMin_;
};

#endif
