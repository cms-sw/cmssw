#ifndef RecoAlgos_PtMinSelector_h
#define RecoAlgos_PtMinSelector_h
/* \class PtMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: PtMinSelector.h,v 1.4 2006/10/03 11:44:47 llista Exp $
 */

template<typename T>
struct PtMinSelector {
  typedef T value_type;
  PtMinSelector( double ptMin ) : 
    ptMin_( ptMin ) { }
  bool operator()( const value_type & t ) const { return t.pt() >= ptMin_; }
private:
  double ptMin_;
};

#endif
