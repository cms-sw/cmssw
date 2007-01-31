#ifndef RecoAlgos_EtaRangeSelector_h
#define RecoAlgos_EtaRangeSelector_h
/* \class EtaRangeSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: EtaRangeSelector.h,v 1.2 2006/10/03 10:34:03 llista Exp $
 */

template<typename T>
struct EtaRangeSelector {
  typedef T value_type;
  EtaRangeSelector( double etaMin, double etaMax ) : 
    etaMin_( etaMin ), etaMax_( etaMax ) { }
  bool operator()( const value_type & t ) const { 
    double eta = t.eta();
    return ( eta >= etaMin_ && eta <= etaMax_ ); 
  }
private:
  double etaMin_, etaMax_;
};

#endif
