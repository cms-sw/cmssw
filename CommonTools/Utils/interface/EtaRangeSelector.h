#ifndef RecoAlgos_EtaRangeSelector_h
#define RecoAlgos_EtaRangeSelector_h
/* \class EtaRangeSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: EtaRangeSelector.h,v 1.1 2009/02/24 14:40:26 llista Exp $
 */

struct EtaRangeSelector {
  EtaRangeSelector( double etaMin, double etaMax ) : 
    etaMin_( etaMin ), etaMax_( etaMax ) { }
  template<typename T>
  bool operator()( const T & t ) const { 
    double eta = t.eta();
    return ( eta >= etaMin_ && eta <= etaMax_ ); 
  }
private:
  double etaMin_, etaMax_;
};

#endif
