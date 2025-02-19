#ifndef hadronic_include_AlphaT_hh
#define hadronic_include_AlphaT_hh

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>
//#include "special_less.h"

/**
   Usage:

   Functor takes as its first argument: 
   std::vector<const LV*> 
   or 
   std::vector<LV>
   where LV is any class that has the the methods LV::Et() and LV::Pt().
   
   eg:
   double alpha_t = AlphaT()( ev.CommonObjects() );
   
   Optionally, a bool can be passed as an argument which determines
   whether jet Et (default=true) or Pt (false) values are used in the
   calculation.
   
   eg (to use Et):
   double alpha_t = AlphaT()( ev.CommonObjects(), true );
   or (equivalently):
   double alpha_t = AlphaT()( ev.CommonObjects() );
   
   eg (to use Pt):
   double alpha_t = AlphaT()( ev.CommonObjects(), false );
   
   Also, the second argument can be a std::vector<bool> (passed by
   reference), which indicates whether the jet of the corresponding
   index was assigned to the peusdo-jet #1 (ie, true) or #2 (false).
   
   eg (third argument, Et or Pt, is optional and defaults to true):
   std::vector<bool> jet1;
   double alpha_t = AlphaT()( ev.CommonObjects(), jet1, false ); 
   
 */

struct AlphaT {
  
  // -----------------------------------------------------------------------------
  //
  struct fabs_less { 
    bool operator()( const double x, const double y ) const { return fabs(x) < fabs(y); }
  };
  
  // -----------------------------------------------------------------------------
  //
  template<class LorentzV>
  double operator()( const std::vector<LorentzV const *>& p4, 
		     bool use_et = true ) const {  

    if ( p4.size() == 0 ) { return 0; }
    
    std::vector<double> et;  
    std::vector<double> px;  
    std::vector<double> py;  

    transform( p4.begin(), p4.end(), back_inserter(et), ( use_et ? std::mem_fun(&LorentzV::Et) : std::mem_fun(&LorentzV::Pt) ) );
    transform( p4.begin(), p4.end(), back_inserter(px), std::mem_fun(&LorentzV::Px) );
    transform( p4.begin(), p4.end(), back_inserter(py), std::mem_fun(&LorentzV::Py) );
    
    return value( et, px, py );

  }

  // -----------------------------------------------------------------------------
  //
  template<class LorentzV>
  double operator()( const std::vector<LorentzV const *>& p4, 
		     std::vector<bool>& pseudo_jet1,
		     bool use_et = true ) const {  
    
    if ( p4.size() == 0 ) { return 0; }
    
    std::vector<double> et;  
    std::vector<double> px;  
    std::vector<double> py;  
    pseudo_jet1.clear();
    
    transform( p4.begin(), p4.end(), back_inserter(et), std::mem_fun( use_et ? &LorentzV::Et : &LorentzV::Pt ) );
    transform( p4.begin(), p4.end(), back_inserter(px), std::mem_fun(&LorentzV::Px) );
    transform( p4.begin(), p4.end(), back_inserter(py), std::mem_fun(&LorentzV::Py) );

    return value( et, px, py, pseudo_jet1 );

  }
  
  // -----------------------------------------------------------------------------
  //
  template<class LorentzV>
  double operator()( const std::vector<LorentzV>& p4, 
		     bool use_et = true ) const {  
    
    if ( p4.size() == 0 ) { return 0; }
    
    std::vector<double> et;  
    std::vector<double> px;  
    std::vector<double> py;  
    
    transform( p4.begin(), p4.end(), back_inserter(et), std::mem_fun_ref( use_et ? &LorentzV::Et : &LorentzV::Pt ) );
    transform( p4.begin(), p4.end(), back_inserter(px), std::mem_fun_ref(&LorentzV::Px) );
    transform( p4.begin(), p4.end(), back_inserter(py), std::mem_fun_ref(&LorentzV::Py) );
    
    return value( et, px, py );
    
  }
  
  
  // -----------------------------------------------------------------------------
  //
  template<class LorentzV>
  double operator()( const std::vector<LorentzV>& p4, 
		     std::vector<bool>& pseudo_jet1,
		     bool use_et = true ) const {  
    
    if ( p4.size() == 0 ) { return 0; }
    
    std::vector<double> et;  
    std::vector<double> px;  
    std::vector<double> py;  
    
    transform( p4.begin(), p4.end(), back_inserter(et), std::mem_fun_ref( use_et ? &LorentzV::Et : &LorentzV::Pt ) );
    transform( p4.begin(), p4.end(), back_inserter(px), std::mem_fun_ref(&LorentzV::Px) );
    transform( p4.begin(), p4.end(), back_inserter(py), std::mem_fun_ref(&LorentzV::Py) );
    
    return value( et, px, py, pseudo_jet1 );
    
  }

  // -----------------------------------------------------------------------------
  //
  static double value( const std::vector<double>& et,
		       const std::vector<double>& px,
		       const std::vector<double>& py,
		       std::vector<bool>& pseudo_jet1,
		       bool list = true ) {

    // Clear pseudo-jet container
    pseudo_jet1.clear();
    
    // Momentum sums in transverse plane
    const double sum_et = accumulate( et.begin(), et.end(), 0. );
    const double sum_px = accumulate( px.begin(), px.end(), 0. );
    const double sum_py = accumulate( py.begin(), py.end(), 0. );
    
    // Minimum Delta Et for two pseudo-jets
    double min_delta_sum_et = -1.;
    for ( unsigned i=0; i < unsigned(1<<(et.size()-1)); i++ ) { //@@ iterate through different combinations
      double delta_sum_et = 0.;
      std::vector<bool> jet;
      for ( unsigned j=0; j < et.size(); j++ ) { //@@ iterate through jets
	delta_sum_et += et[j] * ( 1 - 2 * (int(i>>j)&1) ); 
	if ( list ) { jet.push_back( (int(i>>j)&1) == 0 ); } 
      }
      if ( ( fabs(delta_sum_et) < min_delta_sum_et || min_delta_sum_et < 0. ) ) {
	min_delta_sum_et = fabs(delta_sum_et);
	if ( list && jet.size() == et.size() ) {
	  pseudo_jet1.resize(jet.size());
	  std::copy( jet.begin(), jet.end(), pseudo_jet1.begin() );
	}
      }
    }
    if ( min_delta_sum_et < 0. ) { return 0.; }
    
    // Alpha_T
    return ( 0.5 * ( sum_et - min_delta_sum_et ) / sqrt( sum_et*sum_et - (sum_px*sum_px+sum_py*sum_py) ) );

  }
  
  // -----------------------------------------------------------------------------
  //
  static double value( const std::vector<double>& et,
		       const std::vector<double>& px,
		       const std::vector<double>& py ) {
    std::vector<bool> pseudo_jet1;
    return value( et, px, py, pseudo_jet1, false );
  }
  
};

#endif // hadronic_include_AlphaT_hh


