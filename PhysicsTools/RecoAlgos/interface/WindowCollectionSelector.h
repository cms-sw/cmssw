#ifndef RecoAlgos_WindowCollectionSelector_h
#define RecoAlgos_WindowCollectionSelector_h
/** \class WindowCollectionSelector
 *
 * selects track pairs wose combination lies in a given window.
 * could be invariant mass, deltaR , deltaPhi, etc.
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: WindowCollectionSelector.h,v 1.2 2006/07/21 16:58:23 llista Exp $
 *
 */

#include "PhysicsTools/RecoAlgos/interface/TrackSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
namespace edm { class Event; }

template<typename C, typename M>
struct WindowCollectionSelector {
  typedef C collection;
  typedef std::vector<const typename C::value_type *> container;
  typedef typename container::const_iterator const_iterator;
  WindowCollectionSelector( const edm::ParameterSet & cfg ) : 
    min_( cfg.template getParameter<double>( "min" ) ),
    max_( cfg.template getParameter<double>( "max" ) ) { }
  const_iterator begin() const { return selected_.begin(); }
  const_iterator end() const { return selected_.end(); }
  void select( const reco::TrackCollection & c, const edm::Event & ) {
    unsigned int s = c.size();
    std::vector<bool> v( s, false );
    for( unsigned int i = 0; i < s; ++ i )
      for( unsigned int j = i + 1; j < s; ++ j ) {
	double m = m_( c[ i ], c[ j ] );
	if ( m >= min_ && m <= max_ ) {
	  v[ i ] = v[ j ] = true;
	}
      }
    selected_.clear();
    for( unsigned int i = 0; i < c.size(); ++i )
      if ( v[ i ] ) selected_.push_back( & c[ i ] );
  }
  
private:
  M m_;
  double min_, max_;
  container selected_;
};

#endif
