#ifndef RecoAlgos_SortTrackSelector_h
#define RecoAlgos_SortTrackSelector_h
/** \class SortTrackSelector
 *
 * selects a subset of a track collection. Also clones
 * TrackExtra part and RecHits collection
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: SortTrackSelector.h,v 1.1 2006/07/21 10:27:05 llista Exp $
 *
 */

#include "PhysicsTools/RecoAlgos/interface/TrackSelectorBase.h"
#include <algorithm>

template<typename C>
class SortTrackSelector : public TrackSelectorBase {
public:
  /// constructor 
  explicit SortTrackSelector( const edm::ParameterSet & cfg ) :
    TrackSelectorBase( cfg ),
    max_( cfg.template getParameter<unsigned int>( "max" ) ) { }
  /// destructor
  virtual ~SortTrackSelector() { }

private:
  /// select a track collection
  virtual void select( const reco::TrackCollection & coll , std::vector<const reco::Track *> & sel ) const {
    C c; Comparator cmp( c );
    std::vector<const reco::Track *> v;
    for( reco::TrackCollection::const_iterator i = coll.begin(); i != coll.end(); ++ i )
      v.push_back( & * i );
    std::sort( v.begin(), v.end(), cmp );
    for( unsigned int i = 0; i < max_ && i < v.size(); ++i )
      sel.push_back( v[ i ] );
  }
  /// maximum number of tracks to select
  unsigned int max_;
  /// comparator helper struct
  struct Comparator {
    Comparator( const C & c ) : c_( c ) { }
    bool operator()( const reco::Track * t1, const reco::Track * t2 ) const {
      return c_( * t1, * t2 );
    } 
    C c_;
  };
};

#endif
