#ifndef RecoAlgos_SingleTrackSelector_h
#define RecoAlgos_SingleTrackSelector_h
/** \class SingleTrackSelector
 *
 * selects a subset of a track collection based
 * on single track selection done via functor
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: SingleTrackSelector.h,v 1.1 2006/07/21 10:27:05 llista Exp $
 *
 */

#include "PhysicsTools/RecoAlgos/interface/SingleTrackSelectorBase.h"

template<typename S>
class SingleTrackSelector : public SingleTrackSelectorBase {
public:
  /// constructor 
  explicit SingleTrackSelector( const edm::ParameterSet & cfg ) :
    SingleTrackSelectorBase( cfg ), select_( cfg ) { }
  /// destructor
  virtual ~SingleTrackSelector() { }
  
private:
  /// select one track
  virtual bool select( const reco::Track& t ) const {
    return select_( t );
  }
  /// actual selector object
  S select_;
};

#endif
