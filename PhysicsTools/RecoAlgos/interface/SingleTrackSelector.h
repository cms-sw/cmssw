#ifndef RecoAlgos_SingleTrackSelector_h
#define RecoAlgos_SingleTrackSelector_h
/** \class SingleTrackSelector
 *
 * selects a subset of a track collection based
 * on single track selection done via functor
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: SingleTrackSelector.h,v 1.2 2006/07/21 12:38:50 llista Exp $
 *
 */

#include "PhysicsTools/RecoAlgos/interface/TrackSelector.h"
#include "PhysicsTools/RecoAlgos/interface/SingleElementCollectionSelector.h"

template<typename S>
class SingleTrackSelector : 
  public TrackSelector<SingleElementCollectionSelector<reco::TrackCollection, S> > {
public:
  SingleTrackSelector( const edm::ParameterSet & cfg ) : 
    TrackSelector<SingleElementCollectionSelector<reco::TrackCollection, S> >( cfg ) { }
  ~SingleTrackSelector() { }
};

#endif
