#ifndef RecoAlgos_SingleTrackSelectorBase_h
#define RecoAlgos_SingleTrackSelectorBase_h
/** \class SingleTrackSelectorBase
 *
 * selects a subset of a track collection. Also clones
 * TrackExtra part and RecHits collection
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 * $Id: TrackSelector.h,v 1.3 2006/07/21 06:20:45 llista Exp $
 *
 */

#include "PhysicsTools/RecoAlgos/interface/TrackSelectorBase.h"

class SingleTrackSelectorBase : public TrackSelectorBase {
public:
  /// constructor 
  explicit SingleTrackSelectorBase( const edm::ParameterSet & );
  /// destructor
  virtual ~SingleTrackSelectorBase();

private:
  /// select a track collection
  virtual void select( const reco::TrackCollection &, std::vector<const reco::Track *> & ) const;
  /// select one track
  virtual bool select( const reco::Track& ) const;
};

#endif
