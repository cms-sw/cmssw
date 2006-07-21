#ifndef RecoAlgos_SingleTrackSelectorBase_h
#define RecoAlgos_SingleTrackSelectorBase_h
/** \class SingleTrackSelectorBase
 *
 * selects a subset of a track collection based 
 * on single track selection
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: SingleTrackSelectorBase.h,v 1.1 2006/07/21 10:27:05 llista Exp $
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
