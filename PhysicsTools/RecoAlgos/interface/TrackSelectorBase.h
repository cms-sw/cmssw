#ifndef RecoAlgos_TrackSelectorBase_h
#define RecoAlgos_TrackSelectorBase_h
/** \class TrackSelectorBase
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

#include "FWCore/Framework/interface/EDFilter.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include <vector>

class TrackSelectorBase : public edm::EDFilter {
public:
  /// constructor 
  explicit TrackSelectorBase( const edm::ParameterSet & );
  /// destructor
  virtual ~TrackSelectorBase();
  
private:
  /// process one event
  virtual bool filter( edm::Event&, const edm::EventSetup& );
  /// select a track collection
  virtual void select( const reco::TrackCollection &, std::vector<const reco::Track *> & ) const = 0;
  /// source collection label
  std::string src_;
  /// filter event
  bool filter_;
};

#endif
