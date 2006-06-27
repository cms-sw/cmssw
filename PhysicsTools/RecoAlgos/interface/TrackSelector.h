#ifndef RecoAlgos_TrackSelector_h
#define RecoAlgos_TrackSelector_h
/** \class TrackSelector
 *
 * selects a subset of a track collection. Also clones
 * TrackExtra part and RecHits collection
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: TrackSelector.h,v 1.1 2006/06/27 08:37:19 llista Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"

namespace reco {
  class Track;
}

class TrackSelectorBase : public edm::EDProducer {
public:
  /// constructor 
  explicit TrackSelectorBase( const edm::ParameterSet & );
  /// destructor
  virtual ~TrackSelectorBase();
  
private:
  /// process one event
  virtual void produce( edm::Event&, const edm::EventSetup& );
  /// select one track
  virtual bool select( const reco::Track& ) const;
  /// source collection label
  std::string src_;
  /// output branch label
  std::string alias_;
};

template<typename S>
class TrackSelector : public TrackSelectorBase {
public:
  /// constructor 
  explicit TrackSelector( const edm::ParameterSet & cfg ) :
    TrackSelectorBase( cfg ), select_( cfg ) { }
  /// destructor
  virtual ~TrackSelector() { }
  
private:
  /// select one track
  virtual bool select( const reco::Track& t ) const {
    return select_( t );
  }
  /// actual selector object
  S select_;
};

#endif
