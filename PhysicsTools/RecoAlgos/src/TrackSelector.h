#ifndef RecoAlgos_TrackSelector_h
#define RecoAlgos_TrackSelector_h
/** \class TrackSelector
 *
 * selects a subset of a track collection. Also clones
 * TrackExtra part and RecHits collection
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.5 $
 *
 * $Id: SelectorProducer.h,v 1.5 2006/06/20 09:58:02 llista Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"

namespace reco {
  class Track;
}

class TrackSelector : public edm::EDProducer {
public:
  /// constructor 
  explicit TrackSelector( const edm::ParameterSet & );
  /// destructor
  virtual ~TrackSelector();
  
private:
  /// process one event
  virtual void produce( edm::Event&, const edm::EventSetup& );
  /// select one track
  virtual bool select( const reco::Track& ) const;
  /// source collection label
  std::string src_;
};

#endif
