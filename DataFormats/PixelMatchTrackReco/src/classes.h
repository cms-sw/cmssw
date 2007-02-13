#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/PixelMatchTrackReco/interface/TrackSeedAssociation.h"
#include "DataFormats/PixelMatchTrackReco/interface/GsfTrackSeedAssociation.h"
#include <vector>

namespace {
  namespace {
    reco::TrackSeedAssociationCollection v5;
    edm::Wrapper<reco::TrackSeedAssociationCollection> c5;
    reco::TrackSeedAssociation vv5;
    reco::TrackSeedAssociationRef r5;
    reco::TrackSeedAssociationRefProd rp5;
    reco::TrackSeedAssociationRefVector rv5;
    
    reco::GsfTrackSeedAssociationCollection v55;
    edm::Wrapper<reco::GsfTrackSeedAssociationCollection> c55;
    reco::GsfTrackSeedAssociation vv55;
    reco::GsfTrackSeedAssociationRef r55;
    reco::GsfTrackSeedAssociationRefProd rp55;
    reco::GsfTrackSeedAssociationRefVector rv55;
    
    edm::helpers::Key< edm::RefProd < std::vector < reco::Track > > > rpt1;
    edm::AssociationMap<edm::OneToValue< std::vector<reco::Track>, double, unsigned int > > am1;
    edm::helpers::Key< edm::RefProd < std::vector < reco::GsfTrack > > > rpt11;
    edm::AssociationMap<edm::OneToValue< std::vector<reco::GsfTrack>, double, unsigned int > > am11;


  }
}
