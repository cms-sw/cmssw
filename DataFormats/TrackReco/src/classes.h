#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/OneToValue.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"
//#include "DataFormats/TrackReco/interface/DeDxHitFwd.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/TrackReco/interface/TrackDeDxEstimate.h"

#include <vector>

namespace {
  namespace {
    reco::TrackExtraCollection v3;
    edm::Wrapper<reco::TrackExtraCollection> c3;
    edm::Ref<reco::TrackExtraCollection> r3;
    edm::RefProd<reco::TrackExtraCollection> rp3;
    edm::RefVector<reco::TrackExtraCollection> rv3;

    reco::TrackCollection v1;
    edm::Wrapper<reco::TrackCollection> c1;
    reco::TrackRef r1;
    reco::TrackRefProd rp1;
    reco::TrackRefVector rv1;
    edm::Wrapper<reco::TrackRefVector> wv1;
    std::vector<reco::TrackRef> vr1;

    edm::helpers::Key<edm::RefProd<std::vector<reco::Track> > > rpt1;
    edm::AssociationMap<edm::OneToValue<std::vector<reco::Track>, double, unsigned int> > am1;
    edm::AssociationMap<edm::OneToValue<std::vector<reco::Track>, float, unsigned int> > am2;
    edm::AssociationMap<edm::OneToValue<std::vector<reco::Track>, int, unsigned int> > am3;
    edm::AssociationMap<edm::OneToValue<std::vector<reco::Track>, bool, unsigned int> > am4;
    edm::Wrapper<edm::AssociationMap<edm::OneToValue<std::vector<reco::Track>, double, unsigned int> > > wam1;
    edm::Wrapper<edm::AssociationMap<edm::OneToValue<std::vector<reco::Track>, float, unsigned int> > > wam2;
    edm::Wrapper<edm::AssociationMap<edm::OneToValue<std::vector<reco::Track>, int, unsigned int> > >  wam3;
    edm::Wrapper<edm::AssociationMap<edm::OneToValue<std::vector<reco::Track>, bool, unsigned int> > > wam4;

    edm::helpers::KeyVal<reco::TrackRef,reco::TrackRef> kvtttmap1;
    reco::TrackToTrackMap tttmap1;
    edm::Wrapper<reco::TrackToTrackMap> wtttmap1;

// DEDX containers
    reco::DeDxHit dedx1;
    //reco::DeDxHitCollection dedx2;
    //reco::DeDxHitRef dedx3;
    //reco::DeDxHitRefProd dedx4;
    //reco::DeDxHitRefVector dedx5;

    reco::TrackDeDxHitsCollection dedx6;
    reco::TrackDeDxHits dedx7;
    reco::TrackDeDxHitsRef dedx8;
    reco::TrackDeDxHitsRefProd dedx9;
    reco::TrackDeDxHitsRefVector dedx10;
   std::vector<std::pair<edm::Ref<std::vector<reco::Track>,reco::Track,edm::refhelper::FindUsingAdvance<std::vector<reco::Track>,reco::Track> >,std::vector<reco::DeDxHit> > > dedx11;
   std::vector<std::vector<reco::DeDxHit> > dedx12;
   
   reco::TrackDeDxEstimateCollection dedxe1;
   reco::TrackDeDxEstimate dedxe2;
   reco::TrackDeDxEstimateRef dedxe3;
   reco::TrackDeDxEstimateRefProd dedxe4;
   reco::TrackDeDxEstimateRefVector dedxe5;

   edm::Wrapper<reco::TrackDeDxHitsCollection> dedxw1; 
   edm::Wrapper<reco::TrackDeDxEstimateCollection> dedxw2; 

    // RefToBase Holders for Tracks
    edm::RefToBase<reco::Track>  rtbt;
    edm::reftobase::IndirectHolder<reco::Track> iht;
//

  }
}
