#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "Rtypes.h" 
#include "Math/Cartesian3D.h" 
#include "Math/Polar3D.h" 
#include "Math/CylindricalEta3D.h" 
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h" 
#include "DataFormats/TrackReco/interface/TrackResiduals.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/OneToValue.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"
//#include "DataFormats/TrackReco/interface/DeDxHitFwd.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"


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

    edm::Wrapper<edm::AssociationVector<reco::TrackRefProd,std::vector<double> > > wav1;
    edm::Wrapper<edm::AssociationVector<reco::TrackRefProd,std::vector<float> > > wav2;
    edm::Wrapper<edm::AssociationVector<reco::TrackRefProd,std::vector<int> > > wav3;
    edm::Wrapper<edm::AssociationVector<reco::TrackRefProd,std::vector<bool> > > wav4;

    edm::helpers::KeyVal<reco::TrackRef,reco::TrackRef> kvtttmap1;
    reco::TrackToTrackMap tttmap1;
    edm::Wrapper<reco::TrackToTrackMap> wtttmap1;

    // DEDX containers    
    reco::DeDxDataCollection dedxe1;
    reco::DeDxData dedxe2;
    reco::DeDxDataValueMap dedxe3;
 
    edm::Wrapper<reco::TrackDeDxHitsCollection> dedxw1; 
    edm::Wrapper<reco::DeDxDataCollection> dedxw2; 
    edm::Wrapper<reco::DeDxDataValueMap> dedxw3;



 
    // RefToBase Holders for Tracks
    edm::RefToBase<reco::Track>                         rtb_tk;
    edm::reftobase::IndirectHolder<reco::Track>         ih_tk;
    //edm::reftobase::BaseHolder<reco::Track>             bh_tk;
    edm::reftobase::RefHolder<reco::TrackRef>           rf_tkr;
    edm::reftobase::Holder<reco::Track, reco::TrackRef> h_tk_tkr;
    std::vector< edm::RefToBase<reco::Track> >		rtb_tk_vect;

  }
}
