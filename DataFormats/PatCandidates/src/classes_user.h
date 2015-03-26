#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/PtrVector.h"

#include "DataFormats/PatCandidates/interface/PATObject.h"

#include "DataFormats/PatCandidates/interface/Vertexing.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace DataFormats_PatCandidates {
  struct dictionaryuser {

  /*   UserData: Core   */
  pat::UserDataCollection	         ov_p_ud;
  /*   UserData: Standalone UserData in the event. Needed?   */
  edm::Wrapper<pat::UserDataCollection>	 w_ov_p_ud;
  edm::Wrapper<edm::ValueMap<edm::Ptr<pat::UserData> > > w_vm_ptr_p_ud;
  edm::Ptr<pat::UserData> yadda_pat_ptr_userdata;
  /*   UserData: a few holders   */
  pat::UserHolder<math::XYZVector>	         p_udh_v3d;
  pat::UserHolder<math::XYZPoint>	         p_udh_p3d;
  pat::UserHolder<math::XYZTLorentzVector>	 p_udh_lv;
  pat::UserHolder<math::PtEtaPhiMLorentzVector>	 p_udh_plv;
  pat::UserHolder<AlgebraicSymMatrix22>          p_udh_smat_22;
  pat::UserHolder<AlgebraicSymMatrix33>          p_udh_smat_33;
  pat::UserHolder<AlgebraicSymMatrix44>          p_udh_smat_44;
  pat::UserHolder<AlgebraicSymMatrix55>          p_udh_smat_55;
  pat::UserHolder<AlgebraicVector2>              p_udh_vec_2;
  pat::UserHolder<AlgebraicVector3>              p_udh_vec_3;
  pat::UserHolder<AlgebraicVector4>              p_udh_vec_4;
  pat::UserHolder<AlgebraicVector5>              p_udh_vec_5;
  pat::UserHolder<reco::Track>                   p_udh_tk;
  pat::UserHolder<reco::Vertex>                  p_udh_vtx;
  pat::UserHolder<std::vector<unsigned int> >    p_udh_vunit;

  };

}
