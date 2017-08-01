#include "L1Trigger/L1TMuonBarrel/interface/L1BMTrackCollection.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMSecProcId.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMAddressArray.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMTrackSegLoc.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <vector>

namespace L1Trigger_L1TMuonBarrel {
  struct dictionary {
    L1MuBMTrackSegPhi l1mu_trk_ph;
    L1MuBMTrackSegEta l1mu_trk_th;
    L1MuBMTrack       l1mu_trk_tr;
    L1MuBMSecProcId   l1mu_dt_proc;
    L1MuBMTrackSegLoc  l1mu_dt_segloc;
    L1MuBMAddressArray l1mu_dt_addr;

    std::vector<L1MuBMTrackSegPhi> l1mu_trk_ph_V;
    std::vector<L1MuBMTrackSegEta> l1mu_trk_th_V;
    std::vector<L1MuBMTrack>       l1mu_trk_tr_V;
    std::pair<L1MuDTTrack, std::vector<L1MuDTTrackSegPhi> >         l1_trk_tr;
    std::vector<std::pair<L1MuDTTrack, std::vector<L1MuDTTrackSegPhi> > >         l1_trk_tr_V;

    edm::Wrapper<std::vector<L1MuBMTrackSegPhi> > l1mu_trk_ph_W;
    edm::Wrapper<std::vector<L1MuBMTrackSegEta> > l1mu_trk_th_W;
    edm::Wrapper<std::vector<L1MuBMTrack> >       l1mu_trk_tr_W;
    edm::Wrapper<std::vector<std::pair<L1MuDTTrack, std::vector<L1MuDTTrackSegPhi> > > >           l1_trk_tr_W;
    //edm::Wrapper<L1BMTrackCollection>              l1_trk_col_W;

    /* L1MuBMTrackSegPhiCollection l1mu_trk_ph_K; */
    /* L1MuBMTrackSegEtaCollection l1mu_trk_th_K; */
    /* L1MuBMTrackCollection   l1mu_trk_tr_K; */

    /* edm::Wrapper<L1MuBMTrackSegPhiCollection> l1mu_trk_ph_W; */
    /* edm::Wrapper<L1MuBMTrackSegEtaCollection> l1mu_trk_th_W; */
    /* edm::Wrapper<L1MuBMTrackCollection>   l1mu_trk_tr_W; */

  };
}
