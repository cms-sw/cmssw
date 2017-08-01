#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMTrack.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMTrackSegPhi.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMTrackSegEta.h"
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

    edm::Wrapper<std::vector<L1MuBMTrackSegPhi> > l1mu_trk_ph_W;
    edm::Wrapper<std::vector<L1MuBMTrackSegEta> > l1mu_trk_th_W;
    edm::Wrapper<std::vector<L1MuBMTrack> >       l1mu_trk_tr_W;
  };
}
