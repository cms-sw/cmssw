#ifndef COMMONTOOLS_RECOALGOS_TRACKPUIDMVA
#define COMMONTOOLS_RECOALGOS_TRACKPUIDMVA

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "CommonTools/RecoAlgos/interface/MVAComputer.h"

class TrackPUIDMVA {
public:
  //---ctors---
  TrackPUIDMVA(std::string weights_file);

  //---dtor---
  ~TrackPUIDMVA(){};

  //---getters---
  // 4D
  float operator()(const reco::TrackRef& trk,
                   const reco::TrackRef& ext_trk,
                   edm::ValueMap<float>& btl_chi2s,
                   edm::ValueMap<float>& btl_time_chi2s,
                   edm::ValueMap<float>& etl_chi2s,
                   edm::ValueMap<float>& etl_time_chi2s,
                   edm::ValueMap<float>& tmtds,
                   edm::ValueMap<float>& trk_lengths);

private:
  MVAComputer::mva_variables vars_;
  MVAComputer mva_;
};

#endif
