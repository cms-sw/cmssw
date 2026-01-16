#ifndef RECOMTD_TIMINGIDTOOLS_MTDTRACKQUALITYMVA
#define RECOMTD_TIMINGIDTOOLS_MTDTRACKQUALITYMVA

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "CommonTools/MVAUtils/interface/TMVAEvaluator.h"

#define MTDTRACKQUALITYMVA_VARS(MTDBDTVAR) \
  MTDBDTVAR(Track_pt)                      \
  MTDBDTVAR(Track_eta)                     \
  MTDBDTVAR(Track_phi)                     \
  MTDBDTVAR(Track_dz)                      \
  MTDBDTVAR(Track_dxy)                     \
  MTDBDTVAR(Track_chi2)                    \
  MTDBDTVAR(Track_ndof)                    \
  MTDBDTVAR(Track_npixBarrelValidHits)     \
  MTDBDTVAR(Track_npixEndcapValidHits)     \
  MTDBDTVAR(Track_BTLchi2)                 \
  MTDBDTVAR(Track_BTLtime_chi2)            \
  MTDBDTVAR(Track_ETLchi2)                 \
  MTDBDTVAR(Track_ETLtime_chi2)            \
  MTDBDTVAR(Track_Tmtd)                    \
  MTDBDTVAR(Track_sigmaTmtd)               \
  MTDBDTVAR(Track_length)                  \
  MTDBDTVAR(Track_lHitPos)

#define MTDBDTVAR_ENUM(ENUM) ENUM,
#define MTDBDTVAR_STRING(STRING) #STRING,

class MTDTrackQualityMVA {
public:
  //---ctors---
  MTDTrackQualityMVA(std::string weights_file);

  enum class VarID { MTDTRACKQUALITYMVA_VARS(MTDBDTVAR_ENUM) };

  //---getters---
  // 4D
  float operator()(const reco::TrackRef& trk,
                   const reco::BeamSpot& beamspot,
                   const edm::ValueMap<int>& npixBarrels,
                   const edm::ValueMap<int>& npixEndcaps,
                   const edm::ValueMap<float>& btl_chi2s,
                   const edm::ValueMap<float>& btl_time_chi2s,
                   const edm::ValueMap<float>& etl_chi2s,
                   const edm::ValueMap<float>& etl_time_chi2s,
                   const edm::ValueMap<float>& tmtds,
                   const edm::ValueMap<float>& sigmatmtds,
                   const edm::ValueMap<float>& trk_lengths,
                   const edm::ValueMap<float>& trk_lhitpos) const;

private:
  std::vector<std::string> vars_, spec_vars_;
  std::unique_ptr<TMVAEvaluator> mva_;
};

#endif
