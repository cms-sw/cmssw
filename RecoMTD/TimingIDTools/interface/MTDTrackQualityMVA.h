#ifndef RECOMTD_TIMINGIDTOOLS_MTDTRACKQUALITYMVA
#define RECOMTD_TIMINGIDTOOLS_MTDTRACKQUALITYMVA

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "CommonTools/MVAUtils/interface/TMVAEvaluator.h"

#define MTDTRACKQUALITYMVA_VARS(MTDBDTVAR) \
  MTDBDTVAR(pt)                            \
  MTDBDTVAR(eta)                           \
  MTDBDTVAR(phi)                           \
  MTDBDTVAR(chi2)                          \
  MTDBDTVAR(ndof)                          \
  MTDBDTVAR(numberOfValidHits)             \
  MTDBDTVAR(numberOfValidPixelBarrelHits)  \
  MTDBDTVAR(numberOfValidPixelEndcapHits)  \
  MTDBDTVAR(btlMatchChi2)                  \
  MTDBDTVAR(btlMatchTimeChi2)              \
  MTDBDTVAR(etlMatchChi2)                  \
  MTDBDTVAR(etlMatchTimeChi2)              \
  MTDBDTVAR(mtdt)                          \
  MTDBDTVAR(path_len)

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
                   const reco::TrackRef& ext_trk,
                   const edm::ValueMap<float>& btl_chi2s,
                   const edm::ValueMap<float>& btl_time_chi2s,
                   const edm::ValueMap<float>& etl_chi2s,
                   const edm::ValueMap<float>& etl_time_chi2s,
                   const edm::ValueMap<float>& tmtds,
                   const edm::ValueMap<float>& trk_lengths) const;

private:
  std::vector<std::string> vars_, spec_vars_;
  std::unique_ptr<TMVAEvaluator> mva_;
};

#endif
