#ifndef RECOMTD_TIMINGIDTOOLS_MTDTRACKQUALITYMVA
#define RECOMTD_TIMINGIDTOOLS_MTDTRACKQUALITYMVA

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "CommonTools/MVAUtils/interface/TMVAEvaluator.h"

#define MTDTRACKQUALITYMVA_VARS(VAR) \
  VAR(pt)   \
  VAR(eta)  \
  VAR(phi)   \
  VAR(chi2)  \
  VAR(ndof)  \
  VAR(numberOfValidHits)  \
  VAR(numberOfValidPixelBarrelHits)  \
  VAR(numberOfValidPixelEndcapHits)  \
  VAR(btlMatchChi2)  \
  VAR(btlMatchTimeChi2)  \
  VAR(etlMatchChi2)  \
  VAR(etlMatchTimeChi2)  \
  VAR(mtdt)  \
  VAR(path_len)  \

#define VAR_ENUM(ENUM) ENUM,
#define VAR_STRING(STRING) #STRING,

class MTDTrackQualityMVA {
public:
  //---ctors---
  MTDTrackQualityMVA(std::string weights_file);

  //---dtor---
  ~MTDTrackQualityMVA(){};

  enum varID { 
    MTDTRACKQUALITYMVA_VARS(VAR_ENUM)
  };

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
