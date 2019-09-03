// Author: Felice Pantaleo,Marco Rovere - felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 09/2018

#ifndef __RecoHGCal_TICL_PRbyCA_H__
#define __RecoHGCal_TICL_PRbyCA_H__
#include <memory>  // unique_ptr
#include "RecoHGCal/TICL/interface/PatternRecognitionAlgoBase.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

class HGCGraph;

namespace ticl {
  class PatternRecognitionbyCA final : public PatternRecognitionAlgoBase {
  public:
    PatternRecognitionbyCA(const edm::ParameterSet& conf);
    ~PatternRecognitionbyCA() override;

    void makeTracksters(const PatternRecognitionAlgoBase::Inputs& input, std::vector<Trackster>& result) override;

  private:
    hgcal::RecHitTools rhtools_;
    std::unique_ptr<HGCGraph> theGraph_;
    const bool out_in_dfs_ = false;
    const unsigned int max_out_in_hops_ = 99999;
    float min_cos_theta_;
    float min_cos_pointing_;
    int missing_layers_;
    int min_clusters_per_ntuplet_;
    float max_delta_time_;
  };
}  // namespace ticl
#endif
