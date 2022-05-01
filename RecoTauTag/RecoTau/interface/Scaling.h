#ifndef RecoTauTag_RecoTau_Scaling_h
#define RecoTauTag_RecoTau_Scaling_h

namespace Scaling {
    enum class FeatureT{TauFlat, GridGlobal, PfCand_electron, PfCand_muon, PfCand_chHad, PfCand_nHad, PfCand_gamma, Electron, Muon};
    struct ScalingParams
    {
        const std::vector<std::vector<float>> mean_;
        const std::vector<std::vector<float>> std_;
        const std::vector<std::vector<float>> lim_min_;
        const std::vector<std::vector<float>> lim_max_;
    };

    const std::map<FeatureT, ScalingParams> scalingParamsMap_v2p1 = {
        {FeatureT::TauFlat, {
            // mean_
            {{25.0},{510.0},{0.0},{0.5762},{1.967},{0},{0},{0},{14.32},{0},
                {2.213},{11.36},{0},{0},{0},{1.202},{22.17},{0},{0.002281},{2.392},
                {0},{0.00318},{2.991},{3.212e-05},{0},{16.75},{-0.0008515},{-0.0001629},{-0.0007875},{-5.564},
                {0.5},{0.5},{0.007766},{0.5},{1.672},{0},{0.5},{0},{1.5707963267948966},{2.256},
                {0.0},{0},{0.0002029}},
            // std_ 
            {{25.0},{490.0},{2.3},{0.5293},{1.133},{1},{1},{1},{44.8},{1},
                {6.783},{48.09},{1},{1},{1},{3.739},{13.68},{1},{0.009705},{4.187},
                {1},{0.01452},{4.527},{0.4518},{1},{191.7},{0.4016},{0.4041},{1.157},{8.72},
                {0.5},{0.5},{0.01834},{0.5},{5.058},{1},{0.5},{1},{1.5707963267948966},{2.943},
                {1.0},{1},{0.03612}},
            // lim_min_
            {{-1.0},{-1.0},{-1.0},{-5},{-5},{-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity()},{-5},{-std::numeric_limits<float>::infinity()},
                {-5},{-5},{-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity()},{-5},{-5},{-std::numeric_limits<float>::infinity()},{-5},{-5},
                {-std::numeric_limits<float>::infinity()},{-5},{-5},{-5},{-std::numeric_limits<float>::infinity()},{-5},{-5},{-5},{-5},{-5},
                {-1.0},{-1.0},{-5},{-1.0},{-5},{-std::numeric_limits<float>::infinity()},{-1.0},{-std::numeric_limits<float>::infinity()},{-1.0},{-5},
                {-1.0},{-std::numeric_limits<float>::infinity()},{-5}},
            // lim_max_
            {{1.0},{1.0},{1.0},{5},{5},{std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity()},{5},{std::numeric_limits<float>::infinity()},
                {5},{5},{std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity()},{5},{5},{std::numeric_limits<float>::infinity()},{5},{5},
                {std::numeric_limits<float>::infinity()},{5},{5},{5},{std::numeric_limits<float>::infinity()},{5},{5},{5},{5},{5},
                {1.0},{1.0},{5},{1.0},{5},{std::numeric_limits<float>::infinity()},{1.0},{std::numeric_limits<float>::infinity()},{1.0},{5},
                {1.0},{std::numeric_limits<float>::infinity()},{5}},
            } 
        },
    }; // end scalingParamsMap_v2p1

    const std::map<FeatureT, ScalingParams> scalingParamsMap_v2p5 = {
        {FeatureT::TauFlat, {
            // mean_
            {{25.0},{510.0},{0.0},{0.5762},{1.967},{0},{0},{0},{14.32},{0},
                {2.213},{11.36},{0},{0},{0},{1.202},{22.17},{0},{0.002281},{2.392},
                {0},{0.00318},{2.991},{3.212e-05},{0},{16.75},{-0.0008515},{-0.0001629},{-0.0007875},{-5.564},
                {0.5},{0.5},{0.007766},{0.5},{1.672},{0},{0.5},{0},{1.5707963267948966},{2.256},
                {0.0},{0},{0.0002029}},
            // std_ 
            {{25.0},{490.0},{2.3},{0.5293},{1.133},{1},{1},{1},{44.8},{1},
                {6.783},{48.09},{1},{1},{1},{3.739},{13.68},{1},{0.009705},{4.187},
                {1},{0.01452},{4.527},{0.4518},{1},{191.7},{0.4016},{0.4041},{1.157},{8.72},
                {0.5},{0.5},{0.01834},{0.5},{5.058},{1},{0.5},{1},{1.5707963267948966},{2.943},
                {1.0},{1},{0.03612}},
            // lim_min_
            {{-1.0},{-1.0},{-1.0},{-5},{-5},{-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity()},{-5},{-std::numeric_limits<float>::infinity()},
                {-5},{-5},{-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity()},{-5},{-5},{-std::numeric_limits<float>::infinity()},{-5},{-5},
                {-std::numeric_limits<float>::infinity()},{-5},{-5},{-5},{-std::numeric_limits<float>::infinity()},{-5},{-5},{-5},{-5},{-5},
                {-1.0},{-1.0},{-5},{-1.0},{-5},{-std::numeric_limits<float>::infinity()},{-1.0},{-std::numeric_limits<float>::infinity()},{-1.0},{-5},
                {-1.0},{-std::numeric_limits<float>::infinity()},{-5}},
            // lim_max_
            {{1.0},{1.0},{1.0},{5},{5},{std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity()},{5},{std::numeric_limits<float>::infinity()},
                {5},{5},{std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity()},{5},{5},{std::numeric_limits<float>::infinity()},{5},{5},
                {std::numeric_limits<float>::infinity()},{5},{5},{5},{std::numeric_limits<float>::infinity()},{5},{5},{5},{5},{5},
                {1.0},{1.0},{5},{1.0},{5},{std::numeric_limits<float>::infinity()},{1.0},{std::numeric_limits<float>::infinity()},{1.0},{5},
                {1.0},{std::numeric_limits<float>::infinity()},{5}},
            } 
        },
        {FeatureT::GridGlobal, {
            // mean_
            {{25.0,25.0},{510.0,510.0},{0.0,0.0},{0,0}},
            // std_
            {{25.0,25.0},{490.0,490.0},{2.3,2.3},{1,1}},
            // lim_min_
            {{-1.0,-1.0},{-1.0,-1.0},{-1.0,-1.0},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()}},
            // lim_max_
            {{1.0,1.0},{1.0,1.0},{1.0,1.0},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()}},
            }
        },
    }; // end scalingParamsMap_v2p5

}; // end Scaling namespace

#endif