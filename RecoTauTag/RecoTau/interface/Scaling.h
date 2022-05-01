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
            {{21.49},{20.0},{0.0},{0.},{0.6669},
            {1.},{0},{1},{0},{47.78},
            {0},{9.029},{57.59},{0},{0},
            {0},{1.731},{22.38},{-0.0241},{0.0675},
            {0.7973},{0},{0.0018},{2.26},{0},
            {0.0026},{2.928},{0.},{0},{4.717},
            {-0.0003},{-0.0009},{-0.0022},{0.},{0.},
            {0.},{0.0052},{0.},{1.538},{0},
            {0.},{0},{0.},{2.95},{0.0},
            {0},{0.0042}},

            // std_ 
            {{9.713},{980.0},{2.3},{3.141592653589793},{0.6553},
            {4.2},{1},{2},{2},{123.5},
            {1},{26.42},{155.3},{1},{1},
            {1},{6.846},{16.34},{0.0074},{0.0128},
            {3.456},{1},{0.0085},{4.191},{1},
            {0.0114},{4.466},{0.0190},{1},{11.78},
            {0.7362},{0.7354},{1.993},{1},{1.},
            {1},{0.01433},{1.},{4.401},{1},
            {1},{1},{3.141592653589793},{3.927},{1.},
            {1.0},{0.0323}},

            // lim_min_
            {{-5},{0.},{-1.0},{-1.},{-5},
            {0},{-std::numeric_limits<float>::infinity()},{0},{0},{-5},
            {-std::numeric_limits<float>::infinity()},{-5},{-5},{-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity()},
            {-std::numeric_limits<float>::infinity()},{-5},{-5},{-5},{-5},
            {-5},{-std::numeric_limits<float>::infinity()},{-5},{-5},{-std::numeric_limits<float>::infinity()},
            {-5},{-5},{-5},{-std::numeric_limits<float>::infinity()},{-5},
            {-5},{-5},{-5},{-std::numeric_limits<float>::infinity()},{0},
            {0},{-5},{0},{-5},{-std::numeric_limits<float>::infinity()},
            {0},{-std::numeric_limits<float>::infinity()},{0},{-5},{-1.0},
            {-std::numeric_limits<float>::infinity()},{-5}},

            // lim_max_
            {{5},{1.},{1.0},{1.},{5},
            {1},{std::numeric_limits<float>::infinity()},{1},{1},{5},
            {std::numeric_limits<float>::infinity()},{5},{5},{std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity()},
            {std::numeric_limits<float>::infinity()},{5},{5},{5},{5},
            {5},{std::numeric_limits<float>::infinity()},{5},{5},{std::numeric_limits<float>::infinity()},
            {5},{5},{5},{std::numeric_limits<float>::infinity()},{5},
            {5},{5},{5},{std::numeric_limits<float>::infinity()},{1},
            {1},{5},{1},{5},{std::numeric_limits<float>::infinity()},
            {1},{std::numeric_limits<float>::infinity()},{1},{5},{-1.0},
            {std::numeric_limits<float>::infinity()},{5}},
            } 
        }, // end TauFlat

        {FeatureT::GridGlobal, {
            // mean_
            {{21.49},{20.0},{0.0},{0}},
            // std_
            {{9.713},{980.0},{2.3},{1}},
            // lim_min_
            {{-5},{0.},{-1.0},{-std::numeric_limits<float>::infinity()}},
            // lim_max_
            {{5},{1.},{1.0},{std::numeric_limits<float>::infinity()}},
            }
        }, // end GridGlobal

    }; // end scalingParamsMap_v2p1

    const std::map<FeatureT, ScalingParams> scalingParamsMap_v2p5 = {
        {FeatureT::TauFlat, {
            // mean_
            {{25.0},{510.0},{0.0},{0.5762},{1.967},
            {0},{0},{0},{14.32},{0},
            {2.213},{11.36},{0},{0},{0},
            {1.202},{22.17},{0},{0.002281},{2.392},
            {0},{0.00318},{2.991},{3.212e-05},{0},
            {16.75},{-0.0008515},{-0.0001629},{-0.0007875},{-5.564},
            {0.5},{0.5},{0.007766},{0.5},{1.672},
            {0},{0.5},{0},{1.5707963267948966},{2.256},
            {0.0},{0},{0.0002029}},

            // std_ 
            {{25.0},{490.0},{2.3},{0.5293},{1.133},
            {1},{1},{1},{44.8},{1},
            {6.783},{48.09},{1},{1},{1},
            {3.739},{13.68},{1},{0.009705},{4.187},
            {1},{0.01452},{4.527},{0.4518},{1},
            {191.7},{0.4016},{0.4041},{1.157},{8.72},
            {0.5},{0.5},{0.01834},{0.5},{5.058},
            {1},{0.5},{1},{1.5707963267948966},{2.943},
            {1.0},{1},{0.03612}},

            // lim_min_
            {{-1.0},{-1.0},{-1.0},{-5},{-5},
            {-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity()},{-5},{-std::numeric_limits<float>::infinity()},
            {-5},{-5},{-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity()},
            {-5},{-5},{-std::numeric_limits<float>::infinity()},{-5},{-5},
            {-std::numeric_limits<float>::infinity()},{-5},{-5},{-5},{-std::numeric_limits<float>::infinity()},
            {-5},{-5},{-5},{-5},{-5},
            {-1.0},{-1.0},{-5},{-1.0},{-5},
            {-std::numeric_limits<float>::infinity()},{-1.0},{-std::numeric_limits<float>::infinity()},{-1.0},{-5},
            {-1.0},{-std::numeric_limits<float>::infinity()},{-5}},

            // lim_max_
            {{1.0},{1.0},{1.0},{5},{5},
            {std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity()},{5},{std::numeric_limits<float>::infinity()},
            {5},{5},{std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity()},
            {5},{5},{std::numeric_limits<float>::infinity()},{5},{5},
            {std::numeric_limits<float>::infinity()},{5},{5},{5},{std::numeric_limits<float>::infinity()},
            {5},{5},{5},{5},{5},
            {1.0},{1.0},{5},{1.0},{5},
            {std::numeric_limits<float>::infinity()},{1.0},{std::numeric_limits<float>::infinity()},{1.0},{5},
            {1.0},{std::numeric_limits<float>::infinity()},{5}},
            } 
        }, // end TauFlat

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
        }, // end GridGlobal
    }; // end scalingParamsMap_v2p5

}; // end Scaling namespace

#endif