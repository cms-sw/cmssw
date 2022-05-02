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
            {{21.49,21.49},{20.0,20.0},{0.0,0.0},{0,0}},
            // std_
            {{9.713,9.713},{980.0,980.0},{2.3,2.3},{1,1}},
            // lim_min_
            {{-5,-5},{0.,0.},{-1.0,-1.0},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()}},
            // lim_max_
            {{5,5},{1.,1.},{1.0,1.0},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()}},
            }
        }, // end GridGlobal

        {FeatureT::PfCand_electron, {
            // mean_
            {{0,0},{0.304,0.9792},{0.0,0.0},{0.0,0.0},{0,0},
             {0,0},{0,0},{0,0},{0,0},{0,0},
             {0,0},{0.001,0.001},{0,0},{0.0003,0.0003},{0,0},
             {0,0},{0.,0.},{1.634,1.634},{0.001,0.001},{24.56,24.56},
             {2.272,2.272},{15.18,15.18}},

            // std_
            {{1,1},{1.845,0.5383},{0.5,0.1},{0.5,0.1},{7.,7.},
             {1,1},{1,1},{1,1},{10.0,10.0},{0.1221,0.1221},
             {0.1226,0.1226},{1.024,1.024},{0.3411,0.3411},{0.3385,0.3385},{1.307,1.307},
             {1,1},{0.171,0.171},{6.45,6.45},{1.02,1.02},{210.4,210.4},
             {8.439,8.439},{3.203,3.203}},

            // lim_min_
            {{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-5,-5},{-1.0,-1.0},{-1.0,-1.0},{0.,0.},
            {-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{0.,0.},{-5,-5},
            {-5,-5},{-5,-5},{-5,-5},{-5,-5},{-5,-5},
            {-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-5,-5},{-5,-5},{-5,-5},{-5,-5},
            {-5,-5},{-5,-5}},

            // lim_max_
            {{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{5,5},{1.0,1.0},{1.0,1.0},{1.,1.},
            {std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{1.,1.},{5,5},
            {5,5},{5,5},{5,5},{5,5},{5,5},
            {std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{5,5},{5,5},{5,5},{5,5},
            {5,5},{5,5}}
            }
        }, // end PfCand_electron

        {FeatureT::PfCand_gamma, {
            // mean
            {{0,0},{0.02576,0.6048},{0.0,0.0},{0.0,0.0},{0,0},
             {0,0},{0,0},{0,0},{0,0},{0.,0.},
             {0.,0.},{0.,0.},{0.,0.},{0.001,0.001},{0.0008,0.0008},
             {0.0038,0.0038},{0,0},{0.0004,0.0004},{4.271,4.271},{0.0071,0.0071},
             {162.1,162.1},{4.268,4.268},{12.25,12.25}},
            
            // std
            {{1,1},{0.3833,1.669},{0.5,0.1},{0.5,0.1},{7.,7.},
             {3.,3.},{1,1},{1,1},{1,1},{7.,7.},
             {0.0067,0.0067},{0.0069,0.0069},{0.0578,0.0578},{0.9565,0.9565},{0.9592,0.9592},
             {2.154,2.154},{1,1},{0.882,0.882},{63.78,63.78},{5.285,5.285},
             {622.4,622.4},{15.47,15.47},{4.774,4.774}},

            // lim_min
            {{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-5,-5},{-1.0,-1.0},{-1.0,-1.0},{0.,0.},
             {0.,0.},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{0.,0.},
             {-5,-5},{-5,-5},{-5,-5},{-5,-5},{-5,-5},
             {-5,-5},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-5,-5},{-5,-5},{-5,-5},
             {-5,-5},{-5,-5},{-5,-5}},

            // lim_max
            {{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{5,5},{1.0,1.0},{1.0,1.0},{1.,1.},
             {1.,1.},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{1.,1.},
             {5,5},{5,5},{5,5},{5,5},{5,5},
             {5,5},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{5,5},{5,5},{5,5},
             {5,5},{5,5},{5,5}},
            
            }
        }, // end PfCand_gamma

        {FeatureT::Electron, {
            // mean
            {{0,0},{0.5111,1.067},{0.0,0.0},{0.0,0.0},{0,0},
            {1.729,1.729},{0.1439,0.1439},{1.794,1.794},{1.531,1.531},{1.531,1.531},
            {0.7735,0.7735},{0.7735,0.7735},{1.625,1.625},{1.993,1.993},{70.25,70.25},
            {2.432,2.432},{2.034,2.034},{6.64,6.64},{4.183,4.183},{0.,0.},
            {-0.0001,-0.0001},{-0.0001,-0.0001},{0.0002,0.0002},{0.0001,0.0001},{0.0004,0.0004},
            {0,0},{0,0},{0.0008,0.0008},{14.04,14.04},{0.0099,0.0099},
            {3.049,3.049},{16.52,16.52},{1.355,1.355},{5.046,5.046},{0,0},
            {2.411,2.411},{15.16,15.16}},

            // std 
            {{1,1},{2.765,1.521},{0.5,0.1},{0.5,0.1},{1,1},
            {1.644,1.644},{0.3284,0.3284},{2.079,2.079},{1.424,1.424},{1.424,1.424},
            {0.935,0.935},{0.935,0.935},{1.581,1.581},{1.308,1.308},{58.16,58.16},
            {15.13,15.13},{13.96,13.96},{36.8,36.8},{20.63,20.63},{0.0363,0.0363},
            {0.0512,0.0512},{0.0541,0.0541},{0.0553,0.0553},{0.0523,0.0523},{0.0777,0.0777},
            {1,1},{1,1},{0.0052,0.0052},{69.48,69.48},{0.0851,0.0851},
            {10.39,10.39},{2.806,2.806},{16.81,16.81},{3.119,3.119},{1,1},
            {6.98,6.98},{5.26,5.26}},

            // lim_min
            {{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-5,-5},{-1.0,-1.0},{-1.0,-1.0},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},
            {-5,-5},{-5,-5},{-5,-5},{-5,-5},{-5,-5},
            {-5,-5},{-5,-5},{-5,-5},{-5,-5},{-5,-5},
            {-5,-5},{-5,-5},{-5,-5},{-5,-5},{-5,-5},
            {-5,-5},{-5,-5},{-5,-5},{-5,-5},{-5,-5},
            {-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-5,-5},{-5,-5},{-5,-5},
            {-5,-5},{-5,-5},{-5,-5},{-5,-5},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},
            {-5,-5},{-5,-5}},

            // lim_max
            {{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{5,5},{1.0,1.0},{1.0,1.0},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},
            {5,5},{5,5},{5,5},{5,5},{5,5},
            {5,5},{5,5},{5,5},{5,5},{5,5},
            {5,5},{5,5},{5,5},{5,5},{5,5},
            {5,5},{5,5},{5,5},{5,5},{5,5},
            {std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{5,5},{5,5},{5,5},
            {5,5},{5,5},{5,5},{5,5},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},
            {5,5},{5,5}},
            }
        }, // end Electron

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

        {FeatureT::PfCand_electron, {
            // mean_
            {{0,0},{0.3457,0.9558},{0.0,0.0},{0.0,0.0},{0,0},
             {0,0},{0,0},{0,0},{5.0,5.0},{-0.0008022,-2.888e-06},
             {-2.653e-05,7.215e-06},{0.00382,0.0002156},{0.002371,0.0002385},{0.0003833,6.221e-05},{0.0004431,0.0003546},
             {0,0},{0.000397,3.333e-05},{3.409,1.412},{0.003507,0.0002181},{169.6,21.72},
             {4.561,2.387},{12.6,14.73}},

            // std_
            {{1,1},{1.164,0.2323},{0.5,0.1},{0.5,0.1},{1,1},
             {1,1},{1,1},{1,1},{5.0,5.0},{0.4081,0.03703},
             {0.4056,0.03682},{3.329,0.5552},{0.6623,0.1855},{0.6648,0.1867},{3.548,0.749},
             {1,1},{0.5572,0.05183},{16.07,3.111},{3.3,0.5551},{486.1,230.5},
             {14.8,8.818},{3.729,3.125}},

            // lim_min_
            {{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-5,-5},{-1.0,-1.0},{-1.0,-1.0},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},
            {-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-1.0,-1.0},{-5,-5},
            {-5,-5},{-5,-5},{-5,-5},{-5,-5},{-5,-5},
            {-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-5,-5},{-5,-5},{-5,-5},{-5,-5},
            {-5,-5},{-5,-5}},

            // lim_max_
            {{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{5,5},{1.0,1.0},{1.0,1.0},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},
            {std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{1.0,1.0},{5,5},
            {5,5},{5,5},{5,5},{5,5},{5,5},
            {std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{5,5},{5,5},{5,5},{5,5},
            {5,5},{5,5}}
            }
        }, // end PfCand_electron
    
        {FeatureT::PfCand_gamma, {
            // mean
            {{0,0},{0.02024,0.2681},{0.0,0.0},{0.0,0.0},{0,0},
             {0,0},{0,0},{0,0},{0,0},{3.5,3.5},
             {2.364e-08,-6.701e-06},{-1.355e-07,4.799e-06},{5.947e-07,3.08e-05},{0.001155,0.0009319},{-3.88e-05,-0.0001133},
             {0.001081,0.0007838},{0,0},{0.003532,-0.0003009},{4.09,3.826},{0.02207,0.01115},
             {175.0,114.2},{4.798,4.218},{12.18,12.27}},
            
            // std
            {{1,1},{0.1801,0.5467},{0.5,0.1},{0.5,0.1},{1,1},
             {1,1},{1,1},{1,1},{1,1},{3.5,3.5},
             {0.003674,0.02348},{0.00371,0.02357},{0.02345,0.2203},{0.4628,0.4899},{0.4667,0.4941},
             {1.057,1.284},{1,1},{1.006,0.633},{11.45,20.83},{4.517,4.191},
             {546.1,439.3},{16.85,15.84},{4.741,4.562}},

            // lim_min
            {{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-5,-5},{-1.0,-1.0},{-1.0,-1.0},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},
             {-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-1.0,-1.0},
             {-5,-5},{-5,-5},{-5,-5},{-5,-5},{-5,-5},
             {-5,-5},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-5,-5},{-5,-5},{-5,-5},
             {-5,-5},{-5,-5},{-5,-5}},

            // lim_max
            {{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{5,5},{1.0,1.0},{1.0,1.0},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},
             {std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{1.0,1.0},
             {5,5},{5,5},{5,5},{5,5},{5,5},
             {5,5},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{5,5},{5,5},{5,5},
             {5,5},{5,5},{5,5}},

            }
        }, // end PfCand_gamma

        {FeatureT::Electron, {
            // mean
            {{0,0},{0.3827,0.9372},{0.0,0.0},{0.0,0.0},{0,0},
            {1.37,1.654},{0.3215,0.1878},{1.793,2.055},{1.093,2.593},{1.093,2.593},
            {1.013,1.006},{1.013,1.006},{1.063,1.749},{1.445,2.0},{13.07,59.55},
            {3.797,1.748},{2.624,1.404},{5.68,5.054},{2.231,3.078},{-0.0001921,-4.413e-06},
            {-0.0009969,-1.477e-05},{-0.0008593,9.209e-07},{-0.0008999,0.0001262},{-0.001147,8.781e-05},{-0.001182,0.0003861},
            {0,0},{0,0},{0.001218,0.000632},{31.5,15.88},{0.05644,0.005635},
            {6.344,3.163},{14.65,16.15},{1.917,1.669},{6.866,5.276},{0,0},
            {1.862,2.813},{12.15,14.46}},

            // std 
            {{1,1},{1.272,0.4817},{0.5,0.1},{0.5,0.1},{1,1},
            {8.381,1.104},{0.5275,0.3595},{2.419,2.141},{82.69,1183.0},{82.69,1183.0},
            {673.8,233.5},{673.8,233.5},{5.614,88.75},{2.021,1.278},{27.8,44.9},
            {21.65,2.591},{19.0,2.199},{41.93,14.8},{21.58,10.23},{0.1324,0.0119},
            {0.1474,0.02151},{0.1548,0.02331},{0.1514,0.03042},{0.1452,0.03347},{0.1966,0.05816},
            {1,1},{1,1},{0.00775,0.004139},{82.72,50.36},{0.2343,0.05148},
            {292.7,15.01},{3.103,2.752},{229.2,431.6},{5.051,2.463},{1,1},
            {5.64,8.186},{5.557,5.149}},

            // lim_min
            {{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-5,-5},{-1.0,-1.0},{-1.0,-1.0},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},
            {-5,-5},{-5,-5},{-5,-5},{-5,-5},{-5,-5},
            {-5,-5},{-5,-5},{-5,-5},{-5,-5},{-5,-5},
            {-5,-5},{-5,-5},{-5,-5},{-5,-5},{-5,-5},
            {-5,-5},{-5,-5},{-5,-5},{-5,-5},{-5,-5},
            {-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},{-5,-5},{-5,-5},{-5,-5},
            {-5,-5},{-5,-5},{-5,-5},{-5,-5},{-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()},
            {-5,-5},{-5,-5}},

            // lim_max
            {{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{5,5},{1.0,1.0},{1.0,1.0},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},
            {5,5},{5,5},{5,5},{5,5},{5,5},
            {5,5},{5,5},{5,5},{5,5},{5,5},
            {5,5},{5,5},{5,5},{5,5},{5,5},
            {5,5},{5,5},{5,5},{5,5},{5,5},
            {std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},{5,5},{5,5},{5,5},
            {5,5},{5,5},{5,5},{5,5},{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()},
            {5,5},{5,5}},
            }
        }, // end Electron

    }; // end scalingParamsMap_v2p5

}; // end Scaling namespace

#endif