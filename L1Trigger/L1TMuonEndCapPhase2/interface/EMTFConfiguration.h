#ifndef L1Trigger_L1TMuonEndCapPhase2_EMTFConfiguration_h
#define L1Trigger_L1TMuonEndCapPhase2_EMTFConfiguration_h

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"

namespace emtf::phase2 {

    // Class
    class EMTFConfiguration {

        public:
            EMTFConfiguration(const edm::ParameterSet&);

            ~EMTFConfiguration();

            // Event configuration
            void update(
                    const edm::Event&,
                    const edm::EventSetup&);

            // Config
            int verbosity_;

            // Validation
            std::string validation_dir_;

            // Neural Network
            std::string prompt_graph_path_; 
            std::string displ_graph_path_; 

            // Trigger
            int min_bx_;
            int max_bx_;
            int bx_window_;

            // Subsystems
            bool csc_en_;
            bool rpc_en_;
            bool gem_en_;
            bool me0_en_;
            bool ge0_en_;

            int csc_bx_shift_;
            int rpc_bx_shift_;
            int gem_bx_shift_;
            int me0_bx_shift_;

            // Primitive Selectoin
            bool include_neighbor_en_;
    };
}

#endif  // L1Trigger_L1TMuonEndCapPhase2_EMTFConfiguration_h not defined
