#ifndef L1Trigger_L1TGEM_ME0StubAlgoSubfunction_H
#define L1Trigger_L1TGEM_ME0StubAlgoSubfunction_H

#include <cmath>
#include <vector>
#include <map>
#include <cstdint>
#include <algorithm>
#include <bitset>
#include <iostream>
#include <numeric>
#include "DataFormats/GEMDigi/interface/ME0StubPrimitive.h"

namespace l1t {
    namespace me0 {
        typedef std::bitset<192> UInt192;

        struct Config {
            bool skip_centroids;
            // int32_t ly_thresh;
            std::vector<int32_t> ly_thresh_patid;
            std::vector<int32_t> ly_thresh_eta;
            int32_t max_span;
            int32_t width;
            bool deghost_pre;
            bool deghost_post;
            int32_t group_width;
            int32_t ghost_width;
            bool x_prt_en;
            bool en_non_pointing;
            int32_t cross_part_seg_width;
            int32_t num_outputs;
            bool check_ids;
            int32_t edge_distance;
            int32_t num_or;
        };

        class hi_lo_t {
        private:
        public:
            int hi, lo;
            hi_lo_t(int hi_, int lo_) : hi(hi_), lo(lo_) {}
        };

        class patdef_t {
        private:
        public:
            int id;
            std::vector<hi_lo_t> layers;
            patdef_t(int id_, std::vector<hi_lo_t> layers_) : id(id_), layers(layers_) {}
        };

        class Mask {
        private:
        public:
            int id;
            std::vector<uint64_t> mask;
            Mask(int id_, std::vector<uint64_t> mask_) : id(id_), mask(mask_) {}
            std::string to_string() const;
        };

        hi_lo_t mirror_hi_lo(const hi_lo_t& ly);
        patdef_t mirror_patdef(const patdef_t& pat, int id);
        std::vector<hi_lo_t> create_pat_ly(double lower, double upper);

        int count_ones(uint64_t x);
        int max_cluster_size(uint64_t x);
        UInt192 set_bit(int index, UInt192 num1);
        UInt192 clear_bit(int index, UInt192 num);
        uint64_t one_bit_mask(int num);
        std::vector<int> find_ones(uint64_t& data);
        std::pair<double,std::vector<int>> find_centroid(uint64_t& data);
        std::vector<std::vector<ME0StubPrimitive>> chunk(const std::vector<ME0StubPrimitive>& in_list, int n);
        void segment_sorter(std::vector<ME0StubPrimitive>& segs, int n);
        std::vector<int> concatVector(const std::vector<std::vector<int>>& vec);
        std::vector<ME0StubPrimitive> concatVector(const std::vector<std::vector<ME0StubPrimitive>>& vec);
    }
}
#endif