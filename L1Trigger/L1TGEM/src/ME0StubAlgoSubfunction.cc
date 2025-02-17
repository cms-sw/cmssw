#include "L1Trigger/L1TGEM/interface/ME0StubAlgoSubfunction.h"

using namespace l1t::me0;

//define functions to generate patterns
hi_lo_t l1t::me0::mirror_hi_lo(const hi_lo_t& ly) {
    hi_lo_t mirrored{-1*(ly.lo), -1*(ly.hi)};
    return mirrored;
}
patdef_t l1t::me0::mirror_patdef(const patdef_t& pat, int id) {
    std::vector<hi_lo_t> layers_;
    for (hi_lo_t l : pat.layers) {
        layers_.push_back(mirror_hi_lo(l));
    }
    patdef_t mirrored{id, layers_};
    return mirrored;
}
std::vector<hi_lo_t> l1t::me0::create_pat_ly(double lower, double upper) {
    std::vector<hi_lo_t> layer_list;
    double hi, lo;
    int hi_i, lo_i;
    for (int i=0; i<6; ++i) {
        if (i < 3) {
            hi = lower*(i-2.5);
            lo = upper*(i-2.5);
        }
        else {
            hi = upper*(i-2.5);
            lo = lower*(i-2.5);
        }
        if (std::abs(hi) < 0.1) {hi = 0.0f;}
        if (std::abs(lo) < 0.1) {lo = 0.0f;}
        hi_i = std::ceil(hi);
        lo_i = std::floor(lo);
        layer_list.push_back(hi_lo_t{hi_i, lo_i});
    }
    return layer_list;
}
int l1t::me0::count_ones(u_int64_t x) {
    int cnt = 0;
    while (x > 0) {
        if (x&1) {
            ++cnt;
        }
        x = (x>>1);
    }
    return cnt;
}
int l1t::me0::max_cluster_size(uint64_t x) {
    int size = 0;
    int max_size = 0;
    while (x > 0) {
        if ((x&1) == 1) {size++;}
        else {
            if (size > max_size) {max_size = size;}
            size = 0;
        }
        x = x>>1;
    }
    if (size > max_size) {max_size=size;}
    return max_size;
}
UInt192 l1t::me0::set_bit(int index, UInt192 num1 = UInt192(0)) {
    UInt192 num2 = (UInt192(1) << index);
    UInt192 final_v = num1 | num2;
    return final_v;
}
UInt192 l1t::me0::clear_bit(int index, UInt192 num) {
    UInt192 bit = UInt192(1) & (num >> index);
    return num^(bit << index);
}
uint64_t l1t::me0::one_bit_mask(int num) {
    uint64_t o_mask = 0;
    int bit_num = 0;
    while (num != 0) {
        o_mask |= (1 << bit_num);
        num = (num >> 1);
        ++bit_num;
    }
    return o_mask;
}
std::vector<int> l1t::me0::find_ones(uint64_t& data) {
    std::vector<int> ones;
    int cnt = 0;
    while (data > 0) {
        if ((data & 1)) {ones.push_back(cnt+1);}
        data >>= 1;
        ++cnt;
    }
    return ones;
}
std::pair<double,std::vector<int>> l1t::me0::find_centroid(uint64_t& data) {
    std::vector<int> ones = find_ones(data);
    if (static_cast<int>(ones.size())==0) {return {0.0, ones};}
    int sum = 0;
    for (int n : ones) {sum += n;}
    return {static_cast<double>(sum)/static_cast<double>(ones.size()), ones};
}
std::vector<std::vector<ME0StubPrimitive>> l1t::me0::chunk(const std::vector<ME0StubPrimitive>& in_list, int n) {
    std::vector<std::vector<ME0StubPrimitive>> chunks;
    int size = in_list.size();
    for (int i = 0; i < (size + n - 1) / n; ++i) {
        std::vector<ME0StubPrimitive> chunk(in_list.begin() + i * n, in_list.begin() + std::min((i + 1) * n, size));
        chunks.push_back(chunk);
    }
    return chunks;
}
void l1t::me0::segment_sorter(std::vector<ME0StubPrimitive>& segs, int n) {
    std::sort(segs.begin(), segs.end(),
          [](const ME0StubPrimitive& lhs, const ME0StubPrimitive& rhs) {
            return (lhs.Quality() > rhs.Quality());});
    segs = std::vector<ME0StubPrimitive>(segs.begin(), std::min(segs.begin() + n, segs.end()));
}
std::vector<int> l1t::me0::concatVector(const std::vector<std::vector<int>>& vec) {
    std::vector<int> cat;
    for (auto v : vec) {
        cat.insert(cat.end(), v.begin(), v.end());
    }
    return cat;
}
std::vector<ME0StubPrimitive> l1t::me0::concatVector(const std::vector<std::vector<ME0StubPrimitive>>& vec) {
    std::vector<ME0StubPrimitive> cat;
    for (auto v : vec) {
        cat.insert(cat.end(), v.begin(), v.end());
    }
    return cat;
}