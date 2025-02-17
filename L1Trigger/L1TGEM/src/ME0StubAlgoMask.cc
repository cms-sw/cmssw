#include "L1Trigger/L1TGEM/interface/ME0StubAlgoMask.h"

using namespace l1t::me0;

std::vector<int> l1t::me0::shift_center(const hi_lo_t& ly, int max_span) {
    /*

    Patterns are defined as a +hi and -lo around a center point of a pattern.

    e.g. for a pattern 37 strips wide, there is a central strip,
    and 18 strips to the left and right of it.

    This patterns shifts from a +hi and -lo around the central strip, to an offset +hi and -lo.

    e.g. for (hi, lo) = (1, -1) and a window of 37, this will return (17,19)

    */
    int center = std::floor(max_span/2);
    int hi = ly.hi + center;
    int lo = ly.lo + center;
    std::vector<int> out = {lo, hi};
    return out;
}

uint64_t l1t::me0::set_high_bits(const std::vector<int>& lo_hi_pair) {
    /*
    Given a high bit and low bit, this function will return a bitmask with all the bits in
    between the high and low set to 1
    */
    int lo = lo_hi_pair[0], hi = lo_hi_pair[1];
    uint64_t out = std::pow(2,(hi-lo+1)) - 1;
    out <<= lo;
    return out;
}

Mask l1t::me0::get_ly_mask(const patdef_t& ly_pat, int max_span = 37) {
    /*
    takes in a given layer pattern and returns a list of integer bit masks
    for each layer
    */

    std::vector<std::vector<int>> m_vals;
    std::vector<uint64_t> m_vec;
    
    // for each layer, shift the provided hi and lo values for each layer from
    // pattern definition by center
    for (hi_lo_t ly : ly_pat.layers) {m_vals.push_back(shift_center(ly, max_span));}
    
    // use the high and low indices to determine where the high bits must go for
    // each layer 
    for (std::vector<int> x : m_vals) {m_vec.push_back(set_high_bits(x));}
    
    Mask mask_{ly_pat.id, m_vec};
    return mask_; 
}