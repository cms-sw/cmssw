#ifndef L1Trigger_L1TGEM_ME0StubAlgoMask_H
#define L1Trigger_L1TGEM_ME0StubAlgoMask_H

#include "L1Trigger/L1TGEM/interface/ME0StubAlgoSubfunction.h"

namespace l1t {
  namespace me0 {
    std::vector<int> shift_center(const hi_lo_t& ly, int max_span);
    uint64_t set_high_bits(const std::vector<int>& lo_hi_pair);
    Mask get_ly_mask(const patdef_t& ly_pat, int max_span);

    const patdef_t pat_straight = patdef_t(17, create_pat_ly(-0.4, 0.4));
    const patdef_t pat_l = patdef_t(16, create_pat_ly(0.2, 0.9));
    const patdef_t pat_r = mirror_patdef(pat_l, pat_l.id - 1);
    const patdef_t pat_l2 = patdef_t(14, create_pat_ly(0.9, 1.7));
    const patdef_t pat_r2 = mirror_patdef(pat_l2, pat_l2.id - 1);
    const patdef_t pat_l3 = patdef_t(12, create_pat_ly(1.4, 2.3));
    const patdef_t pat_r3 = mirror_patdef(pat_l3, pat_l3.id - 1);
    const patdef_t pat_l4 = patdef_t(10, create_pat_ly(2.0, 3.0));
    const patdef_t pat_r4 = mirror_patdef(pat_l4, pat_l4.id - 1);
    const patdef_t pat_l5 = patdef_t(8, create_pat_ly(2.7, 3.8));
    const patdef_t pat_r5 = mirror_patdef(pat_l5, pat_l5.id - 1);
    const patdef_t pat_l6 = patdef_t(6, create_pat_ly(3.5, 4.7));
    const patdef_t pat_r6 = mirror_patdef(pat_l6, pat_l6.id - 1);
    const patdef_t pat_l7 = patdef_t(4, create_pat_ly(4.3, 5.5));
    const patdef_t pat_r7 = mirror_patdef(pat_l7, pat_l7.id - 1);
    const patdef_t pat_l8 = patdef_t(2, create_pat_ly(5.4, 7.0));
    const patdef_t pat_r8 = mirror_patdef(pat_l8, pat_l8.id - 1);

    const std::vector<Mask> LAYER_MASK{get_ly_mask(pat_straight, 37),
                                       get_ly_mask(pat_l, 37),
                                       get_ly_mask(pat_r, 37),
                                       get_ly_mask(pat_l2, 37),
                                       get_ly_mask(pat_r2, 37),
                                       get_ly_mask(pat_l3, 37),
                                       get_ly_mask(pat_r3, 37),
                                       get_ly_mask(pat_l4, 37),
                                       get_ly_mask(pat_r4, 37),
                                       get_ly_mask(pat_l5, 37),
                                       get_ly_mask(pat_r5, 37),
                                       get_ly_mask(pat_l6, 37),
                                       get_ly_mask(pat_r6, 37),
                                       get_ly_mask(pat_l7, 37),
                                       get_ly_mask(pat_r7, 37),
                                       get_ly_mask(pat_l8, 37),
                                       get_ly_mask(pat_r8, 37)};
  }  // namespace me0
}  // namespace l1t
#endif