#ifndef L1Trigger_L1TGEM_ME0StubAlgoMask_H
#define L1Trigger_L1TGEM_ME0StubAlgoMask_H

#include "L1Trigger/L1TGEM/interface/ME0StubAlgoSubfunction.h"

namespace l1t {
  namespace me0 {
    std::vector<int> shift_center(const hi_lo_t& ly, int max_span);
    uint64_t set_high_bits(const std::vector<int>& lo_hi_pair);
    Mask get_ly_mask(const patdef_t& ly_pat, int max_span);

    /*
    create_pat_ly(low, high) returns a vector of hi_lo_t objects with the given low and high values.
    low and high are relative distances from the center of the pattern (strip 18 for max_span=37).

    For example, create_pat_ly(0.2, 0.9) returns a vector of hi_lo_t objects with the following values:
    { [ hi: 0, lo: -3 ], [ hi: 0, lo: -2 ], [ hi: 0, lo: -1 ], [ hi: 1, lo: 0 ], [ hi: 2, lo: 0 ], [ hi: 3, lo: 0 ]}

    patdef_t(id, layers) saves the pattern ID and the vector of hi_lo_t objects for each layer.
    
    get_ly_mask(patdef_f pat, int pat_width) returns a Mask object with the given pattern and max_span values.
    example:

    get_ly_mask(pat_l, 37) returns a Mask object with the following values:
    Pattern ID: 16
    {0b0000000000000000001111000000000000000,  // ly5 
     0b0000000000000000001110000000000000000,  // ly4 
     0b0000000000000000001100000000000000000,  // ly3 
     0b0000000000000000011000000000000000000,  // ly2 
     0b0000000000000000111000000000000000000,  // ly1 
     0b0000000000000001111000000000000000000}  // ly0 
    */

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

/*
patlist:
Pattern ID: 17
ly5 -----------------XXX-----------------

ly4 -----------------XXX-----------------

ly3 -----------------XXX-----------------

ly2 -----------------XXX-----------------

ly1 -----------------XXX-----------------

ly0 -----------------XXX-----------------



Pattern ID: 16
ly5 ------------------XXXX---------------

ly4 ------------------XXX----------------

ly3 ------------------XX-----------------

ly2 -----------------XX------------------

ly1 ----------------XXX------------------

ly0 ---------------XXXX------------------



Pattern ID: 15
ly5 ---------------XXXX------------------

ly4 ----------------XXX------------------

ly3 -----------------XX------------------

ly2 ------------------XX-----------------

ly1 ------------------XXX----------------

ly0 ------------------XXXX---------------



Pattern ID: 14
ly5 --------------------XXXX-------------

ly4 -------------------XXX---------------

ly3 ------------------XX-----------------

ly2 -----------------XX------------------

ly1 ---------------XXX-------------------

ly0 -------------XXXX--------------------



Pattern ID: 13
ly5 -------------XXXX--------------------

ly4 ---------------XXX-------------------

ly3 -----------------XX------------------

ly2 ------------------XX-----------------

ly1 -------------------XXX---------------

ly0 --------------------XXXX-------------



Pattern ID: 12
ly5 ---------------------XXXX------------

ly4 --------------------XXX--------------

ly3 ------------------XXX----------------

ly2 ----------------XXX------------------

ly1 --------------XXX--------------------

ly0 ------------XXXX---------------------



Pattern ID: 11
ly5 ------------XXXX---------------------

ly4 --------------XXX--------------------

ly3 ----------------XXX------------------

ly2 ------------------XXX----------------

ly1 --------------------XXX--------------

ly0 ---------------------XXXX------------



Pattern ID: 10
ly5 -----------------------XXXX----------

ly4 ---------------------XXX-------------

ly3 -------------------XX----------------

ly2 ----------------XX-------------------

ly1 -------------XXX---------------------

ly0 ----------XXXX-----------------------



Pattern ID: 9
ly5 ----------XXXX-----------------------

ly4 -------------XXX---------------------

ly3 ----------------XX-------------------

ly2 -------------------XX----------------

ly1 ---------------------XXX-------------

ly0 -----------------------XXXX----------



Pattern ID: 8
ly5 ------------------------XXXXX--------

ly4 ----------------------XXX------------

ly3 -------------------XX----------------

ly2 ----------------XX-------------------

ly1 ------------XXX----------------------

ly0 --------XXXXX------------------------



Pattern ID: 7
ly5 --------XXXXX------------------------

ly4 ------------XXX----------------------

ly3 ----------------XX-------------------

ly2 -------------------XX----------------

ly1 ----------------------XXX------------

ly0 ------------------------XXXXX--------



Pattern ID: 6
ly5 --------------------------XXXXX------

ly4 -----------------------XXXX----------

ly3 -------------------XXX---------------

ly2 ---------------XXX-------------------

ly1 ----------XXXX-----------------------

ly0 ------XXXXX--------------------------



Pattern ID: 5
ly5 ------XXXXX--------------------------

ly4 ----------XXXX-----------------------

ly3 ---------------XXX-------------------

ly2 -------------------XXX---------------

ly1 -----------------------XXXX----------

ly0 --------------------------XXXXX------



Pattern ID: 4
ly5 ----------------------------XXXXX----

ly4 ------------------------XXXX---------

ly3 --------------------XX---------------

ly2 ---------------XX--------------------

ly1 ---------XXXX------------------------

ly0 ----XXXXX----------------------------



Pattern ID: 3
ly5 ----XXXXX----------------------------

ly4 ---------XXXX------------------------

ly3 ---------------XX--------------------

ly2 --------------------XX---------------

ly1 ------------------------XXXX---------

ly0 ----------------------------XXXXX----



Pattern ID: 2
ly5 -------------------------------XXXXXX

ly4 --------------------------XXXX-------

ly3 --------------------XXX--------------

ly2 --------------XXX--------------------

ly1 -------XXXX--------------------------

ly0 XXXXXX-------------------------------



Pattern ID: 1
ly5 XXXXXX-------------------------------

ly4 -------XXXX--------------------------

ly3 --------------XXX--------------------

ly2 --------------------XXX--------------

ly1 --------------------------XXXX-------

ly0 -------------------------------XXXXXX

*/