#ifndef L1Trigger_L1TGEM_ME0StubAlgoMask_H
#define L1Trigger_L1TGEM_ME0StubAlgoMask_H

#include "L1Trigger/L1TGEM/interface/ME0StubAlgoSubfunction.h"

namespace l1t {
  namespace me0 {
    class HiLo {
    private:
    public:
      int hi, lo;
      HiLo(int hi_, int lo_) : hi(hi_), lo(lo_) {}
    };

    class PatternDefinition {
    private:
    public:
      int id;
      std::vector<HiLo> layers;
      PatternDefinition(int id_, std::vector<HiLo> layers_) : id(id_), layers(layers_) {}
    };

    class Mask {
    private:
    public:
      int id;
      std::vector<uint64_t> mask;
      Mask(int id_, std::vector<uint64_t> mask_) : id(id_), mask(mask_) {}
      std::string toString() const;  // not implemented, for debugging purposes
    };

    std::vector<int> shiftCenter(const HiLo& layer, int maxSpan);
    uint64_t setHighBits(const std::vector<int>& loHiPair);
    Mask getLayerMask(const PatternDefinition& layerPattern, const std::vector<int> layerSpans);

    HiLo mirrorHiLo(const HiLo& layer);
    PatternDefinition mirrorPatternDefinition(const PatternDefinition& pattern, int id);
    std::vector<HiLo> createPatternLayer(double lower, double upper);

    /*
    createPatternLayer(low, high) returns a vector of HiLo objects with the given low and high values.
    low and high are relative distances from the center of the pattern (strip 18 for maxSpan=37).

    For example, createPatternLayer(0.2, 0.9) returns a vector of HiLo objects with the following values:
    { [ hi: 0, lo: -3 ], [ hi: 0, lo: -2 ], [ hi: 0, lo: -1 ], [ hi: 1, lo: 0 ], [ hi: 2, lo: 0 ], [ hi: 3, lo: 0 ]}

    PatternDefinition(id, layers) saves the pattern ID and the vector of HiLo objects for each layer.
    
    getLayerMask(PatternDefinition pattern, const std::vector<int>& layerSpans) returns a Mask object with the given pattern and layerSpans values.
    example:

    getLayerMask(patternLeft, {37, 37, 37, 37, 37, 37}) returns a Mask object with the following values:
    Pattern ID: 16
    {0b0000000000000000001111000000000000000,  // ly5 
     0b0000000000000000001110000000000000000,  // ly4 
     0b0000000000000000001100000000000000000,  // ly3 
     0b0000000000000000011000000000000000000,  // ly2 
     0b0000000000000000111000000000000000000,  // ly1 
     0b0000000000000001111000000000000000000}  // ly0 
    */

    const PatternDefinition kPatternStraight = PatternDefinition(17, createPatternLayer(-0.4, 0.4));
    const PatternDefinition kPatternLeft = PatternDefinition(16, createPatternLayer(0.2, 0.9));
    const PatternDefinition kPatternRight = mirrorPatternDefinition(kPatternLeft, kPatternLeft.id - 1);
    const PatternDefinition kPatternLeft2 = PatternDefinition(14, createPatternLayer(0.9, 1.7));
    const PatternDefinition kPatternRight2 = mirrorPatternDefinition(kPatternLeft2, kPatternLeft2.id - 1);
    const PatternDefinition kPatternLeft3 = PatternDefinition(12, createPatternLayer(1.4, 2.3));
    const PatternDefinition kPatternRight3 = mirrorPatternDefinition(kPatternLeft3, kPatternLeft3.id - 1);
    const PatternDefinition kPatternLeft4 = PatternDefinition(10, createPatternLayer(2.0, 3.0));
    const PatternDefinition kPatternRight4 = mirrorPatternDefinition(kPatternLeft4, kPatternLeft4.id - 1);
    const PatternDefinition kPatternLeft5 = PatternDefinition(8, createPatternLayer(2.7, 3.8));
    const PatternDefinition kPatternRight5 = mirrorPatternDefinition(kPatternLeft5, kPatternLeft5.id - 1);
    const PatternDefinition kPatternLeft6 = PatternDefinition(6, createPatternLayer(3.5, 4.7));
    const PatternDefinition kPatternRight6 = mirrorPatternDefinition(kPatternLeft6, kPatternLeft6.id - 1);
    const PatternDefinition kPatternLeft7 = PatternDefinition(4, createPatternLayer(4.3, 5.5));
    const PatternDefinition kPatternRight7 = mirrorPatternDefinition(kPatternLeft7, kPatternLeft7.id - 1);
    const PatternDefinition kPatternLeft8 = PatternDefinition(2, createPatternLayer(5.4, 7.0));
    const PatternDefinition kPatternRight8 = mirrorPatternDefinition(kPatternLeft8, kPatternLeft8.id - 1);

    const std::vector<PatternDefinition> kPatternList{kPatternRight8,
                                                      kPatternLeft8,
                                                      kPatternRight7,
                                                      kPatternLeft7,
                                                      kPatternRight6,
                                                      kPatternLeft6,
                                                      kPatternRight5,
                                                      kPatternLeft5,
                                                      kPatternRight4,
                                                      kPatternLeft4,
                                                      kPatternRight3,
                                                      kPatternLeft3,
                                                      kPatternRight2,
                                                      kPatternLeft2,
                                                      kPatternRight,
                                                      kPatternLeft,
                                                      kPatternStraight};

    std::vector<int> calculateLayerSpans(const std::vector<PatternDefinition>& patternList);
    const std::vector<int> kLayerSpans = calculateLayerSpans(kPatternList);

    std::vector<int> calculatePatternSpans(const std::vector<PatternDefinition>& patternList);
    const std::vector<int> kPatSpans = calculatePatternSpans(kPatternList);

    std::vector<std::vector<int>> calculatePatternOffsets(const std::vector<PatternDefinition>& patternList,
                                                          const std::vector<int>& patternSpans,
                                                          const std::vector<int>& layerSpans);
    const std::vector<std::vector<int>> kPatOffsets = calculatePatternOffsets(kPatternList, kPatSpans, kLayerSpans);

    const std::vector<Mask> kLayerMask{getLayerMask(kPatternRight8, kLayerSpans),
                                       getLayerMask(kPatternLeft8, kLayerSpans),
                                       getLayerMask(kPatternRight7, kLayerSpans),
                                       getLayerMask(kPatternLeft7, kLayerSpans),
                                       getLayerMask(kPatternRight6, kLayerSpans),
                                       getLayerMask(kPatternLeft6, kLayerSpans),
                                       getLayerMask(kPatternRight5, kLayerSpans),
                                       getLayerMask(kPatternLeft5, kLayerSpans),
                                       getLayerMask(kPatternRight4, kLayerSpans),
                                       getLayerMask(kPatternLeft4, kLayerSpans),
                                       getLayerMask(kPatternRight3, kLayerSpans),
                                       getLayerMask(kPatternLeft3, kLayerSpans),
                                       getLayerMask(kPatternRight2, kLayerSpans),
                                       getLayerMask(kPatternLeft2, kLayerSpans),
                                       getLayerMask(kPatternRight, kLayerSpans),
                                       getLayerMask(kPatternLeft, kLayerSpans),
                                       getLayerMask(kPatternStraight, kLayerSpans)};
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
