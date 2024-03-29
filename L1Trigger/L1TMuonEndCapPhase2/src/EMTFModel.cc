#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DebugUtils.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFModel.h"

using namespace emtf::phase2;
using namespace emtf::phase2::model;

EMTFModel::EMTFModel(const EMTFContext& context) : context_(context) {
  // ===========================================================================
  // Zone 0
  // ===========================================================================

  // clang-format off
    zones::hitmap_t&& zone0_hm = {
        { // Row 0
            {
                site_id_t::kME0, {
                    {114,   38,     90 },
                    {108,   75,     127},
                    {109,   113,    165},
                    {110,   150,    202},
                    {111,   188,    240},
                    {112,   225,    277},
                    {113,   263,    315}
                }
            }
        },
        { // Row 1
            {
                site_id_t::kGE11, {
                    {99 ,   38,     90 },
                    {54 ,   75,     127},
                    {55 ,   113,    165},
                    {56 ,   150,    202},
                    {63 ,   188,    240},
                    {64 ,   225,    277},
                    {65 ,   263,    315}
                }
            }
        },
        { // Row 2
            {
                site_id_t::kME11, {
                    {45 ,   38,     90 },
                    {0  ,   75,     127},
                    {1  ,   113,    165},
                    {2  ,   150,    202},
                    {9  ,   188,    240},
                    {10 ,   225,    277},
                    {11 ,   263,    315}
                }
            }
        },
        { // Row 3
            {
                site_id_t::kGE21, {
                    {102,   0   ,   90 },
                    {72 ,   75  ,   165},
                    {73 ,   150 ,   240},
                    {74 ,   225 ,   315}
                }
            }
        },
        { // Row 4
            {
                site_id_t::kME2, {
                    {48 ,   0   ,   90 },
                    {18 ,   75  ,   165},
                    {19 ,   150 ,   240},
                    {20 ,   225 ,   315}
                }
            }
        },
        { // Row 5
            {
                site_id_t::kME3, {
                    {50 ,   0   ,   90 },
                    {27 ,   75  ,   165},
                    {28 ,   150 ,   240},
                    {29 ,   225 ,   315}
                }
            }
        },
        { // Row 6
            {
                site_id_t::kRE3, {
                    {104,   0   ,   90 },
                    {81 ,   75  ,   165},
                    {82 ,   150 ,   240},
                    {83 ,   225 ,   315}
                }
            }
        },
        { // Row 7
            {
                site_id_t::kME4, {
                    {52 ,   0   ,   90 },
                    {36 ,   75  ,   165},
                    {37 ,   150 ,   240},
                    {38 ,   225 ,   315}
                }
            },
            {
                site_id_t::kRE4, {
                    {106,   0   ,   90 },
                    {90 ,   75  ,   165},
                    {91 ,   150 ,   240},
                    {92 ,   225 ,   315}
                }
            }
        },
    };

    std::vector<zones::pattern_t> zone0_prompt_pd = {
        {{49, 55, 61}, {49, 55, 61}, {50, 55, 60}, {53, 55, 57}, {53, 55, 57}, {54, 55, 56}, {54, 55, 56}, {53, 55, 57}},
        {{42, 47, 52}, {42, 47, 52}, {45, 49, 53}, {53, 54, 56}, {53, 55, 57}, {55, 56, 57}, {54, 56, 58}, {54, 56, 59}},
        {{58, 63, 68}, {58, 63, 68}, {57, 61, 65}, {54, 56, 57}, {53, 55, 57}, {53, 54, 55}, {52, 54, 56}, {51, 54, 56}},
        {{35, 42, 49}, {36, 42, 48}, {39, 45, 51}, {52, 54, 56}, {54, 55, 56}, {54, 57, 59}, {54, 57, 60}, {52, 57, 61}},
        {{61, 68, 75}, {62, 68, 74}, {59, 65, 71}, {54, 56, 58}, {54, 55, 56}, {51, 53, 56}, {50, 53, 56}, {49, 53, 58}},
        {{21, 33, 45}, {22, 33, 43}, {29, 39, 49}, {51, 54, 56}, {54, 55, 56}, {52, 57, 62}, {51, 57, 63}, {46, 55, 65}},
        {{65, 77, 89}, {67, 77, 88}, {61, 71, 81}, {54, 56, 59}, {54, 55, 56}, {48, 53, 58}, {47, 53, 59}, {45, 55, 64}}
    };

    zones::quality_lut_t zone0_prompt_ql = {
        0, 3, 3, 4, 3, 5, 4, 6, 1, 6, 6, 7, 21, 26, 24, 27, 1, 21, 21, 24, 22, 26, 24, 29,
        2, 25, 25, 27, 26, 30, 28, 31, 0, 17, 17, 20, 18, 34, 20, 37, 2, 42, 28, 45, 42, 46, 45, 49,
        9, 42, 42, 45, 43, 47, 46, 49, 10, 45, 45, 48, 47, 50, 49, 51, 0, 5, 5, 6, 17, 33, 19, 36,
        2, 7, 7, 7, 29, 46, 30, 49, 3, 42, 29, 46, 43, 47, 46, 50, 4, 45, 31, 49, 47, 50, 50, 51,
        1, 20, 19, 23, 22, 38, 23, 39, 4, 45, 30, 48, 46, 49, 48, 51, 11, 46, 46, 49, 47, 50, 50, 51,
        13, 48, 49, 51, 50, 51, 51, 51, 0, 16, 16, 18, 16, 32, 18, 35, 2, 41, 27, 43, 42, 44, 44, 47,
        9, 42, 41, 44, 42, 45, 44, 48, 10, 44, 44, 47, 44, 48, 48, 50, 8, 40, 40, 41, 40, 41, 40, 43,
        11, 52, 52, 55, 53, 57, 54, 61, 12, 52, 52, 55, 53, 58, 56, 61, 14, 55, 55, 60, 58, 62, 59, 63,
        1, 40, 23, 40, 40, 41, 40, 43, 5, 52, 31, 54, 53, 57, 54, 60, 12, 52, 53, 56, 53, 58, 57, 62,
        14, 55, 56, 60, 57, 61, 60, 63, 8, 41, 41, 43, 41, 43, 43, 47, 13, 54, 54, 59, 57, 61, 58, 63,
        15, 56, 56, 59, 58, 61, 60, 63, 15, 59, 59, 62, 62, 63, 62, 63
    };

    std::vector<zones::pattern_t> zone0_disp_pd = {
        {{50, 55, 60}, {50, 55, 60}, {50, 55, 60}, {53, 55, 57}, {53, 55, 57}, {54, 55, 56}, {53, 55, 57}, {53, 55, 57}},
        {{53, 61, 67}, {53, 61, 65}, {53, 60, 65}, {54, 56, 57}, {54, 56, 57}, {52, 54, 56}, {52, 54, 56}, {49, 53, 56}},
        {{43, 49, 57}, {45, 49, 57}, {45, 50, 57}, {53, 54, 56}, {53, 54, 56}, {54, 56, 58}, {54, 56, 58}, {54, 57, 61}},
        {{54, 63, 72}, {54, 63, 70}, {54, 63, 71}, {54, 57, 58}, {54, 56, 56}, {49, 53, 56}, {49, 52, 56}, {45, 51, 56}},
        {{38, 47, 56}, {40, 47, 56}, {39, 47, 56}, {52, 53, 56}, {54, 54, 56}, {54, 57, 61}, {54, 58, 61}, {54, 59, 65}},
        {{54, 64, 77}, {54, 66, 74}, {54, 66, 76}, {54, 57, 59}, {54, 56, 56}, {46, 52, 56}, {45, 50, 56}, {40, 48, 56}},
        {{33, 46, 56}, {36, 44, 56}, {34, 44, 56}, {51, 53, 56}, {54, 54, 56}, {54, 58, 64}, {54, 60, 65}, {54, 62, 70}}
    };

    zones::quality_lut_t zone0_disp_ql = {
        0, 3, 3, 4, 3, 5, 4, 5, 1, 6, 6, 7, 21, 26, 26, 30, 1, 22, 21, 25, 22, 26, 24, 29,
        2, 24, 24, 27, 25, 29, 29, 31, 0, 17, 17, 20, 18, 34, 20, 38, 2, 42, 28, 47, 42, 47, 46, 50,
        9, 42, 42, 46, 43, 47, 46, 49, 10, 45, 45, 49, 46, 49, 49, 51, 0, 6, 5, 6, 17, 33, 19, 36,
        2, 7, 7, 7, 28, 46, 31, 50, 3, 42, 27, 46, 43, 47, 46, 50, 4, 45, 30, 50, 46, 50, 50, 51,
        1, 20, 19, 23, 21, 37, 23, 39, 4, 45, 30, 48, 45, 49, 48, 51, 11, 45, 45, 49, 47, 50, 49, 51,
        13, 48, 48, 51, 49, 51, 51, 51, 0, 16, 16, 18, 16, 32, 18, 35, 2, 41, 27, 44, 41, 44, 44, 48,
        9, 42, 41, 44, 42, 45, 44, 48, 10, 44, 43, 47, 44, 48, 47, 50, 8, 40, 40, 41, 40, 41, 41, 43,
        11, 52, 52, 56, 53, 57, 57, 61, 12, 53, 52, 57, 53, 58, 58, 62, 14, 56, 55, 60, 57, 62, 61, 63,
        1, 40, 23, 40, 40, 41, 40, 43, 5, 52, 31, 55, 52, 55, 56, 60, 12, 53, 52, 56, 53, 58, 57, 61,
        13, 55, 54, 60, 55, 61, 60, 63, 8, 41, 40, 43, 42, 43, 43, 47, 14, 54, 54, 59, 54, 59, 60, 62,
        15, 56, 54, 59, 58, 62, 61, 63, 15, 59, 58, 62, 59, 63, 63, 63
    };
  // clang-format on

  zones_.push_back({zone0_hm, zone0_prompt_pd, zone0_prompt_ql, zone0_disp_pd, zone0_disp_ql});

  // ===========================================================================
  // Zone 1
  // ===========================================================================

  // clang-format off
    zones::hitmap_t&& zone1_hm = {
        { // Row 0
            {
                site_id_t::kGE11, {
                    {99 ,   38,     90 },
                    {54 ,   75,     127},
                    {55 ,   113,    165},
                    {56 ,   150,    202},
                    {63 ,   188,    240},
                    {64 ,   225,    277},
                    {65 ,   263,    315}
                }
            }
        },
        { // Row 1
            {
                site_id_t::kME11, {
                    {45 ,   38,     90 },
                    {0  ,   75,     127},
                    {1  ,   113,    165},
                    {2  ,   150,    202},
                    {9  ,   188,    240},
                    {10 ,   225,    277},
                    {11 ,   263,    315}
                }
            }
        },
        { // Row 2
            {
                site_id_t::kME12, {
                    {46 ,   38,     90 },
                    {3  ,   75,     127},
                    {4  ,   113,    165},
                    {5  ,   150,    202},
                    {12 ,   188,    240},
                    {13 ,   225,    277},
                    {14 ,   263,    315}
                }
            },
            {
                site_id_t::kRE1, {
                    {100,   38,     90 },
                    {57 ,   75,     127},
                    {58 ,   113,    165},
                    {59 ,   150,    202},
                    {66 ,   188,    240},
                    {67 ,   225,    277},
                    {68 ,   263,    315}
                }
            }
        },
        { // Row 3
            {
                site_id_t::kGE21, {
                    {102,   0   ,   90 },
                    {72 ,   75  ,   165},
                    {73 ,   150 ,   240},
                    {74 ,   225 ,   315}
                }
            }
        },
        { // Row 4
            {
                site_id_t::kME2, {
                    {48 ,   0   ,   90 },
                    {18 ,   75  ,   165},
                    {19 ,   150 ,   240},
                    {20 ,   225 ,   315}
                }
            }
        },
        { // Row 5
            {
                site_id_t::kME3, {
                    // ME3/1
                    {50 ,   0   ,   90 },
                    {27 ,   75  ,   165},
                    {28 ,   150 ,   240},
                    {29 ,   225 ,   315},
                    // ME3/2
                    {51 ,   38,     90 },
                    {30 ,   75,     127},
                    {31 ,   113,    165},
                    {32 ,   150,    202},
                    {33 ,   188,    240},
                    {34 ,   225,    277},
                    {35 ,   263,    315}
                }
            }
        },
        { // Row 6
            {
                site_id_t::kRE3, {
                    // RE3/1
                    {104,   0   ,   90 },
                    {81 ,   75  ,   165},
                    {82 ,   150 ,   240},
                    {83 ,   225 ,   315},
                    // RE3/2
                    {105,   38,     90 },
                    {84 ,   75,     127},
                    {85 ,   113,    165},
                    {86 ,   150,    202},
                    {87 ,   188,    240},
                    {88 ,   225,    277},
                    {89 ,   263,    315}
                }
            }
        },
        { // Row 7
            {
                site_id_t::kME4, {
                    // ME4/1
                    {52 ,   0   ,   90 },
                    {36 ,   75  ,   165},
                    {37 ,   150 ,   240},
                    {38 ,   225 ,   315},
                    // ME4/2
                    {53 ,   38,     90 },
                    {39 ,   75,     127},
                    {40 ,   113,    165},
                    {41 ,   150,    202},
                    {42 ,   188,    240},
                    {43 ,   225,    277},
                    {44 ,   263,    315}
                }
            },        
            {
                site_id_t::kRE4, {
                    // RE4/1
                    {106,   0   ,   90 },
                    {90 ,   75  ,   165},
                    {91 ,   150 ,   240},
                    {92 ,   225 ,   315},
                    // RE4/2
                    {107,   38,     90 },
                    {93 ,   75,     127},
                    {94 ,   113,    165},
                    {95 ,   150,    202},
                    {96 ,   188,    240},
                    {97 ,   225,    277},
                    {98 ,   263,    315}
                }
            }        
        }
    };

    std::vector<zones::pattern_t> zone1_prompt_pd = {
        {{47, 55, 63}, {48, 55, 62}, {51, 55, 59}, {53, 55, 57}, {53, 55, 57}, {54, 55, 56}, {53, 55, 57}, {53, 55, 57}},
        {{38, 44, 50}, {41, 46, 51}, {49, 52, 54}, {53, 54, 56}, {53, 55, 57}, {54, 56, 57}, {54, 56, 58}, {53, 56, 58}},
        {{60, 66, 72}, {59, 64, 69}, {56, 58, 61}, {54, 56, 57}, {53, 55, 57}, {53, 54, 56}, {52, 54, 56}, {52, 54, 57}},
        {{29, 37, 44}, {32, 40, 47}, {46, 50, 53}, {52, 54, 56}, {54, 55, 56}, {53, 56, 59}, {51, 55, 59}, {48, 54, 59}},
        {{66, 73, 81}, {63, 70, 78}, {57, 60, 64}, {54, 56, 58}, {54, 55, 56}, {51, 54, 57}, {51, 55, 59}, {51, 56, 62}},
        {{16, 27, 39}, {21, 32, 42}, {43, 48, 53}, {52, 55, 57}, {54, 55, 56}, {44, 52, 59}, {40, 49, 59}, {31, 44, 59}},
        {{71, 83, 94}, {68, 78, 89}, {57, 62, 67}, {53, 55, 58}, {54, 55, 56}, {51, 58, 66}, {51, 61, 70}, {51, 66, 79}}
    };

    zones::quality_lut_t zone1_prompt_ql = {
        0, 3, 4, 6, 3, 5, 5, 7, 1, 7, 21, 25, 19, 23, 25, 27, 1, 20, 22, 25, 34, 36, 37, 39,
        2, 24, 26, 28, 36, 38, 39, 39, 0, 16, 18, 21, 32, 33, 34, 37, 4, 27, 42, 46, 42, 45, 45, 49,
        9, 42, 43, 46, 43, 46, 47, 50, 10, 45, 46, 50, 45, 48, 49, 51, 0, 7, 17, 20, 17, 18, 20, 23,
        3, 7, 28, 31, 27, 29, 30, 31, 4, 29, 43, 47, 42, 45, 47, 50, 6, 30, 46, 50, 46, 48, 49, 51,
        1, 19, 22, 24, 34, 35, 37, 38, 5, 29, 46, 49, 45, 48, 49, 51, 11, 46, 47, 50, 47, 49, 50, 51,
        13, 49, 50, 51, 48, 51, 51, 51, 0, 16, 16, 18, 32, 32, 33, 35, 2, 26, 42, 44, 41, 43, 44, 48,
        9, 41, 42, 45, 42, 44, 45, 48, 10, 44, 44, 48, 44, 47, 48, 50, 8, 40, 40, 41, 40, 40, 41, 43,
        11, 52, 53, 56, 52, 54, 56, 60, 12, 52, 53, 57, 53, 56, 58, 62, 14, 55, 57, 61, 57, 59, 61, 63,
        2, 22, 40, 41, 40, 40, 41, 43, 6, 31, 52, 55, 52, 54, 55, 59, 12, 52, 53, 57, 53, 54, 58, 61,
        14, 54, 57, 61, 55, 59, 60, 63, 8, 40, 41, 43, 41, 43, 44, 47, 13, 54, 56, 60, 56, 58, 60, 62,
        15, 55, 58, 62, 58, 59, 62, 63, 15, 59, 61, 63, 60, 62, 63, 63
    };

    std::vector<zones::pattern_t> zone1_disp_pd = {
        {{50, 55, 60}, {50, 55, 60}, {52, 55, 58}, {53, 55, 57}, {53, 55, 57}, {54, 55, 56}, {53, 55, 57}, {53, 55, 57}},
        {{53, 60, 65}, {53, 60, 65}, {54, 58, 61}, {54, 56, 57}, {54, 56, 57}, {53, 54, 56}, {52, 54, 56}, {50, 53, 56}},
        {{45, 50, 57}, {45, 50, 57}, {49, 52, 56}, {53, 54, 56}, {53, 54, 56}, {54, 56, 57}, {54, 56, 58}, {54, 57, 60}},
        {{53, 63, 71}, {53, 62, 69}, {54, 59, 63}, {54, 57, 58}, {54, 56, 56}, {50, 54, 56}, {50, 53, 56}, {47, 51, 56}},
        {{39, 47, 57}, {41, 48, 57}, {47, 51, 56}, {52, 53, 56}, {54, 54, 56}, {54, 56, 60}, {54, 57, 60}, {54, 59, 63}},
        {{54, 65, 73}, {54, 65, 74}, {54, 61, 67}, {54, 57, 59}, {54, 56, 56}, {47, 53, 56}, {47, 51, 56}, {43, 49, 56}},
        {{37, 45, 56}, {36, 45, 56}, {43, 49, 56}, {51, 53, 56}, {54, 54, 56}, {54, 57, 63}, {54, 59, 63}, {54, 61, 67}}
    };

    zones::quality_lut_t zone1_disp_ql = {
        0, 3, 4, 6, 4, 5, 6, 7, 1, 7, 22, 26, 20, 24, 25, 29, 1, 21, 22, 25, 34, 37, 37, 39,
        2, 23, 25, 28, 36, 38, 39, 39, 0, 17, 18, 20, 32, 33, 34, 37, 3, 27, 42, 46, 42, 45, 46, 50,
        9, 42, 43, 46, 43, 46, 47, 50, 10, 45, 46, 50, 45, 48, 49, 51, 0, 7, 17, 18, 16, 19, 20, 24,
        3, 7, 27, 31, 27, 30, 29, 31, 4, 28, 43, 47, 42, 47, 47, 50, 5, 30, 46, 50, 45, 49, 49, 51,
        1, 19, 21, 23, 34, 35, 36, 38, 5, 29, 45, 49, 45, 49, 48, 51, 11, 46, 47, 50, 46, 48, 50, 51,
        13, 49, 49, 51, 48, 51, 51, 51, 0, 16, 16, 18, 32, 32, 33, 35, 2, 26, 41, 44, 41, 44, 44, 48,
        9, 42, 42, 45, 42, 44, 45, 48, 10, 44, 44, 48, 44, 48, 47, 50, 8, 40, 40, 41, 40, 41, 41, 43,
        11, 52, 52, 57, 52, 55, 56, 61, 12, 53, 53, 57, 53, 57, 58, 62, 14, 55, 56, 61, 55, 61, 60, 63,
        2, 22, 40, 40, 40, 40, 41, 43, 6, 31, 52, 57, 52, 54, 55, 60, 12, 52, 53, 58, 53, 56, 58, 62,
        14, 56, 56, 61, 54, 59, 60, 63, 8, 40, 41, 43, 41, 43, 43, 47, 13, 54, 54, 60, 54, 58, 59, 63,
        15, 55, 57, 61, 58, 60, 62, 63, 15, 59, 59, 63, 59, 62, 62, 63
    };
  // clang-format on

  zones_.push_back({zone1_hm, zone1_prompt_pd, zone1_prompt_ql, zone1_disp_pd, zone1_disp_ql});

  // ===========================================================================
  // Zone 2
  // ===========================================================================

  // clang-format off
    zones::hitmap_t&& zone2_hm = {
        { // Row 0
            {
                site_id_t::kME12, {
                    {46 ,   38,     90 },
                    {3  ,   75,     127},
                    {4  ,   113,    165},
                    {5  ,   150,    202},
                    {12 ,   188,    240},
                    {13 ,   225,    277},
                    {14 ,   263,    315}
                }
            }
        },
        { // Row 1
            {
                site_id_t::kRE1, {
                    {100,   38,     90 },
                    {57 ,   75,     127},
                    {58 ,   113,    165},
                    {59 ,   150,    202},
                    {66 ,   188,    240},
                    {67 ,   225,    277},
                    {68 ,   263,    315}
                }
            }
        },
        { // Row 2
            {
                site_id_t::kRE2, {
                    {103,   38,     90 },
                    {75 ,   75,     127},
                    {76 ,   113,    165},
                    {77 ,   150,    202},
                    {78 ,   188,    240},
                    {79 ,   225,    277},
                    {80 ,   263,    315}
                }
            }
        },
        { // Row 3
            {
                site_id_t::kME2, {
                    {49 ,   38,     90 },
                    {21 ,   75,     127},
                    {22 ,   113,    165},
                    {23 ,   150,    202},
                    {24 ,   188,    240},
                    {25 ,   225,    277},
                    {26 ,   263,    315}
                }
            }
        },
        { // Row 4
            {
                site_id_t::kME3, {
                    {51 ,   38,     90 },
                    {30 ,   75,     127},
                    {31 ,   113,    165},
                    {32 ,   150,    202},
                    {33 ,   188,    240},
                    {34 ,   225,    277},
                    {35 ,   263,    315}
                }
            }
        },
        { // Row 5
            {
                site_id_t::kRE3, {
                    {105,   38,     90 },
                    {84 ,   75,     127},
                    {85 ,   113,    165},
                    {86 ,   150,    202},
                    {87 ,   188,    240},
                    {88 ,   225,    277},
                    {89 ,   263,    315}
                }
            }
        },
        { // Row 6
            {
                site_id_t::kME4, {
                    {53 ,   38,     90 },
                    {39 ,   75,     127},
                    {40 ,   113,    165},
                    {41 ,   150,    202},
                    {42 ,   188,    240},
                    {43 ,   225,    277},
                    {44 ,   263,    315}
                }
            }
        },
        { // Row 7
            {
                site_id_t::kRE4, {
                    {107,   38,     90 },
                    {93 ,   75,     127},
                    {94 ,   113,    165},
                    {95 ,   150,    202},
                    {96 ,   188,    240},
                    {97 ,   225,    277},
                    {98 ,   263,    315},
                }
            }
        }
    };

    std::vector<zones::pattern_t> zone2_prompt_pd = { // Pattern N: Row0..RowM
        {{52, 55, 58}, {52, 55, 58}, {53, 55, 57}, {53, 55, 57}, {54, 55, 56}, {53, 55, 57}, {53, 55, 57}, {53, 55, 57}},
        {{54, 57, 61}, {53, 57, 60}, {54, 56, 58}, {54, 56, 58}, {53, 55, 56}, {53, 54, 56}, {52, 54, 56}, {52, 54, 56}},
        {{49, 53, 56}, {50, 53, 57}, {52, 54, 56}, {52, 54, 56}, {54, 55, 57}, {54, 56, 57}, {54, 56, 58}, {54, 56, 58}},
        {{54, 57, 61}, {54, 57, 61}, {54, 56, 59}, {54, 56, 59}, {53, 54, 56}, {52, 54, 56}, {51, 54, 56}, {51, 53, 56}},
        {{49, 53, 56}, {49, 53, 56}, {51, 54, 56}, {51, 54, 56}, {54, 56, 57}, {54, 56, 58}, {54, 56, 59}, {54, 57, 59}},
        {{54, 59, 65}, {54, 60, 64}, {54, 57, 59}, {54, 56, 56}, {50, 54, 56}, {49, 53, 56}, {47, 52, 56}, {46, 51, 56}},
        {{45, 51, 56}, {46, 50, 56}, {51, 53, 56}, {54, 54, 56}, {54, 56, 60}, {54, 57, 61}, {54, 58, 63}, {54, 59, 64}}
    };

    zones::quality_lut_t zone2_prompt_ql = {
        0, 3, 3, 5, 1, 23, 7, 26, 1, 36, 22, 38, 2, 39, 25, 39, 0, 32, 18, 35, 4, 44, 44, 48,
        9, 44, 44, 49, 10, 49, 48, 51, 0, 17, 6, 21, 3, 28, 7, 30, 4, 44, 43, 48, 5, 48, 48, 51,
        1, 34, 21, 37, 4, 47, 47, 50, 10, 49, 48, 51, 11, 51, 50, 51, 0, 32, 16, 33, 3, 43, 43, 47,
        8, 43, 43, 47, 9, 47, 46, 50, 8, 40, 40, 42, 11, 53, 53, 57, 11, 54, 53, 58, 13, 58, 56, 62,
        2, 40, 40, 41, 5, 52, 52, 54, 11, 53, 52, 57, 12, 58, 54, 61, 8, 42, 42, 45, 12, 56, 55, 59,
        14, 57, 56, 61, 15, 62, 59, 63, 0, 16, 5, 19, 2, 27, 7, 29, 3, 43, 42, 46, 4, 46, 45, 50,
        1, 40, 40, 41, 6, 52, 52, 54, 10, 53, 52, 58, 12, 55, 54, 60, 1, 24, 7, 26, 5, 31, 7, 31,
        6, 53, 52, 57, 6, 58, 54, 60, 2, 41, 41, 45, 7, 55, 55, 59, 13, 56, 56, 61, 14, 60, 59, 63,
        0, 34, 20, 37, 4, 46, 46, 49, 9, 47, 46, 50, 10, 50, 49, 51, 8, 42, 42, 45, 12, 57, 56, 61,
        13, 57, 55, 62, 15, 62, 59, 63, 2, 41, 41, 44, 6, 55, 54, 59, 13, 58, 56, 61, 14, 61, 59, 63,
        9, 45, 45, 49, 14, 60, 60, 62, 15, 61, 60, 63, 15, 63, 62, 63
    };

    std::vector<zones::pattern_t> zone2_disp_pd = { // Pattern N: Row0..RowM
        {{52, 55, 58}, {52, 55, 58}, {53, 55, 57}, {53, 55, 57}, {54, 55, 56}, {53, 55, 57}, {53, 55, 57}, {53, 55, 57}},
        {{54, 57, 61}, {54, 57, 61}, {54, 56, 59}, {54, 56, 59}, {53, 55, 56}, {53, 54, 56}, {52, 54, 56}, {51, 54, 56}},
        {{49, 53, 56}, {49, 53, 56}, {51, 54, 56}, {51, 54, 56}, {54, 55, 57}, {54, 56, 57}, {54, 56, 58}, {54, 56, 59}},
        {{54, 58, 62}, {54, 58, 62}, {54, 56, 58}, {54, 56, 57}, {51, 54, 56}, {51, 53, 56}, {49, 53, 56}, {48, 52, 56}},
        {{48, 52, 56}, {48, 52, 56}, {52, 54, 56}, {53, 54, 56}, {54, 56, 59}, {54, 57, 59}, {54, 57, 61}, {54, 58, 62}},
        {{54, 60, 66}, {54, 60, 65}, {54, 57, 59}, {54, 56, 56}, {49, 53, 56}, {48, 52, 56}, {46, 52, 56}, {45, 51, 56}},
        {{44, 50, 56}, {45, 50, 56}, {51, 53, 56}, {54, 54, 56}, {54, 57, 61}, {54, 58, 62}, {54, 58, 64}, {54, 59, 65}}
    };

    zones::quality_lut_t zone2_disp_ql = {
        0, 3, 3, 5, 1, 23, 7, 26, 1, 36, 22, 38, 2, 39, 25, 39, 0, 32, 18, 35, 4, 44, 44, 48,
        9, 44, 44, 49, 10, 49, 48, 51, 0, 17, 6, 21, 3, 28, 7, 30, 4, 44, 43, 48, 5, 48, 48, 51,
        1, 34, 21, 37, 4, 47, 48, 50, 10, 49, 47, 51, 11, 51, 50, 51, 0, 32, 16, 33, 3, 43, 43, 47,
        8, 43, 43, 47, 9, 47, 46, 50, 8, 40, 40, 42, 11, 53, 52, 57, 11, 53, 52, 58, 13, 58, 56, 61,
        2, 40, 40, 41, 5, 52, 52, 55, 11, 53, 53, 57, 12, 56, 55, 61, 8, 42, 41, 45, 12, 56, 54, 59,
        14, 58, 56, 62, 15, 61, 59, 63, 0, 16, 6, 20, 2, 27, 7, 29, 3, 43, 42, 46, 4, 46, 46, 50,
        1, 40, 40, 41, 5, 53, 52, 54, 10, 53, 52, 58, 12, 55, 54, 60, 1, 24, 7, 26, 5, 31, 7, 31,
        6, 53, 52, 57, 6, 57, 54, 61, 2, 42, 41, 45, 7, 54, 55, 59, 13, 58, 55, 60, 14, 60, 59, 63,
        0, 34, 19, 37, 4, 46, 45, 49, 9, 47, 46, 50, 10, 50, 49, 51, 8, 42, 42, 45, 12, 57, 56, 61,
        13, 57, 56, 62, 15, 62, 60, 63, 2, 41, 41, 44, 6, 56, 54, 59, 13, 58, 55, 61, 14, 61, 59, 63,
        9, 45, 45, 49, 14, 60, 59, 62, 15, 62, 60, 63, 15, 63, 62, 63
    };
  // clang-format on

  zones_.push_back({zone2_hm, zone2_prompt_pd, zone2_prompt_ql, zone2_disp_pd, zone2_disp_ql});

  // ===========================================================================
  // Features
  // ===========================================================================
  // feat       | ME1/1 | ME1/2 |  ME2  |  ME3  |  ME4  |  RE1  |  RE2  |  RE3  |  RE4  | GE1/1 | GE2/1 |  ME0
  // -----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------
  // emtf_phi   |   *   |   *   |   *   |   *   |   *   |   *   |   *   |   *   |   *   |   *   |   *   |   *
  // emtf_theta |   *   |   *   |   *   |   *   |   *   |   *   |   *   |   *   |   *   |   *   |   *   |   *
  // emtf_bend  |   *   |   *   |   *   |   *   |   *   |       |       |       |       |       |       |   *
  // emtf_qual  |   *   |   *   |   *   |   *   |   *   |       |       |       |       |       |       |   *
  // emtf_time  |       |       |       |       |       |       |       |       |       |       |       |

  // clang-format off
    features_ = {
        {
            feature_id_t::kPhi, {
                site_id_t::kME11, site_id_t::kME12, site_id_t::kME2, site_id_t::kME3, site_id_t::kME4,
                site_id_t::kRE1 , site_id_t::kRE2 , site_id_t::kRE3, site_id_t::kRE4,
                site_id_t::kGE11, site_id_t::kGE21, site_id_t::kME0
            }
        }, 
        {
            feature_id_t::kTheta, {
                site_id_t::kME11, site_id_t::kME12, site_id_t::kME2, site_id_t::kME3, site_id_t::kME4,
                site_id_t::kRE1 , site_id_t::kRE2 , site_id_t::kRE3, site_id_t::kRE4,
                site_id_t::kGE11, site_id_t::kGE21, site_id_t::kME0
            }
        }, 
        {
            feature_id_t::kBend, {
                site_id_t::kME11, site_id_t::kME12, site_id_t::kME2, site_id_t::kME3, site_id_t::kME4,
                site_id_t::kME0
            }
        }, 
        {
            feature_id_t::kQuality, {
                site_id_t::kME11, site_id_t::kME12, site_id_t::kME2, site_id_t::kME3, site_id_t::kME4,
                site_id_t::kME0
            }
        }, 
    };
  // clang-format on

  // ===========================================================================
  // Theta Options
  // ===========================================================================

  // clang-format off
    theta_medians_ = {
        // ME2_t1, ME3_t1, ME4_t1, ME2_t2, ME3_t2, ME4_t2, GE21, RE3, RE4
        {
            {
                {site_id_t::kME2, theta_id_t::kTheta1},
                {site_id_t::kME3, theta_id_t::kTheta1},
                {site_id_t::kME4, theta_id_t::kTheta1}
            },
            {
                {site_id_t::kME2, theta_id_t::kTheta2},
                {site_id_t::kME3, theta_id_t::kTheta2},
                {site_id_t::kME4, theta_id_t::kTheta2}
            },
            {
                {site_id_t::kGE21, theta_id_t::kTheta1},
                {site_id_t::kRE3, theta_id_t::kTheta1},
                {site_id_t::kRE4, theta_id_t::kTheta1}
            },
        },
        // ME2_t1, ME3_t1, ME4_t1, ME2_t2, ME3_t2, ME4_t2, RE2, RE3, RE4
        {
            {
                {site_id_t::kME2, theta_id_t::kTheta1},
                {site_id_t::kME3, theta_id_t::kTheta1},
                {site_id_t::kME4, theta_id_t::kTheta1}
            },
            {
                {site_id_t::kME2, theta_id_t::kTheta2},
                {site_id_t::kME3, theta_id_t::kTheta2},
                {site_id_t::kME4, theta_id_t::kTheta2}
            },
            {
                {site_id_t::kRE2, theta_id_t::kTheta1},
                {site_id_t::kRE3, theta_id_t::kTheta1},
                {site_id_t::kRE4, theta_id_t::kTheta1}
            },
        },
        // ME12_t1, ME11_t1, ME0_t2, ME12_t2, ME11_t2, ME0_t2, RE1, GE11, ME0_t1
        {
            {
                {site_id_t::kME12, theta_id_t::kTheta1},
                {site_id_t::kME11, theta_id_t::kTheta1},
                {site_id_t::kME0 , theta_id_t::kTheta2}
            },
            {
                {site_id_t::kME12, theta_id_t::kTheta2},
                {site_id_t::kME11, theta_id_t::kTheta2},
                {site_id_t::kME0 , theta_id_t::kTheta2}
            },
            {
                {site_id_t::kRE1, theta_id_t::kTheta1},
                {site_id_t::kGE11, theta_id_t::kTheta1},
                {site_id_t::kME0, theta_id_t::kTheta1}
            },
        },
    };
  // clang-format on

  // ===========================================================================
  // Site Reduction
  // ===========================================================================
  // Site (out) | Site (in)
  // -----------|-------------------------------------------
  // ME1        | ME1/1, GE1/1, ME1/2, RE1/2
  // ME2        | ME2, GE2/1, RE2/2
  // ME3        | ME3, RE3
  // ME4        | ME4, RE4
  // ME0        | ME0

  // clang-format off
    reduced_sites_ = {
        {reduced_site_id_t::kME1, {site_id_t::kME11, site_id_t::kGE11, site_id_t::kME12, site_id_t::kRE1}},
        {reduced_site_id_t::kME2, {site_id_t::kME2, site_id_t::kGE21, site_id_t::kRE2}},
        {reduced_site_id_t::kME3, {site_id_t::kME3, site_id_t::kRE3}},
        {reduced_site_id_t::kME4, {site_id_t::kME4, site_id_t::kRE4}},
        {reduced_site_id_t::kME0, {site_id_t::kME0}},
    };
  // clang-format on
}

EMTFModel::~EMTFModel() {
  // Do Nothing
}
