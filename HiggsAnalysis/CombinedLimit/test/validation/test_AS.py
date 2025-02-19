from TestClasses import *

suite = []
M='Asymptotic'
suite += [ (M, '*', MultiDatacardWithExpectedTest("Counting",     datacardGlob("simple-counting/counting-B5p5-Obs[16]*.txt"),M,""))]

suite += [ (M, '*', MultiDatacardWithExpectedTest("Asymm",      datacardGlob("asymm/counting-B5p5-Obs6*txt"),  M, "")) ]

suite += [ (M, '*', MultiDatacardWithExpectedTest("Gamma",     datacardGlob("gammas/counting-*.txt"), M, "")) ]

suite += [ (M, '*', MultiDatacardWithExpectedTest("HWW",      datacardGlob("hww4ch-1fb-B*.txt"), M,'')) ]
suite += [ (M, '*', MultiDatacardWithExpectedTest("HWW_S0",   datacardGlob("hww4ch-1fb-B*.txt"), M,'-S 0')) ]

suite += [ (M, 'summer11', SingleDatacardWithExpectedTest("Summer11_HGG_115",      "summer11/hgg/hgg_8cats.txt",            M, '', 115)) ]
suite += [ (M, 'summer11', SingleDatacardWithExpectedTest("Summer11_HWWC_170_CNT", "summer11/hwwc.170/comb_hww_cnt.txt",    M, '', 170)) ]
suite += [ (M, 'summer11', SingleDatacardWithExpectedTest("Summer11_HWWC_170_EXT", "summer11/hwwc.170/comb_hww_ext.txt",    M, '', 170)) ]
suite += [ (M, 'summer11', SingleDatacardWithExpectedTest("Summer11_HWWS_130",     "summer11/hwws.130/comb_hww.txt",        M, '', 130)) ]
suite += [ (M, 'summer11', SingleDatacardWithExpectedTest("Summer11_HWWS_130_0J",  "summer11/hwws.130/comb_hww0j.txt",      M, '', 130)) ]
suite += [ (M, 'summer11', SingleDatacardWithExpectedTest("Summer11_HZZ4L_145",    "summer11/hzz4l.145/comb_hzz4l.txt",     M, '', 145)) ]
suite += [ (M, 'summer11', SingleDatacardWithExpectedTest("Summer11_HZZ2L2Q_300",  "summer11/hzz2l2q.300/comb_hzz2l2q.txt", M, '', 300)) ]
suite += [ (M, 'summer11', SingleDatacardWithExpectedTest("Summer11_HTT_125",      "summer11/htt.125/comb_htt.txt",         M, '', 125)) ]

suite += [ (M, 'complex', SingleDatacardWithExpectedTest("Binary_ZZ_mh140",     "binary/toy-ZZ-mh140.root",     M, '')) ]
suite += [ (M, 'complex', SingleDatacardWithExpectedTest("Binary_WW_mh140",     "binary/toy-WW-mh140.root",     M, '')) ]
suite += [ (M, 'complex', SingleDatacardWithExpectedTest("Binary_gg_mh140",     "binary/toy-gg-mh140.root",     M, '')) ]
suite += [ (M, 'complex', SingleDatacardWithExpectedTest("Binary_WWZZgg_mh140", "binary/toy-WWZZgg-mh140.root", M, '')) ]
