from TestClasses import *

suite = []
M='BayesianToyMC'

opts="-H ProfileLikelihood -i 20 --tries 5"
suite += [ (M, '*', MultiDatacardTest("Counting", datacardGlob("simple-counting/counting-B5p5-Obs[16]*.txt"),M, opts))]
suite += [ (M, '*', MultiDatacardTest("Gamma",    datacardGlob("gammas/counting-*.txt"),                     M, opts)) ]

opts="-H ProfileLikelihood -i 250 --tries 5"
suite += [ (M, '*', MultiDatacardTest("HWW",      datacardGlob("hww4ch-1fb-B-mH1[46]0.txt"), M, opts)) ]

opts="-H ProfileLikelihood -i 250 --tries 5"
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HGG_115",      "summer11/hgg/hgg_8cats.txt",            M, opts, 115)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWC_170_CNT", "summer11/hwwc.170/comb_hww_cnt.txt",    M, opts, 170)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWC_170_EXT", "summer11/hwwc.170/comb_hww_ext.txt",    M, opts, 170)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWS_130",     "summer11/hwws.130/comb_hww.txt",        M, opts, 130)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWS_130_0J",  "summer11/hwws.130/comb_hww0j.txt",      M, opts, 130)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HZZ4L_145",    "summer11/hzz4l.145/comb_hzz4l.txt",     M, opts, 145)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HZZ2L2Q_300",  "summer11/hzz2l2q.300/comb_hzz2l2q.txt", M, opts, 300)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HTT_125",      "summer11/htt.125/comb_htt.txt",         M, opts, 125)) ]
