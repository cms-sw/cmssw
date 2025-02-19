from TestClasses import *

suite = []
M='ProfileLikelihood'
suite += [ (M, '*', MultiDatacardTest("Counting",     datacardGlob("simple-counting/counting-B5p5-Obs[16]*.txt"),M,""))]
suite += [ (M, '*', MultiDatacardTest("Counting_Sig", datacardGlob("simple-counting/counting-B5p5-Obs[12][10]-S*.txt"),M,"--signif"))]

suite += [ (M, '*', MultiDatacardTest("Asymm",      datacardGlob("asymm/counting-B5p5-Obs6*txt"),  M, "")) ]
suite += [ (M, '*', MultiDatacardTest("Asymm_Sig",  datacardGlob("asymm/counting-B5p5-Obs20*txt"), M, "--signif")) ]

suite += [ (M, '*', MultiDatacardTest("Gamma",     datacardGlob("gammas/counting-*.txt"), M, "")) ]
suite += [ (M, '*', MultiDatacardTest("Gamma_Sig", [ "gammas/counting-S1-Sideband1-alpha0p3-Obs2.txt",
                                                     "gammas/counting-S1-Sideband2-alpha0p3-Obs6.txt",
                                                     "gammas/counting-Sideband11-alpha0p5-Obs11.txt",
                                                     "gammas/counting-Sideband11-alpha0p5-Obs20.txt"], M, "--signif")) ]

suite += [ (M, '*', MultiDatacardTest("HWW",         datacardGlob("hww4ch-1fb-B*.txt"), M,'')) ]
suite += [ (M, '*', MultiDatacardTest("HWW_S0",      datacardGlob("hww4ch-1fb-B*.txt"), M,'-S 0')) ]
suite += [ (M, '*', MultiDatacardTest("HWW_Sig",     datacardGlob("hww4ch-1fb-S*.txt"), M,'--signif')) ]
suite += [ (M, '*', MultiDatacardTest("HWW_Sig_S0",  datacardGlob("hww4ch-1fb-S*.txt"), M,'-S 0  --signif')) ]

suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HGG_115",      "summer11/hgg/hgg_8cats.txt",            M, '', 115)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWC_170_CNT", "summer11/hwwc.170/comb_hww_cnt.txt",    M, '', 170)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWC_170_EXT", "summer11/hwwc.170/comb_hww_ext.txt",    M, '', 170)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWS_130",     "summer11/hwws.130/comb_hww.txt",        M, '', 130)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWS_130_0J",  "summer11/hwws.130/comb_hww0j.txt",      M, '', 130)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HZZ4L_145",    "summer11/hzz4l.145/comb_hzz4l.txt",     M, '', 145)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HZZ2L2Q_300",  "summer11/hzz2l2q.300/comb_hzz2l2q.txt", M, '', 300)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HTT_125",      "summer11/htt.125/comb_htt.txt",         M, '', 125)) ]

suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HGG_115_PLC",      "summer11/hgg/hgg_8cats.txt",            M, '--usePLC', 115)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWC_170_CNT_PLC", "summer11/hwwc.170/comb_hww_cnt.txt",    M, '--usePLC', 170)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWC_170_EXT_PLC", "summer11/hwwc.170/comb_hww_ext.txt",    M, '--usePLC', 170)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWS_130_PLC",     "summer11/hwws.130/comb_hww.txt",        M, '--usePLC', 130)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWS_130_0J_PLC",  "summer11/hwws.130/comb_hww0j.txt",      M, '--usePLC', 130)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HZZ4L_145_PLC",    "summer11/hzz4l.145/comb_hzz4l.txt",     M, '--usePLC', 145)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HZZ2L2Q_300_PLC",  "summer11/hzz2l2q.300/comb_hzz2l2q.txt", M, '--usePLC', 300)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HTT_125_PLC",      "summer11/htt.125/comb_htt.txt",         M, '--usePLC', 125)) ]


suite += [ (M, 'summer11', SingleDatacardTest("Summer11_Sig_HGG_115",      "summer11/hgg/hgg_8cats.txt",            M, '--signif', 115)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_Sig_HWWC_170_CNT", "summer11/hwwc.170/comb_hww_cnt.txt",    M, '--signif', 170)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_Sig_HWWC_170_EXT", "summer11/hwwc.170/comb_hww_ext.txt",    M, '--signif', 170)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_Sig_HWWS_130",     "summer11/hwws.130/comb_hww.txt",        M, '--signif', 130)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_Sig_HWWS_130_0J",  "summer11/hwws.130/comb_hww0j.txt",      M, '--signif', 130)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_Sig_HZZ4L_145",    "summer11/hzz4l.145/comb_hzz4l.txt",     M, '--signif', 145)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_Sig_HZZ2L2Q_300",  "summer11/hzz2l2q.300/comb_hzz2l2q.txt", M, '--signif', 300)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_Sig_HTT_125",      "summer11/htt.125/comb_htt.txt",         M, '--signif', 125)) ]

suite += [ (M, 'summer11', SingleDatacardTest("Summer11_Sig_HGG_115_PLC",      "summer11/hgg/hgg_8cats.txt",            M, '--signif --usePLC', 115)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_Sig_HWWC_170_CNT_PLC", "summer11/hwwc.170/comb_hww_cnt.txt",    M, '--signif --usePLC', 170)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_Sig_HWWC_170_EXT_PLC", "summer11/hwwc.170/comb_hww_ext.txt",    M, '--signif --usePLC', 170)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_Sig_HWWS_130_PLC",     "summer11/hwws.130/comb_hww.txt",        M, '--signif --usePLC', 130)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_Sig_HWWS_130_0J_PLC",  "summer11/hwws.130/comb_hww0j.txt",      M, '--signif --usePLC', 130)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_Sig_HZZ4L_145_PLC",    "summer11/hzz4l.145/comb_hzz4l.txt",     M, '--signif --usePLC', 145)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_Sig_HZZ2L2Q_300_PLC",  "summer11/hzz2l2q.300/comb_hzz2l2q.txt", M, '--signif --usePLC', 300)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_Sig_HTT_125_PLC",      "summer11/htt.125/comb_htt.txt",         M, '--signif --usePLC', 125)) ]



suite += [ (M, 'complex', SingleDatacardTest("Binary_ZZ_mh140",     "binary/toy-ZZ-mh140.root",     M, '')) ]
suite += [ (M, 'complex', SingleDatacardTest("Binary_WW_mh140",     "binary/toy-WW-mh140.root",     M, '')) ]
suite += [ (M, 'complex', SingleDatacardTest("Binary_gg_mh140",     "binary/toy-gg-mh140.root",     M, '')) ]
suite += [ (M, 'complex', SingleDatacardTest("Binary_WWZZgg_mh140", "binary/toy-WWZZgg-mh140.root", M, '')) ]
