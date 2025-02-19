from TestClasses import *

suite = []
M='MarkovChainMC'
suite += [ (M, 'fast', MultiDatacardTest("Counting", datacardGlob("simple-counting/counting-B5p5-Obs6*.txt"),M,"-H ProfileLikelihood")) ]
suite += [ (M, 'full', MultiDatacardTest("Counting", datacardGlob("simple-counting/counting-B5p5-Obs[16]*.txt"),M,"-H ProfileLikelihood --tries 50"))]

suite += [ (M, 'fast', SingleDatacardTest("Gamma", "gammas/counting-S1-Sideband10-alpha0p2-Obs2.txt", M, "-H ProfileLikelihood")) ]
suite += [ (M, 'full', SingleDatacardTest("Gamma", "gammas/counting-S1-Sideband10-alpha0p2-Obs2.txt", M, "-H ProfileLikelihood --tries 50")) ]

suite += [ (M, 'fast', MultiDatacardTest("HWW",     datacardGlob("hww4ch-1fb-B-mH1[46]0.txt"), M,'-H ProfileLikelihood      --tries 20')) ]
suite += [ (M, 'fast', MultiDatacardTest("HWW_S0",  datacardGlob("hww4ch-1fb-B-mH1[46]0.txt"), M,'-H ProfileLikelihood -S 0 --tries 50')) ]
for d in datacardGlob("hww4ch-1fb-B*mH*.txt"):
    n = re.sub("hww4ch-1fb-(.*mH..0).txt","\\1",d)
    suite += [ (M, 'full',  SingleDatacardTest('HWW_'+n, d, M,'-H ProfileLikelihood --tries 200 -i 20000')) ]

suite += [ ('MarkovChainMC', 'fast', 
            MultiOptionTest("Proposals", "simple-counting/counting-B5p5-Obs6-Syst30B.txt", "MarkovChainMC", "--tries 20",
                            { 'Uniform':'--proposal=uniform', 'Gaus':'--proposal=gaus', 'Ortho':'--proposal=ortho'} )) ]
suite += [ ('MarkovChainMC', 'full', 
            MultiOptionTest("Proposals", "simple-counting/counting-B5p5-Obs6-Syst30U.txt", "MarkovChainMC", "--tries 100 -i 20000",
                            { 'Uniform':'--proposal=uniform', 'Gaus':'--proposal=gaus', 'Ortho':'--proposal=ortho'} )) ]

opts  = "--tries 4  -i  50000"
optsB = "--tries 10 -i 100000"
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HGG_115",      "summer11/hgg/hgg_8cats.txt",            M, opts,  115)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWC_170_CNT", "summer11/hwwc.170/comb_hww_cnt.txt",    M, optsB, 170)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWC_170_EXT", "summer11/hwwc.170/comb_hww_ext.txt",    M, optsB, 170)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWS_130",     "summer11/hwws.130/comb_hww.txt",        M, optsB, 130)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWS_130_0J",  "summer11/hwws.130/comb_hww0j.txt",      M, optsB, 130)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HZZ4L_145",    "summer11/hzz4l.145/comb_hzz4l.txt",     M, opts,  145)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HZZ2L2Q_300",  "summer11/hzz2l2q.300/comb_hzz2l2q.txt", M, opts,  300)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HTT_125",      "summer11/htt.125/comb_htt.txt",         M, optsB, 125)) ]


