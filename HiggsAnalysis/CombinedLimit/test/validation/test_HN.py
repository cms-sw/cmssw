from TestClasses import *

suite = []
M='HybridNew'

### Test the test statistics
suite += [ (M, '*', MultiOptionTest("Counting_TestStats", "simple-counting/counting-B5p5-Obs6-Syst30B.txt", M,
                "--singlePoint 5 --onlyTestStat",
                {'Atlas':'--testStat=Atlas', 'LEP':'--testStat=LEP', 'TEV':'--testStat=TEV'})) ]
suite += [ (M, '*', MultiOptionTest("Counting_TestStats_BelowB", "simple-counting/counting-B5p5-Obs1-Syst30B.txt", M,
                "--singlePoint 5 --onlyTestStat",
                {'Atlas':'--testStat=Atlas', 'LEP':'--testStat=LEP', 'TEV':'--testStat=TEV'})) ]
suite += [ (M, '*', MultiOptionTest("Counting_TestStats_AboveB", "simple-counting/counting-B5p5-Obs11-Syst30B.txt", M,
                "--singlePoint 10 --onlyTestStat",
                {'Atlas':'--testStat=Atlas', 'LEP':'--testStat=LEP', 'TEV':'--testStat=TEV', 'Profile':'--testStat=Profile'})) ]
suite += [ (M, '*', MultiOptionTest("Counting_TestStats_AboveSB", "simple-counting/counting-B5p5-Obs11-Syst30B.txt", M,
                "--singlePoint 2 --onlyTestStat",
                {'Atlas':'--testStat=Atlas', 'LEP':'--testStat=LEP', 'TEV':'--testStat=TEV', 'Profile':'--testStat=Profile'})) ]

suite += [ (M, '*', MultiOptionTest("HWW_TestStats", "hww4ch-1fb-B-mH140.txt", M,
                "--singlePoint 1 --onlyTestStat",
                {'Atlas':'--testStat=Atlas', 'LEP':'--testStat=LEP', 'TEV':'--testStat=TEV'})) ]

### Test the p-values 
for R in [ 'CLs', 'CLsplusb' ]:
    suite += [ (M, 'fast', MultiOptionTest("Counting_pValues_%s" % R, "simple-counting/counting-B5p5-Obs6-Syst30B.txt", M,
                            "--singlePoint 5 --fork 2 --clsAcc=0 -i 5  --rule=%s" % R,
                            {'Atlas':'--testStat=Atlas', 'LEP':'--testStat=LEP', 'TEV':'--testStat=TEV'})) ]
    (fork, iters) = (2, 25)  # fast default
    #(fork, iters) = (4, 200) # accurate, to get reference value
    suite += [ (M, 'fast', SingleDatacardTest("Counting_pValues_Freq_0_%s" % R, "simple-counting/counting-B5p5-Obs6-StatOnly.txt", M, "--freq --singlePoint 5 --clsAcc 0 --fork %d -i %d --rule=%s" % (fork,iters,R))) ]
    suite += [ (M, 'fast', SingleDatacardTest("Counting_pValues_Freq_B_%s" % R, "simple-counting/counting-B5p5-Obs6-Syst30B.txt",  M, "--freq --singlePoint 5 --clsAcc 0 --fork %d -i %d --rule=%s" % (fork,iters,R))) ]
    suite += [ (M, 'fast', SingleDatacardTest("Counting_pValues_Freq_S_%s" % R, "simple-counting/counting-B5p5-Obs6-Syst30S.txt",  M, "--freq --singlePoint 5 --clsAcc 0 --fork %d -i %d --rule=%s" % (fork,iters,R))) ]
    suite += [ (M, 'fast', SingleDatacardTest("Counting_pValues_Freq_U_%s" % R, "simple-counting/counting-B5p5-Obs6-Syst30U.txt",  M, "--freq --singlePoint 5 --clsAcc 0 --fork %d -i %d --rule=%s" % (fork,iters,R))) ]
    suite += [ (M, 'full', MultiOptionTest("Counting_pValues_%s" % R, "simple-counting/counting-B5p5-Obs6-Syst30B.txt", M,
                            "--singlePoint 5 --fork 6 --clsAcc=0.005 --rule=%s" % R,
                            {'Atlas':'--testStat=Atlas', 'LEP':'--testStat=LEP', 'TEV':'--testStat=TEV'})) ]

for X in [ "LHC", "LEP", "TEV" ]:
        suite += [ (M, 'fast', SingleDatacardTest("HWW_pValues_%s"%X, "hww4ch-1fb-B-mH140.txt", M, 
                        "--singlePoint 1 --fork 2 --clsAcc=0.01  --testStat=%s"%X )) ]
        suite += [ (M, 'full', SingleDatacardTest("HWW_pValues_%s"%X, "hww4ch-1fb-B-mH140.txt", M, 
                        "--singlePoint 1 --fork 6 --clsAcc=0.005 --testStat=%s"%X )) ]


### Test the limits
optionsFast="--fork 2 -H ProfileLikelihood --clsAcc=0.008 --rRelAcc=0.02 --rAbsAcc=0"
optionsFull="--fork 6 -H ProfileLikelihood --rAbsAcc=0.05 --rRelAcc=0.02"
suite += [ (M, 'fast', MultiDatacardTest("Counting", [ "simple-counting/counting-B5p5-Obs6-Syst30B.txt",
                                                       "simple-counting/counting-B5p5-Obs1-Syst30B.txt",
                                                       "simple-counting/counting-B5p5-Obs6-Syst30S.txt",
                                                       "simple-counting/counting-B5p5-Obs6-StatOnly.txt",
                                                       "simple-counting/counting-B5p5-Obs6-Syst30U.txt",
                                                       "simple-counting/counting-B5p5-Obs11-Syst30B.txt"], M, optionsFast)) ]
suite += [ (M, 'fast', MultiDatacardTest("Counting_Freq", [ "simple-counting/counting-B5p5-Obs6-Syst30B.txt",
                                                            "simple-counting/counting-B5p5-Obs1-Syst30B.txt",
                                                            "simple-counting/counting-B5p5-Obs6-Syst30S.txt",
                                                            "simple-counting/counting-B5p5-Obs6-Syst30U.txt",
                                                            "simple-counting/counting-B5p5-Obs6-StatOnly.txt",
                                                            "simple-counting/counting-B5p5-Obs11-Syst30B.txt"], M, "--freq "+optionsFast)) ]
suite += [ (M, 'fast', SingleDatacardTest("HWW", "hww4ch-1fb-B-mH140.txt",  M, optionsFast)) ]

suite += [ (M, 'full', MultiDatacardTest("Counting",  datacardGlob("simple-counting/counting-B5p5-Obs[16]*-S*[Uy].txt"), M, optionsFast)) ]
for d in datacardGlob("hww4ch-1fb-B*mH*.txt"):
    n = re.sub("hww4ch-1fb-(.*mH..0).txt","\\1",d)
    suite += [ (M, 'full',  SingleDatacardTest('HWW_'+n, d, M, optionsFull)) ]

### Test significances
suite += [ (M, '*', MultiDatacardTest("Counting_Sig_2",   datacardGlob("simple-counting/counting-B5p5-Obs11-S*.txt"),M,"--signif -T 2000 -i 4  "+optionsFast))]
suite += [ (M, '*', MultiDatacardTest("Counting_Sig_3p5", datacardGlob("simple-counting/counting-B5p5-Obs20-Syst30[BUC].txt"),M,"--signif -T 4000 -i 100 "+optionsFast))]
suite += [ (M, '*', MultiDatacardTest("Counting_Freq_Sig_2",   datacardGlob("simple-counting/counting-B5p5-Obs11-S*.txt"),M,"--signif --freq -T 2002 -i 4  "+optionsFast))]
suite += [ (M, '*', MultiDatacardTest("Counting_Freq_Sig_3p5", datacardGlob("simple-counting/counting-B5p5-Obs20-Syst30[BUC].txt"),M,"--signif --freq -T 4000 -i 100 "+optionsFast))]

opts = "-T 400 --fork 4 -H ProfileLikelihood --rAbsAcc=0 --rRelAcc=0.05 --clsAcc=0.008"
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HGG_115",      "summer11/hgg/hgg_8cats.txt",            M, opts, 115)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWC_170_CNT", "summer11/hwwc.170/comb_hww_cnt.txt",    M, opts, 170)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWC_170_EXT", "summer11/hwwc.170/comb_hww_ext.txt",    M, opts, 170)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWS_130",     "summer11/hwws.130/comb_hww.txt",        M, opts, 130)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWS_130_0J",  "summer11/hwws.130/comb_hww0j.txt",      M, opts, 130)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HZZ4L_145",    "summer11/hzz4l.145/comb_hzz4l.txt",     M, opts, 145)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HZZ2L2Q_300",  "summer11/hzz2l2q.300/comb_hzz2l2q.txt", M, opts, 300)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HTT_125",      "summer11/htt.125/comb_htt.txt",         M, opts, 125)) ]

opts += " --expectedFromGrid=0.5"
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HGG_115_Median",      "summer11/hgg/hgg_8cats.txt",            M, opts, 115)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWC_170_CNT_Median", "summer11/hwwc.170/comb_hww_cnt.txt",    M, opts, 170)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWC_170_EXT_Median", "summer11/hwwc.170/comb_hww_ext.txt",    M, opts, 170)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWS_130_Median",     "summer11/hwws.130/comb_hww.txt",        M, opts, 130)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWS_130_0J_Median",  "summer11/hwws.130/comb_hww0j.txt",      M, opts, 130)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HZZ4L_145_Median",    "summer11/hzz4l.145/comb_hzz4l.txt",     M, opts, 145)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HZZ2L2Q_300_Median",  "summer11/hzz2l2q.300/comb_hzz2l2q.txt", M, opts, 300)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HTT_125_Median",      "summer11/htt.125/comb_htt.txt",         M, opts, 125)) ]

opts = "--freq -T 400 --fork 4 -H ProfileLikelihood --rAbsAcc=0 --rRelAcc=0.1 --clsAcc=0.01"
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HGG_115_Freq",      "summer11/hgg/hgg_8cats.txt",            M, opts, 115)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWC_170_CNT_Freq", "summer11/hwwc.170/comb_hww_cnt.txt",    M, opts, 170)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWC_170_EXT_Freq", "summer11/hwwc.170/comb_hww_ext.txt",    M, opts, 170)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWS_130_Freq",     "summer11/hwws.130/comb_hww.txt",        M, opts, 130)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWS_130_0J_Freq",  "summer11/hwws.130/comb_hww0j.txt",      M, opts, 130)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HZZ4L_145_Freq",    "summer11/hzz4l.145/comb_hzz4l.txt",     M, opts, 145)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HZZ2L2Q_300_Freq",  "summer11/hzz2l2q.300/comb_hzz2l2q.txt", M, opts, 300)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HTT_125_Freq",      "summer11/htt.125/comb_htt.txt",         M, opts, 125)) ]

opts += " --expectedFromGrid=0.5"
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HGG_115_Freq_Median",      "summer11/hgg/hgg_8cats.txt",            M, opts, 115)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWC_170_CNT_Freq_Median", "summer11/hwwc.170/comb_hww_cnt.txt",    M, opts, 170)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWC_170_EXT_Freq_Median", "summer11/hwwc.170/comb_hww_ext.txt",    M, opts, 170)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWS_130_Freq_Median",     "summer11/hwws.130/comb_hww.txt",        M, opts, 130)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HWWS_130_0J_Freq_Median",  "summer11/hwws.130/comb_hww0j.txt",      M, opts, 130)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HZZ4L_145_Freq_Median",    "summer11/hzz4l.145/comb_hzz4l.txt",     M, opts, 145)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HZZ2L2Q_300_Freq_Median",  "summer11/hzz2l2q.300/comb_hzz2l2q.txt", M, opts, 300)) ]
suite += [ (M, 'summer11', SingleDatacardTest("Summer11_HTT_125_Freq_Median",      "summer11/htt.125/comb_htt.txt",         M, opts, 125)) ]





