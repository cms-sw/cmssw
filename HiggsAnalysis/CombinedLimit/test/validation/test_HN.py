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
                            "--singlePoint 5 --fork 2 --clsAcc=0.01  --rule=%s" % R,
                            {'Atlas':'--testStat=Atlas', 'LEP':'--testStat=LEP', 'TEV':'--testStat=TEV'})) ]
    suite += [ (M, 'full', MultiOptionTest("Counting_pValues_%s" % R, "simple-counting/counting-B5p5-Obs6-Syst30B.txt", M,
                            "--singlePoint 5 --fork 6 --clsAcc=0.005 --rule=%s" % R,
                            {'Atlas':'--testStat=Atlas', 'LEP':'--testStat=LEP', 'TEV':'--testStat=TEV'})) ]

for X in [ "Atlas", "LEP", "TEV" ]:
        suite += [ (M, 'fast', SingleDatacardTest("HWW_pValues_%s"%X, "hww4ch-1fb-B-mH140.txt", M, 
                        "--singlePoint 1 --fork 2 --clsAcc=0.01  --testStat=%s"%X )) ]
        suite += [ (M, 'full', SingleDatacardTest("HWW_pValues_%s"%X, "hww4ch-1fb-B-mH140.txt", M, 
                        "--singlePoint 1 --fork 6 --clsAcc=0.005 --testStat=%s"%X )) ]


### Test the limits
optionsFast="--fork 2 -H ProfileLikelihood"
optionsFull="--fork 6 -H ProfileLikelihood --rAbsAcc=0.05 --rRelAcc=0.02"
suite += [ (M, 'fast', MultiDatacardTest("Counting", [ "simple-counting/counting-B5p5-Obs6-Syst30B.txt",
                                                       "simple-counting/counting-B5p5-Obs1-Syst30B.txt",
                                                       "simple-counting/counting-B5p5-Obs11-Syst30B.txt"], M, optionsFast)) ]
suite += [ (M, 'fast', SingleDatacardTest("HWW", "hww4ch-1fb-B-mH140.txt",  M, optionsFast)) ]

suite += [ (M, 'full', MultiDatacardTest("Counting",  datacardGlob("simple-counting/counting-B5p5-Obs[16]*-S*[Uy].txt"), M, optionsFast)) ]
for d in datacardGlob("hww4ch-1fb-B*mH*.txt"):
    n = re.sub("hww4ch-1fb-(.*mH..0).txt","\\1",d)
    suite += [ (M, 'full',  SingleDatacardTest('HWW_'+n, d, M, optionsFull)) ]

### Custom suite for just the ATLAS test statistics
suite += [ (M, 'atlas', SingleDatacardTest("TestStat_Atlas", "simple-counting/counting-B5p5-Obs6-Syst30B.txt", M, "--singlePoint 5 --onlyTestStat --testStat=Atlas")) ]
suite += [ (M, 'atlas', SingleDatacardTest("TestStat_LEP",   "simple-counting/counting-B5p5-Obs6-Syst30B.txt", M, "--singlePoint 5 --onlyTestStat --testStat=LEP"  )) ]
suite += [ (M, 'atlas', SingleDatacardTest("pValue_Atlas",   "simple-counting/counting-B5p5-Obs6-Syst30B.txt", M, "--singlePoint 5                --testStat=Atlas --fork 4 ")) ]
suite += [ (M, 'atlas', SingleDatacardTest("pValue_LEP",     "simple-counting/counting-B5p5-Obs6-Syst30B.txt", M, "--singlePoint 5                --testStat=LEP   --fork 4 ")) ]
