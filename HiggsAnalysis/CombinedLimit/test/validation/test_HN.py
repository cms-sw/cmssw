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
                            "--singlePoint 5 --fork 4 -T 500 --clsAcc=0.01  --rule=%s" % R,
                            {'Atlas':'--testStat=Atlas', 'LEP':'--testStat=LEP', 'TEV':'--testStat=TEV'})) ]
    suite += [ (M, 'full', MultiOptionTest("Counting_pValues_%s" % R, "simple-counting/counting-B5p5-Obs6-Syst30B.txt", M,
                            "--singlePoint 5 --fork 4 -T 500 --clsAcc=0.001 --rule=%s" % R,
                            {'Atlas':'--testStat=Atlas', 'LEP':'--testStat=LEP', 'TEV':'--testStat=TEV'})) ]

for X in [ "Atlas", "LEP", "TEV" ]:
        suite += [ (M, 'fast', SingleDatacardTest("HWW_pValues_%s"%X, "hww4ch-1fb-B-mH140.txt", M, 
                        "--singlePoint 2 --fork 2 -T 100 --clsAcc=1 --testStat=%s"%X )) ]
        suite += [ (M, 'full', SingleDatacardTest("HWW_pValues_%s"%X, "hww4ch-1fb-B-mH140.txt", M, 
                        "--singlePoint 2 --fork 6 -T 250 --clsAcc=1 --testStat=%s"%X )) ]

### Test the limits
options="--fork 4 -T 50 -H ProfileLikelihood"
suite += [ (M, 'fast', SingleDatacardTest("Counting", "simple-counting/counting-B5p5-Obs6-Syst30B.txt", M, options)) ]
suite += [ (M, 'fast', SingleDatacardTest("HWW",      "hww4ch-1fb-B-mH140.txt",                         M, options)) ]

suite += [ (M, 'full', MultiDatacardTest("Counting",  datacardGlob("simple-counting/counting-B5p5-Obs[16]*-S*[Uy].txt"), M, options)) ]
for d in datacardGlob("hww4ch-1fb-B*mH*.txt"):
    n = re.sub("hww4ch-1fb-(.*mH..0).txt","\\1",d)
    suite += [ (M, 'full',  SingleDatacardTest('HWW_'+n, d, M, options)) ]

### Custom suite for just the ATLAS test statistics
suite += [ (M, 'atlas', SingleDatacardTest("TestStat_Atlas", "simple-counting/counting-B5p5-Obs6-Syst30B.txt", M, "--singlePoint 5 --onlyTestStat --testStat=Atlas")) ]
suite += [ (M, 'atlas', SingleDatacardTest("TestStat_LEP",   "simple-counting/counting-B5p5-Obs6-Syst30B.txt", M, "--singlePoint 5 --onlyTestStat --testStat=LEP"  )) ]
suite += [ (M, 'atlas', SingleDatacardTest("pValue_Atlas",   "simple-counting/counting-B5p5-Obs6-Syst30B.txt", M, "--singlePoint 5                --testStat=Atlas --fork 4 -T 500")) ]
suite += [ (M, 'atlas', SingleDatacardTest("pValue_LEP",     "simple-counting/counting-B5p5-Obs6-Syst30B.txt", M, "--singlePoint 5                --testStat=LEP   --fork 4 -T 500")) ]
