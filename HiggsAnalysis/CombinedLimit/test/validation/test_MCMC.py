from TestClasses import *

suite = []
M='MarkovChainMC'
suite += [ (M, 'fast', MultiDatacardTest("Counting", datacardGlob("simple-counting/counting-B5p5-Obs6*.txt"),M,"")) ]
suite += [ (M, 'full', MultiDatacardTest("Counting", datacardGlob("simple-counting/counting-B5p5-Obs[16]*.txt"),M,"--tries 50"))]

suite += [ (M, 'fast', SingleDatacardTest("Gamma", "gammas/counting-S1-Sideband10-alpha0p2-Obs2.txt", M, "")) ]
suite += [ (M, 'full', SingleDatacardTest("Gamma", "gammas/counting-S1-Sideband10-alpha0p2-Obs2.txt", M, "--tries 50")) ]

suite += [ (M, 'fast', MultiDatacardTest("HWW",     datacardGlob("hww4ch-1fb-B-mH1[46]0.txt"), M,'     --tries 20')) ]
suite += [ (M, 'fast', MultiDatacardTest("HWW_S0",  datacardGlob("hww4ch-1fb-B-mH1[46]0.txt"), M,'-S 0 --tries 50')) ]
for d in datacardGlob("hww4ch-1fb-B*mH*.txt"):
    n = re.sub("hww4ch-1fb-(.*mH..0).txt","\\1",d)
    suite += [ (M, 'full',  SingleDatacardTest('HWW_'+n, d, M,'--tries 100')) ]

suite += [ ('MarkovChainMC', 'fast', 
            MultiOptionTest("Proposals", "simple-counting/counting-B5p5-Obs6-Syst30B.txt", "MarkovChainMC", "--tries 20",
                            { 'Uniform':'--proposal=uniform', 'Gaus':'--proposal=gaus', 'Ortho':'--proposal=ortho'} )) ]
suite += [ ('MarkovChainMC', 'full', 
            MultiOptionTest("Proposals", "simple-counting/counting-B5p5-Obs6-Syst30U.txt", "MarkovChainMC", "--tries 100 -i 20000",
                            { 'Uniform':'--proposal=uniform', 'Gaus':'--proposal=gaus', 'Ortho':'--proposal=ortho'} )) ]

