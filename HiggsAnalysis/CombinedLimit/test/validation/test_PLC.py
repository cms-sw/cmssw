from TestClasses import *

suite = []
M='ProfileLikelihood'
suite += [ (M, '*', MultiDatacardTest("Counting",     datacardGlob("simple-counting/counting-B5p5-Obs[16]*.txt"),M,""))]
suite += [ (M, '*', MultiDatacardTest("Counting_Sig", datacardGlob("simple-counting/counting-B5p5-Obs[12][10]-S*.txt"),M,"--signif"))]

suite += [ (M, '*', MultiDatacardTest("Gamma",     datacardGlob("gammas/counting-*.txt"), M, "")) ]
suite += [ (M, '*', MultiDatacardTest("Gamma_Sig", [ "gammas/counting-S1-Sideband1-alpha0p3-Obs2.txt",
                                                     "gammas/counting-S1-Sideband2-alpha0p3-Obs6.txt",
                                                     "gammas/counting-Sideband11-alpha0p5-Obs11.txt",
                                                     "gammas/counting-Sideband11-alpha0p5-Obs20.txt"], M, "--signif")) ]

suite += [ (M, '*', MultiDatacardTest("HWW",         datacardGlob("hww4ch-1fb-B*.txt"), M,'')) ]
suite += [ (M, '*', MultiDatacardTest("HWW_S0",      datacardGlob("hww4ch-1fb-B*.txt"), M,'-S 0')) ]
suite += [ (M, '*', MultiDatacardTest("HWW_Sig",     datacardGlob("hww4ch-1fb-S*.txt"), M,'--signif')) ]
suite += [ (M, '*', MultiDatacardTest("HWW_Sig_S0",  datacardGlob("hww4ch-1fb-S*.txt"), M,'-S 0  --signif')) ]
