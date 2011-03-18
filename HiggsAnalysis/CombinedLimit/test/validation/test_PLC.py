from TestClasses import *

suite = []
M='ProfileLikelihood'
suite += [ (M, '*', MultiDatacardTest("Counting",     datacardGlob("simple-counting/counting-B5p5-Obs[16]*.txt"),M,""))]
suite += [ (M, '*', MultiDatacardTest("Counting_Sig", datacardGlob("simple-counting/counting-B5p5-Obs[12][10]-S*.txt"),M,"--signif"))]

suite += [ (M, '*', MultiDatacardTest("Gamma",     datacardGlob("gammas/counting-*.txt"), M, "")) ]
suite += [ (M, '*', MultiDatacardTest("Gamma_Sig", datacardGlob("gammas/counting-S*.txt"), M, "--signif")) ]

suite += [ (M, '*', MultiDatacardTest("HWW",         datacardGlob("hww4ch-1fb-B*.txt"), M,'')) ]
suite += [ (M, '*', MultiDatacardTest("HWW_S0",      datacardGlob("hww4ch-1fb-B*.txt"), M,'-S 0')) ]
suite += [ (M, '*', MultiDatacardTest("HWW_Sig",     datacardGlob("hww4ch-1fb-S*.txt"), M,'--signif')) ]
suite += [ (M, '*', MultiDatacardTest("HWW_Sig_S0",  datacardGlob("hww4ch-1fb-S*.txt"), M,'-S 0  --signif')) ]
