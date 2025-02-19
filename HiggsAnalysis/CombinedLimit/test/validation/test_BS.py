from TestClasses import *

suite = []
M='BayesianSimple'
suite += [ (M, '*', MultiDatacardTest("Counting", datacardGlob("simple-counting/counting-B5p5-Obs[16]*.txt"),M,""))]
suite += [ (M, '*', MultiDatacardTest("Gamma",    datacardGlob("gammas/counting-*.txt"), M, "")) ]
suite += [ (M, '*', MultiDatacardTest("HWW_S0",   datacardGlob("hww4ch-1fb-B*.txt"), M,'-S 0')) ]
suite += [ (M, '*', MultiDatacardTest("Asymm",    datacardGlob("asymm/counting-B5p5-Obs6*txt"), M, "")) ]
