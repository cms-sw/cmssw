from TestClasses import *

suite = []
M='FeldmanCousins'
suite += [ (M, '*', MultiDatacardTest("Counting",       datacardGlob("simple-counting/counting-B5p5-Obs[16]*-Syst30U.txt"),M,""))]
suite += [ (M, '*', MultiDatacardTest("Counting_Lower", datacardGlob("simple-counting/counting-B5p5-Obs[12][10]-Syst30U.txt"),M,"--lowerLimit"))]
suite += [ (M, '*', MultiDatacardTest("Counting_ZeroB",       datacardGlob("simple-counting/counting-B0-*.txt"),M,""))]
suite += [ (M, '*', MultiDatacardTest("Counting_ZeroB_Lower", datacardGlob("simple-counting/counting-B0-*.txt"),M,"--lowerLimit"))]

suite += [ (M, 'fast', MultiDatacardTest("HWW",           datacardGlob("hww4ch-1fb-B-mH1[46]0.txt"), M,'')) ]
suite += [ (M, 'full', MultiDatacardTest("HWW",           datacardGlob("hww4ch-1fb-B*mH*.txt"),      M,'')) ]
suite += [ (M, 'full', MultiDatacardTest("HWW_Sig",       datacardGlob("hww4ch-1fb-S*.txt"), M,'')) ]
suite += [ (M, 'full', MultiDatacardTest("HWW_Sig_Lower", datacardGlob("hww4ch-1fb-S*.txt"), M,'--lowerLimit')) ]
