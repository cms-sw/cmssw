from TestClasses import *

suite = []
M='FeldmanCousins'
suite += [ (M, '*', MultiDatacardTest("Counting",       datacardGlob("simple-counting/counting-B5p5-Obs[16]*-Syst30U.txt"),M,""))]
suite += [ (M, '*', MultiDatacardTest("Counting_Lower", datacardGlob("simple-counting/counting-B5p5-Obs[12][10]-Syst30U.txt"),M,"--lowerLimit"))]
suite += [ (M, '*', MultiDatacardTest("Counting_ZeroB",       datacardGlob("simple-counting/counting-B0-*.txt"),M,""))]
suite += [ (M, '*', MultiDatacardTest("Counting_ZeroB_Lower", datacardGlob("simple-counting/counting-B0-*.txt"),M,"--lowerLimit"))]

for d in datacardGlob("hww4ch-1fb-B*mH*.txt"):
    n = re.sub("hww4ch-1fb-(.*mH..0).txt","\\1",d)
    suite += [ (M, 'full',  SingleDatacardTest('HWW_'+n, d, M, "-H ProfileLikelihood")) ]
for d in datacardGlob("hww4ch-1fb-S*mH*.txt"):
    n = re.sub("hww4ch-1fb-(.*mH..0).txt","\\1",d)
    suite += [ (M, 'full',  SingleDatacardTest('HWW_Sig_'+n,       d, M, "-H ProfileLikelihood")) ]
    suite += [ (M, 'full',  SingleDatacardTest('HWW_Sig_Lower_'+n, d, M, "-H ProfileLikelihood --lowerLimit")) ]
