from DQM.L1TMonitor.L1TFED_cfi import *

l1tStage2Fed = l1tfed.clone()
l1tStage2Fed.L1FEDS = cms.vint32(
    1354, 1356, 1358, # CALOL1
    1360,             # CALOL2
    1376, 1377,       # BMTF
    1380, 1381,       # OMTF
    1384, 1385,       # EMTF
    1386,             # CPPF
    1402,             # GMT
    1404,             # UGT
    1405)             # UGTSPARE

