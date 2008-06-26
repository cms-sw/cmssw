
from Configuration.Generator.BsMM_cfi import *
from Configuration.Generator.BsMM_filt_cfi import *

ProductionFilterSequence = cms.Sequence(MuFilter+MuMuFilter)
