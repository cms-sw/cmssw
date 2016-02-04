
from Configuration.Generator.bJpsiX_cfi import *
from Configuration.Generator.bJpsiX_filt_cfi import *

ProductionFilterSequence = cms.Sequence(bfilter+jpsifilter+mumufilter)
