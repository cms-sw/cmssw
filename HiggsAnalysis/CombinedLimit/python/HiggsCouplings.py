# Benchmark Higgs models as defined in (put ref to LHCXSWG document)

# the model equivalent to mu
from HiggsAnalysis.CombinedLimit.HiggsBenchmarkModels.CSquared import CSquaredHiggs
cSq = CSquaredHiggs()

# CVCF models
from HiggsAnalysis.CombinedLimit.HiggsBenchmarkModels.VectorsAndFermionsModels import CvCfHiggs, CvCfXgHiggs, CfXgHiggs
cVcF = CvCfHiggs()
#cVcFxG = CvCfXgHiggs()
#cFxG = CfXgHiggs()

# Models probing the Fermion sector
from HiggsAnalysis.CombinedLimit.HiggsBenchmarkModels.FermionSectorModels import C5qlHiggs, C5udHiggs
c5ql = C5qlHiggs()
c5ud = C5udHiggs()

# Models to test Custodial symmetry
from HiggsAnalysis.CombinedLimit.HiggsBenchmarkModels.CustodialSymmetryModels import CwzHiggs, CzwHiggs, RzwHiggs, RwzHiggs
cWZ  = CwzHiggs() 
cZW  = CzwHiggs() 
rZW  = RzwHiggs()
rWZ  = RwzHiggs()

