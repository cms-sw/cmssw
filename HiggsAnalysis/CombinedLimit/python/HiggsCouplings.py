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
from HiggsAnalysis.CombinedLimit.HiggsBenchmarkModels.FermionSectorModels import C5qlHiggs, C5udHiggs, LambdaduHiggs, LambdalqHiggs
lambdadu = LambdaduHiggs()
lambdalq = LambdalqHiggs()
c5ql = C5qlHiggs()
c5ud = C5udHiggs()

# Models to test Custodial symmetry
from HiggsAnalysis.CombinedLimit.HiggsBenchmarkModels.CustodialSymmetryModels import CwzHiggs, CzwHiggs, RzwHiggs, RwzHiggs, LambdaWZHiggs
lambdaWZ  = LambdaWZHiggs() 
cWZ       = CwzHiggs() 
cZW       = CzwHiggs() 
rZW       = RzwHiggs()
rWZ       = RwzHiggs()

# Models probing the loops structure
from HiggsAnalysis.CombinedLimit.HiggsBenchmarkModels.LoopAndInvisibleModel import HiggsLoops, HiggsLoopsInvisible
higgsLoops  = HiggsLoops() 
higgsLoopsInvisible  = HiggsLoopsInvisible() 

# Model with full LO parametrization 
from HiggsAnalysis.CombinedLimit.LOFullParametrization import C5, C6
c5 = C5()
c6 = C6()


