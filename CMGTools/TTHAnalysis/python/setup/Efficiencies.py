from CMGTools.RootTools.fwlite.Config import CFG
from CMGTools.TTHAnalysis.tools.EfficiencyCorrector import EfficiencyCorrector


eff2012 = CFG(
    name='eff',
    muonFile = 'data/eff_mu12.root',
    muonHisto = 'TH2D_ALL_2012',
    eleFile = 'data/eff_ele12.root',    
    eleHisto = 'h_electronScaleFactor_RecoIdIsoSip'
    )


