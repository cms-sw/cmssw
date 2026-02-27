import FWCore.ParameterSet.Config as cms
import RecoLocalCalo.HcalRecProducers.zdcRecoRun3_cfi
# clone (new) zdcrecoRun3  from imtermediate zdcRecoRun3
zdcrecoRun3 = RecoLocalCalo.HcalRecProducers.zdcRecoRun3_cfi.zdcRecoRun3.clone()
# apply modifier(s)
from Configuration.ProcessModifiers.rpdReco_cff  import rpdReco
rpdReco.toModify(zdcrecoRun3, skipRPD = False) 
