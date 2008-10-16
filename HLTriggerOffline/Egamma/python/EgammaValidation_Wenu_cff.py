
from HLTriggerOffline.Egamma.veryHighEtDQM_cfi import *
from HLTriggerOffline.Egamma.singlePhotonRelaxedDQM_cfi import *
from HLTriggerOffline.Egamma.singlePhotonDQM_cfi import *
from HLTriggerOffline.Egamma.singleElectronRelaxedDQM_cfi import *
from HLTriggerOffline.Egamma.singleElectronDQM_cfi import *
from HLTriggerOffline.Egamma.singleElectronRelaxedLargeWindowDQM_cfi import *
from HLTriggerOffline.Egamma.singleElectronLargeWindowDQM_cfi import *
from HLTriggerOffline.Egamma.highEtDQM_cfi import *
from HLTriggerOffline.Egamma.doublePhotonRelaxedDQM_cfi import *
from HLTriggerOffline.Egamma.doublePhotonDQM_cfi import *
from HLTriggerOffline.Egamma.doubleElectronRelaxedDQM_cfi import *
from HLTriggerOffline.Egamma.doubleElectronDQM_cfi import *

highEtDQM.pdgGen = 11
veryHighEtDQM.pdgGen = 11

leptons = cms.EDFilter("PdgIdAndStatusCandViewSelector",
    status = cms.vint32(1),
    src = cms.InputTag("genParticles"),
    pdgId = cms.vint32(11)
)



cut = cms.EDFilter("EtaPtMinCandViewSelector",
    src = cms.InputTag("leptons"),
    etaMin = cms.double(-2.5),
    etaMax = cms.double(2.5),
    ptMin = cms.double(2.0)
)

sel = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("cut"),
    minNumber = cms.uint32(1)
)

egammavalWenu = cms.Sequence(leptons*cut*sel*(doubleElectronDQM+doubleElectronRelaxedDQM+doublePhotonDQM+doublePhotonRelaxedDQM+highEtDQM+veryHighEtDQM+singleElectronDQM+singleElectronLargeWindowDQM+singleElectronRelaxedDQM+singleElectronRelaxedLargeWindowDQM+singlePhotonRelaxedDQM+singlePhotonDQM)
)
