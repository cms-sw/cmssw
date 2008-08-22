
from HLTriggerOffline.Egamma.veryHighEtDQM_cfi import *
from HLTriggerOffline.Egamma.singlePhotonRelaxedDQM_cfi import *
from HLTriggerOffline.Egamma.singlePhotonDQM_cfi import *
from HLTriggerOffline.Egamma.singleElectronRelaxedDQM_cfi import *
from HLTriggerOffline.Egamma.singleElectronDQM_cfi import *
from HLTriggerOffline.Egamma.highEtDQM_cfi import *
from HLTriggerOffline.Egamma.doublePhotonRelaxedDQM_cfi import *
from HLTriggerOffline.Egamma.doublePhotonDQM_cfi import *
from HLTriggerOffline.Egamma.doubleElectronRelaxedDQM_cfi import *
from HLTriggerOffline.Egamma.doubleElectronDQM_cfi import *
#from HLTriggerOffline.Egamma.mcSinglePhotonEtaFilter_cfi import *
#from HLTriggerOffline.Egamma.mcSingleElectronEtaFilter_cfi import *
#from HLTriggerOffline.Egamma.mcDoublePhotonEtaFilter_cfi import *
#from HLTriggerOffline.Egamma.mcDoubleElectronEtaFilter_cfi import *

egammaval = cms.Sequence(
    doubleElectronDQM
    *doubleElectronRelaxedDQM
    *doublePhotonDQM
    *doublePhotonRelaxedDQM
    *highEtDQM
    *singleElectronDQM
    *singleElectronRelaxedDQM
    *singlePhotonDQM
    *singlePhotonRelaxedDQM
    *veryHighEtDQM
    #*mcDoubleElectronEtaFilter
    #*mcDoublePhotonEtaFilter
    #*mcSingleElectronEtaFilter
    #*mcSinglePhotonEtaFilter
)
