import FWCore.ParameterSet.Config as cms

import DQM.Physics.qcdPhotonsDQM_cfi
qcdPhotonsCosmicDQM = DQM.Physics.qcdPhotonsDQM_cfi.qcdPhotonsDQM.clone(
    barrelRecHitTag           = cms.InputTag("ecalRecHit:EcalRecHitsEB"),
    endcapRecHitTag           = cms.InputTag("ecalRecHit:EcalRecHitsEE")
)
