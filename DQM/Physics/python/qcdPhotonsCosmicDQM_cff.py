import FWCore.ParameterSet.Config as cms

import DQM.Physics.qcdPhotonsDQM_cfi
qcdPhotonsCosmicDQM = DQM.Physics.qcdPhotonsDQM_cfi.qcdPhotonsDQM.clone(
    barrelRecHitTag           = "ecalRecHit:EcalRecHitsEB",
    endcapRecHitTag           = "ecalRecHit:EcalRecHitsEE"
)
