import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HcalRecAlgos.hcalRecAlgos_cfi import hcalRecAlgos as _hcalRecAlgos

hltPhase2hcalRecAlgos = _hcalRecAlgos.clone(
    RecoveredRecHitBits=[""],
    SeverityLevels={
        2: dict(RecHitFlags=cms.vstring("HBHEIsolatedNoise", "HFAnomalousHit")),
        3: dict(
            RecHitFlags=cms.vstring(
                "HBHEHpdHitMultiplicity",
                "HBHEFlatNoise",
                "HBHESpikeNoise",
                "HBHETS4TS5Noise",
                "HBHENegativeNoise",
                "HBHEOOTPU",
            )
        ),
        4: dict(
            RecHitFlags=cms.vstring(
                "HFLongShort", "HFS8S1Ratio", "HFPET", "HFSignalAsymmetry"
            )
        ),
    },
    phase=1,
)
