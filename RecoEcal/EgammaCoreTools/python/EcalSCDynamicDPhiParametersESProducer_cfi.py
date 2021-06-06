import FWCore.ParameterSet.Config as cms

ecalSCDynamicDPhiParametersESProducer = cms.ESProducer("EcalSCDynamicDPhiParametersESProducer",
    # Parameters from the analysis by L. Zygala [https://indico.cern.ch/event/949294/contributions/3988389/attachments/2091573/3514649/2020_08_26_Clustering.pdf]
    # dynamic dPhi parameters depending on cluster energy and seed crystal eta
    dynamicDPhiParameterSets = cms.VPSet(
        cms.PSet(
            eMin = cms.double(0.),
            etaMin = cms.double(2.),
            yoffset = cms.double(0.0928887),
            scale = cms.double(1.22321),
            xoffset = cms.double(-0.260256),
            width = cms.double(0.345852),
            saturation = cms.double(0.12),
            cutoff = cms.double(0.3)
        ),
        cms.PSet(
            eMin = cms.double(0.),
            etaMin = cms.double(1.75),
            yoffset = cms.double(0.05643),
            scale = cms.double(1.60429),
            xoffset = cms.double(-0.642352),
            width = cms.double(0.458106),
            saturation = cms.double(0.12),
            cutoff = cms.double(0.45)
        ),
        cms.PSet(
            eMin = cms.double(0.),
            etaMin = cms.double(1.479),
            yoffset = cms.double(0.0497038),
            scale = cms.double(0.975707),
            xoffset = cms.double(-0.18149),
            width = cms.double(0.431729),
            saturation = cms.double(0.14),
            cutoff = cms.double(0.55)
        ),
        cms.PSet(
            eMin = cms.double(0.),
            etaMin = cms.double(0.),
            yoffset = cms.double(0.0280506),
            scale = cms.double(0.946048),
            xoffset = cms.double(-0.101172),
            width = cms.double(0.432767),
            saturation = cms.double(0.14),
            cutoff = cms.double(0.6)
        )
    )
)

