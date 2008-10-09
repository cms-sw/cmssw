import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.SISConeJetParameters_cfi import *
FastjetWithAreaPU = cms.PSet(
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    UE_Subtraction = cms.string('no')
)
UEAnalysisSISConeJetParameters = cms.PSet(
    verbose = cms.untracked.bool(False),
    jetPtMin = cms.double(5.0),
    inputEtMin = cms.double(0.9),
    coneRadius = cms.double(0.7),
    jetType = cms.untracked.string('GenJet'),
    inputEMin = cms.double(0.0)
)
ueSisCone7GenJet = cms.EDProducer("SISConeJetProducer",
    UEAnalysisSISConeJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    src = cms.InputTag("goodParticles")
)

ueSisCone7ChgGenJet = cms.EDProducer("SISConeJetProducer",
    UEAnalysisSISConeJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    src = cms.InputTag("chargeParticles")
)

ueSisCone7TracksJet = cms.EDProducer("SISConeJetProducer",
    UEAnalysisSISConeJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    src = cms.InputTag("goodTracks")
)

ueSisCone7GenJet500 = cms.EDProducer("SISConeJetProducer",
    UEAnalysisSISConeJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    src = cms.InputTag("goodParticles")
)

ueSisCone7ChgGenJet500 = cms.EDProducer("SISConeJetProducer",
    UEAnalysisSISConeJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    src = cms.InputTag("chargeParticles")
)

ueSisCone7TracksJet500 = cms.EDProducer("SISConeJetProducer",
    UEAnalysisSISConeJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    src = cms.InputTag("goodTracks")
)
ueSisCone7GenJet500.inputEtMin    = 0.5
ueSisCone7ChgGenJet500.inputEtMin = 0.5
ueSisCone7TracksJet500.jetType    = 'BasicJet'
ueSisCone7TracksJet500.inputEtMin = 0.5

ueSisCone7GenJet1500 = cms.EDProducer("SISConeJetProducer",
                                         UEAnalysisSISConeJetParameters,
                                         SISConeJetParameters,
                                         FastjetNoPU,
                                         src = cms.InputTag("goodParticles")
                                     )

ueSisCone7ChgGenJet1500 = cms.EDProducer("SISConeJetProducer",
                                            UEAnalysisSISConeJetParameters,
                                            SISConeJetParameters,
                                            FastjetNoPU,
                                            src = cms.InputTag("chargeParticles")
                                        )

ueSisCone7TracksJet1500 = cms.EDProducer("SISConeJetProducer",
                                            UEAnalysisSISConeJetParameters,
                                            SISConeJetParameters,
                                            FastjetNoPU,
                                            src = cms.InputTag("goodTracks")
                                        )
ueSisCone7GenJet1500.inputEtMin    = 1.5
ueSisCone7ChgGenJet1500.inputEtMin = 1.5
ueSisCone7TracksJet1500.jetType    = 'BasicJet'
ueSisCone7TracksJet1500.inputEtMin = 1.5


UEAnalysisJetsOnlyMC = cms.Sequence(ueSisCone7GenJet*ueSisCone7ChgGenJet*ueSisCone7GenJet500*ueSisCone7ChgGenJet500*ueSisCone7GenJet1500*ueSisCone7ChgGenJet1500)
UEAnalysisJetsOnlyReco = cms.Sequence(ueSisCone7TracksJet*ueSisCone7TracksJet500*ueSisCone7TracksJet1500)
UEAnalysisJets = cms.Sequence(UEAnalysisJetsOnlyMC*UEAnalysisJetsOnlyReco)
ueSisCone7TracksJet.jetType = 'BasicJet'


