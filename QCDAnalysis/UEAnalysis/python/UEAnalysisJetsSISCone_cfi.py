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
    jetPtMin = cms.double(1.0),
    inputEtMin = cms.double(0.9),
    coneRadius = cms.double(0.5),
    jetType = cms.untracked.string('GenJet'),
    inputEMin = cms.double(0.0)
)
ueSisCone5GenJet = cms.EDProducer("SISConeJetProducer",
    UEAnalysisSISConeJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    src = cms.InputTag("goodParticles")
)
ueSisCone5ChgGenJet = cms.EDProducer("SISConeJetProducer",
    UEAnalysisSISConeJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    src = cms.InputTag("chargeParticles")
)
ueSisCone5TracksJet = cms.EDProducer("SISConeJetProducer",
    UEAnalysisSISConeJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    src = cms.InputTag("goodTracks")
)
ueSisCone5TracksJet.jetType = 'BasicJet'

ueSisCone5GenJet500 = cms.EDProducer("SISConeJetProducer",
    UEAnalysisSISConeJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    src = cms.InputTag("goodParticles")
)

ueSisCone5ChgGenJet500 = cms.EDProducer("SISConeJetProducer",
    UEAnalysisSISConeJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    src = cms.InputTag("chargeParticles")
)

ueSisCone5TracksJet500 = cms.EDProducer("SISConeJetProducer",
    UEAnalysisSISConeJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    src = cms.InputTag("goodTracks")
)
ueSisCone5GenJet500.inputEtMin    = 0.5
ueSisCone5ChgGenJet500.inputEtMin = 0.5
ueSisCone5TracksJet500.jetType    = 'BasicJet'
ueSisCone5TracksJet500.inputEtMin = 0.5

ueSisCone5GenJet700 = cms.EDProducer("SISConeJetProducer",
    UEAnalysisSISConeJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    src = cms.InputTag("goodParticles")
)

ueSisCone5ChgGenJet700 = cms.EDProducer("SISConeJetProducer",
    UEAnalysisSISConeJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    src = cms.InputTag("chargeParticles")
)

ueSisCone5TracksJet700 = cms.EDProducer("SISConeJetProducer",
    UEAnalysisSISConeJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    src = cms.InputTag("goodTracks")
)
ueSisCone5GenJet700.inputEtMin    = 0.7
ueSisCone5ChgGenJet700.inputEtMin = 0.7
ueSisCone5TracksJet700.jetType    = 'BasicJet'
ueSisCone5TracksJet700.inputEtMin = 0.7


ueSisCone5GenJet1100 = cms.EDProducer("SISConeJetProducer",
    UEAnalysisSISConeJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    src = cms.InputTag("goodParticles")
)

ueSisCone5ChgGenJet1100 = cms.EDProducer("SISConeJetProducer",
    UEAnalysisSISConeJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    src = cms.InputTag("chargeParticles")
)

ueSisCone5TracksJet1100 = cms.EDProducer("SISConeJetProducer",
    UEAnalysisSISConeJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    src = cms.InputTag("goodTracks")
)
ueSisCone5GenJet1100.inputEtMin    = 1.1
ueSisCone5ChgGenJet1100.inputEtMin = 1.1
ueSisCone5TracksJet1100.jetType    = 'BasicJet'
ueSisCone5TracksJet1100.inputEtMin = 1.1

ueSisCone5GenJet1500 = cms.EDProducer("SISConeJetProducer",
                                         UEAnalysisSISConeJetParameters,
                                         SISConeJetParameters,
                                         FastjetNoPU,
                                         src = cms.InputTag("goodParticles")
                                     )

ueSisCone5ChgGenJet1500 = cms.EDProducer("SISConeJetProducer",
                                            UEAnalysisSISConeJetParameters,
                                            SISConeJetParameters,
                                            FastjetNoPU,
                                            src = cms.InputTag("chargeParticles")
                                        )

ueSisCone5TracksJet1500 = cms.EDProducer("SISConeJetProducer",
                                            UEAnalysisSISConeJetParameters,
                                            SISConeJetParameters,
                                            FastjetNoPU,
                                            src = cms.InputTag("goodTracks")
                                        )
ueSisCone5GenJet1500.inputEtMin    = 1.5
ueSisCone5ChgGenJet1500.inputEtMin = 1.5
ueSisCone5TracksJet1500.jetType    = 'BasicJet'
ueSisCone5TracksJet1500.inputEtMin = 1.5


UEAnalysisJetsOnlyMC = cms.Sequence(ueSisCone5GenJet*ueSisCone5ChgGenJet*ueSisCone5GenJet500*ueSisCone5ChgGenJet500*ueSisCone5GenJet1500*ueSisCone5ChgGenJet1500*ueSisCone5GenJet700*ueSisCone5ChgGenJet700*ueSisCone5GenJet1100*ueSisCone5ChgGenJet1100)
UEAnalysisJetsOnlyReco = cms.Sequence(ueSisCone5TracksJet*ueSisCone5TracksJet500*ueSisCone5TracksJet1500*ueSisCone5TracksJet700*ueSisCone5TracksJet1100)

UEAnalysisJets = cms.Sequence(UEAnalysisJetsOnlyMC*UEAnalysisJetsOnlyReco)


