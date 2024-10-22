import FWCore.ParameterSet.Config as cms

from DQM.Physics.HeavyFlavorDQMAnalyzer_cfi import *
from DQM.Physics.vertexSelectForHeavyFlavorDQM_cfi import recoSelectForHeavyFlavorDQM

bphWriteSpecificDecayForDQM = cms.EDProducer('BPHWriteSpecificDecay',
    pVertexLabel = cms.string('offlinePrimaryVertices'),
    pfCandsLabel = cms.string('particleFlow'),
    patMuonLabel = cms.string('selectedPatMuons'),
    k0CandsLabel = cms.string('generalV0Candidates:Kshort'),
    l0CandsLabel = cms.string('generalV0Candidates:Lambda'),
    oniaName  = cms.string('OniaToMuMuCands'),
    sdName    = cms.string('Kx0ToKPiCands'),
    ssName    = cms.string('PhiToKKCands'),
    buName    = cms.string('BuToJPsiKCands'),
    bpName    = cms.string('BuToPsi2SKCands'),
    bdName    = cms.string('BdToJPsiKx0Cands'),
    bsName    = cms.string('BsToJPsiPhiCands'),
    k0Name    = cms.string('K0sToPiPiCands'),
    l0Name    = cms.string('Lambda0ToPPiCands'),
    b0Name    = cms.string('BdToJPsiK0sCands'),
    lbName    = cms.string('LambdaBToJPsiLambda0Cands'),
    bcName    = cms.string('BcToJPsiPiCands'),
    psi2SName = cms.string('Psi2SToJPsiPiPiCands'),
    writeVertex   = cms.bool( True ),
    writeMomentum = cms.bool( True ),
    recoSelect = cms.VPSet(recoSelectForHeavyFlavorDQM)
)

heavyFlavorDQM = HeavyFlavorDQMAnalyzer.clone(
    pvCollection = cms.InputTag('offlinePrimaryVertices'),
    beamSpot = cms.InputTag('offlineBeamSpot'),
    OniaToMuMuCands            = cms.InputTag('bphWriteSpecificDecayForDQM:OniaToMuMuCands'),
    Kx0ToKPiCands              = cms.InputTag('bphWriteSpecificDecayForDQM:Kx0ToKPiCands'),
    PhiToKKCands               = cms.InputTag('bphWriteSpecificDecayForDQM:PhiToKKCands'),
    BuToJPsiKCands             = cms.InputTag('bphWriteSpecificDecayForDQM:BuToJPsiKCands'),
    #BuToPsi2SKCands            = cms.InputTag('bphWriteSpecificDecayForDQM:BuToPsi2SKCands'),
    BdToJPsiKx0Cands           = cms.InputTag('bphWriteSpecificDecayForDQM:BdToJPsiKx0Cands'),
    BsToJPsiPhiCands           = cms.InputTag('bphWriteSpecificDecayForDQM:BsToJPsiPhiCands'),
    K0sToPiPiCands             = cms.InputTag('bphWriteSpecificDecayForDQM:K0sToPiPiCands'),
    Lambda0ToPPiCands          = cms.InputTag('bphWriteSpecificDecayForDQM:Lambda0ToPPiCands'),
    BdToJPsiK0sCands           = cms.InputTag('bphWriteSpecificDecayForDQM:BdToJPsiK0sCands'),
    LambdaBToJPsiLambda0Cands  = cms.InputTag('bphWriteSpecificDecayForDQM:LambdaBToJPsiLambda0Cands'),
    BcToJPsiPiCands            = cms.InputTag('bphWriteSpecificDecayForDQM:BcToJPsiPiCands'),
    Psi2SToJPsiPiPiCands       = cms.InputTag('bphWriteSpecificDecayForDQM:Psi2SToJPsiPiPiCands'),
)

heavyFlavorDQMSource = cms.Sequence(bphWriteSpecificDecayForDQM * heavyFlavorDQM)
