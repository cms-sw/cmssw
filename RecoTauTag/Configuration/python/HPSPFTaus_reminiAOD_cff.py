import FWCore.ParameterSet.Config as cms
import copy

'''

Sequences for HPS taus

'''

# Define the discriminators for this tau
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi                      import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackFinding_cfi            import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectron_cfi                  import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectronMVA5_cfi              import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectronMVA6_cfi              import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectronDeadECAL_cfi          import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon_cfi                      import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon2_cfi                     import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuonMVA_cfi                   import *

from RecoTauTag.RecoTau.RecoTauDiscriminantCutMultiplexer_cfi import *

# Load helper functions to change the source of the discriminants
from RecoTauTag.RecoTau.TauDiscriminatorTools import *

# Load PFjet input parameters
from RecoTauTag.RecoTau.PFRecoTauPFJetInputs_cfi import PFRecoTauPFJetInputs

# deltaBeta correction factor
ak4dBetaCorrection = 0.20

# Load MVAs from SQLlite file/prep. DB
from RecoTauTag.Configuration.loadRecoTauTagMVAsFromPrepDB_cfi import *

# Select those taus that pass the HPS selections
#  - pt > 15, mass cuts, tauCone cut
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByHPSSelection_cfi import hpsSelectionDiscriminator, decayMode_1Prong0Pi0, decayMode_1Prong1Pi0, decayMode_1Prong2Pi0, decayMode_2Prong0Pi0, decayMode_2Prong1Pi0, decayMode_3Prong0Pi0
hpsSelectionDiscriminator76xReMiniAOD = hpsSelectionDiscriminator.clone()

hpsPFTauDiscriminationByDecayModeFindingNewDMs76xReMiniAOD = hpsSelectionDiscriminator76xReMiniAOD.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    #----------------------------------------------------------------------------
    # CV: disable 3Prong1Pi0 decay mode
    decayModes = cms.VPSet(
        decayMode_1Prong0Pi0,
        decayMode_1Prong1Pi0,
        decayMode_1Prong2Pi0,
        decayMode_2Prong0Pi0,
        decayMode_2Prong1Pi0,
        decayMode_3Prong0Pi0
    )
    #----------------------------------------------------------------------------
)
hpsPFTauDiscriminationByDecayModeFindingOldDMs76xReMiniAOD = hpsSelectionDiscriminator76xReMiniAOD.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    decayModes = cms.VPSet(
        decayMode_1Prong0Pi0,
        decayMode_1Prong1Pi0,
        decayMode_1Prong2Pi0,
        decayMode_3Prong0Pi0
    ),
    requireTauChargedHadronsToBeChargedPFCands = cms.bool(True)
)
hpsPFTauDiscriminationByDecayModeFinding76xReMiniAOD = hpsPFTauDiscriminationByDecayModeFindingOldDMs76xReMiniAOD.clone() # CV: kept for backwards compatibility

# Define decay mode prediscriminant
requireDecayMode76xReMiniAOD = cms.PSet(
    BooleanOperator = cms.string("and"),
    decayMode = cms.PSet(
        Producer = cms.InputTag('hpsPFTauDiscriminationByDecayModeFindingNewDMs76xReMiniAOD'),
        cut = cms.double(0.5)
    )
)

#Building the prototype for  the Discriminator by Isolation
hpsPFTauDiscriminationByLooseIsolation76xReMiniAOD = pfRecoTauDiscriminationByIsolation.clone(
    PFTauProducer = cms.InputTag("hpsPFTauProducer76xReMiniAOD"),
    Prediscriminants = requireDecayMode76xReMiniAOD.clone(),
    ApplyDiscriminationByTrackerIsolation = False,
    ApplyDiscriminationByECALIsolation = True,
    applyOccupancyCut = True
)
hpsPFTauDiscriminationByLooseIsolation76xReMiniAOD.Prediscriminants.preIso = cms.PSet(
    Producer = cms.InputTag("hpsPFTauDiscriminationByLooseChargedIsolation76xReMiniAOD"),
    cut = cms.double(0.5))

# Make an even looser discriminator
hpsPFTauDiscriminationByVLooseIsolation76xReMiniAOD = hpsPFTauDiscriminationByLooseIsolation76xReMiniAOD.clone(
    customOuterCone = cms.double(0.3),
    isoConeSizeForDeltaBeta = cms.double(0.3),
)
hpsPFTauDiscriminationByVLooseIsolation76xReMiniAOD.qualityCuts.isolationQualityCuts.minTrackPt = 1.5
hpsPFTauDiscriminationByVLooseIsolation76xReMiniAOD.qualityCuts.isolationQualityCuts.minGammaEt = 2.0
hpsPFTauDiscriminationByVLooseIsolation76xReMiniAOD.Prediscriminants.preIso.Producer =  cms.InputTag("hpsPFTauDiscriminationByVLooseChargedIsolation76xReMiniAOD")

hpsPFTauDiscriminationByMediumIsolation76xReMiniAOD = hpsPFTauDiscriminationByLooseIsolation76xReMiniAOD.clone()
hpsPFTauDiscriminationByMediumIsolation76xReMiniAOD.qualityCuts.isolationQualityCuts.minTrackPt = 0.8
hpsPFTauDiscriminationByMediumIsolation76xReMiniAOD.qualityCuts.isolationQualityCuts.minGammaEt = 0.8
hpsPFTauDiscriminationByMediumIsolation76xReMiniAOD.Prediscriminants.preIso.Producer = cms.InputTag("hpsPFTauDiscriminationByMediumChargedIsolation76xReMiniAOD")

hpsPFTauDiscriminationByTightIsolation76xReMiniAOD = hpsPFTauDiscriminationByLooseIsolation76xReMiniAOD.clone()
hpsPFTauDiscriminationByTightIsolation76xReMiniAOD.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByTightIsolation76xReMiniAOD.qualityCuts.isolationQualityCuts.minGammaEt = 0.5
hpsPFTauDiscriminationByTightIsolation76xReMiniAOD.Prediscriminants.preIso.Producer = cms.InputTag("hpsPFTauDiscriminationByTightChargedIsolation76xReMiniAOD")

hpsPFTauDiscriminationByIsolationSeq76xReMiniAOD = cms.Sequence(
    hpsPFTauDiscriminationByVLooseIsolation76xReMiniAOD*
    hpsPFTauDiscriminationByLooseIsolation76xReMiniAOD*
    hpsPFTauDiscriminationByMediumIsolation76xReMiniAOD*
    hpsPFTauDiscriminationByTightIsolation76xReMiniAOD
)

_isolation_types = ['VLoose', 'Loose', 'Medium', 'Tight']
# Now build the sequences that apply PU corrections

# Make Delta Beta corrections (on SumPt quantity)
hpsPFTauDiscriminationByVLooseIsolationDBSumPtCorr76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolation76xReMiniAOD.clone(
    deltaBetaPUTrackPtCutOverride = cms.double(0.5),
    applyDeltaBetaCorrection = True,
    isoConeSizeForDeltaBeta = 0.8,
    deltaBetaFactor = "%0.4f"%(0.0123/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
)
hpsPFTauDiscriminationByVLooseIsolationDBSumPtCorr76xReMiniAOD.maximumSumPtCut = hpsPFTauDiscriminationByVLooseIsolationDBSumPtCorr76xReMiniAOD.qualityCuts.isolationQualityCuts.minGammaEt

hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr76xReMiniAOD = hpsPFTauDiscriminationByLooseIsolation76xReMiniAOD.clone(
    deltaBetaPUTrackPtCutOverride = cms.double(0.5),
    applyDeltaBetaCorrection = True,
    isoConeSizeForDeltaBeta = 0.8,
    deltaBetaFactor = "%0.4f"%(0.0123/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
)
hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr76xReMiniAOD.maximumSumPtCut = hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr76xReMiniAOD.qualityCuts.isolationQualityCuts.minGammaEt

hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr76xReMiniAOD = hpsPFTauDiscriminationByMediumIsolation76xReMiniAOD.clone(
    deltaBetaPUTrackPtCutOverride = cms.double(0.5),
    applyDeltaBetaCorrection = True,
    isoConeSizeForDeltaBeta = 0.8,
    deltaBetaFactor = "%0.4f"%(0.0462/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
)
hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr76xReMiniAOD.maximumSumPtCut = hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr76xReMiniAOD.qualityCuts.isolationQualityCuts.minGammaEt

hpsPFTauDiscriminationByTightIsolationDBSumPtCorr76xReMiniAOD = hpsPFTauDiscriminationByTightIsolation76xReMiniAOD.clone(
    deltaBetaPUTrackPtCutOverride = cms.double(0.5),
    applyDeltaBetaCorrection = True,
    isoConeSizeForDeltaBeta = 0.8,
    deltaBetaFactor = "%0.4f"%(ak4dBetaCorrection),
    applyOccupancyCut = False,
    applySumPtCut = True,
)
hpsPFTauDiscriminationByTightIsolationDBSumPtCorr76xReMiniAOD.maximumSumPtCut = hpsPFTauDiscriminationByTightIsolationDBSumPtCorr76xReMiniAOD.qualityCuts.isolationQualityCuts.minGammaEt

hpsPFTauDiscriminationByIsolationSeqDBSumPtCorr76xReMiniAOD = cms.Sequence(
    hpsPFTauDiscriminationByVLooseIsolationDBSumPtCorr76xReMiniAOD*
    hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr76xReMiniAOD*
    hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr76xReMiniAOD*
    hpsPFTauDiscriminationByTightIsolationDBSumPtCorr76xReMiniAOD
)

hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationDBSumPtCorr76xReMiniAOD.clone(
    ApplyDiscriminationByTrackerIsolation = True,
    ApplyDiscriminationByECALIsolation = True,
    deltaBetaFactor = "%0.4f"%((0.09/0.25)*(ak4dBetaCorrection)),
    applyOccupancyCut = False,
    applySumPtCut = True,
    maximumSumPtCut = 3.5,
    Prediscriminants = requireDecayMode76xReMiniAOD.clone()
)
hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr76xReMiniAOD.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr76xReMiniAOD.qualityCuts.isolationQualityCuts.minGammaEt = 0.5

hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr76xReMiniAOD = hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr76xReMiniAOD.clone(
    ApplyDiscriminationByTrackerIsolation = True,
    ApplyDiscriminationByECALIsolation = True,
    deltaBetaFactor = "%0.4f"%(ak4dBetaCorrection),
    applyOccupancyCut = False,
    applySumPtCut = True,
    maximumSumPtCut = 2.5,
    Prediscriminants = requireDecayMode76xReMiniAOD.clone()
)
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr76xReMiniAOD.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr76xReMiniAOD.qualityCuts.isolationQualityCuts.minGammaEt = 0.5

hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr76xReMiniAOD = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr76xReMiniAOD.clone(
    applySumPtCut = False,
    storeRawSumPt = cms.bool(True)
)

hpsPFTauDiscriminationByRawChargedIsolationDBSumPtCorr76xReMiniAOD = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr76xReMiniAOD.clone(
    applySumPtCut = False,
    ApplyDiscriminationByECALIsolation = False,
    storeRawSumPt = cms.bool(True)
)

hpsPFTauDiscriminationByRawGammaIsolationDBSumPtCorr76xReMiniAOD = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr76xReMiniAOD.clone(
    applySumPtCut = False,
    ApplyDiscriminationByTrackerIsolation = False,
    storeRawSumPt = cms.bool(True)
)

hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr76xReMiniAOD = hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr76xReMiniAOD.clone(
    ApplyDiscriminationByTrackerIsolation = True,
    ApplyDiscriminationByECALIsolation = True,
    deltaBetaFactor = "%0.4f"%(ak4dBetaCorrection),
    applyOccupancyCut = False,
    applySumPtCut = True,
    maximumSumPtCut = 1.5,
    Prediscriminants = requireDecayMode76xReMiniAOD.clone()
)
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr76xReMiniAOD.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr76xReMiniAOD.qualityCuts.isolationQualityCuts.minGammaEt = 0.5

hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr76xReMiniAOD = hpsPFTauDiscriminationByTightIsolationDBSumPtCorr76xReMiniAOD.clone(
    ApplyDiscriminationByTrackerIsolation = True,
    ApplyDiscriminationByECALIsolation = True,
    deltaBetaFactor = "%0.4f"%(ak4dBetaCorrection),
    applyOccupancyCut = False,
    applySumPtCut = True,
    maximumSumPtCut = 0.8,
    Prediscriminants = requireDecayMode76xReMiniAOD.clone()
)
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr76xReMiniAOD.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr76xReMiniAOD.qualityCuts.isolationQualityCuts.minGammaEt = 0.5

hpsPFTauDiscriminationByCombinedIsolationSeqDBSumPtCorr76xReMiniAOD = cms.Sequence(
    hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr76xReMiniAOD*
    hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr76xReMiniAOD*
    hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr76xReMiniAOD*
    hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr76xReMiniAOD
)

#Charge isolation based on combined isolation
hpsPFTauDiscriminationByVLooseChargedIsolation76xReMiniAOD = hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr76xReMiniAOD.clone(
    ApplyDiscriminationByECALIsolation = False
)

hpsPFTauDiscriminationByLooseChargedIsolation76xReMiniAOD = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr76xReMiniAOD.clone(
    ApplyDiscriminationByECALIsolation = False
)

hpsPFTauDiscriminationByMediumChargedIsolation76xReMiniAOD = hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr76xReMiniAOD.clone(
    ApplyDiscriminationByECALIsolation = False
)
hpsPFTauDiscriminationByTightChargedIsolation76xReMiniAOD = hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr76xReMiniAOD.clone(
    ApplyDiscriminationByECALIsolation = False
)

hpsPFTauDiscriminationByChargedIsolationSeq76xReMiniAOD = cms.Sequence(
    hpsPFTauDiscriminationByVLooseChargedIsolation76xReMiniAOD*
    hpsPFTauDiscriminationByLooseChargedIsolation76xReMiniAOD*
    hpsPFTauDiscriminationByMediumChargedIsolation76xReMiniAOD*
    hpsPFTauDiscriminationByTightChargedIsolation76xReMiniAOD
)

#copying discriminator against electrons and muons
hpsPFTauDiscriminationByLooseElectronRejection76xReMiniAOD = pfRecoTauDiscriminationAgainstElectron.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = noPrediscriminants,
    PFElectronMVA_maxValue = cms.double(0.6)
)
hpsPFTauDiscriminationByMediumElectronRejection76xReMiniAOD = pfRecoTauDiscriminationAgainstElectron.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = noPrediscriminants,
    ApplyCut_EcalCrackCut = cms.bool(True)
)
hpsPFTauDiscriminationByTightElectronRejection76xReMiniAOD = pfRecoTauDiscriminationAgainstElectron.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = noPrediscriminants,
    ApplyCut_EcalCrackCut = cms.bool(True),
    ApplyCut_BremCombined = cms.bool(True)
)

hpsPFTauDiscriminationByLooseMuonRejection76xReMiniAOD = pfRecoTauDiscriminationAgainstMuon.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = noPrediscriminants
)
hpsPFTauDiscriminationByMediumMuonRejection76xReMiniAOD = pfRecoTauDiscriminationAgainstMuon.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = noPrediscriminants,
    discriminatorOption = cms.string('noAllArbitrated')
)
hpsPFTauDiscriminationByTightMuonRejection76xReMiniAOD = pfRecoTauDiscriminationAgainstMuon.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = noPrediscriminants,
    discriminatorOption = cms.string('noAllArbitratedWithHOP')
)

hpsPFTauDiscriminationByLooseMuonRejection276xReMiniAOD = pfRecoTauDiscriminationAgainstMuon2.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = noPrediscriminants
)
hpsPFTauDiscriminationByMediumMuonRejection276xReMiniAOD = pfRecoTauDiscriminationAgainstMuon2.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = noPrediscriminants,
    discriminatorOption = cms.string('medium')
)
hpsPFTauDiscriminationByTightMuonRejection276xReMiniAOD = pfRecoTauDiscriminationAgainstMuon2.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = noPrediscriminants,
    discriminatorOption = cms.string('tight')
)

hpsPFTauDiscriminationByLooseMuonRejection376xReMiniAOD = pfRecoTauDiscriminationAgainstMuon2.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = noPrediscriminants,
    discriminatorOption = cms.string('custom'),
    maxNumberOfMatches = cms.int32(1),
    doCaloMuonVeto = cms.bool(True),
    maxNumberOfHitsLast2Stations = cms.int32(-1)
)
hpsPFTauDiscriminationByTightMuonRejection376xReMiniAOD = hpsPFTauDiscriminationByLooseMuonRejection376xReMiniAOD.clone(
    maxNumberOfHitsLast2Stations = cms.int32(0)
)

hpsPFTauDiscriminationByMVArawMuonRejection76xReMiniAOD = pfRecoTauDiscriminationAgainstMuonMVA.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = requireDecayMode76xReMiniAOD.clone(),
    loadMVAfromDB = cms.bool(True),
    returnMVA = cms.bool(True),
    mvaName = cms.string("RecoTauTag_againstMuonMVAv1")
)
##hpsPFTauDiscriminationByMVALooseMuonRejection76xReMiniAOD = hpsPFTauDiscriminationByMVArawMuonRejection76xReMiniAOD.clone(
##    returnMVA = cms.bool(False),
##    mvaMin = cms.double(0.75)
##)
##hpsPFTauDiscriminationByMVAMediumMuonRejection76xReMiniAOD = hpsPFTauDiscriminationByMVALooseMuonRejection76xReMiniAOD.clone(
##    mvaMin = cms.double(0.950)
##)
##hpsPFTauDiscriminationByMVATightMuonRejection76xReMiniAOD = hpsPFTauDiscriminationByMVALooseMuonRejection76xReMiniAOD.clone(
##    mvaMin = cms.double(0.975)
##)
hpsPFTauDiscriminationByMVALooseMuonRejection76xReMiniAOD = recoTauDiscriminantCutMultiplexer.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = requireDecayMode76xReMiniAOD.clone(),    
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByMVArawMuonRejection76xReMiniAOD'),
    key = cms.InputTag('hpsPFTauDiscriminationByMVArawMuonRejection76xReMiniAOD:category'),
    loadMVAfromDB = cms.bool(True),
    mvaOutput_normalization = cms.string("RecoTauTag_againstMuonMVAv1_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_againstMuonMVAv1_WPeff99_5"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByMVAMediumMuonRejection76xReMiniAOD = hpsPFTauDiscriminationByMVALooseMuonRejection76xReMiniAOD.clone()
hpsPFTauDiscriminationByMVAMediumMuonRejection76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_againstMuonMVAv1_WPeff99_0")
hpsPFTauDiscriminationByMVATightMuonRejection76xReMiniAOD = hpsPFTauDiscriminationByMVALooseMuonRejection76xReMiniAOD.clone()
hpsPFTauDiscriminationByMVATightMuonRejection76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_againstMuonMVAv1_WPeff98_0")

hpsPFTauDiscriminationByMVA5rawElectronRejection76xReMiniAOD = pfRecoTauDiscriminationAgainstElectronMVA5.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = requireDecayMode76xReMiniAOD.clone(),
    loadMVAfromDB = cms.bool(True),
    mvaName_NoEleMatch_woGwoGSF_BL = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_BL"),
    mvaName_NoEleMatch_woGwGSF_BL = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_BL"),
    mvaName_NoEleMatch_wGwoGSF_BL = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_BL"),
    mvaName_NoEleMatch_wGwGSF_BL = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_BL"),
    mvaName_woGwoGSF_BL = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_BL"),
    mvaName_woGwGSF_BL = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_BL"),
    mvaName_wGwoGSF_BL = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_BL"),
    mvaName_wGwGSF_BL = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_BL"),
    mvaName_NoEleMatch_woGwoGSF_EC = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_EC"),
    mvaName_NoEleMatch_woGwGSF_EC = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_EC"),
    mvaName_NoEleMatch_wGwoGSF_EC = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_EC"),
    mvaName_NoEleMatch_wGwGSF_EC = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_EC"),
    mvaName_woGwoGSF_EC = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_EC"),
    mvaName_woGwGSF_EC = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_EC"),
    mvaName_wGwoGSF_EC = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_EC"),
    mvaName_wGwGSF_EC = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_EC")
)

hpsPFTauDiscriminationByMVA5VLooseElectronRejection76xReMiniAOD = recoTauDiscriminantCutMultiplexer.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = requireDecayMode76xReMiniAOD.clone(),
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByMVA5rawElectronRejection76xReMiniAOD'),
    key = cms.InputTag('hpsPFTauDiscriminationByMVA5rawElectronRejection76xReMiniAOD:category'),
    loadMVAfromDB = cms.bool(True),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0), # minMVANoEleMatchWOgWOgsfBL
            cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_BL_WPeff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(1), # minMVANoEleMatchWOgWgsfBL
            cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_BL_WPeff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(2), # minMVANoEleMatchWgWOgsfBL
            cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_BL_WPeff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(3), # minMVANoEleMatchWgWgsfBL
            cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_BL_WPeff99"),
            variable = cms.string("pt")
        ),
         cms.PSet(
            category = cms.uint32(4), # minMVAWOgWOgsfBL
            cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_BL_WPeff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(5), # minMVAWOgWgsfBL
            cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_BL_WPeff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(6), # minMVAWgWOgsfBL
            cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_BL_WPeff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(7), # minMVAWgWgsfBL
            cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_BL_WPeff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(8), # minMVANoEleMatchWOgWOgsfEC
            cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_EC_WPeff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(9), # minMVANoEleMatchWOgWgsfEC
            cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_EC_WPeff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(10), # minMVANoEleMatchWgWOgsfEC
            cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_EC_WPeff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(11), # minMVANoEleMatchWgWgsfEC
            cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_EC_WPeff99"),
            variable = cms.string("pt")
        ),
         cms.PSet(
            category = cms.uint32(12), # minMVAWOgWOgsfEC
            cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_EC_WPeff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(13), # minMVAWOgWgsfEC
            cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_EC_WPeff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(14), # minMVAWgWOgsfEC
            cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_EC_WPeff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(15), # minMVAWgWgsfEC
            cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_EC_WPeff99"),
            variable = cms.string("pt")
        )
    )
)

hpsPFTauDiscriminationByMVA5LooseElectronRejection76xReMiniAOD = copy.deepcopy(hpsPFTauDiscriminationByMVA5VLooseElectronRejection76xReMiniAOD)
hpsPFTauDiscriminationByMVA5LooseElectronRejection76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_BL_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection76xReMiniAOD.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_BL_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection76xReMiniAOD.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_BL_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection76xReMiniAOD.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_BL_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection76xReMiniAOD.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_BL_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection76xReMiniAOD.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_BL_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection76xReMiniAOD.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_BL_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection76xReMiniAOD.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_BL_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection76xReMiniAOD.mapping[8].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_EC_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection76xReMiniAOD.mapping[9].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_EC_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection76xReMiniAOD.mapping[10].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_EC_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection76xReMiniAOD.mapping[11].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_EC_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection76xReMiniAOD.mapping[12].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_EC_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection76xReMiniAOD.mapping[13].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_EC_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection76xReMiniAOD.mapping[14].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_EC_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection76xReMiniAOD.mapping[15].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_EC_WPeff96")

hpsPFTauDiscriminationByMVA5MediumElectronRejection76xReMiniAOD = copy.deepcopy(hpsPFTauDiscriminationByMVA5VLooseElectronRejection76xReMiniAOD)
hpsPFTauDiscriminationByMVA5MediumElectronRejection76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_BL_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection76xReMiniAOD.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_BL_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection76xReMiniAOD.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_BL_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection76xReMiniAOD.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_BL_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection76xReMiniAOD.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_BL_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection76xReMiniAOD.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_BL_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection76xReMiniAOD.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_BL_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection76xReMiniAOD.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_BL_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection76xReMiniAOD.mapping[8].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_EC_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection76xReMiniAOD.mapping[9].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_EC_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection76xReMiniAOD.mapping[10].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_EC_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection76xReMiniAOD.mapping[11].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_EC_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection76xReMiniAOD.mapping[12].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_EC_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection76xReMiniAOD.mapping[13].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_EC_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection76xReMiniAOD.mapping[14].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_EC_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection76xReMiniAOD.mapping[15].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_EC_WPeff91")

hpsPFTauDiscriminationByMVA5TightElectronRejection76xReMiniAOD = copy.deepcopy(hpsPFTauDiscriminationByMVA5VLooseElectronRejection76xReMiniAOD)
hpsPFTauDiscriminationByMVA5TightElectronRejection76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_BL_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection76xReMiniAOD.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_BL_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection76xReMiniAOD.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_BL_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection76xReMiniAOD.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_BL_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection76xReMiniAOD.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_BL_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection76xReMiniAOD.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_BL_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection76xReMiniAOD.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_BL_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection76xReMiniAOD.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_BL_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection76xReMiniAOD.mapping[8].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_EC_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection76xReMiniAOD.mapping[9].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_EC_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection76xReMiniAOD.mapping[10].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_EC_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection76xReMiniAOD.mapping[11].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_EC_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection76xReMiniAOD.mapping[12].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_EC_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection76xReMiniAOD.mapping[13].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_EC_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection76xReMiniAOD.mapping[14].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_EC_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection76xReMiniAOD.mapping[15].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_EC_WPeff85")

hpsPFTauDiscriminationByMVA5VTightElectronRejection76xReMiniAOD = copy.deepcopy(hpsPFTauDiscriminationByMVA5VLooseElectronRejection76xReMiniAOD)
hpsPFTauDiscriminationByMVA5VTightElectronRejection76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_BL_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection76xReMiniAOD.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_BL_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection76xReMiniAOD.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_BL_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection76xReMiniAOD.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_BL_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection76xReMiniAOD.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_BL_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection76xReMiniAOD.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_BL_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection76xReMiniAOD.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_BL_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection76xReMiniAOD.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_BL_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection76xReMiniAOD.mapping[8].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_EC_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection76xReMiniAOD.mapping[9].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_EC_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection76xReMiniAOD.mapping[10].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_EC_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection76xReMiniAOD.mapping[11].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_EC_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection76xReMiniAOD.mapping[12].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_EC_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection76xReMiniAOD.mapping[13].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_EC_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection76xReMiniAOD.mapping[14].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_EC_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection76xReMiniAOD.mapping[15].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_EC_WPeff79")

hpsPFTauDiscriminationByMVA6rawElectronRejection76xReMiniAOD = pfRecoTauDiscriminationAgainstElectronMVA6.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = requireDecayMode76xReMiniAOD.clone(),
    loadMVAfromDB = cms.bool(True),
    mvaName_NoEleMatch_woGwoGSF_BL = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_BL"),
    mvaName_NoEleMatch_wGwoGSF_BL = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_BL"),
    mvaName_woGwGSF_BL = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_BL"),
    mvaName_wGwGSF_BL = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_BL"),
    mvaName_NoEleMatch_woGwoGSF_EC = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_EC"),
    mvaName_NoEleMatch_wGwoGSF_EC = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_EC"),
    mvaName_woGwGSF_EC = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_EC"),
    mvaName_wGwGSF_EC = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_EC")
)

hpsPFTauDiscriminationByMVA6VLooseElectronRejection76xReMiniAOD = recoTauDiscriminantCutMultiplexer.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = requireDecayMode76xReMiniAOD.clone(),
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByMVA6rawElectronRejection76xReMiniAOD'),
    key = cms.InputTag('hpsPFTauDiscriminationByMVA6rawElectronRejection76xReMiniAOD:category'),
    loadMVAfromDB = cms.bool(True),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0), # minMVANoEleMatchWOgWOgsfBL
            cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_BL_WPEff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(2), # minMVANoEleMatchWgWOgsfBL
            cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_BL_WPEff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(5), # minMVAWOgWgsfBL
            cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_BL_WPEff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(7), # minMVAWgWgsfBL
            cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_BL_WPEff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(8), # minMVANoEleMatchWOgWOgsfEC
            cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_EC_WPEff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(10), # minMVANoEleMatchWgWOgsfEC
            cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_EC_WPEff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(13), # minMVAWOgWgsfEC
            cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_EC_WPEff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(15), # minMVAWgWgsfEC
            cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_EC_WPEff99"),
            variable = cms.string("pt")
        )
    )
)

hpsPFTauDiscriminationByMVA6LooseElectronRejection76xReMiniAOD = copy.deepcopy(hpsPFTauDiscriminationByMVA6VLooseElectronRejection76xReMiniAOD)
hpsPFTauDiscriminationByMVA6LooseElectronRejection76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_BL_WPEff96")
hpsPFTauDiscriminationByMVA6LooseElectronRejection76xReMiniAOD.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_BL_WPEff96")
hpsPFTauDiscriminationByMVA6LooseElectronRejection76xReMiniAOD.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_BL_WPEff96")
hpsPFTauDiscriminationByMVA6LooseElectronRejection76xReMiniAOD.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_BL_WPEff96")
hpsPFTauDiscriminationByMVA6LooseElectronRejection76xReMiniAOD.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_EC_WPEff96")
hpsPFTauDiscriminationByMVA6LooseElectronRejection76xReMiniAOD.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_EC_WPEff96")
hpsPFTauDiscriminationByMVA6LooseElectronRejection76xReMiniAOD.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_EC_WPEff96")
hpsPFTauDiscriminationByMVA6LooseElectronRejection76xReMiniAOD.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_EC_WPEff96")

hpsPFTauDiscriminationByMVA6MediumElectronRejection76xReMiniAOD = copy.deepcopy(hpsPFTauDiscriminationByMVA6VLooseElectronRejection76xReMiniAOD)
hpsPFTauDiscriminationByMVA6MediumElectronRejection76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_BL_WPEff91")
hpsPFTauDiscriminationByMVA6MediumElectronRejection76xReMiniAOD.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_BL_WPEff91")
hpsPFTauDiscriminationByMVA6MediumElectronRejection76xReMiniAOD.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_BL_WPEff91")
hpsPFTauDiscriminationByMVA6MediumElectronRejection76xReMiniAOD.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_BL_WPEff91")
hpsPFTauDiscriminationByMVA6MediumElectronRejection76xReMiniAOD.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_EC_WPEff91")
hpsPFTauDiscriminationByMVA6MediumElectronRejection76xReMiniAOD.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_EC_WPEff91")
hpsPFTauDiscriminationByMVA6MediumElectronRejection76xReMiniAOD.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_EC_WPEff91")
hpsPFTauDiscriminationByMVA6MediumElectronRejection76xReMiniAOD.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_EC_WPEff91")

hpsPFTauDiscriminationByMVA6TightElectronRejection76xReMiniAOD = copy.deepcopy(hpsPFTauDiscriminationByMVA6VLooseElectronRejection76xReMiniAOD)
hpsPFTauDiscriminationByMVA6TightElectronRejection76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_BL_WPEff85")
hpsPFTauDiscriminationByMVA6TightElectronRejection76xReMiniAOD.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_BL_WPEff85")
hpsPFTauDiscriminationByMVA6TightElectronRejection76xReMiniAOD.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_BL_WPEff85")
hpsPFTauDiscriminationByMVA6TightElectronRejection76xReMiniAOD.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_BL_WPEff85")
hpsPFTauDiscriminationByMVA6TightElectronRejection76xReMiniAOD.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_EC_WPEff85")
hpsPFTauDiscriminationByMVA6TightElectronRejection76xReMiniAOD.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_EC_WPEff85")
hpsPFTauDiscriminationByMVA6TightElectronRejection76xReMiniAOD.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_EC_WPEff85")
hpsPFTauDiscriminationByMVA6TightElectronRejection76xReMiniAOD.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_EC_WPEff85")

hpsPFTauDiscriminationByMVA6VTightElectronRejection76xReMiniAOD = copy.deepcopy(hpsPFTauDiscriminationByMVA6VLooseElectronRejection76xReMiniAOD)
hpsPFTauDiscriminationByMVA6VTightElectronRejection76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_BL_WPEff79")
hpsPFTauDiscriminationByMVA6VTightElectronRejection76xReMiniAOD.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_BL_WPEff79")
hpsPFTauDiscriminationByMVA6VTightElectronRejection76xReMiniAOD.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_BL_WPEff79")
hpsPFTauDiscriminationByMVA6VTightElectronRejection76xReMiniAOD.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_BL_WPEff79")
hpsPFTauDiscriminationByMVA6VTightElectronRejection76xReMiniAOD.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_EC_WPEff79")
hpsPFTauDiscriminationByMVA6VTightElectronRejection76xReMiniAOD.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_EC_WPEff79")
hpsPFTauDiscriminationByMVA6VTightElectronRejection76xReMiniAOD.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_EC_WPEff79")
hpsPFTauDiscriminationByMVA6VTightElectronRejection76xReMiniAOD.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_EC_WPEff79")

hpsPFTauDiscriminationByDeadECALElectronRejection76xReMiniAOD = pfRecoTauDiscriminationAgainstElectronDeadECAL.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = requireDecayMode76xReMiniAOD.clone()
)

#Define new sequence that is using smaller number on hits cut
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr76xReMiniAOD.clone()
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD = hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr76xReMiniAOD.clone()
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD = hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr76xReMiniAOD.clone()

hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD.qualityCuts.isolationQualityCuts.minTrackHits = cms.uint32(3)
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD.applyPhotonPtSumOutsideSignalConeCut = cms.bool(True)
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD.qualityCuts.isolationQualityCuts.minTrackHits = cms.uint32(3)
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD.applyPhotonPtSumOutsideSignalConeCut = cms.bool(True)
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD.qualityCuts.isolationQualityCuts.minTrackHits = cms.uint32(3)
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD.applyPhotonPtSumOutsideSignalConeCut = cms.bool(True)

hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD.clone(
    applySumPtCut = False,
    storeRawSumPt = cms.bool(True)
)

hpsPFTauDiscriminationByCombinedIsolationSeqDBSumPtCorr3Hits76xReMiniAOD = cms.Sequence(
    hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD*
    hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD*
    hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD*
    hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD
)

hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3HitsdR0376xReMiniAOD = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD.clone()
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3HitsdR0376xReMiniAOD = hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD.clone()
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3HitsdR0376xReMiniAOD = hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD.clone()

hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3HitsdR0376xReMiniAOD.deltaBetaFactor = cms.string('0.0720') # 0.2*(0.3/0.5)^2
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3HitsdR0376xReMiniAOD.customOuterCone = cms.double(0.3)
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3HitsdR0376xReMiniAOD.deltaBetaFactor = cms.string('0.0720') # 0.2*(0.3/0.5)^2
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3HitsdR0376xReMiniAOD.customOuterCone = cms.double(0.3)
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3HitsdR0376xReMiniAOD.deltaBetaFactor = cms.string('0.0720') # 0.2*(0.3/0.5)^2
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3HitsdR0376xReMiniAOD.customOuterCone = cms.double(0.3)

hpsPFTauDiscriminationByCombinedIsolationSeqDBSumPtCorr3HitsdR0376xReMiniAOD = cms.Sequence(
    hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3HitsdR0376xReMiniAOD*
    hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3HitsdR0376xReMiniAOD*
    hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3HitsdR0376xReMiniAOD
)

hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits76xReMiniAOD = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD.clone(
    ApplyDiscriminationByECALIsolation = cms.bool(False),
    applyDeltaBetaCorrection = cms.bool(False),
    ApplyDiscriminationByWeightedECALIsolation = cms.bool(True),
    UseAllPFCandsForWeights = cms.bool(True),
    applyFootprintCorrection = cms.bool(True),
    applyPhotonPtSumOutsideSignalConeCut = cms.bool(True)    
)

hpsPFTauDiscriminationByMediumPileupWeightedIsolation3Hits76xReMiniAOD = hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits76xReMiniAOD.clone(
    maximumSumPtCut = hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD.maximumSumPtCut
)

hpsPFTauDiscriminationByTightPileupWeightedIsolation3Hits76xReMiniAOD = hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits76xReMiniAOD.clone(
    maximumSumPtCut = hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD.maximumSumPtCut
)

hpsPFTauDiscriminationByPhotonPtSumOutsideSignalCone76xReMiniAOD = hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits76xReMiniAOD.clone(
    applySumPtCut = cms.bool(False)
)

hpsPFTauDiscriminationByRawPileupWeightedIsolation3Hits76xReMiniAOD = hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits76xReMiniAOD.clone(
    Prediscriminants = cms.PSet(
        BooleanOperator = cms.string("and"),
        decayMode = cms.PSet(
            Producer = cms.InputTag('hpsPFTauDiscriminationByPhotonPtSumOutsideSignalCone76xReMiniAOD'),
            cut = cms.double(0.5)
        )
    ),
    applySumPtCut = cms.bool(False),
    storeRawSumPt = cms.bool(True)
)

hpsPFTauDiscriminationByPileupWeightedIsolationSeq3Hits76xReMiniAOD = cms.Sequence(
    hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits76xReMiniAOD*
    hpsPFTauDiscriminationByMediumPileupWeightedIsolation3Hits76xReMiniAOD*
    hpsPFTauDiscriminationByTightPileupWeightedIsolation3Hits76xReMiniAOD*
    hpsPFTauDiscriminationByPhotonPtSumOutsideSignalCone76xReMiniAOD*
    hpsPFTauDiscriminationByRawPileupWeightedIsolation3Hits76xReMiniAOD
)

# Define the HPS selection discriminator used in cleaning
hpsSelectionDiscriminator76xReMiniAOD.PFTauProducer = cms.InputTag("combinatoricRecoTaus76xReMiniAOD")
#----------------------------------------------------------------------------
# CV: disable 3Prong1Pi0 decay mode
hpsSelectionDiscriminator76xReMiniAOD.decayModes = cms.VPSet(
    decayMode_1Prong0Pi0,
    decayMode_1Prong1Pi0,
    decayMode_1Prong2Pi0,
    decayMode_2Prong0Pi0,
    decayMode_2Prong1Pi0,
    decayMode_3Prong0Pi0
)
#----------------------------------------------------------------------------

from RecoTauTag.RecoTau.RecoTauCleaner_cfi import RecoTauCleaner
hpsPFTauProducerSansRefs76xReMiniAOD = RecoTauCleaner.clone(
    src = cms.InputTag("combinatoricRecoTaus76xReMiniAOD")
)
hpsPFTauProducerSansRefs76xReMiniAOD.cleaners[1].src = cms.InputTag("hpsSelectionDiscriminator76xReMiniAOD")

from RecoTauTag.RecoTau.RecoTauPiZeroUnembedder_cfi import RecoTauPiZeroUnembedder
hpsPFTauProducer76xReMiniAOD = RecoTauPiZeroUnembedder.clone(
    src = cms.InputTag("hpsPFTauProducerSansRefs76xReMiniAOD")
)

from RecoTauTag.RecoTau.PFTauPrimaryVertexProducer_cfi      import *
from RecoTauTag.RecoTau.PFTauSecondaryVertexProducer_cfi    import *
from RecoTauTag.RecoTau.PFTauTransverseImpactParameters_cfi import *
hpsPFTauPrimaryVertexProducer76xReMiniAOD = PFTauPrimaryVertexProducer.clone(
    PFTauTag = cms.InputTag("hpsPFTauProducer76xReMiniAOD"),
    ElectronTag = cms.InputTag(""),
    MuonTag = cms.InputTag(""),
    PVTag = cms.InputTag("offlinePrimaryVertices"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    TrackCollectionTag = cms.InputTag("generalTracks"),
    Algorithm = cms.int32(1),
    useBeamSpot = cms.bool(True),
    RemoveMuonTracks = cms.bool(False),
    RemoveElectronTracks = cms.bool(False),
    useSelectedTaus = cms.bool(False),
    discriminators = cms.VPSet(
        cms.PSet(
            discriminator = cms.InputTag('hpsPFTauDiscriminationByDecayModeFindingNewDMs76xReMiniAOD'),
            selectionCut = cms.double(0.5)
        )
    ),
    cut = cms.string("pt > 18.0 & abs(eta) < 2.4")
)

hpsPFTauSecondaryVertexProducer76xReMiniAOD = PFTauSecondaryVertexProducer.clone(
    PFTauTag = cms.InputTag("hpsPFTauProducer76xReMiniAOD")
)
hpsPFTauTransverseImpactParameters76xReMiniAOD = PFTauTransverseImpactParameters.clone(
    PFTauTag = cms.InputTag("hpsPFTauProducer76xReMiniAOD"),
    PFTauPVATag = cms.InputTag("hpsPFTauPrimaryVertexProducer76xReMiniAOD"),
    PFTauSVATag = cms.InputTag("hpsPFTauSecondaryVertexProducer76xReMiniAOD"),
    useFullCalculation = cms.bool(True)
)
hpsPFTauVertexAndImpactParametersSeq76xReMiniAOD = cms.Sequence(
    hpsPFTauPrimaryVertexProducer76xReMiniAOD*
    hpsPFTauSecondaryVertexProducer76xReMiniAOD*
    hpsPFTauTransverseImpactParameters76xReMiniAOD
)

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByMVAIsolation2_cff import *
hpsPFTauChargedIsoPtSum76xReMiniAOD = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = requireDecayMode76xReMiniAOD.clone(),
    ApplyDiscriminationByECALIsolation = cms.bool(False),
    ApplyDiscriminationByTrackerIsolation = cms.bool(True),
    applySumPtCut = cms.bool(False),
    applyDeltaBetaCorrection = cms.bool(False),
    storeRawSumPt = cms.bool(True),
    storeRawPUsumPt = cms.bool(False),     
    customOuterCone = PFRecoTauPFJetInputs.isolationConeSize,
    isoConeSizeForDeltaBeta = cms.double(0.8),
    verbosity = cms.int32(0)
)
hpsPFTauNeutralIsoPtSum76xReMiniAOD = hpsPFTauChargedIsoPtSum76xReMiniAOD.clone(
    ApplyDiscriminationByECALIsolation = cms.bool(True),
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    verbosity = cms.int32(0)
)
hpsPFTauPUcorrPtSum76xReMiniAOD = hpsPFTauChargedIsoPtSum76xReMiniAOD.clone(
    ApplyDiscriminationByECALIsolation = cms.bool(False),
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    applyDeltaBetaCorrection = cms.bool(True),
    storeRawSumPt = cms.bool(False),
    storeRawPUsumPt = cms.bool(True),
    verbosity = cms.int32(0)
)
hpsPFTauNeutralIsoPtSumWeight76xReMiniAOD = hpsPFTauChargedIsoPtSum76xReMiniAOD.clone(
    ApplyDiscriminationByWeightedECALIsolation = cms.bool(True),
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    UseAllPFCandsForWeights = cms.bool(True),
    verbosity = cms.int32(0)
)
hpsPFTauFootprintCorrection76xReMiniAOD = hpsPFTauChargedIsoPtSum76xReMiniAOD.clone(    
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    storeRawSumPt = cms.bool(False),
    storeRawFootprintCorrection = cms.bool(True),
    verbosity = cms.int32(0)
)
hpsPFTauPhotonPtSumOutsideSignalCone76xReMiniAOD = hpsPFTauChargedIsoPtSum76xReMiniAOD.clone(    
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    storeRawSumPt = cms.bool(False),
    storeRawPhotonSumPt_outsideSignalCone = cms.bool(True),
    verbosity = cms.int32(0)
)

hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw76xReMiniAOD = discriminationByIsolationMVA2raw.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = requireDecayMode76xReMiniAOD.clone(),
    loadMVAfromDB = cms.bool(True),
    mvaName = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1"),
    mvaOpt = cms.string("oldDMwoLT"),
    srcTauTransverseImpactParameters = cms.InputTag('hpsPFTauTransverseImpactParameters76xReMiniAOD'),    
    srcChargedIsoPtSum = cms.InputTag('hpsPFTauChargedIsoPtSum76xReMiniAOD'),
    srcNeutralIsoPtSum = cms.InputTag('hpsPFTauNeutralIsoPtSum76xReMiniAOD'),
    srcPUcorrPtSum = cms.InputTag('hpsPFTauPUcorrPtSum76xReMiniAOD'),
    verbosity = cms.int32(0)
)
hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT76xReMiniAOD = discriminationByIsolationMVA2VLoose.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = requireDecayMode76xReMiniAOD.clone(),    
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw76xReMiniAOD'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw76xReMiniAOD:category'),
    loadMVAfromDB = cms.bool(True),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_WPEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwoLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwoLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_WPEff80")
hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwoLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwoLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_WPEff70")
hpsPFTauDiscriminationByTightIsolationMVA3oldDMwoLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByTightIsolationMVA3oldDMwoLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_WPEff60")
hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwoLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwoLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_WPEff50")
hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwoLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwoLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_WPEff40")
hpsPFTauDiscriminationByIsolationMVA3oldDMwLTraw76xReMiniAOD = hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw76xReMiniAOD.clone(
    mvaName = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1"),
    mvaOpt = cms.string("oldDMwLT"),
    verbosity = cms.int32(0)
)
hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT76xReMiniAOD.clone(
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3oldDMwLTraw76xReMiniAOD'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3oldDMwLTraw76xReMiniAOD:category'),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_WPEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_WPEff80")
hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_WPEff70")
hpsPFTauDiscriminationByTightIsolationMVA3oldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByTightIsolationMVA3oldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_WPEff60")
hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_WPEff50")
hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_WPEff40")
hpsPFTauDiscriminationByIsolationMVA3newDMwoLTraw76xReMiniAOD = hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw76xReMiniAOD.clone(
    mvaName = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1"),
    mvaOpt = cms.string("newDMwoLT"),
    verbosity = cms.int32(0)
)
hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT76xReMiniAOD.clone(
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3newDMwoLTraw76xReMiniAOD'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3newDMwoLTraw76xReMiniAOD:category'),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1_WPEff90"),
            variable = cms.string("pt")
        )
    ),
    verbosity = cms.int32(0)
)
hpsPFTauDiscriminationByLooseIsolationMVA3newDMwoLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByLooseIsolationMVA3newDMwoLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1_WPEff80")
hpsPFTauDiscriminationByMediumIsolationMVA3newDMwoLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByMediumIsolationMVA3newDMwoLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1_WPEff70")
hpsPFTauDiscriminationByTightIsolationMVA3newDMwoLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByTightIsolationMVA3newDMwoLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1_WPEff60")
hpsPFTauDiscriminationByVTightIsolationMVA3newDMwoLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByVTightIsolationMVA3newDMwoLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1_WPEff50")
hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwoLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwoLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1_WPEff40")
hpsPFTauDiscriminationByIsolationMVA3newDMwLTraw76xReMiniAOD = hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw76xReMiniAOD.clone(
    mvaName = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1"),
    mvaOpt = cms.string("newDMwLT"),
    verbosity = cms.int32(0)
)
hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT76xReMiniAOD.clone(
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3newDMwLTraw76xReMiniAOD'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3newDMwLTraw76xReMiniAOD:category'),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_WPEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVA3newDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByLooseIsolationMVA3newDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_WPEff80")
hpsPFTauDiscriminationByMediumIsolationMVA3newDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByMediumIsolationMVA3newDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_WPEff70")
hpsPFTauDiscriminationByTightIsolationMVA3newDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByTightIsolationMVA3newDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_WPEff60")
hpsPFTauDiscriminationByVTightIsolationMVA3newDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByVTightIsolationMVA3newDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_WPEff50")
hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_WPEff40")

#Define new Run2 MVA isolations
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByMVAIsolationRun2_cff import *
hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw76xReMiniAOD = discriminationByIsolationMVArun2v1raw.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = requireDecayMode76xReMiniAOD.clone(),
    loadMVAfromDB = cms.bool(True),
    mvaName = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1"),
    mvaOpt = cms.string("DBoldDMwLT"),
    srcTauTransverseImpactParameters = cms.InputTag('hpsPFTauTransverseImpactParameters76xReMiniAOD'),
    srcChargedIsoPtSum = cms.InputTag('hpsPFTauChargedIsoPtSum76xReMiniAOD'),
    srcNeutralIsoPtSum = cms.InputTag('hpsPFTauNeutralIsoPtSum76xReMiniAOD'),
    srcPUcorrPtSum = cms.InputTag('hpsPFTauPUcorrPtSum76xReMiniAOD'),
    srcPhotonPtSumOutsideSignalCone = cms.InputTag('hpsPFTauPhotonPtSumOutsideSignalCone76xReMiniAOD'),
    srcFootprintCorrection = cms.InputTag('hpsPFTauFootprintCorrection76xReMiniAOD'),
    verbosity = cms.int32(0)
)

hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT76xReMiniAOD = discriminationByIsolationMVArun2v1VLoose.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = requireDecayMode76xReMiniAOD.clone(),
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw76xReMiniAOD'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw76xReMiniAOD:category'),
    loadMVAfromDB = cms.bool(True),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff80")
hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff70")
hpsPFTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff60")
hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff50")
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff40")

hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw76xReMiniAOD = hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw76xReMiniAOD.clone(
    mvaName = cms.string("RecoTauTag_tauIdMVADBnewDMwLTv1"),
    mvaOpt = cms.string("DBnewDMwLT"),
    verbosity = cms.int32(0)
)

hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT76xReMiniAOD.clone(
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw76xReMiniAOD'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw76xReMiniAOD:category'),
    loadMVAfromDB = cms.bool(True),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVADBnewDMwLTv1_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVADBnewDMwLTv1_WPEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBnewDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBnewDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBnewDMwLTv1_WPEff80")
hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBnewDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBnewDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBnewDMwLTv1_WPEff70")
hpsPFTauDiscriminationByTightIsolationMVArun2v1DBnewDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByTightIsolationMVArun2v1DBnewDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBnewDMwLTv1_WPEff60")
hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBnewDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBnewDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBnewDMwLTv1_WPEff50")
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBnewDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBnewDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBnewDMwLTv1_WPEff40")

hpsPFTauDiscriminationByIsolationMVArun2v1PWoldDMwLTraw76xReMiniAOD = hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw76xReMiniAOD.clone(
    mvaName = cms.string("RecoTauTag_tauIdMVAPWoldDMwLTv1"),
    mvaOpt = cms.string("PWoldDMwLT"),
    srcPUcorrPtSum = cms.InputTag('hpsPFTauNeutralIsoPtSumWeight76xReMiniAOD'),
    verbosity = cms.int32(0)
)

hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWoldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT76xReMiniAOD.clone(
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1PWoldDMwLTraw76xReMiniAOD'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1PWoldDMwLTraw76xReMiniAOD:category'),
    loadMVAfromDB = cms.bool(True),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAPWoldDMwLTv1_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVAPWoldDMwLTv1_WPEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWoldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWoldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWoldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWoldDMwLTv1_WPEff80")
hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWoldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWoldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWoldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWoldDMwLTv1_WPEff70")
hpsPFTauDiscriminationByTightIsolationMVArun2v1PWoldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWoldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByTightIsolationMVArun2v1PWoldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWoldDMwLTv1_WPEff60")
hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWoldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWoldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWoldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWoldDMwLTv1_WPEff50")
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWoldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWoldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWoldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWoldDMwLTv1_WPEff40")

hpsPFTauDiscriminationByIsolationMVArun2v1PWnewDMwLTraw76xReMiniAOD = hpsPFTauDiscriminationByIsolationMVArun2v1PWoldDMwLTraw76xReMiniAOD.clone(
    mvaName = cms.string("RecoTauTag_tauIdMVAPWnewDMwLTv1"),
    mvaOpt = cms.string("PWnewDMwLT"),
    verbosity = cms.int32(0)
)

hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWnewDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWoldDMwLT76xReMiniAOD.clone(
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1PWnewDMwLTraw76xReMiniAOD'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1PWnewDMwLTraw76xReMiniAOD:category'),
    loadMVAfromDB = cms.bool(True),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAPWnewDMwLTv1_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVAPWnewDMwLTv1_WPEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWnewDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWnewDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWnewDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWnewDMwLTv1_WPEff80")
hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWnewDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWnewDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWnewDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWnewDMwLTv1_WPEff70")
hpsPFTauDiscriminationByTightIsolationMVArun2v1PWnewDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWnewDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByTightIsolationMVArun2v1PWnewDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWnewDMwLTv1_WPEff60")
hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWnewDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWnewDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWnewDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWnewDMwLTv1_WPEff50")
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWnewDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWnewDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWnewDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWnewDMwLTv1_WPEff40")

hpsPFTauChargedIsoPtSumdR0376xReMiniAOD = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits76xReMiniAOD.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = requireDecayMode76xReMiniAOD.clone(),
    ApplyDiscriminationByECALIsolation = cms.bool(False),
    ApplyDiscriminationByTrackerIsolation = cms.bool(True),
    applySumPtCut = cms.bool(False),
    applyDeltaBetaCorrection = cms.bool(False),
    storeRawSumPt = cms.bool(True),
    storeRawPUsumPt = cms.bool(False),
    customOuterCone = cms.double(0.3),
    isoConeSizeForDeltaBeta = cms.double(0.8),
    verbosity = cms.int32(0)
)
hpsPFTauNeutralIsoPtSumdR0376xReMiniAOD = hpsPFTauChargedIsoPtSumdR0376xReMiniAOD.clone(
    ApplyDiscriminationByECALIsolation = cms.bool(True),
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    verbosity = cms.int32(0)
)
hpsPFTauPUcorrPtSumdR0376xReMiniAOD = hpsPFTauChargedIsoPtSumdR0376xReMiniAOD.clone(
    ApplyDiscriminationByECALIsolation = cms.bool(False),
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    applyDeltaBetaCorrection = cms.bool(True),
    storeRawSumPt = cms.bool(False),
    storeRawPUsumPt = cms.bool(True),
    verbosity = cms.int32(0)
)
hpsPFTauNeutralIsoPtSumWeightdR0376xReMiniAOD = hpsPFTauChargedIsoPtSumdR0376xReMiniAOD.clone(
    ApplyDiscriminationByWeightedECALIsolation = cms.bool(True),
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    UseAllPFCandsForWeights = cms.bool(True),
    verbosity = cms.int32(0)
)
hpsPFTauFootprintCorrectiondR0376xReMiniAOD = hpsPFTauChargedIsoPtSumdR0376xReMiniAOD.clone(
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    storeRawSumPt = cms.bool(False),
    storeRawFootprintCorrection = cms.bool(True),
    verbosity = cms.int32(0)
)
hpsPFTauPhotonPtSumOutsideSignalConedR0376xReMiniAOD = hpsPFTauChargedIsoPtSumdR0376xReMiniAOD.clone(
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    storeRawSumPt = cms.bool(False),
    storeRawPhotonSumPt_outsideSignalCone = cms.bool(True),
    verbosity = cms.int32(0)
)

hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTraw76xReMiniAOD = hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw76xReMiniAOD.clone(
    mvaName = cms.string("RecoTauTag_tauIdMVADBdR03oldDMwLTv1"),
    mvaOpt = cms.string("DBoldDMwLT"),
    srcChargedIsoPtSum = cms.InputTag('hpsPFTauChargedIsoPtSumdR0376xReMiniAOD'),
    srcNeutralIsoPtSum = cms.InputTag('hpsPFTauNeutralIsoPtSumdR0376xReMiniAOD'),
    srcPUcorrPtSum = cms.InputTag('hpsPFTauPUcorrPtSumdR0376xReMiniAOD'),
    srcPhotonPtSumOutsideSignalCone = cms.InputTag('hpsPFTauPhotonPtSumOutsideSignalConedR0376xReMiniAOD'),
    srcFootprintCorrection = cms.InputTag('hpsPFTauFootprintCorrectiondR0376xReMiniAOD'),
    verbosity = cms.int32(0)
)
hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT76xReMiniAOD.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer76xReMiniAOD'),
    Prediscriminants = requireDecayMode76xReMiniAOD.clone(),
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTraw76xReMiniAOD'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTraw76xReMiniAOD:category'),
    loadMVAfromDB = cms.bool(True),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVADBdR03oldDMwLTv1_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVADBdR03oldDMwLTv1_WPEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBdR03oldDMwLTv1_WPEff80")
hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBdR03oldDMwLTv1_WPEff70")
hpsPFTauDiscriminationByTightIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByTightIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBdR03oldDMwLTv1_WPEff60")
hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBdR03oldDMwLTv1_WPEff50")
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBdR03oldDMwLTv1_WPEff40")

hpsPFTauDiscriminationByIsolationMVArun2v1PWdR03oldDMwLTraw76xReMiniAOD = hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTraw76xReMiniAOD.clone(
    mvaName = cms.string("RecoTauTag_tauIdMVAPWdR03oldDMwLTv1"),
    mvaOpt = cms.string("PWoldDMwLT"),
    srcPUcorrPtSum = cms.InputTag('hpsPFTauNeutralIsoPtSumWeightdR0376xReMiniAOD'),
    verbosity = cms.int32(0)
)
hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD.clone(
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1PWdR03oldDMwLTraw76xReMiniAOD'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1PWdR03oldDMwLTraw76xReMiniAOD:category'),
    loadMVAfromDB = cms.bool(True),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAPWdR03oldDMwLTv1_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVAPWdR03oldDMwLTv1_WPEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWdR03oldDMwLTv1_WPEff80")
hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWdR03oldDMwLTv1_WPEff70")
hpsPFTauDiscriminationByTightIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByTightIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWdR03oldDMwLTv1_WPEff60")
hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWdR03oldDMwLTv1_WPEff50")
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD.clone()
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWdR03oldDMwLTv1_WPEff40")

hpsPFTauMVAIsolation2Seq76xReMiniAOD = cms.Sequence(
    hpsPFTauChargedIsoPtSum76xReMiniAOD
   + hpsPFTauNeutralIsoPtSum76xReMiniAOD
   + hpsPFTauPUcorrPtSum76xReMiniAOD
   + hpsPFTauNeutralIsoPtSumWeight76xReMiniAOD
   + hpsPFTauFootprintCorrection76xReMiniAOD
   + hpsPFTauPhotonPtSumOutsideSignalCone76xReMiniAOD
   #+ hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw76xReMiniAOD
   #+ hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT76xReMiniAOD
   #+ hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwoLT76xReMiniAOD
   #+ hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwoLT76xReMiniAOD
   #+ hpsPFTauDiscriminationByTightIsolationMVA3oldDMwoLT76xReMiniAOD
   #+ hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwoLT76xReMiniAOD
   #+ hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwoLT76xReMiniAOD  
   + hpsPFTauDiscriminationByIsolationMVA3oldDMwLTraw76xReMiniAOD
   + hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByTightIsolationMVA3oldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwLT76xReMiniAOD
   #+ hpsPFTauDiscriminationByIsolationMVA3newDMwoLTraw76xReMiniAOD
   #+ hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT76xReMiniAOD
   #+ hpsPFTauDiscriminationByLooseIsolationMVA3newDMwoLT76xReMiniAOD
   #+ hpsPFTauDiscriminationByMediumIsolationMVA3newDMwoLT76xReMiniAOD
   #+ hpsPFTauDiscriminationByTightIsolationMVA3newDMwoLT76xReMiniAOD
   #+ hpsPFTauDiscriminationByVTightIsolationMVA3newDMwoLT76xReMiniAOD
   #+ hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwoLT76xReMiniAOD
   + hpsPFTauDiscriminationByIsolationMVA3newDMwLTraw76xReMiniAOD
   + hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByLooseIsolationMVA3newDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByMediumIsolationMVA3newDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByTightIsolationMVA3newDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByVTightIsolationMVA3newDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwLT76xReMiniAOD  
   # new MVA isolations for Run2
   + hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw76xReMiniAOD
   + hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw76xReMiniAOD
   + hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBnewDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBnewDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByTightIsolationMVArun2v1DBnewDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBnewDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBnewDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByIsolationMVArun2v1PWoldDMwLTraw76xReMiniAOD
   + hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWoldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWoldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWoldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByTightIsolationMVArun2v1PWoldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWoldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWoldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByIsolationMVArun2v1PWnewDMwLTraw76xReMiniAOD
   + hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWnewDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWnewDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWnewDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByTightIsolationMVArun2v1PWnewDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWnewDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWnewDMwLT76xReMiniAOD
   + hpsPFTauChargedIsoPtSumdR0376xReMiniAOD
   + hpsPFTauNeutralIsoPtSumdR0376xReMiniAOD
   + hpsPFTauPUcorrPtSumdR0376xReMiniAOD
   + hpsPFTauNeutralIsoPtSumWeightdR0376xReMiniAOD
   + hpsPFTauFootprintCorrectiondR0376xReMiniAOD
   + hpsPFTauPhotonPtSumOutsideSignalConedR0376xReMiniAOD
   + hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTraw76xReMiniAOD
   + hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByTightIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBdR03oldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByIsolationMVArun2v1PWdR03oldDMwLTraw76xReMiniAOD
   + hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByTightIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD
   + hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWdR03oldDMwLT76xReMiniAOD
)    

produceHPSPFTaus76xReMiniAOD = cms.Sequence(
    hpsSelectionDiscriminator76xReMiniAOD
    #*hpsTightIsolationCleaner76xReMiniAOD
    #*hpsMediumIsolationCleaner76xReMiniAOD
    #*hpsLooseIsolationCleaner76xReMiniAOD
    #*hpsVLooseIsolationCleaner76xReMiniAOD
    *hpsPFTauProducerSansRefs76xReMiniAOD
    *hpsPFTauProducer76xReMiniAOD
)

produceAndDiscriminateHPSPFTaus76xReMiniAOD = cms.Sequence(
    produceHPSPFTaus76xReMiniAOD*
    hpsPFTauDiscriminationByDecayModeFindingNewDMs76xReMiniAOD*
    hpsPFTauDiscriminationByDecayModeFindingOldDMs76xReMiniAOD*
    hpsPFTauDiscriminationByDecayModeFinding76xReMiniAOD* # CV: kept for backwards compatibility
    hpsPFTauDiscriminationByChargedIsolationSeq76xReMiniAOD*
    hpsPFTauDiscriminationByIsolationSeq76xReMiniAOD*
    #hpsPFTauDiscriminationByIsolationSeqRhoCorr76xReMiniAOD*
    #hpsPFTauDiscriminationByIsolationSeqCustomRhoCorr76xReMiniAOD*
    hpsPFTauDiscriminationByIsolationSeqDBSumPtCorr76xReMiniAOD*
    
    hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr76xReMiniAOD*
    hpsPFTauDiscriminationByRawChargedIsolationDBSumPtCorr76xReMiniAOD*
    hpsPFTauDiscriminationByRawGammaIsolationDBSumPtCorr76xReMiniAOD*

    hpsPFTauDiscriminationByCombinedIsolationSeqDBSumPtCorr76xReMiniAOD*
    hpsPFTauDiscriminationByCombinedIsolationSeqDBSumPtCorr3Hits76xReMiniAOD*
    hpsPFTauDiscriminationByCombinedIsolationSeqDBSumPtCorr3HitsdR0376xReMiniAOD*
    hpsPFTauDiscriminationByPileupWeightedIsolationSeq3Hits76xReMiniAOD*
    
    hpsPFTauDiscriminationByLooseElectronRejection76xReMiniAOD*
    hpsPFTauDiscriminationByMediumElectronRejection76xReMiniAOD*
    hpsPFTauDiscriminationByTightElectronRejection76xReMiniAOD*
    hpsPFTauDiscriminationByMVA5rawElectronRejection76xReMiniAOD*
    hpsPFTauDiscriminationByMVA5VLooseElectronRejection76xReMiniAOD*
    hpsPFTauDiscriminationByMVA5LooseElectronRejection76xReMiniAOD*
    hpsPFTauDiscriminationByMVA5MediumElectronRejection76xReMiniAOD*
    hpsPFTauDiscriminationByMVA5TightElectronRejection76xReMiniAOD*
    hpsPFTauDiscriminationByMVA5VTightElectronRejection76xReMiniAOD*
    hpsPFTauDiscriminationByMVA6rawElectronRejection76xReMiniAOD*
    hpsPFTauDiscriminationByMVA6VLooseElectronRejection76xReMiniAOD*
    hpsPFTauDiscriminationByMVA6LooseElectronRejection76xReMiniAOD*
    hpsPFTauDiscriminationByMVA6MediumElectronRejection76xReMiniAOD*
    hpsPFTauDiscriminationByMVA6TightElectronRejection76xReMiniAOD*
    hpsPFTauDiscriminationByMVA6VTightElectronRejection76xReMiniAOD*
    hpsPFTauDiscriminationByDeadECALElectronRejection76xReMiniAOD*
    hpsPFTauDiscriminationByLooseMuonRejection76xReMiniAOD*
    hpsPFTauDiscriminationByMediumMuonRejection76xReMiniAOD*
    hpsPFTauDiscriminationByTightMuonRejection76xReMiniAOD*
    hpsPFTauDiscriminationByLooseMuonRejection276xReMiniAOD*
    hpsPFTauDiscriminationByMediumMuonRejection276xReMiniAOD*
    hpsPFTauDiscriminationByTightMuonRejection276xReMiniAOD*
    hpsPFTauDiscriminationByLooseMuonRejection376xReMiniAOD*
    hpsPFTauDiscriminationByTightMuonRejection376xReMiniAOD*
    hpsPFTauDiscriminationByMVArawMuonRejection76xReMiniAOD*
    hpsPFTauDiscriminationByMVALooseMuonRejection76xReMiniAOD*
    hpsPFTauDiscriminationByMVAMediumMuonRejection76xReMiniAOD*
    hpsPFTauDiscriminationByMVATightMuonRejection76xReMiniAOD*

    hpsPFTauVertexAndImpactParametersSeq76xReMiniAOD*

    hpsPFTauMVAIsolation2Seq76xReMiniAOD
)


