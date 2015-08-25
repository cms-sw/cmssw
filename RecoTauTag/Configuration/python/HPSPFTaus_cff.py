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
ak4dBetaCorrection=0.20

# Load MVAs from SQLlite file/prep. DB
from RecoTauTag.Configuration.loadRecoTauTagMVAsFromPrepDB_cfi import *

# Select those taus that pass the HPS selections
#  - pt > 15, mass cuts, tauCone cut
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByHPSSelection_cfi import hpsSelectionDiscriminator, decayMode_1Prong0Pi0, decayMode_1Prong1Pi0, decayMode_1Prong2Pi0, decayMode_3Prong0Pi0
hpsPFTauDiscriminationByDecayModeFindingNewDMs = hpsSelectionDiscriminator.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer')
)
hpsPFTauDiscriminationByDecayModeFindingOldDMs = hpsSelectionDiscriminator.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    decayModes = cms.VPSet(
        decayMode_1Prong0Pi0,
        decayMode_1Prong1Pi0,
        decayMode_1Prong2Pi0,
        decayMode_3Prong0Pi0
    ),
    requireTauChargedHadronsToBeChargedPFCands = cms.bool(True)
)
hpsPFTauDiscriminationByDecayModeFinding = hpsPFTauDiscriminationByDecayModeFindingOldDMs.clone() # CV: kept for backwards compatibility

# Define decay mode prediscriminant
requireDecayMode = cms.PSet(
    BooleanOperator = cms.string("and"),
    decayMode = cms.PSet(
        Producer = cms.InputTag('hpsPFTauDiscriminationByDecayModeFindingNewDMs'),
        cut = cms.double(0.5)
    )
)

#Building the prototype for  the Discriminator by Isolation
hpsPFTauDiscriminationByLooseIsolation = pfRecoTauDiscriminationByIsolation.clone(
    PFTauProducer = cms.InputTag("hpsPFTauProducer"),
    Prediscriminants = requireDecayMode.clone(),
    ApplyDiscriminationByTrackerIsolation = False,
    ApplyDiscriminationByECALIsolation = True,
    applyOccupancyCut = True
)
hpsPFTauDiscriminationByLooseIsolation.Prediscriminants.preIso = cms.PSet(
    Producer = cms.InputTag("hpsPFTauDiscriminationByLooseChargedIsolation"),
    cut = cms.double(0.5))

# Make an even looser discriminator
hpsPFTauDiscriminationByVLooseIsolation = hpsPFTauDiscriminationByLooseIsolation.clone(
    customOuterCone = cms.double(0.3),
    isoConeSizeForDeltaBeta = cms.double(0.3),
)
hpsPFTauDiscriminationByVLooseIsolation.qualityCuts.isolationQualityCuts.minTrackPt = 1.5
hpsPFTauDiscriminationByVLooseIsolation.qualityCuts.isolationQualityCuts.minGammaEt = 2.0
hpsPFTauDiscriminationByVLooseIsolation.Prediscriminants.preIso.Producer =  cms.InputTag("hpsPFTauDiscriminationByVLooseChargedIsolation")

hpsPFTauDiscriminationByMediumIsolation = hpsPFTauDiscriminationByLooseIsolation.clone()
hpsPFTauDiscriminationByMediumIsolation.qualityCuts.isolationQualityCuts.minTrackPt = 0.8
hpsPFTauDiscriminationByMediumIsolation.qualityCuts.isolationQualityCuts.minGammaEt = 0.8
hpsPFTauDiscriminationByMediumIsolation.Prediscriminants.preIso.Producer = cms.InputTag("hpsPFTauDiscriminationByMediumChargedIsolation")

hpsPFTauDiscriminationByTightIsolation = hpsPFTauDiscriminationByLooseIsolation.clone()
hpsPFTauDiscriminationByTightIsolation.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByTightIsolation.qualityCuts.isolationQualityCuts.minGammaEt = 0.5
hpsPFTauDiscriminationByTightIsolation.Prediscriminants.preIso.Producer = cms.InputTag("hpsPFTauDiscriminationByTightChargedIsolation")

hpsPFTauDiscriminationByIsolationSeq = cms.Sequence(
    hpsPFTauDiscriminationByVLooseIsolation*
    hpsPFTauDiscriminationByLooseIsolation*
    hpsPFTauDiscriminationByMediumIsolation*
    hpsPFTauDiscriminationByTightIsolation
)

_isolation_types = ['VLoose', 'Loose', 'Medium', 'Tight']
# Now build the sequences that apply PU corrections

# Make Delta Beta corrections (on SumPt quantity)
hpsPFTauDiscriminationByVLooseIsolationDBSumPtCorr = hpsPFTauDiscriminationByVLooseIsolation.clone(
    deltaBetaPUTrackPtCutOverride = cms.double(0.5),
    applyDeltaBetaCorrection = True,
    isoConeSizeForDeltaBeta = 0.8,
    deltaBetaFactor = "%0.4f"%(0.0123/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
)
hpsPFTauDiscriminationByVLooseIsolationDBSumPtCorr.maximumSumPtCut = hpsPFTauDiscriminationByVLooseIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt

hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr = hpsPFTauDiscriminationByLooseIsolation.clone(
    deltaBetaPUTrackPtCutOverride = cms.double(0.5),
    applyDeltaBetaCorrection = True,
    isoConeSizeForDeltaBeta = 0.8,
    deltaBetaFactor = "%0.4f"%(0.0123/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
)
hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr.maximumSumPtCut = hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt

hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr = hpsPFTauDiscriminationByMediumIsolation.clone(
    deltaBetaPUTrackPtCutOverride = cms.double(0.5),
    applyDeltaBetaCorrection = True,
    isoConeSizeForDeltaBeta = 0.8,
    deltaBetaFactor = "%0.4f"%(0.0462/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
)
hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr.maximumSumPtCut = hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt

hpsPFTauDiscriminationByTightIsolationDBSumPtCorr = hpsPFTauDiscriminationByTightIsolation.clone(
    deltaBetaPUTrackPtCutOverride = cms.double(0.5),
    applyDeltaBetaCorrection = True,
    isoConeSizeForDeltaBeta = 0.8,
    deltaBetaFactor = "%0.4f"%(ak4dBetaCorrection),
    applyOccupancyCut = False,
    applySumPtCut = True,
)
hpsPFTauDiscriminationByTightIsolationDBSumPtCorr.maximumSumPtCut = hpsPFTauDiscriminationByTightIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt

hpsPFTauDiscriminationByIsolationSeqDBSumPtCorr = cms.Sequence(
    hpsPFTauDiscriminationByVLooseIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByTightIsolationDBSumPtCorr
)

hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr = hpsPFTauDiscriminationByVLooseIsolationDBSumPtCorr.clone(
    ApplyDiscriminationByTrackerIsolation = True,
    ApplyDiscriminationByECALIsolation = True,
    deltaBetaFactor = "%0.4f"%((0.09/0.25)*(ak4dBetaCorrection)),
    applyOccupancyCut = False,
    applySumPtCut = True,
    maximumSumPtCut = 3.5,
    Prediscriminants = requireDecayMode.clone()
)
hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt = 0.5

hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr = hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr.clone(
    ApplyDiscriminationByTrackerIsolation = True,
    ApplyDiscriminationByECALIsolation = True,
    deltaBetaFactor = "%0.4f"%(ak4dBetaCorrection),
    applyOccupancyCut = False,
    applySumPtCut = True,
    maximumSumPtCut = 2.5,
    Prediscriminants = requireDecayMode.clone()
)
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt = 0.5

hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.clone(
    applySumPtCut = False,
    storeRawSumPt = cms.bool(True)
)

hpsPFTauDiscriminationByRawChargedIsolationDBSumPtCorr = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.clone(
    applySumPtCut = False,
    ApplyDiscriminationByECALIsolation = False,
    storeRawSumPt = cms.bool(True)
)

hpsPFTauDiscriminationByRawGammaIsolationDBSumPtCorr = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.clone(
    applySumPtCut = False,
    ApplyDiscriminationByTrackerIsolation = False,
    storeRawSumPt = cms.bool(True)
)

hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr = hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr.clone(
    ApplyDiscriminationByTrackerIsolation = True,
    ApplyDiscriminationByECALIsolation = True,
    deltaBetaFactor = "%0.4f"%(ak4dBetaCorrection),
    applyOccupancyCut = False,
    applySumPtCut = True,
    maximumSumPtCut = 1.5,
    Prediscriminants = requireDecayMode.clone()
)
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt = 0.5

hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr = hpsPFTauDiscriminationByTightIsolationDBSumPtCorr.clone(
    ApplyDiscriminationByTrackerIsolation = True,
    ApplyDiscriminationByECALIsolation = True,
    deltaBetaFactor = "%0.4f"%(ak4dBetaCorrection),
    applyOccupancyCut = False,
    applySumPtCut = True,
    maximumSumPtCut = 0.8,
    Prediscriminants = requireDecayMode.clone()
)
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt = 0.5

hpsPFTauDiscriminationByCombinedIsolationSeqDBSumPtCorr = cms.Sequence(
    hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr
)

#Charge isolation based on combined isolation
hpsPFTauDiscriminationByVLooseChargedIsolation = hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr.clone(
    ApplyDiscriminationByECALIsolation = False
)

hpsPFTauDiscriminationByLooseChargedIsolation = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.clone(
    ApplyDiscriminationByECALIsolation = False
)

hpsPFTauDiscriminationByMediumChargedIsolation = hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr.clone(
    ApplyDiscriminationByECALIsolation = False
)
hpsPFTauDiscriminationByTightChargedIsolation = hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr.clone(
    ApplyDiscriminationByECALIsolation = False
)

hpsPFTauDiscriminationByChargedIsolationSeq = cms.Sequence(
    hpsPFTauDiscriminationByVLooseChargedIsolation*
    hpsPFTauDiscriminationByLooseChargedIsolation*
    hpsPFTauDiscriminationByMediumChargedIsolation*
    hpsPFTauDiscriminationByTightChargedIsolation
)

#copying discriminator against electrons and muons
hpsPFTauDiscriminationByLooseElectronRejection = pfRecoTauDiscriminationAgainstElectron.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants,
    PFElectronMVA_maxValue = cms.double(0.6)
)
hpsPFTauDiscriminationByMediumElectronRejection = pfRecoTauDiscriminationAgainstElectron.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants,
    ApplyCut_EcalCrackCut = cms.bool(True)
)
hpsPFTauDiscriminationByTightElectronRejection = pfRecoTauDiscriminationAgainstElectron.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants,
    ApplyCut_EcalCrackCut = cms.bool(True),
    ApplyCut_BremCombined = cms.bool(True)
)

hpsPFTauDiscriminationByLooseMuonRejection = pfRecoTauDiscriminationAgainstMuon.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants
)
hpsPFTauDiscriminationByMediumMuonRejection = pfRecoTauDiscriminationAgainstMuon.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants,
    discriminatorOption = cms.string('noAllArbitrated')
)
hpsPFTauDiscriminationByTightMuonRejection = pfRecoTauDiscriminationAgainstMuon.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants,
    discriminatorOption = cms.string('noAllArbitratedWithHOP')
)

hpsPFTauDiscriminationByLooseMuonRejection2 = pfRecoTauDiscriminationAgainstMuon2.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants
)
hpsPFTauDiscriminationByMediumMuonRejection2 = pfRecoTauDiscriminationAgainstMuon2.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants,
    discriminatorOption = cms.string('medium')
)
hpsPFTauDiscriminationByTightMuonRejection2 = pfRecoTauDiscriminationAgainstMuon2.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants,
    discriminatorOption = cms.string('tight')
)

hpsPFTauDiscriminationByLooseMuonRejection3 = pfRecoTauDiscriminationAgainstMuon2.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants,
    discriminatorOption = cms.string('custom'),
    maxNumberOfMatches = cms.int32(1),
    doCaloMuonVeto = cms.bool(True),
    maxNumberOfHitsLast2Stations = cms.int32(-1)
)
hpsPFTauDiscriminationByTightMuonRejection3 = hpsPFTauDiscriminationByLooseMuonRejection3.clone(
    maxNumberOfHitsLast2Stations = cms.int32(0)
)

hpsPFTauDiscriminationByMVArawMuonRejection = pfRecoTauDiscriminationAgainstMuonMVA.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    loadMVAfromDB = cms.bool(True),
    returnMVA = cms.bool(True),
    mvaName = cms.string("RecoTauTag_againstMuonMVAv1")
)
##hpsPFTauDiscriminationByMVALooseMuonRejection = hpsPFTauDiscriminationByMVArawMuonRejection.clone(
##    returnMVA = cms.bool(False),
##    mvaMin = cms.double(0.75)
##)
##hpsPFTauDiscriminationByMVAMediumMuonRejection = hpsPFTauDiscriminationByMVALooseMuonRejection.clone(
##    mvaMin = cms.double(0.950)
##)
##hpsPFTauDiscriminationByMVATightMuonRejection = hpsPFTauDiscriminationByMVALooseMuonRejection.clone(
##    mvaMin = cms.double(0.975)
##)
hpsPFTauDiscriminationByMVALooseMuonRejection = recoTauDiscriminantCutMultiplexer.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),    
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByMVArawMuonRejection'),
    key = cms.InputTag('hpsPFTauDiscriminationByMVArawMuonRejection:category'),
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
hpsPFTauDiscriminationByMVAMediumMuonRejection = hpsPFTauDiscriminationByMVALooseMuonRejection.clone()
hpsPFTauDiscriminationByMVAMediumMuonRejection.mapping[0].cut = cms.string("RecoTauTag_againstMuonMVAv1_WPeff99_0")
hpsPFTauDiscriminationByMVATightMuonRejection = hpsPFTauDiscriminationByMVALooseMuonRejection.clone()
hpsPFTauDiscriminationByMVATightMuonRejection.mapping[0].cut = cms.string("RecoTauTag_againstMuonMVAv1_WPeff98_0")

hpsPFTauDiscriminationByMVA5rawElectronRejection = pfRecoTauDiscriminationAgainstElectronMVA5.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
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

hpsPFTauDiscriminationByMVA5VLooseElectronRejection = recoTauDiscriminantCutMultiplexer.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByMVA5rawElectronRejection'),
    key = cms.InputTag('hpsPFTauDiscriminationByMVA5rawElectronRejection:category'),
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

hpsPFTauDiscriminationByMVA5LooseElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA5VLooseElectronRejection)
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_BL_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_BL_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_BL_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_BL_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_BL_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_BL_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_BL_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_BL_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[8].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_EC_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[9].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_EC_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[10].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_EC_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[11].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_EC_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[12].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_EC_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[13].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_EC_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[14].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_EC_WPeff96")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[15].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_EC_WPeff96")

hpsPFTauDiscriminationByMVA5MediumElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA5VLooseElectronRejection)
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_BL_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_BL_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_BL_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_BL_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_BL_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_BL_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_BL_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_BL_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[8].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_EC_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[9].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_EC_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[10].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_EC_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[11].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_EC_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[12].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_EC_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[13].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_EC_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[14].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_EC_WPeff91")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[15].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_EC_WPeff91")

hpsPFTauDiscriminationByMVA5TightElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA5VLooseElectronRejection)
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_BL_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_BL_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_BL_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_BL_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_BL_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_BL_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_BL_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_BL_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[8].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_EC_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[9].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_EC_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[10].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_EC_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[11].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_EC_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[12].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_EC_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[13].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_EC_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[14].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_EC_WPeff85")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[15].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_EC_WPeff85")

hpsPFTauDiscriminationByMVA5VTightElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA5VLooseElectronRejection)
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_BL_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_BL_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_BL_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_BL_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_BL_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_BL_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_BL_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_BL_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[8].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwoGSF_EC_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[9].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_woGwGSF_EC_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[10].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwoGSF_EC_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[11].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_NoEleMatch_wGwGSF_EC_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[12].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwoGSF_EC_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[13].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_woGwGSF_EC_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[14].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwoGSF_EC_WPeff79")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[15].cut = cms.string("RecoTauTag_antiElectronMVA5v1_gbr_wGwGSF_EC_WPeff79")

hpsPFTauDiscriminationByDeadECALElectronRejection = pfRecoTauDiscriminationAgainstElectronDeadECAL.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone()
)

#Define new sequence that is using smaller number on hits cut
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.clone()
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits = hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr.clone()
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits = hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr.clone()

hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits.qualityCuts.isolationQualityCuts.minTrackHits = cms.uint32(3)
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits.applyPhotonPtSumOutsideSignalConeCut = cms.bool(True)
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits.qualityCuts.isolationQualityCuts.minTrackHits = cms.uint32(3)
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits.applyPhotonPtSumOutsideSignalConeCut = cms.bool(True)
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits.qualityCuts.isolationQualityCuts.minTrackHits = cms.uint32(3)
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits.applyPhotonPtSumOutsideSignalConeCut = cms.bool(True)

hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr3Hits = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits.clone(
    applySumPtCut = False,
    storeRawSumPt = cms.bool(True)
)

hpsPFTauDiscriminationByCombinedIsolationSeqDBSumPtCorr3Hits = cms.Sequence(
    hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits*
    hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits*
    hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits*
    hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr3Hits
)

hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits.clone(
    ApplyDiscriminationByECALIsolation = cms.bool(False),
    applyDeltaBetaCorrection = cms.bool(False),
    ApplyDiscriminationByWeightedECALIsolation = cms.bool(True),
    UseAllPFCandsForWeights = cms.bool(True),
    applyFootprintCorrection = cms.bool(True),
    applyPhotonPtSumOutsideSignalConeCut = cms.bool(True)    
)

hpsPFTauDiscriminationByMediumPileupWeightedIsolation3Hits = hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits.clone(
    maximumSumPtCut = hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits.maximumSumPtCut
)

hpsPFTauDiscriminationByTightPileupWeightedIsolation3Hits = hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits.clone(
    maximumSumPtCut = hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits.maximumSumPtCut
)

hpsPFTauDiscriminationByPhotonPtSumOutsideSignalCone = hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits.clone(
    applySumPtCut = cms.bool(False)
)

hpsPFTauDiscriminationByRawPileupWeightedIsolation3Hits = hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits.clone(
    Prediscriminants = cms.PSet(
        BooleanOperator = cms.string("and"),
        decayMode = cms.PSet(
            Producer = cms.InputTag('hpsPFTauDiscriminationByPhotonPtSumOutsideSignalCone'),
            cut = cms.double(0.5)
        )
    ),
    applySumPtCut = cms.bool(False),
    storeRawSumPt = cms.bool(True)
)

hpsPFTauDiscriminationByPileupWeightedIsolationSeq3Hits = cms.Sequence(
    hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits*
    hpsPFTauDiscriminationByMediumPileupWeightedIsolation3Hits*
    hpsPFTauDiscriminationByTightPileupWeightedIsolation3Hits*
    hpsPFTauDiscriminationByPhotonPtSumOutsideSignalCone*
    hpsPFTauDiscriminationByRawPileupWeightedIsolation3Hits
)

# Define the HPS selection discriminator used in cleaning
hpsSelectionDiscriminator.PFTauProducer = cms.InputTag("combinatoricRecoTaus")

from RecoTauTag.RecoTau.RecoTauCleaner_cfi import RecoTauCleaner
hpsPFTauProducerSansRefs = RecoTauCleaner.clone(
    src = cms.InputTag("combinatoricRecoTaus")
)


from RecoTauTag.RecoTau.RecoTauPiZeroUnembedder_cfi import RecoTauPiZeroUnembedder
hpsPFTauProducer = RecoTauPiZeroUnembedder.clone(
    src = cms.InputTag("hpsPFTauProducerSansRefs")
)


from RecoTauTag.RecoTau.PFTauPrimaryVertexProducer_cfi      import *
from RecoTauTag.RecoTau.PFTauSecondaryVertexProducer_cfi    import *
from RecoTauTag.RecoTau.PFTauTransverseImpactParameters_cfi import *
hpsPFTauPrimaryVertexProducer = PFTauPrimaryVertexProducer.clone(
    PFTauTag = cms.InputTag("hpsPFTauProducer"),
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
            discriminator = cms.InputTag('hpsPFTauDiscriminationByDecayModeFindingNewDMs'),
            selectionCut = cms.double(0.5)
        )
    ),
    cut = cms.string("pt > 18.0 & abs(eta) < 2.4")
)

hpsPFTauSecondaryVertexProducer = PFTauSecondaryVertexProducer.clone(
    PFTauTag = cms.InputTag("hpsPFTauProducer")
)
hpsPFTauTransverseImpactParameters = PFTauTransverseImpactParameters.clone(
    PFTauTag = cms.InputTag("hpsPFTauProducer"),
    PFTauPVATag = cms.InputTag("hpsPFTauPrimaryVertexProducer"),
    PFTauSVATag = cms.InputTag("hpsPFTauSecondaryVertexProducer"),
    useFullCalculation = cms.bool(False)
)
hpsPFTauVertexAndImpactParametersSeq = cms.Sequence(
    hpsPFTauPrimaryVertexProducer*
    hpsPFTauSecondaryVertexProducer*
    hpsPFTauTransverseImpactParameters
)

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByMVAIsolation2_cff import *
hpsPFTauChargedIsoPtSum = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
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
hpsPFTauNeutralIsoPtSum = hpsPFTauChargedIsoPtSum.clone(
    ApplyDiscriminationByECALIsolation = cms.bool(True),
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    verbosity = cms.int32(0)
)
hpsPFTauPUcorrPtSum = hpsPFTauChargedIsoPtSum.clone(
    ApplyDiscriminationByECALIsolation = cms.bool(False),
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    applyDeltaBetaCorrection = cms.bool(True),
    storeRawSumPt = cms.bool(False),
    storeRawPUsumPt = cms.bool(True),
    verbosity = cms.int32(0)
)
hpsPFTauNeutralIsoPtSumWeight = hpsPFTauChargedIsoPtSum.clone(
    ApplyDiscriminationByWeightedECALIsolation = cms.bool(True),
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    UseAllPFCandsForWeights = cms.bool(True),
    verbosity = cms.int32(0)
)
hpsPFTauFootprintCorrection = hpsPFTauChargedIsoPtSum.clone(    
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    storeRawSumPt = cms.bool(False),
    storeRawFootprintCorrection = cms.bool(True),
    verbosity = cms.int32(0)
)
hpsPFTauPhotonPtSumOutsideSignalCone = hpsPFTauChargedIsoPtSum.clone(    
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    storeRawSumPt = cms.bool(False),
    storeRawPhotonSumPt_outsideSignalCone = cms.bool(True),
    verbosity = cms.int32(0)
)

hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw = discriminationByIsolationMVA2raw.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    loadMVAfromDB = cms.bool(True),
    mvaName = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1"),
    mvaOpt = cms.string("oldDMwoLT"),
    srcTauTransverseImpactParameters = cms.InputTag('hpsPFTauTransverseImpactParameters'),    
    srcChargedIsoPtSum = cms.InputTag('hpsPFTauChargedIsoPtSum'),
    srcNeutralIsoPtSum = cms.InputTag('hpsPFTauNeutralIsoPtSum'),
    srcPUcorrPtSum = cms.InputTag('hpsPFTauPUcorrPtSum'),
    verbosity = cms.int32(0)
)
hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT = discriminationByIsolationMVA2VLoose.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),    
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw:category'),
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
hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone()
hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_WPEff80")
hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone()
hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_WPEff70")
hpsPFTauDiscriminationByTightIsolationMVA3oldDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone()
hpsPFTauDiscriminationByTightIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_WPEff60")
hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone()
hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_WPEff50")
hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone()
hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwoLTv1_WPEff40")
hpsPFTauDiscriminationByIsolationMVA3oldDMwLTraw = hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw.clone(
    mvaName = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1"),
    mvaOpt = cms.string("oldDMwLT"),
    verbosity = cms.int32(0)
)
hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone(
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3oldDMwLTraw'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3oldDMwLTraw:category'),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_WPEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT.clone()
hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_WPEff80")
hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT.clone()
hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_WPEff70")
hpsPFTauDiscriminationByTightIsolationMVA3oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT.clone()
hpsPFTauDiscriminationByTightIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_WPEff60")
hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT.clone()
hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_WPEff50")
hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT.clone()
hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAoldDMwLTv1_WPEff40")
hpsPFTauDiscriminationByIsolationMVA3newDMwoLTraw = hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw.clone(
    mvaName = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1"),
    mvaOpt = cms.string("newDMwoLT"),
    verbosity = cms.int32(0)
)
hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone(
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3newDMwoLTraw'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3newDMwoLTraw:category'),
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
hpsPFTauDiscriminationByLooseIsolationMVA3newDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT.clone()
hpsPFTauDiscriminationByLooseIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1_WPEff80")
##hpsPFTauDiscriminationByLooseIsolationMVA3newDMwoLT.verbosity = cms.int32(1)
hpsPFTauDiscriminationByMediumIsolationMVA3newDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT.clone()
hpsPFTauDiscriminationByMediumIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1_WPEff70")
hpsPFTauDiscriminationByTightIsolationMVA3newDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT.clone()
hpsPFTauDiscriminationByTightIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1_WPEff60")
hpsPFTauDiscriminationByVTightIsolationMVA3newDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT.clone()
hpsPFTauDiscriminationByVTightIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1_WPEff50")
hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT.clone()
hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwoLTv1_WPEff40")
hpsPFTauDiscriminationByIsolationMVA3newDMwLTraw = hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw.clone(
    mvaName = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1"),
    mvaOpt = cms.string("newDMwLT"),
    verbosity = cms.int32(0)
)
hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone(
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3newDMwLTraw'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3newDMwLTraw:category'),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_WPEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVA3newDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT.clone()
hpsPFTauDiscriminationByLooseIsolationMVA3newDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_WPEff80")
hpsPFTauDiscriminationByMediumIsolationMVA3newDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT.clone()
hpsPFTauDiscriminationByMediumIsolationMVA3newDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_WPEff70")
hpsPFTauDiscriminationByTightIsolationMVA3newDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT.clone()
hpsPFTauDiscriminationByTightIsolationMVA3newDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_WPEff60")
hpsPFTauDiscriminationByVTightIsolationMVA3newDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT.clone()
hpsPFTauDiscriminationByVTightIsolationMVA3newDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_WPEff50")
hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT.clone()
hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAnewDMwLTv1_WPEff40")

hpsPFTauMVAIsolation2Seq = cms.Sequence(
    hpsPFTauChargedIsoPtSum
   + hpsPFTauNeutralIsoPtSum
   + hpsPFTauPUcorrPtSum
   + hpsPFTauNeutralIsoPtSumWeight
   + hpsPFTauFootprintCorrection
   + hpsPFTauPhotonPtSumOutsideSignalCone
   + hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw
   + hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT
   + hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwoLT
   + hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwoLT
   + hpsPFTauDiscriminationByTightIsolationMVA3oldDMwoLT
   + hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwoLT
   + hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwoLT    
   + hpsPFTauDiscriminationByIsolationMVA3oldDMwLTraw
   + hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT
   + hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwLT
   + hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwLT
   + hpsPFTauDiscriminationByTightIsolationMVA3oldDMwLT
   + hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwLT
   + hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwLT
   + hpsPFTauDiscriminationByIsolationMVA3newDMwoLTraw
   + hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT
   + hpsPFTauDiscriminationByLooseIsolationMVA3newDMwoLT
   + hpsPFTauDiscriminationByMediumIsolationMVA3newDMwoLT
   + hpsPFTauDiscriminationByTightIsolationMVA3newDMwoLT
   + hpsPFTauDiscriminationByVTightIsolationMVA3newDMwoLT
   + hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwoLT 
   + hpsPFTauDiscriminationByIsolationMVA3newDMwLTraw
   + hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT
   + hpsPFTauDiscriminationByLooseIsolationMVA3newDMwLT
   + hpsPFTauDiscriminationByMediumIsolationMVA3newDMwLT
   + hpsPFTauDiscriminationByTightIsolationMVA3newDMwLT
   + hpsPFTauDiscriminationByVTightIsolationMVA3newDMwLT
   + hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwLT    
)    

produceHPSPFTaus = cms.Sequence(
    hpsSelectionDiscriminator
    #*hpsTightIsolationCleaner
    #*hpsMediumIsolationCleaner
    #*hpsLooseIsolationCleaner
    #*hpsVLooseIsolationCleaner
    *hpsPFTauProducerSansRefs
    *hpsPFTauProducer
)

produceAndDiscriminateHPSPFTaus = cms.Sequence(
    produceHPSPFTaus*
    hpsPFTauDiscriminationByDecayModeFindingNewDMs*
    hpsPFTauDiscriminationByDecayModeFindingOldDMs*
    hpsPFTauDiscriminationByDecayModeFinding* # CV: kept for backwards compatibility
    hpsPFTauDiscriminationByChargedIsolationSeq*
    hpsPFTauDiscriminationByIsolationSeq*
    #hpsPFTauDiscriminationByIsolationSeqRhoCorr*
    #hpsPFTauDiscriminationByIsolationSeqCustomRhoCorr*
    hpsPFTauDiscriminationByIsolationSeqDBSumPtCorr*
    
    hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByRawChargedIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByRawGammaIsolationDBSumPtCorr*

    hpsPFTauDiscriminationByCombinedIsolationSeqDBSumPtCorr*
    hpsPFTauDiscriminationByCombinedIsolationSeqDBSumPtCorr3Hits*
    hpsPFTauDiscriminationByPileupWeightedIsolationSeq3Hits*
    
    hpsPFTauDiscriminationByLooseElectronRejection*
    hpsPFTauDiscriminationByMediumElectronRejection*
    hpsPFTauDiscriminationByTightElectronRejection*
    hpsPFTauDiscriminationByMVA5rawElectronRejection*
    hpsPFTauDiscriminationByMVA5VLooseElectronRejection*
    hpsPFTauDiscriminationByMVA5LooseElectronRejection*
    hpsPFTauDiscriminationByMVA5MediumElectronRejection*
    hpsPFTauDiscriminationByMVA5TightElectronRejection*
    hpsPFTauDiscriminationByMVA5VTightElectronRejection*
    hpsPFTauDiscriminationByDeadECALElectronRejection*
    hpsPFTauDiscriminationByLooseMuonRejection*
    hpsPFTauDiscriminationByMediumMuonRejection*
    hpsPFTauDiscriminationByTightMuonRejection*
    hpsPFTauDiscriminationByLooseMuonRejection2*
    hpsPFTauDiscriminationByMediumMuonRejection2*
    hpsPFTauDiscriminationByTightMuonRejection2*
    hpsPFTauDiscriminationByLooseMuonRejection3*
    hpsPFTauDiscriminationByTightMuonRejection3*
    hpsPFTauDiscriminationByMVArawMuonRejection*
    hpsPFTauDiscriminationByMVALooseMuonRejection*
    hpsPFTauDiscriminationByMVAMediumMuonRejection*
    hpsPFTauDiscriminationByMVATightMuonRejection*

    hpsPFTauVertexAndImpactParametersSeq*

    hpsPFTauMVAIsolation2Seq
)
