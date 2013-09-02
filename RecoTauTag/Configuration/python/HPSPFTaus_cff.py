import FWCore.ParameterSet.Config as cms
import copy

'''

Sequences for HPS taus

'''

# Define the discriminators for this tau
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi                      import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByMVAIsolation_cfi                   import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackFinding_cfi            import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectron_cfi                  import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectronMVA3GBR_cfi           import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectronMVA4GBR_cfi           import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectronMVA5GBR_cfi           import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectronDeadECAL_cfi          import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon_cfi                      import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon2_cfi                     import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuonMVA_cfi                   import *

from RecoTauTag.RecoTau.RecoTauDiscriminantCutMultiplexer_cfi import *

# Load helper functions to change the source of the discriminants
from RecoTauTag.RecoTau.TauDiscriminatorTools import *

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
hpsPFTauDiscriminationByVLooseIsolationDBSumPtCorr.maximumSumPtCut=hpsPFTauDiscriminationByVLooseIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt

hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr = hpsPFTauDiscriminationByLooseIsolation.clone(
    deltaBetaPUTrackPtCutOverride = cms.double(0.5),
    applyDeltaBetaCorrection = True,
    isoConeSizeForDeltaBeta = 0.8,
    deltaBetaFactor = "%0.4f"%(0.0123/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
)
hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr.maximumSumPtCut=hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt

hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr = hpsPFTauDiscriminationByMediumIsolation.clone(
    deltaBetaPUTrackPtCutOverride = cms.double(0.5),
    applyDeltaBetaCorrection = True,
    isoConeSizeForDeltaBeta = 0.8,
    deltaBetaFactor = "%0.4f"%(0.0462/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
)
hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr.maximumSumPtCut=hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt

hpsPFTauDiscriminationByTightIsolationDBSumPtCorr = hpsPFTauDiscriminationByTightIsolation.clone(
    deltaBetaPUTrackPtCutOverride = cms.double(0.5),
    applyDeltaBetaCorrection = True,
    isoConeSizeForDeltaBeta = 0.8,
    deltaBetaFactor = "%0.4f"%(0.0772/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
)
hpsPFTauDiscriminationByTightIsolationDBSumPtCorr.maximumSumPtCut=hpsPFTauDiscriminationByTightIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt

hpsPFTauDiscriminationByIsolationSeqDBSumPtCorr = cms.Sequence(
    hpsPFTauDiscriminationByVLooseIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByTightIsolationDBSumPtCorr
)

hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr = hpsPFTauDiscriminationByVLooseIsolationDBSumPtCorr.clone(
    ApplyDiscriminationByTrackerIsolation = True,
    ApplyDiscriminationByECALIsolation = True,
    deltaBetaFactor = "%0.4f"%((0.09/0.25)*(0.0772/0.1687)),
    applyOccupancyCut = False,
    applySumPtCut = True,
    maximumSumPtCut = 3.0,
    Prediscriminants = requireDecayMode.clone()
)
hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt = 0.5

hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr = hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr.clone(
    ApplyDiscriminationByTrackerIsolation = True,
    ApplyDiscriminationByECALIsolation = True,
    deltaBetaFactor = "%0.4f"%(0.0772/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
    maximumSumPtCut = 2.0,
    Prediscriminants = requireDecayMode.clone()
)
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt = 0.5

hpsPFTauDiscriminationByRelLooseCombinedIsolationDBSumPtCorr = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.clone(
    applySumPtCut = False,
    applyRelativeSumPtCut = True,
    relativeSumPtCut = 0.09
)

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
    deltaBetaFactor = "%0.4f"%(0.0772/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
    maximumSumPtCut = 1.0,
    Prediscriminants = requireDecayMode.clone()
)
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt = 0.5

hpsPFTauDiscriminationByRelMediumCombinedIsolationDBSumPtCorr = hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr.clone(
    applySumPtCut = False,
    applyRelativeSumPtCut = True,
    relativeSumPtCut = 0.06
)

hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr = hpsPFTauDiscriminationByTightIsolationDBSumPtCorr.clone(
    ApplyDiscriminationByTrackerIsolation = True,
    ApplyDiscriminationByECALIsolation = True,
    deltaBetaFactor = "%0.4f"%(0.0772/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
    maximumSumPtCut = 0.8,
    Prediscriminants = requireDecayMode.clone()
)
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt = 0.5

hpsPFTauDiscriminationByRelTightCombinedIsolationDBSumPtCorr = hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr.clone(
    applySumPtCut = False,
    applyRelativeSumPtCut = True,
    relativeSumPtCut = 0.03
)

hpsPFTauDiscriminationByCombinedIsolationSeqDBSumPtCorr = cms.Sequence(
    hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr*hpsPFTauDiscriminationByRelLooseCombinedIsolationDBSumPtCorr*    
    hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr*hpsPFTauDiscriminationByRelMediumCombinedIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr*hpsPFTauDiscriminationByRelTightCombinedIsolationDBSumPtCorr
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

# Define MVA based isolation discrimators
#   MVA Isolation Version 1
hpsPFTauDiscriminationByIsolationMVAraw = pfRecoTauDiscriminationByMVAIsolation.clone(
    PFTauProducer = cms.InputTag("hpsPFTauProducer"),
    Prediscriminants = requireDecayMode.clone(),
    returnMVA = cms.bool(True),
)

hpsPFTauDiscriminationByLooseIsolationMVA = hpsPFTauDiscriminationByDecayModeFindingNewDMs.clone(
    Prediscriminants = cms.PSet(
        BooleanOperator = cms.string("and"),
        mva = cms.PSet(
            Producer = cms.InputTag('hpsPFTauDiscriminationByIsolationMVAraw'),
            cut = cms.double(0.795)
        )
    )
)
hpsPFTauDiscriminationByMediumIsolationMVA = copy.deepcopy(hpsPFTauDiscriminationByLooseIsolationMVA)
hpsPFTauDiscriminationByMediumIsolationMVA.Prediscriminants.mva.cut = cms.double(0.884)
hpsPFTauDiscriminationByTightIsolationMVA = copy.deepcopy(hpsPFTauDiscriminationByLooseIsolationMVA)
hpsPFTauDiscriminationByTightIsolationMVA.Prediscriminants.mva.cut = cms.double(0.921)

#   MVA Isolation Version 2
hpsPFTauDiscriminationByIsolationMVA2raw = pfRecoTauDiscriminationByMVAIsolation.clone(
    PFTauProducer = cms.InputTag("hpsPFTauProducer"),
    Prediscriminants = requireDecayMode.clone(),
    returnMVA = cms.bool(True),
    gbrfFilePath = cms.FileInPath('RecoTauTag/RecoTau/data/gbrfTauIso_v2.root')
)

hpsPFTauDiscriminationByLooseIsolationMVA2 = hpsPFTauDiscriminationByDecayModeFindingNewDMs.clone(
    Prediscriminants = cms.PSet(
        BooleanOperator = cms.string("and"),
        mva = cms.PSet(
            Producer = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA2raw'),
            cut = cms.double(0.85)
        )
    )
)
hpsPFTauDiscriminationByMediumIsolationMVA2 = copy.deepcopy(hpsPFTauDiscriminationByLooseIsolationMVA2)
hpsPFTauDiscriminationByMediumIsolationMVA2.Prediscriminants.mva.cut = cms.double(0.90)
hpsPFTauDiscriminationByTightIsolationMVA2 = copy.deepcopy(hpsPFTauDiscriminationByLooseIsolationMVA2)
hpsPFTauDiscriminationByTightIsolationMVA2.Prediscriminants.mva.cut = cms.double(0.94)

from RecoJets.Configuration.RecoPFJets_cff import kt6PFJets as _dummy
kt6PFJetsForRhoComputationVoronoi = _dummy.clone(
    doRhoFastjet = True,
    voronoiRfact = 0.9
)

hpsPFTauDiscriminationByMVAIsolationSeq = cms.Sequence(
    kt6PFJetsForRhoComputationVoronoi*
    hpsPFTauDiscriminationByIsolationMVAraw*
    hpsPFTauDiscriminationByLooseIsolationMVA*
    hpsPFTauDiscriminationByMediumIsolationMVA*
    hpsPFTauDiscriminationByTightIsolationMVA*
    hpsPFTauDiscriminationByIsolationMVA2raw*
    hpsPFTauDiscriminationByLooseIsolationMVA2*
    hpsPFTauDiscriminationByMediumIsolationMVA2*
    hpsPFTauDiscriminationByTightIsolationMVA2
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
    returnMVA = cms.bool(True)
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
    inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByMVAMuonRejection.root'),
    mvaOutput_normalization = cms.string("mvaOutput_normalization_opt2"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("opt2eff99_5"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByMVAMediumMuonRejection = hpsPFTauDiscriminationByMVALooseMuonRejection.clone()
hpsPFTauDiscriminationByMVAMediumMuonRejection.mapping[0].cut = cms.string("opt2eff99_0")
hpsPFTauDiscriminationByMVATightMuonRejection = hpsPFTauDiscriminationByMVALooseMuonRejection.clone()
hpsPFTauDiscriminationByMVATightMuonRejection.mapping[0].cut = cms.string("opt2eff98_0")

hpsPFTauDiscriminationByMVA3rawElectronRejection = pfRecoTauDiscriminationAgainstElectronMVA3GBR.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone()
)

hpsPFTauDiscriminationByMVA3LooseElectronRejection = recoTauDiscriminantCutMultiplexer.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByMVA3rawElectronRejection'),
    key = cms.InputTag('hpsPFTauDiscriminationByMVA3rawElectronRejection:category'),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0), # minMVA1prongNoEleMatchWOgWOgsfBL
            cut = cms.double(0.835)
        ),
        cms.PSet(
            category = cms.uint32(1), # minMVA1prongNoEleMatchWOgWgsfBL
            cut = cms.double(0.831)
        ),
        cms.PSet(
            category = cms.uint32(2), # minMVA1prongNoEleMatchWgWOgsfBL
            cut = cms.double(0.849)
        ),
        cms.PSet(
            category = cms.uint32(3), # minMVA1prongNoEleMatchWgWgsfBL
            cut = cms.double(0.859)
        ),
         cms.PSet(
            category = cms.uint32(4), # minMVA1prongWOgWOgsfBL
            cut = cms.double(0.873)
        ),
        cms.PSet(
            category = cms.uint32(5), # minMVA1prongWOgWgsfBL
            cut = cms.double(0.823)
        ),
        cms.PSet(
            category = cms.uint32(6), # minMVA1prongWgWOgsfBL
            cut = cms.double(0.85)
        ),
        cms.PSet(
            category = cms.uint32(7), # minMVA1prongWgWgsfBL
            cut = cms.double(0.855)
        ),
        cms.PSet(
            category = cms.uint32(8), # minMVA1prongNoEleMatchWOgWOgsfEC
            cut = cms.double(0.816)
        ),
        cms.PSet(
            category = cms.uint32(9), # minMVA1prongNoEleMatchWOgWgsfEC
            cut = cms.double(0.861)
        ),
        cms.PSet(
            category = cms.uint32(10), # minMVA1prongNoEleMatchWgWOgsfEC
            cut = cms.double(0.862)
        ),
        cms.PSet(
            category = cms.uint32(11), # minMVA1prongNoEleMatchWgWgsfEC
            cut = cms.double(0.847)
        ),
         cms.PSet(
            category = cms.uint32(12), # minMVA1prongWOgWOgsfEC
            cut = cms.double(0.893)
        ),
        cms.PSet(
            category = cms.uint32(13), # minMVA1prongWOgWgsfEC
            cut = cms.double(0.82)
        ),
        cms.PSet(
            category = cms.uint32(14), # minMVA1prongWgWOgsfEC
            cut = cms.double(0.845)
        ),
        cms.PSet(
            category = cms.uint32(15), # minMVA1prongWgWgsfEC
            cut = cms.double(0.851)
        ),
        cms.PSet(
            category = cms.uint32(16), # minMVA3prongMatch
            cut = cms.double(-1.)
        ),
        cms.PSet(
            category = cms.uint32(17), # minMVA3prongNoMatch
            cut = cms.double(-1.)
        )
    )
)

hpsPFTauDiscriminationByMVA3MediumElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA3LooseElectronRejection)
hpsPFTauDiscriminationByMVA3MediumElectronRejection.mapping[0].cut = cms.double(0.937)
hpsPFTauDiscriminationByMVA3MediumElectronRejection.mapping[1].cut = cms.double(0.949)
hpsPFTauDiscriminationByMVA3MediumElectronRejection.mapping[2].cut = cms.double(0.955)
hpsPFTauDiscriminationByMVA3MediumElectronRejection.mapping[3].cut = cms.double(0.956)
hpsPFTauDiscriminationByMVA3MediumElectronRejection.mapping[4].cut = cms.double(0.962)
hpsPFTauDiscriminationByMVA3MediumElectronRejection.mapping[5].cut = cms.double(0.934)
hpsPFTauDiscriminationByMVA3MediumElectronRejection.mapping[6].cut = cms.double(0.946)
hpsPFTauDiscriminationByMVA3MediumElectronRejection.mapping[7].cut = cms.double(0.948)
hpsPFTauDiscriminationByMVA3MediumElectronRejection.mapping[8].cut = cms.double(0.959)
hpsPFTauDiscriminationByMVA3MediumElectronRejection.mapping[9].cut = cms.double(0.95)
hpsPFTauDiscriminationByMVA3MediumElectronRejection.mapping[10].cut = cms.double(0.954)
hpsPFTauDiscriminationByMVA3MediumElectronRejection.mapping[11].cut = cms.double(0.954)
hpsPFTauDiscriminationByMVA3MediumElectronRejection.mapping[12].cut = cms.double(0.897)
hpsPFTauDiscriminationByMVA3MediumElectronRejection.mapping[13].cut = cms.double(0.951)
hpsPFTauDiscriminationByMVA3MediumElectronRejection.mapping[14].cut = cms.double(0.948)
hpsPFTauDiscriminationByMVA3MediumElectronRejection.mapping[15].cut = cms.double(0.953)
hpsPFTauDiscriminationByMVA3MediumElectronRejection.mapping[16].cut = cms.double(-1.)
hpsPFTauDiscriminationByMVA3MediumElectronRejection.mapping[17].cut = cms.double(-1.)
 
hpsPFTauDiscriminationByMVA3TightElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA3LooseElectronRejection)
hpsPFTauDiscriminationByMVA3TightElectronRejection.mapping[0].cut = cms.double(0.974)
hpsPFTauDiscriminationByMVA3TightElectronRejection.mapping[1].cut = cms.double(0.976)
hpsPFTauDiscriminationByMVA3TightElectronRejection.mapping[2].cut = cms.double(0.978)
hpsPFTauDiscriminationByMVA3TightElectronRejection.mapping[3].cut = cms.double(0.978)
hpsPFTauDiscriminationByMVA3TightElectronRejection.mapping[4].cut = cms.double(0.971)
hpsPFTauDiscriminationByMVA3TightElectronRejection.mapping[5].cut = cms.double(0.969)
hpsPFTauDiscriminationByMVA3TightElectronRejection.mapping[6].cut = cms.double(0.982)
hpsPFTauDiscriminationByMVA3TightElectronRejection.mapping[7].cut = cms.double(0.972)
hpsPFTauDiscriminationByMVA3TightElectronRejection.mapping[8].cut = cms.double(0.982)
hpsPFTauDiscriminationByMVA3TightElectronRejection.mapping[9].cut = cms.double(0.977)
hpsPFTauDiscriminationByMVA3TightElectronRejection.mapping[10].cut = cms.double(0.981)
hpsPFTauDiscriminationByMVA3TightElectronRejection.mapping[11].cut = cms.double(0.978)
hpsPFTauDiscriminationByMVA3TightElectronRejection.mapping[12].cut = cms.double(0.897)
hpsPFTauDiscriminationByMVA3TightElectronRejection.mapping[13].cut = cms.double(0.976)
hpsPFTauDiscriminationByMVA3TightElectronRejection.mapping[14].cut = cms.double(0.975)
hpsPFTauDiscriminationByMVA3TightElectronRejection.mapping[15].cut = cms.double(0.977)
hpsPFTauDiscriminationByMVA3TightElectronRejection.mapping[16].cut = cms.double(-1.)
hpsPFTauDiscriminationByMVA3TightElectronRejection.mapping[17].cut = cms.double(-1.)
 
hpsPFTauDiscriminationByMVA3VTightElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA3LooseElectronRejection)
hpsPFTauDiscriminationByMVA3VTightElectronRejection.mapping[0].cut = cms.double(0.986)
hpsPFTauDiscriminationByMVA3VTightElectronRejection.mapping[1].cut = cms.double(0.986)
hpsPFTauDiscriminationByMVA3VTightElectronRejection.mapping[2].cut = cms.double(0.986)
hpsPFTauDiscriminationByMVA3VTightElectronRejection.mapping[3].cut = cms.double(0.99)
hpsPFTauDiscriminationByMVA3VTightElectronRejection.mapping[4].cut = cms.double(0.983)
hpsPFTauDiscriminationByMVA3VTightElectronRejection.mapping[5].cut = cms.double(0.977)
hpsPFTauDiscriminationByMVA3VTightElectronRejection.mapping[6].cut = cms.double(0.992)
hpsPFTauDiscriminationByMVA3VTightElectronRejection.mapping[7].cut = cms.double(0.981)
hpsPFTauDiscriminationByMVA3VTightElectronRejection.mapping[8].cut = cms.double(0.989)
hpsPFTauDiscriminationByMVA3VTightElectronRejection.mapping[9].cut = cms.double(0.989)
hpsPFTauDiscriminationByMVA3VTightElectronRejection.mapping[10].cut = cms.double(0.987)
hpsPFTauDiscriminationByMVA3VTightElectronRejection.mapping[11].cut = cms.double(0.987)
hpsPFTauDiscriminationByMVA3VTightElectronRejection.mapping[12].cut = cms.double(0.976)
hpsPFTauDiscriminationByMVA3VTightElectronRejection.mapping[13].cut = cms.double(0.991)
hpsPFTauDiscriminationByMVA3VTightElectronRejection.mapping[14].cut = cms.double(0.984)
hpsPFTauDiscriminationByMVA3VTightElectronRejection.mapping[15].cut = cms.double(0.986)
hpsPFTauDiscriminationByMVA3VTightElectronRejection.mapping[16].cut = cms.double(-1.)
hpsPFTauDiscriminationByMVA3VTightElectronRejection.mapping[17].cut = cms.double(-1.)

hpsPFTauDiscriminationByMVA4rawElectronRejection = pfRecoTauDiscriminationAgainstElectronMVA4GBR.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone()
)

hpsPFTauDiscriminationByMVA4LooseElectronRejection = recoTauDiscriminantCutMultiplexer.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByMVA4rawElectronRejection'),
    key = cms.InputTag('hpsPFTauDiscriminationByMVA4rawElectronRejection:category'),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0), # minMVANoEleMatchWOgWOgsfBL
            cut = cms.double(0.841)
        ),
        cms.PSet(
            category = cms.uint32(1), # minMVANoEleMatchWOgWgsfBL
            cut = cms.double(0.853)
        ),
        cms.PSet(
            category = cms.uint32(2), # minMVANoEleMatchWgWOgsfBL
            cut = cms.double(0.862)
        ),
        cms.PSet(
            category = cms.uint32(3), # minMVANoEleMatchWgWgsfBL
            cut = cms.double(0.864)
        ),
         cms.PSet(
            category = cms.uint32(4), # minMVAWOgWOgsfBL
            cut = cms.double(0.858)
        ),
        cms.PSet(
            category = cms.uint32(5), # minMVAWOgWgsfBL
            cut = cms.double(0.838)
        ),
        cms.PSet(
            category = cms.uint32(6), # minMVAWgWOgsfBL
            cut = cms.double(0.834)
        ),
        cms.PSet(
            category = cms.uint32(7), # minMVAWgWgsfBL
            cut = cms.double(0.864)
        ),
        cms.PSet(
            category = cms.uint32(8), # minMVANoEleMatchWOgWOgsfEC
            cut = cms.double(0.887)
        ),
        cms.PSet(
            category = cms.uint32(9), # minMVANoEleMatchWOgWgsfEC
            cut = cms.double(0.866)
        ),
        cms.PSet(
            category = cms.uint32(10), # minMVANoEleMatchWgWOgsfEC
            cut = cms.double(0.846)
        ),
        cms.PSet(
            category = cms.uint32(11), # minMVANoEleMatchWgWgsfEC
            cut = cms.double(0.869)
        ),
         cms.PSet(
            category = cms.uint32(12), # minMVAWOgWOgsfEC
            cut = cms.double(0.889)
        ),
        cms.PSet(
            category = cms.uint32(13), # minMVAWOgWgsfEC
            cut = cms.double(0.857)
        ),
        cms.PSet(
            category = cms.uint32(14), # minMVAWgWOgsfEC
            cut = cms.double(0.882)
        ),
        cms.PSet(
            category = cms.uint32(15), # minMVAWgWgsfEC
            cut = cms.double(0.863)
        )
    )
)

hpsPFTauDiscriminationByMVA4MediumElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA4LooseElectronRejection)
hpsPFTauDiscriminationByMVA4MediumElectronRejection.mapping[0].cut = cms.double(0.932)
hpsPFTauDiscriminationByMVA4MediumElectronRejection.mapping[1].cut = cms.double(0.944)
hpsPFTauDiscriminationByMVA4MediumElectronRejection.mapping[2].cut = cms.double(0.949)
hpsPFTauDiscriminationByMVA4MediumElectronRejection.mapping[3].cut = cms.double(0.949)
hpsPFTauDiscriminationByMVA4MediumElectronRejection.mapping[4].cut = cms.double(0.94)
hpsPFTauDiscriminationByMVA4MediumElectronRejection.mapping[5].cut = cms.double(0.935)
hpsPFTauDiscriminationByMVA4MediumElectronRejection.mapping[6].cut = cms.double(0.943)
hpsPFTauDiscriminationByMVA4MediumElectronRejection.mapping[7].cut = cms.double(0.951)
hpsPFTauDiscriminationByMVA4MediumElectronRejection.mapping[8].cut = cms.double(0.965)
hpsPFTauDiscriminationByMVA4MediumElectronRejection.mapping[9].cut = cms.double(0.959)
hpsPFTauDiscriminationByMVA4MediumElectronRejection.mapping[10].cut = cms.double(0.952)
hpsPFTauDiscriminationByMVA4MediumElectronRejection.mapping[11].cut = cms.double(0.954)
hpsPFTauDiscriminationByMVA4MediumElectronRejection.mapping[12].cut = cms.double(0.966)
hpsPFTauDiscriminationByMVA4MediumElectronRejection.mapping[13].cut = cms.double(0.963)
hpsPFTauDiscriminationByMVA4MediumElectronRejection.mapping[14].cut = cms.double(0.943)
hpsPFTauDiscriminationByMVA4MediumElectronRejection.mapping[15].cut = cms.double(0.958)

hpsPFTauDiscriminationByMVA4TightElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA4LooseElectronRejection)
hpsPFTauDiscriminationByMVA4TightElectronRejection.mapping[0].cut = cms.double(0.969)
hpsPFTauDiscriminationByMVA4TightElectronRejection.mapping[1].cut = cms.double(0.967)
hpsPFTauDiscriminationByMVA4TightElectronRejection.mapping[2].cut = cms.double(0.972)
hpsPFTauDiscriminationByMVA4TightElectronRejection.mapping[3].cut = cms.double(0.974)
hpsPFTauDiscriminationByMVA4TightElectronRejection.mapping[4].cut = cms.double(0.965)
hpsPFTauDiscriminationByMVA4TightElectronRejection.mapping[5].cut = cms.double(0.966)
hpsPFTauDiscriminationByMVA4TightElectronRejection.mapping[6].cut = cms.double(0.973)
hpsPFTauDiscriminationByMVA4TightElectronRejection.mapping[7].cut = cms.double(0.971)
hpsPFTauDiscriminationByMVA4TightElectronRejection.mapping[8].cut = cms.double(0.975)
hpsPFTauDiscriminationByMVA4TightElectronRejection.mapping[9].cut = cms.double(0.972)
hpsPFTauDiscriminationByMVA4TightElectronRejection.mapping[10].cut = cms.double(0.975)
hpsPFTauDiscriminationByMVA4TightElectronRejection.mapping[11].cut = cms.double(0.975)
hpsPFTauDiscriminationByMVA4TightElectronRejection.mapping[12].cut = cms.double(0.968)
hpsPFTauDiscriminationByMVA4TightElectronRejection.mapping[13].cut = cms.double(0.972)
hpsPFTauDiscriminationByMVA4TightElectronRejection.mapping[14].cut = cms.double(0.962)
hpsPFTauDiscriminationByMVA4TightElectronRejection.mapping[15].cut = cms.double(0.971)

hpsPFTauDiscriminationByMVA4VTightElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA4LooseElectronRejection)
hpsPFTauDiscriminationByMVA4VTightElectronRejection.mapping[0].cut = cms.double(0.986)
hpsPFTauDiscriminationByMVA4VTightElectronRejection.mapping[1].cut = cms.double(0.977)
hpsPFTauDiscriminationByMVA4VTightElectronRejection.mapping[2].cut = cms.double(0.98)
hpsPFTauDiscriminationByMVA4VTightElectronRejection.mapping[3].cut = cms.double(0.985)
hpsPFTauDiscriminationByMVA4VTightElectronRejection.mapping[4].cut = cms.double(0.977)
hpsPFTauDiscriminationByMVA4VTightElectronRejection.mapping[5].cut = cms.double(0.979)
hpsPFTauDiscriminationByMVA4VTightElectronRejection.mapping[6].cut = cms.double(0.983)
hpsPFTauDiscriminationByMVA4VTightElectronRejection.mapping[7].cut = cms.double(0.98)
hpsPFTauDiscriminationByMVA4VTightElectronRejection.mapping[8].cut = cms.double(0.982)
hpsPFTauDiscriminationByMVA4VTightElectronRejection.mapping[9].cut = cms.double(0.986)
hpsPFTauDiscriminationByMVA4VTightElectronRejection.mapping[10].cut = cms.double(0.984)
hpsPFTauDiscriminationByMVA4VTightElectronRejection.mapping[11].cut = cms.double(0.983)
hpsPFTauDiscriminationByMVA4VTightElectronRejection.mapping[12].cut = cms.double(0.968)
hpsPFTauDiscriminationByMVA4VTightElectronRejection.mapping[13].cut = cms.double(0.973)
hpsPFTauDiscriminationByMVA4VTightElectronRejection.mapping[14].cut = cms.double(0.981)
hpsPFTauDiscriminationByMVA4VTightElectronRejection.mapping[15].cut = cms.double(0.979)

hpsPFTauDiscriminationByMVA5rawElectronRejection = pfRecoTauDiscriminationAgainstElectronMVA4GBR.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    method = cms.string("BDTG"),
    gbrFile = cms.FileInPath('RecoTauTag/RecoTau/data/gbrDiscriminationAgainstElectronMVA5.root'),
)

hpsPFTauDiscriminationByMVA5VLooseElectronRejection = recoTauDiscriminantCutMultiplexer.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByMVA5rawElectronRejection'),
    key = cms.InputTag('hpsPFTauDiscriminationByMVA5rawElectronRejection:category'),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0), # minMVANoEleMatchWOgWOgsfBL
            cut = cms.double(-0.156)
        ),
        cms.PSet(
            category = cms.uint32(1), # minMVANoEleMatchWOgWgsfBL
            cut = cms.double(0.060)
        ),
        cms.PSet(
            category = cms.uint32(2), # minMVANoEleMatchWgWOgsfBL
            cut = cms.double(-0.078)
        ),
        cms.PSet(
            category = cms.uint32(3), # minMVANoEleMatchWgWgsfBL
            cut = cms.double(0.004)
        ),
         cms.PSet(
            category = cms.uint32(4), # minMVAWOgWOgsfBL
            cut = cms.double(0.499)
        ),
        cms.PSet(
            category = cms.uint32(5), # minMVAWOgWgsfBL
            cut = cms.double(-0.205)
        ),
        cms.PSet(
            category = cms.uint32(6), # minMVAWgWOgsfBL
            cut = cms.double(-0.033)
        ),
        cms.PSet(
            category = cms.uint32(7), # minMVAWgWgsfBL
            cut = cms.double(-0.087)
        ),
        cms.PSet(
            category = cms.uint32(8), # minMVANoEleMatchWOgWOgsfEC
            cut = cms.double(0.344)
        ),
        cms.PSet(
            category = cms.uint32(9), # minMVANoEleMatchWOgWgsfEC
            cut = cms.double(0.314)
        ),
        cms.PSet(
            category = cms.uint32(10), # minMVANoEleMatchWgWOgsfEC
            cut = cms.double(-0.035)
        ),
        cms.PSet(
            category = cms.uint32(11), # minMVANoEleMatchWgWgsfEC
            cut = cms.double(-0.051)
        ),
         cms.PSet(
            category = cms.uint32(12), # minMVAWOgWOgsfEC
            cut = cms.double(0.505)
        ),
        cms.PSet(
            category = cms.uint32(13), # minMVAWOgWgsfEC
            cut = cms.double(-0.139)
        ),
        cms.PSet(
            category = cms.uint32(14), # minMVAWgWOgsfEC
            cut = cms.double(0.133)
        ),
        cms.PSet(
            category = cms.uint32(15), # minMVAWgWgsfEC
            cut = cms.double(-0.070)
        )
    )
)

hpsPFTauDiscriminationByMVA5LooseElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA5VLooseElectronRejection)
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[0].cut = cms.double(0.702)
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[1].cut = cms.double(0.648)
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[2].cut = cms.double(0.662)
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[3].cut = cms.double(0.715)
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[4].cut = cms.double(0.765)
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[5].cut = cms.double(0.634)
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[6].cut = cms.double(0.718)
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[7].cut = cms.double(0.696)
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[8].cut = cms.double(0.747)
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[9].cut = cms.double(0.767)
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[10].cut = cms.double(0.694)
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[11].cut = cms.double(0.719)
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[12].cut = cms.double(0.684)
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[13].cut = cms.double(0.667)
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[14].cut = cms.double(0.740)
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[15].cut = cms.double(0.724)

hpsPFTauDiscriminationByMVA5MediumElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA5LooseElectronRejection)
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[0].cut = cms.double(0.845)
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[1].cut = cms.double(0.865)
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[2].cut = cms.double(0.902)
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[3].cut = cms.double(0.905)
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[4].cut = cms.double(0.897)
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[5].cut = cms.double(0.867)
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[6].cut = cms.double(0.887)
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[7].cut = cms.double(0.906)
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[8].cut = cms.double(0.927)
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[9].cut = cms.double(0.925)
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[10].cut = cms.double(0.909)
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[11].cut = cms.double(0.912)
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[12].cut = cms.double(0.900)
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[13].cut = cms.double(0.904)
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[14].cut = cms.double(0.902)
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[15].cut = cms.double(0.908)

hpsPFTauDiscriminationByMVA5TightElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA5LooseElectronRejection)
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[0].cut = cms.double(0.958)
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[1].cut = cms.double(0.948)
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[2].cut = cms.double(0.957)
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[3].cut = cms.double(0.953)
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[4].cut = cms.double(0.946)
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[5].cut = cms.double(0.924)
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[6].cut = cms.double(0.952)
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[7].cut = cms.double(0.954)
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[8].cut = cms.double(0.959)
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[9].cut = cms.double(0.947)
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[10].cut = cms.double(0.958)
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[11].cut = cms.double(0.959)
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[12].cut = cms.double(0.931)
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[13].cut = cms.double(0.936)
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[14].cut = cms.double(0.942)
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[15].cut = cms.double(0.952)

hpsPFTauDiscriminationByMVA5VTightElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA5LooseElectronRejection)
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[0].cut = cms.double(0.981)
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[1].cut = cms.double(0.967)
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[2].cut = cms.double(0.974)
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[3].cut = cms.double(0.975)
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[4].cut = cms.double(0.954)
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[5].cut = cms.double(0.946)
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[6].cut = cms.double(0.972)
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[7].cut = cms.double(0.969)
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[8].cut = cms.double(0.975)
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[9].cut = cms.double(0.973)
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[10].cut = cms.double(0.976)
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[11].cut = cms.double(0.973)
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[12].cut = cms.double(0.957)
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[13].cut = cms.double(0.966)
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[14].cut = cms.double(0.968)
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[15].cut = cms.double(0.967)

hpsPFTauDiscriminationByMVA6rawElectronRejection = pfRecoTauDiscriminationAgainstElectronMVA5GBR.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    method = cms.string("BDTG"),
    gbrFile = cms.FileInPath('RecoTauTag/RecoTau/data/gbrDiscriminationAgainstElectronMVA6.root'),
)

hpsPFTauDiscriminationByMVA6VLooseElectronRejection = recoTauDiscriminantCutMultiplexer.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByMVA6rawElectronRejection'),
    key = cms.InputTag('hpsPFTauDiscriminationByMVA6rawElectronRejection:category'),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0), # minMVANoEleMatchWOgWOgsfBL
            cut = cms.double(-0.050)
        ),
        cms.PSet(
            category = cms.uint32(1), # minMVANoEleMatchWOgWgsfBL
            cut = cms.double(0.224)
        ),
        cms.PSet(
            category = cms.uint32(2), # minMVANoEleMatchWgWOgsfBL
            cut = cms.double(-0.038)
        ),
        cms.PSet(
            category = cms.uint32(3), # minMVANoEleMatchWgWgsfBL
            cut = cms.double(-0.012)
        ),
         cms.PSet(
            category = cms.uint32(4), # minMVAWOgWOgsfBL
            cut = cms.double(0.436)
        ),
        cms.PSet(
            category = cms.uint32(5), # minMVAWOgWgsfBL
            cut = cms.double(-0.154)
        ),
        cms.PSet(
            category = cms.uint32(6), # minMVAWgWOgsfBL
            cut = cms.double(0.132)
        ),
        cms.PSet(
            category = cms.uint32(7), # minMVAWgWgsfBL
            cut = cms.double(-0.086)
        ),
        cms.PSet(
            category = cms.uint32(8), # minMVANoEleMatchWOgWOgsfEC
            cut = cms.double(0.546)
        ),
        cms.PSet(
            category = cms.uint32(9), # minMVANoEleMatchWOgWgsfEC
            cut = cms.double(0.302)
        ),
        cms.PSet(
            category = cms.uint32(10), # minMVANoEleMatchWgWOgsfEC
            cut = cms.double(0.189)
        ),
        cms.PSet(
            category = cms.uint32(11), # minMVANoEleMatchWgWgsfEC
            cut = cms.double(0.065)
        ),
         cms.PSet(
            category = cms.uint32(12), # minMVAWOgWOgsfEC
            cut = cms.double(0.815)
        ),
        cms.PSet(
            category = cms.uint32(13), # minMVAWOgWgsfEC
            cut = cms.double(-0.167)
        ),
        cms.PSet(
            category = cms.uint32(14), # minMVAWgWOgsfEC
            cut = cms.double(0.422)
        ),
        cms.PSet(
            category = cms.uint32(15), # minMVAWgWgsfEC
            cut = cms.double(0.032)
        )
    )
)

hpsPFTauDiscriminationByMVA6LooseElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA6VLooseElectronRejection)
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[0].cut = cms.double(0.718)
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[1].cut = cms.double(0.712)
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[2].cut = cms.double(0.755)
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[3].cut = cms.double(0.753)
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[4].cut = cms.double(0.745)
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[5].cut = cms.double(0.680)
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[6].cut = cms.double(0.720)
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[7].cut = cms.double(0.736)
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[8].cut = cms.double(0.777)
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[9].cut = cms.double(0.789)
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[10].cut = cms.double(0.767)
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[11].cut = cms.double(0.730)
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[12].cut = cms.double(0.820)
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[13].cut = cms.double(0.674)
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[14].cut = cms.double(0.771)
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[15].cut = cms.double(0.710)

hpsPFTauDiscriminationByMVA6MediumElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA6LooseElectronRejection)
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[0].cut = cms.double(0.907)
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[1].cut = cms.double(0.902)
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[2].cut = cms.double(0.905)
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[3].cut = cms.double(0.926)
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[4].cut = cms.double(0.929)
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[5].cut = cms.double(0.901)
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[6].cut = cms.double(0.918)
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[7].cut = cms.double(0.917)
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[8].cut = cms.double(0.927)
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[9].cut = cms.double(0.924)
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[10].cut = cms.double(0.931)
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[11].cut = cms.double(0.916)
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[12].cut = cms.double(0.843)
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[13].cut = cms.double(0.911)
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[14].cut = cms.double(0.916)
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[15].cut = cms.double(0.931)

hpsPFTauDiscriminationByMVA6TightElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA6LooseElectronRejection)
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[0].cut = cms.double(0.957)
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[1].cut = cms.double(0.964)
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[2].cut = cms.double(0.962)
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[3].cut = cms.double(0.964)
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[4].cut = cms.double(0.961)
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[5].cut = cms.double(0.950)
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[6].cut = cms.double(0.955)
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[7].cut = cms.double(0.955)
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[8].cut = cms.double(0.962)
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[9].cut = cms.double(0.972)
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[10].cut = cms.double(0.967)
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[11].cut = cms.double(0.964)
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[12].cut = cms.double(0.955)
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[13].cut = cms.double(0.963)
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[14].cut = cms.double(0.952)
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[15].cut = cms.double(0.964)

hpsPFTauDiscriminationByMVA6VTightElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA6LooseElectronRejection)
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[0].cut = cms.double(0.977)
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[1].cut = cms.double(0.980)
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[2].cut = cms.double(0.978)
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[3].cut = cms.double(0.979)
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[4].cut = cms.double(0.977)
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[5].cut = cms.double(0.963)
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[6].cut = cms.double(0.979)
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[7].cut = cms.double(0.974)
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[8].cut = cms.double(0.979)
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[9].cut = cms.double(0.980)
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[10].cut = cms.double(0.979)
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[11].cut = cms.double(0.978)
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[12].cut = cms.double(0.955)
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[13].cut = cms.double(0.983)
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[14].cut = cms.double(0.969)
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[15].cut = cms.double(0.977)

hpsPFTauDiscriminationByDeadECALElectronRejection = pfRecoTauDiscriminationAgainstElectronDeadECAL.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone()
)

#Define new sequence that is using smaller number on hits cut
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.clone()
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits = hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr.clone()
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits = hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr.clone()

hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits.qualityCuts.isolationQualityCuts.minTrackHits = cms.uint32(3)
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits.qualityCuts.isolationQualityCuts.minTrackHits = cms.uint32(3)
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits.qualityCuts.isolationQualityCuts.minTrackHits = cms.uint32(3)

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

# Define the HPS selection discriminator used in cleaning
hpsSelectionDiscriminator.PFTauProducer = cms.InputTag("combinatoricRecoTaus")

import RecoTauTag.RecoTau.RecoTauCleanerPlugins as cleaners

hpsPFTauProducerSansRefs = cms.EDProducer(
    "RecoTauCleaner",
    src = cms.InputTag("combinatoricRecoTaus"),
    cleaners = cms.VPSet(
        # Reject taus that have charge == 3
        cleaners.unitCharge,
         # Ignore taus reconstructed in pi0 decay modes in which the highest Pt ("leading") pi0 has pt below 2.5 GeV
         # (in order to make decay mode reconstruction less sensitive to pile-up)
         # NOTE: strips are sorted by decreasing pt
        cms.PSet(
            name = cms.string("leadStripPtLt2_5"),
            plugin = cms.string("RecoTauStringCleanerPlugin"),
            selection = cms.string("signalPiZeroCandidates().size() = 0 | signalPiZeroCandidates().at(0).pt() > 2.5"),
            selectionPassFunction = cms.string("0"),
            selectionFailValue = cms.double(1e3)
        ),
        # Reject taus that are not within DR<0.1 of the jet axis
        #cleaners.matchingConeCut,
        # Reject taus that fail HPS selections
        cms.PSet(
            name = cms.string("HPS_Select"),
            plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
            src = cms.InputTag("hpsSelectionDiscriminator"),
        ),
        # CV: Take highes pT tau (use for testing of new high pT tau reconstruction and check if it can become the new default)
        cleaners.pt,
        # CV: in case two candidates have the same Pt,
        #     prefer candidates in which PFGammas are part of strips (rather than being merged with PFRecoTauChargedHadrons)
        cleaners.stripMultiplicity,
        # Take most isolated tau
        cleaners.combinedIsolation
    )
)

hpsPFTauProducer = cms.EDProducer(
    "RecoTauPiZeroUnembedder",
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
hpsPFTauMVA3IsolationChargedIsoPtSum = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    ApplyDiscriminationByECALIsolation = cms.bool(False),
    ApplyDiscriminationByTrackerIsolation = cms.bool(True),
    applySumPtCut = cms.bool(False),
    applyDeltaBetaCorrection = cms.bool(False),
    storeRawSumPt = cms.bool(True),
    storeRawPUsumPt = cms.bool(False),
    customOuterCone = cms.double(0.5),
    isoConeSizeForDeltaBeta = cms.double(0.8),
    verbosity = cms.int32(0)
)
hpsPFTauMVA3IsolationNeutralIsoPtSum = hpsPFTauMVA3IsolationChargedIsoPtSum.clone(
    ApplyDiscriminationByECALIsolation = cms.bool(True),
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    verbosity = cms.int32(0)
)
hpsPFTauMVA3IsolationPUcorrPtSum = hpsPFTauMVA3IsolationChargedIsoPtSum.clone(
    ApplyDiscriminationByECALIsolation = cms.bool(False),
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    applyDeltaBetaCorrection = cms.bool(True),
    storeRawSumPt = cms.bool(False),
    storeRawPUsumPt = cms.bool(True),
    verbosity = cms.int32(0)
)
hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw = discriminationByIsolationMVA2raw.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/gbrDiscriminationByIsolationMVA3_oldDMwoLT.root'),
    mvaName = cms.string("tauIdMVAoldDMwoLT"),
    mvaOpt = cms.string("oldDMwoLT"),
    srcTauTransverseImpactParameters = cms.InputTag('hpsPFTauTransverseImpactParameters'),    
    srcChargedIsoPtSum = cms.InputTag('hpsPFTauMVA3IsolationChargedIsoPtSum'),
    srcNeutralIsoPtSum = cms.InputTag('hpsPFTauMVA3IsolationNeutralIsoPtSum'),
    srcPUcorrPtSum = cms.InputTag('hpsPFTauMVA3IsolationPUcorrPtSum'),
    verbosity = cms.int32(0)
)
hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT = discriminationByIsolationMVA2VLoose.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),    
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw:category'),
    inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_oldDMwoLT.root'),
    mvaOutput_normalization = cms.string("mvaOutput_normalization_opt2a"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("opt2aEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone()
hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("opt2aEff80")
hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone()
hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("opt2aEff70")
hpsPFTauDiscriminationByTightIsolationMVA3oldDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone()
hpsPFTauDiscriminationByTightIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("opt2aEff60")
hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone()
hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("opt2aEff50")
hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone()
hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("opt2aEff40")
hpsPFTauDiscriminationByIsolationMVA3oldDMwLTraw = hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw.clone(
    inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/gbrDiscriminationByIsolationMVA3_oldDMwLT.root'),
    mvaName = cms.string("tauIdMVAoldDMwLT"),
    mvaOpt = cms.string("oldDMwLT"),
    verbosity = cms.int32(0)
)
hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone(
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3oldDMwLTraw'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3oldDMwLTraw:category'),
    inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_oldDMwLT.root'),
    mvaOutput_normalization = cms.string("mvaOutput_normalization_opt2aLT"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("opt2aLTEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT.clone()
hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("opt2aLTEff80")
hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT.clone()
hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("opt2aLTEff70")
hpsPFTauDiscriminationByTightIsolationMVA3oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT.clone()
hpsPFTauDiscriminationByTightIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("opt2aLTEff60")
hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT.clone()
hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("opt2aLTEff50")
hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT.clone()
hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("opt2aLTEff40")
hpsPFTauDiscriminationByIsolationMVA3newDMwoLTraw = hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw.clone(
    inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/gbrDiscriminationByIsolationMVA3_newDMwoLT.root'),
    mvaName = cms.string("tauIdMVAnewDMwoLT"),
    mvaOpt = cms.string("newDMwoLT"),
    verbosity = cms.int32(0)
)
hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone(
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3newDMwoLTraw'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3newDMwoLTraw:category'),
    inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_newDMwoLT.root'),
    mvaOutput_normalization = cms.string("mvaOutput_normalization_opt2b"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("opt2bEff90"),
            variable = cms.string("pt")
        )
    ),
    verbosity = cms.int32(0)
)
hpsPFTauDiscriminationByLooseIsolationMVA3newDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT.clone()
hpsPFTauDiscriminationByLooseIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("opt2bEff80")
##hpsPFTauDiscriminationByLooseIsolationMVA3newDMwoLT.verbosity = cms.int32(1)
hpsPFTauDiscriminationByMediumIsolationMVA3newDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT.clone()
hpsPFTauDiscriminationByMediumIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("opt2bEff70")
hpsPFTauDiscriminationByTightIsolationMVA3newDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT.clone()
hpsPFTauDiscriminationByTightIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("opt2bEff60")
hpsPFTauDiscriminationByVTightIsolationMVA3newDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT.clone()
hpsPFTauDiscriminationByVTightIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("opt2bEff50")
hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT.clone()
hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("opt2bEff40")
hpsPFTauDiscriminationByIsolationMVA3newDMwLTraw = hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw.clone(
    inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/gbrDiscriminationByIsolationMVA3_newDMwLT.root'),
    mvaName = cms.string("tauIdMVAnewDMwLT"),
    mvaOpt = cms.string("newDMwLT"),
    verbosity = cms.int32(0)
)
hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone(
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3newDMwLTraw'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVA3newDMwLTraw:category'),
    inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_newDMwLT.root'),
    mvaOutput_normalization = cms.string("mvaOutput_normalization_opt2bLT"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("opt2bLTEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVA3newDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT.clone()
hpsPFTauDiscriminationByLooseIsolationMVA3newDMwLT.mapping[0].cut = cms.string("opt2bLTEff80")
hpsPFTauDiscriminationByMediumIsolationMVA3newDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT.clone()
hpsPFTauDiscriminationByMediumIsolationMVA3newDMwLT.mapping[0].cut = cms.string("opt2bLTEff70")
hpsPFTauDiscriminationByTightIsolationMVA3newDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT.clone()
hpsPFTauDiscriminationByTightIsolationMVA3newDMwLT.mapping[0].cut = cms.string("opt2bLTEff60")
hpsPFTauDiscriminationByVTightIsolationMVA3newDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT.clone()
hpsPFTauDiscriminationByVTightIsolationMVA3newDMwLT.mapping[0].cut = cms.string("opt2bLTEff50")
hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT.clone()
hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwLT.mapping[0].cut = cms.string("opt2bLTEff40")

hpsPFTauMVAIsolation2Seq = cms.Sequence(
    hpsPFTauMVA3IsolationChargedIsoPtSum
   + hpsPFTauMVA3IsolationNeutralIsoPtSum
   + hpsPFTauMVA3IsolationPUcorrPtSum
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
    hpsPFTauDiscriminationByChargedIsolationSeq*
    hpsPFTauDiscriminationByIsolationSeq*
    #hpsPFTauDiscriminationByIsolationSeqRhoCorr*
    #hpsPFTauDiscriminationByIsolationSeqCustomRhoCorr*
    hpsPFTauDiscriminationByIsolationSeqDBSumPtCorr*
    hpsPFTauDiscriminationByMVAIsolationSeq*

    hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByRawChargedIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByRawGammaIsolationDBSumPtCorr*

    hpsPFTauDiscriminationByCombinedIsolationSeqDBSumPtCorr*
    hpsPFTauDiscriminationByCombinedIsolationSeqDBSumPtCorr3Hits*
    
    hpsPFTauDiscriminationByLooseElectronRejection*
    hpsPFTauDiscriminationByMediumElectronRejection*
    hpsPFTauDiscriminationByTightElectronRejection*
    hpsPFTauDiscriminationByMVA3rawElectronRejection*
    hpsPFTauDiscriminationByMVA3LooseElectronRejection*
    hpsPFTauDiscriminationByMVA3MediumElectronRejection*
    hpsPFTauDiscriminationByMVA3TightElectronRejection*
    hpsPFTauDiscriminationByMVA3VTightElectronRejection*
    hpsPFTauDiscriminationByMVA4rawElectronRejection*
    hpsPFTauDiscriminationByMVA4LooseElectronRejection*
    hpsPFTauDiscriminationByMVA4MediumElectronRejection*
    hpsPFTauDiscriminationByMVA4TightElectronRejection*
    hpsPFTauDiscriminationByMVA4VTightElectronRejection*
    hpsPFTauDiscriminationByMVA5rawElectronRejection*
    hpsPFTauDiscriminationByMVA5VLooseElectronRejection*
    hpsPFTauDiscriminationByMVA5LooseElectronRejection*
    hpsPFTauDiscriminationByMVA5MediumElectronRejection*
    hpsPFTauDiscriminationByMVA5TightElectronRejection*
    hpsPFTauDiscriminationByMVA5VTightElectronRejection*
    hpsPFTauDiscriminationByMVA6rawElectronRejection*
    hpsPFTauDiscriminationByMVA6VLooseElectronRejection*
    hpsPFTauDiscriminationByMVA6LooseElectronRejection*
    hpsPFTauDiscriminationByMVA6MediumElectronRejection*
    hpsPFTauDiscriminationByMVA6TightElectronRejection*
    hpsPFTauDiscriminationByMVA6VTightElectronRejection*
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
