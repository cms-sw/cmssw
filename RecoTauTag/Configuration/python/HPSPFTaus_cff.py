import FWCore.ParameterSet.Config as cms
import copy

'''

Sequences for HPS taus

'''

# Define the discriminators for this tau
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi                      import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackFinding_cfi            import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectron_cfi                  import *
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

hpsPFTauDiscriminationByMVA5rawElectronRejection = pfRecoTauDiscriminationAgainstElectronMVA5GBR.clone(
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
    inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationAgainstElectronMVA5.root'),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0), # minMVANoEleMatchWOgWOgsfBL
            cut = cms.string("eff99cat0"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(1), # minMVANoEleMatchWOgWgsfBL
            cut = cms.string("eff99cat1"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(2), # minMVANoEleMatchWgWOgsfBL
            cut = cms.string("eff99cat2"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(3), # minMVANoEleMatchWgWgsfBL
            cut = cms.string("eff99cat3"),
            variable = cms.string("pt")
        ),
         cms.PSet(
            category = cms.uint32(4), # minMVAWOgWOgsfBL
            cut = cms.string("eff99cat4"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(5), # minMVAWOgWgsfBL
            cut = cms.string("eff99cat5"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(6), # minMVAWgWOgsfBL
            cut = cms.string("eff99cat6"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(7), # minMVAWgWgsfBL
            cut = cms.string("eff99cat7"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(8), # minMVANoEleMatchWOgWOgsfEC
            cut = cms.string("eff99cat8"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(9), # minMVANoEleMatchWOgWgsfEC
            cut = cms.string("eff99cat9"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(10), # minMVANoEleMatchWgWOgsfEC
            cut = cms.string("eff99cat10"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(11), # minMVANoEleMatchWgWgsfEC
            cut = cms.string("eff99cat11"),
            variable = cms.string("pt")
        ),
         cms.PSet(
            category = cms.uint32(12), # minMVAWOgWOgsfEC
            cut = cms.string("eff99cat12"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(13), # minMVAWOgWgsfEC
            cut = cms.string("eff99cat13"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(14), # minMVAWgWOgsfEC
            cut = cms.string("eff99cat14"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(15), # minMVAWgWgsfEC
            cut = cms.string("eff99cat15"),
            variable = cms.string("pt")
        )
    )
)

hpsPFTauDiscriminationByMVA5LooseElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA5VLooseElectronRejection)
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[0].cut = cms.string("eff96cat0")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[1].cut = cms.string("eff96cat1")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[2].cut = cms.string("eff96cat2")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[3].cut = cms.string("eff96cat3")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[4].cut = cms.string("eff96cat4")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[5].cut = cms.string("eff96cat5")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[6].cut = cms.string("eff96cat6")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[7].cut = cms.string("eff96cat7")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[8].cut = cms.string("eff96cat8")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[9].cut = cms.string("eff96cat9")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[10].cut = cms.string("eff96cat10")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[11].cut = cms.string("eff96cat11")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[12].cut = cms.string("eff96cat12")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[13].cut = cms.string("eff96cat13")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[14].cut = cms.string("eff96cat14")
hpsPFTauDiscriminationByMVA5LooseElectronRejection.mapping[15].cut = cms.string("eff96cat15")

hpsPFTauDiscriminationByMVA5MediumElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA5VLooseElectronRejection)
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[0].cut = cms.string("eff91cat0")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[1].cut = cms.string("eff91cat1")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[2].cut = cms.string("eff91cat2")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[3].cut = cms.string("eff91cat3")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[4].cut = cms.string("eff91cat4")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[5].cut = cms.string("eff91cat5")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[6].cut = cms.string("eff91cat6")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[7].cut = cms.string("eff91cat7")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[8].cut = cms.string("eff91cat8")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[9].cut = cms.string("eff91cat9")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[10].cut = cms.string("eff91cat10")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[11].cut = cms.string("eff91cat11")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[12].cut = cms.string("eff91cat12")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[13].cut = cms.string("eff91cat13")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[14].cut = cms.string("eff91cat14")
hpsPFTauDiscriminationByMVA5MediumElectronRejection.mapping[15].cut = cms.string("eff91cat15")

hpsPFTauDiscriminationByMVA5TightElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA5VLooseElectronRejection)
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[0].cut = cms.string("eff85cat0")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[1].cut = cms.string("eff85cat1")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[2].cut = cms.string("eff85cat2")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[3].cut = cms.string("eff85cat3")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[4].cut = cms.string("eff85cat4")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[5].cut = cms.string("eff85cat5")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[6].cut = cms.string("eff85cat6")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[7].cut = cms.string("eff85cat7")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[8].cut = cms.string("eff85cat8")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[9].cut = cms.string("eff85cat9")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[10].cut = cms.string("eff85cat10")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[11].cut = cms.string("eff85cat11")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[12].cut = cms.string("eff85cat12")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[13].cut = cms.string("eff85cat13")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[14].cut = cms.string("eff85cat14")
hpsPFTauDiscriminationByMVA5TightElectronRejection.mapping[15].cut = cms.string("eff85cat15")

hpsPFTauDiscriminationByMVA5VTightElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA5VLooseElectronRejection)
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[0].cut = cms.string("eff79cat0")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[1].cut = cms.string("eff79cat1")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[2].cut = cms.string("eff79cat2")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[3].cut = cms.string("eff79cat3")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[4].cut = cms.string("eff79cat4")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[5].cut = cms.string("eff79cat5")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[6].cut = cms.string("eff79cat6")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[7].cut = cms.string("eff79cat7")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[8].cut = cms.string("eff79cat8")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[9].cut = cms.string("eff79cat9")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[10].cut = cms.string("eff79cat10")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[11].cut = cms.string("eff79cat11")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[12].cut = cms.string("eff79cat12")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[13].cut = cms.string("eff79cat13")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[14].cut = cms.string("eff79cat14")
hpsPFTauDiscriminationByMVA5VTightElectronRejection.mapping[15].cut = cms.string("eff79cat15")

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
    mvaOutput_normalization = cms.string("mvaOutput_normalization_oldDMwoLT"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("oldDMwoLTEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone()
hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("oldDMwoLTEff80")
hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone()
hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("oldDMwoLTEff70")
hpsPFTauDiscriminationByTightIsolationMVA3oldDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone()
hpsPFTauDiscriminationByTightIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("oldDMwoLTEff60")
hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone()
hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("oldDMwoLTEff50")
hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT.clone()
hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwoLT.mapping[0].cut = cms.string("oldDMwoLTEff40")
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
    mvaOutput_normalization = cms.string("mvaOutput_normalization_oldDMwLT"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("oldDMwLTEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT.clone()
hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("oldDMwLTEff80")
hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT.clone()
hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("oldDMwLTEff70")
hpsPFTauDiscriminationByTightIsolationMVA3oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT.clone()
hpsPFTauDiscriminationByTightIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("oldDMwLTEff60")
hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT.clone()
hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("oldDMwLTEff50")
hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT.clone()
hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwLT.mapping[0].cut = cms.string("oldDMwLTEff40")
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
    mvaOutput_normalization = cms.string("mvaOutput_normalization_newDMwoLT"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("newDMwoLTEff90"),
            variable = cms.string("pt")
        )
    ),
    verbosity = cms.int32(0)
)
hpsPFTauDiscriminationByLooseIsolationMVA3newDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT.clone()
hpsPFTauDiscriminationByLooseIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("newDMwoLTEff80")
##hpsPFTauDiscriminationByLooseIsolationMVA3newDMwoLT.verbosity = cms.int32(1)
hpsPFTauDiscriminationByMediumIsolationMVA3newDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT.clone()
hpsPFTauDiscriminationByMediumIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("newDMwoLTEff70")
hpsPFTauDiscriminationByTightIsolationMVA3newDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT.clone()
hpsPFTauDiscriminationByTightIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("newDMwoLTEff60")
hpsPFTauDiscriminationByVTightIsolationMVA3newDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT.clone()
hpsPFTauDiscriminationByVTightIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("newDMwoLTEff50")
hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwoLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT.clone()
hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwoLT.mapping[0].cut = cms.string("newDMwoLTEff40")
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
    mvaOutput_normalization = cms.string("mvaOutput_normalization_newDMwLT"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("newDMwLTEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVA3newDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT.clone()
hpsPFTauDiscriminationByLooseIsolationMVA3newDMwLT.mapping[0].cut = cms.string("newDMwLTEff80")
hpsPFTauDiscriminationByMediumIsolationMVA3newDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT.clone()
hpsPFTauDiscriminationByMediumIsolationMVA3newDMwLT.mapping[0].cut = cms.string("newDMwLTEff70")
hpsPFTauDiscriminationByTightIsolationMVA3newDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT.clone()
hpsPFTauDiscriminationByTightIsolationMVA3newDMwLT.mapping[0].cut = cms.string("newDMwLTEff60")
hpsPFTauDiscriminationByVTightIsolationMVA3newDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT.clone()
hpsPFTauDiscriminationByVTightIsolationMVA3newDMwLT.mapping[0].cut = cms.string("newDMwLTEff50")
hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT.clone()
hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwLT.mapping[0].cut = cms.string("newDMwLTEff40")

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
