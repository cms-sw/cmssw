import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.coreTools import *
from FWCore.GuiBrowsers.ConfigToolBase import *
from PhysicsTools.PatAlgos.tools.helpers import applyPostfix
from PhysicsTools.PatAlgos.tools.helpers import cloneProcessingSnippet
from RecoTauTag.RecoTau.TauDiscriminatorTools import *

def redoPFTauDiscriminators(process,
                            oldPFTauLabel = cms.InputTag('hpsPFTauProducer'),
                            newPFTauLabel = cms.InputTag('hpsPFTauProducer'),
                            tauType = 'hpsPFTau', postfix = ""):
    print 'Tau discriminators: ', oldPFTauLabel, '->', newPFTauLabel
    print 'Tau type: ', tauType
    #oldPFTauLabel.setModuleLabel(oldPFTauLabel.getModuleLabel()+postfix)
    tauSrc = 'PFTauProducer'

    tauDiscriminationSequence = None

    if tauType == 'hpsPFTau':
        if hasattr(process,"updateHPSPFTaus"+postfix):
            tauDiscriminationSequence = getattr(process,"patHPSPFTauDiscriminationUpdate"+postfix)
        else:
            #        remove producers
            process.patHPSPFTauDiscrimination = process.produceAndDiscriminateHPSPFTaus.copy()
            for iname in process.patHPSPFTauDiscrimination.moduleNames():
                if not (iname.find("DiscriminationBy")>-1 or iname.find("DiscriminationAgainst")>-1 or iname.find("kt6PFJetsForRhoComputationVoronoi")>-1):
                    process.patHPSPFTauDiscrimination.remove(getattr(process,iname) )
            tauDiscriminationSequence = cloneProcessingSnippet(process, process.patHPSPFTauDiscrimination, postfix)

    elif tauType == 'hpsTancTaus': #to be checked if correct
        process.patHPSTaNCPFTauDiscrimination = process.hpsTancTauInitialSequence.copy()
        process.patHPSTaNCPFTauDiscrimination *= process.hpsTancTauDiscriminantSequence
        # remove producers
        for iname in process.patHPSTaNCPFTauDiscrimination.moduleNames():
            if not (iname.find("DiscriminationBy")>-1 or iname.find("DiscriminationAgainst")>-1):
                process.patHPSTaNCPFTauDiscrimination.remove(getattr(process,iname) )
        tauDiscriminationSequence = cloneProcessingSnippet(process, process.patHPSTaNCPFTauDiscrimination, postfix)

    elif tauType == 'fixedConePFTau':
        process.patFixedConePFTauDiscrimination = process.produceAndDiscriminateFixedConePFTaus.copy()
        # remove producers
        for iname in process.patFixedConePFTauDiscrimination.moduleNames():
            if not (iname.find("DiscriminationBy")>-1 or iname.find("DiscriminationAgainst")>-1):
                process.patFixedConePFTauDiscrimination.remove(getattr(process,iname) )
        tauDiscriminationSequence =  cloneProcessingSnippet(process, process.patFixedConePFTauDiscrimination, postfix)

    elif tauType == 'shrinkingConePFTau': #shr cone with TaNC
        process.patShrinkingConePFTauDiscrimination = process.produceAndDiscriminateShrinkingConePFTaus.copy()
        process.patShrinkingConePFTauDiscrimination *= process.produceShrinkingConeDiscriminationByTauNeuralClassifier
        # remove producers
        for iname in process.patShrinkingConePFTauDiscrimination.moduleNames():
            if not (iname.find("DiscriminationBy")>-1 or iname.find("DiscriminationAgainst")>-1):
                process.patShrinkingConePFTauDiscrimination.remove(getattr(process,iname) )
        tauDiscriminationSequence =  cloneProcessingSnippet(process, process.patShrinkingConePFTauDiscrimination, postfix)

    elif tauType == 'caloTau':
        # fill calo sequence by discriminants
        process.patCaloTauDiscrimination = process.tautagging.copy()
        # remove producers
        for iname in process.patCaloTauDiscrimination.moduleNames():
            if not (iname.find("DiscriminationBy")>-1 or iname.find("DiscriminationAgainst")>-1):
                process.patCaloTauDiscrimination.remove(getattr(process,iname) )
        tauDiscriminationSequence =  cloneProcessingSnippet(process, process.patCaloTauDiscrimination, postfix)
        tauSrc = 'CaloTauProducer'
    else:
        raise StandardError, "Unkown tauType: '%s'"%tauType

    if not hasattr(process,"updateHPSPFTaus"+postfix):
        applyPostfix(process,"patDefaultSequence",postfix).replace(
            applyPostfix(process,"patTaus",postfix),
            tauDiscriminationSequence*applyPostfix(process,"patTaus",postfix)
            )

    massSearchReplaceParam(tauDiscriminationSequence, tauSrc, oldPFTauLabel, newPFTauLabel)

# switch to CaloTau collection
def switchToCaloTau(process,
                    pfTauLabelOld = cms.InputTag('hpsPFTauProducer'),
                    pfTauLabelNew = cms.InputTag('caloRecoTauProducer'),
                    patTauLabel = "",
                    postfix = ""):
    print ' Taus: ', pfTauLabelOld, '->', pfTauLabelNew

    caloTauLabel = pfTauLabelNew
    applyPostfix(process, "tauMatch" + patTauLabel, postfix).src = caloTauLabel
    applyPostfix(process, "tauGenJetMatch"+ patTauLabel, postfix).src = caloTauLabel

    applyPostfix(process, "patTaus" + patTauLabel, postfix).tauSource = caloTauLabel
    applyPostfix(process, "patTaus" + patTauLabel, postfix).tauIDSources = _buildIDSourcePSet('caloRecoTau', classicTauIDSources, postfix)
#    applyPostfix(process, "patTaus" + patTauLabel, postfix).tauIDSources = cms.PSet(
#        leadingTrackFinding = cms.InputTag("caloRecoTauDiscriminationByLeadingTrackFinding" + postfix),
#        leadingTrackPtCut   = cms.InputTag("caloRecoTauDiscriminationByLeadingTrackPtCut" + postfix),
#        trackIsolation      = cms.InputTag("caloRecoTauDiscriminationByTrackIsolation" + postfix),
#        ecalIsolation       = cms.InputTag("caloRecoTauDiscriminationByECALIsolation" + postfix),
#        byIsolation         = cms.InputTag("caloRecoTauDiscriminationByIsolation" + postfix),
#        againstElectron     = cms.InputTag("caloRecoTauDiscriminationAgainstElectron" + postfix),
#        againstMuon         = cms.InputTag("caloRecoTauDiscriminationAgainstMuon" + postfix)
#    )
    ## Isolation is somewhat an issue, so we start just by turning it off
    print "NO PF Isolation will be computed for CaloTau (this could be improved later)"
    applyPostfix(process, "patTaus" + patTauLabel, postfix).isolation   = cms.PSet()
    applyPostfix(process, "patTaus" + patTauLabel, postfix).isoDeposits = cms.PSet()
    applyPostfix(process, "patTaus" + patTauLabel, postfix).userIsolation = cms.PSet()

    ## no tau-jet energy corrections determined for CaloTaus yet
#    applyPostfix(process, "patTauJetCorrFactors" + patTauLabel, postfix).src = caloTauLabel
#    applyPostfix(process, "patTaus" + patTauLabel, postfix).addTauJetCorrFactors = cms.bool(False)

    ## adapt cleanPatTaus
    if hasattr(process, "cleanPatTaus" + patTauLabel + postfix):
        getattr(process, "cleanPatTaus" + patTauLabel + postfix).preselection = \
      'tauID("leadingTrackFinding") > 0.5 & tauID("leadingTrackPtCut") > 0.5' \
     + ' & tauID("byIsolation") > 0.5 & tauID("againstElectron") > 0.5 & (signalTracks.size() = 1 | signalTracks.size() = 3)'

def _buildIDSourcePSet(pfTauType, idSources, postfix =""):
    """ Build a PSet defining the tau ID sources to embed into the pat::Tau """
    output = cms.PSet()
    for label, discriminator in idSources:
        if ":" in discriminator:
          discr = discriminator.split(":")
          setattr(output, label, cms.InputTag(pfTauType + discr[0] + postfix + ":" + discr[1]))
        else:  
          setattr(output, label, cms.InputTag(pfTauType + discriminator + postfix))
    return output

def _switchToPFTau(process,
                   pfTauLabelOld,
                   pfTauLabelNew,
                   pfTauType,
                   idSources,
                   jecLevels, jecPayloadMapping,
                   patTauLabel = "",
                   postfix = ""):
    """internal auxiliary function to switch to **any** PFTau collection"""
    print ' Taus: ', pfTauLabelOld, '->', pfTauLabelNew

    applyPostfix(process, "tauMatch" + patTauLabel, postfix).src = pfTauLabelNew
    applyPostfix(process, "tauGenJetMatch" + patTauLabel, postfix).src = pfTauLabelNew

    applyPostfix(process, "tauIsoDepositPFCandidates" + patTauLabel, postfix).src = pfTauLabelNew
    applyPostfix(process, "tauIsoDepositPFCandidates" + patTauLabel, postfix).ExtractorPSet.tauSource = pfTauLabelNew
    applyPostfix(process, "tauIsoDepositPFChargedHadrons" + patTauLabel, postfix).src = pfTauLabelNew
    applyPostfix(process, "tauIsoDepositPFChargedHadrons" + patTauLabel, postfix).ExtractorPSet.tauSource = pfTauLabelNew
    applyPostfix(process, "tauIsoDepositPFNeutralHadrons" + patTauLabel, postfix).src = pfTauLabelNew
    applyPostfix(process, "tauIsoDepositPFNeutralHadrons" + patTauLabel, postfix).ExtractorPSet.tauSource = pfTauLabelNew
    applyPostfix(process, "tauIsoDepositPFGammas" + patTauLabel, postfix).src = pfTauLabelNew
    applyPostfix(process, "tauIsoDepositPFGammas" + patTauLabel, postfix).ExtractorPSet.tauSource = pfTauLabelNew

#    applyPostfix(process, "patTauJetCorrFactors" + patTauLabel, postfix).src = pfTauLabelNew
#    if len(jecLevels) > 0:
#        applyPostfix(process, "patTaus" + patTauLabel, postfix).addTauJetCorrFactors = cms.bool(True)
#        applyPostfix(process, "patTauJetCorrFactors" + patTauLabel, postfix).parameters = jecPayloadMapping
#    else:
#        applyPostfix(process, "patTaus" + patTauLabel, postfix).addTauJetCorrFactors = cms.bool(False)

    applyPostfix(process, "patTaus" + patTauLabel, postfix).tauSource = pfTauLabelNew
    applyPostfix(process, "patTaus" + patTauLabel, postfix).tauIDSources = _buildIDSourcePSet(pfTauType, idSources, postfix)

    if hasattr(process, "cleanPatTaus" + patTauLabel + postfix):
        getattr(process, "cleanPatTaus" + patTauLabel + postfix).preselection = \
          'tauID("leadingTrackFinding") > 0.5 & tauID("leadingPionPtCut") > 0.5 & tauID("byIsolationUsingLeadingPion") > 0.5' \
         + ' & tauID("againstMuon") > 0.5 & tauID("againstElectron") > 0.5' \
         + ' & (signalPFChargedHadrCands.size() = 1 | signalPFChargedHadrCands.size() = 3)'

# Name mapping for classic tau ID sources (present for fixed and shrinkingCones)
classicTauIDSources = [
    ("leadingTrackFinding", "DiscriminationByLeadingTrackFinding"),
    ("leadingTrackPtCut", "DiscriminationByLeadingTrackPtCut"),
    ("trackIsolation", "DiscriminationByTrackIsolation"),
    ("ecalIsolation", "DiscriminationByECALIsolation"),
    ("byIsolation", "DiscriminationByIsolation"),
    ("againstElectron", "DiscriminationAgainstElectron"),
    ("againstMuon", "DiscriminationAgainstMuon") ]

classicPFTauIDSources = [
    ("leadingPionPtCut", "DiscriminationByLeadingPionPtCut"),
    ("trackIsolationUsingLeadingPion", "DiscriminationByTrackIsolationUsingLeadingPion"),
    ("ecalIsolationUsingLeadingPion", "DiscriminationByECALIsolationUsingLeadingPion"),
    ("byIsolationUsingLeadingPion", "DiscriminationByIsolationUsingLeadingPion")]

# Tau Neural Classifier Discriminators
tancTauIDSources = [
    ("byTaNC", "DiscriminationByTaNC"),
    ("byTaNCfrOnePercent", "DiscriminationByTaNCfrOnePercent"),
    ("byTaNCfrHalfPercent", "DiscriminationByTaNCfrHalfPercent"),
    ("byTaNCfrQuarterPercent", "DiscriminationByTaNCfrQuarterPercent"),
    ("byTaNCfrTenthPercent", "DiscriminationByTaNCfrTenthPercent") ]

# Hadron-plus-strip(s) (HPS) Tau Discriminators
hpsTauIDSources = [
    ("decayModeFinding", "DiscriminationByDecayModeFinding"),
    ("byVLooseIsolation", "DiscriminationByVLooseIsolation"),
    ("byLooseIsolation", "DiscriminationByLooseIsolation"),
    ("byMediumIsolation", "DiscriminationByMediumIsolation"),
    ("byTightIsolation", "DiscriminationByTightIsolation"),
    ("byVLooseIsolationDeltaBetaCorr", "DiscriminationByVLooseIsolationDBSumPtCorr"),
    ("byLooseIsolationDeltaBetaCorr", "DiscriminationByLooseIsolationDBSumPtCorr"),
    ("byMediumIsolationDeltaBetaCorr", "DiscriminationByMediumIsolationDBSumPtCorr"),
    ("byTightIsolationDeltaBetaCorr", "DiscriminationByTightIsolationDBSumPtCorr"),
    ("byVLooseCombinedIsolationDeltaBetaCorr", "DiscriminationByVLooseCombinedIsolationDBSumPtCorr"),
    ("byLooseCombinedIsolationDeltaBetaCorr", "DiscriminationByLooseCombinedIsolationDBSumPtCorr"),
    ("byMediumCombinedIsolationDeltaBetaCorr", "DiscriminationByMediumCombinedIsolationDBSumPtCorr"),
    ("byTightCombinedIsolationDeltaBetaCorr", "DiscriminationByTightCombinedIsolationDBSumPtCorr"),
    ("byCombinedIsolationDeltaBetaCorrRaw", "DiscriminationByRawCombinedIsolationDBSumPtCorr"),
#    ("byLooseCombinedIsolationDeltaBetaCorr3Hits", "DiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits"),
#    ("byMediumCombinedIsolationDeltaBetaCorr3Hits", "DiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits"),
#    ("byTightCombinedIsolationDeltaBetaCorr3Hits", "DiscriminationByTightCombinedIsolationDBSumPtCorr3Hits"),    
    ("byIsolationMVAraw", "DiscriminationByIsolationMVAraw"),
    ("byLooseIsolationMVA", "DiscriminationByLooseIsolationMVA"),
    ("byMediumIsolationMVA", "DiscriminationByMediumIsolationMVA"),
    ("byTightIsolationMVA", "DiscriminationByTightIsolationMVA"),
    ("againstElectronLoose", "DiscriminationByLooseElectronRejection"),
    ("againstElectronMedium", "DiscriminationByMediumElectronRejection"),
    ("againstElectronTight", "DiscriminationByTightElectronRejection"),
    ("againstElectronMVA", "DiscriminationByMVAElectronRejection"),
    ("againstElectronMVA2raw", "DiscriminationByMVA2rawElectronRejection"),
    ("againstElectronMVA2category", "DiscriminationByMVA2rawElectronRejection:category"),
    ("againstElectronVLooseMVA2", "DiscriminationByMVA2VLooseElectronRejection"),
    ("againstElectronLooseMVA2", "DiscriminationByMVA2LooseElectronRejection"),
    ("againstElectronMediumMVA2", "DiscriminationByMVA2MediumElectronRejection"),
    ("againstElectronTightMVA2", "DiscriminationByMVA2TightElectronRejection"),
#    ("againstElectronMVA3raw", "DiscriminationByMVA3rawElectronRejection"),
#    ("againstElectronMVA3category", "DiscriminationByMVA3rawElectronRejection:category"),
#    ("againstElectronLooseMVA3", "DiscriminationByMVA3LooseElectronRejection"),
#    ("againstElectronMediumMVA3", "DiscriminationByMVA3MediumElectronRejection"),
#    ("againstElectronTightMVA3", "DiscriminationByMVA3TightElectronRejection"),
#    ("againstElectronDeadECAL", "DiscriminationByDeadECALElectronRejection"),
    ("againstMuonLoose", "DiscriminationByLooseMuonRejection"),
    ("againstMuonMedium", "DiscriminationByMediumMuonRejection"),
    ("againstMuonTight", "DiscriminationByTightMuonRejection") ]

# Discriminators of new HPS + TaNC combined Tau id. algorithm
hpsTancTauIDSources = [
    ("leadingTrackFinding", "DiscriminationByLeadingTrackFinding"),
    ("leadingTrackPtCut", "DiscriminationByLeadingTrackPtCut"),
    ("leadingPionPtCut", "DiscriminationByLeadingPionPtCut"),
    ("byTaNCraw", "DiscriminationByTancRaw"),
    ("byTaNC", "DiscriminationByTanc"),
    ("byTaNCvloose", "DiscriminationByTancVLoose"),
    ("byTaNCloose", "DiscriminationByTancLoose"),
    ("byTaNCmedium", "DiscriminationByTancMedium"),
    ("byTaNCtight", "DiscriminationByTancTight"),
    ("decayModeFinding", "DiscriminationByDecayModeSelection"),
    ("byHPSvloose", "DiscriminationByVLooseIsolation"),
    ("byHPSloose", "DiscriminationByLooseIsolation"),
    ("byHPSmedium", "DiscriminationByMediumIsolation"),
    ("byHPStight", "DiscriminationByTightIsolation"),
    ("againstElectronLoose", "DiscriminationByLooseElectronRejection"),
    ("againstElectronMedium", "DiscriminationByMediumElectronRejection"),
    ("againstElectronTight", "DiscriminationByTightElectronRejection"),
    ("againstMuonLoose", "DiscriminationByLooseMuonRejection"),
    ("againstMuonTight", "DiscriminationByTightMuonRejection") ]

# use tau-jet energy corrections determined for HPS taus for all PFTaus
from RecoTauTag.TauTagTools.tauDecayModes_cfi import *
pfTauJECpayloadMapping = cms.VPSet(
    cms.PSet(
        payload    = cms.string('AK5tauHPSlooseCombDBcorrOneProng0Pi0'),
        decayModes = cms.vstring('%i' % tauToOneProng0PiZero)
    ),
    cms.PSet(
        payload    = cms.string('AK5tauHPSlooseCombDBcorrOneProng1Pi0'),
        decayModes = cms.vstring('%i' % tauToOneProng1PiZero)
    ),
    cms.PSet(
        payload    = cms.string('AK5tauHPSlooseCombDBcorrOneProng2Pi0'),
        decayModes = cms.vstring('%i' % tauToOneProng2PiZero)
    ),
    cms.PSet(
        payload    = cms.string('AK5tauHPSlooseCombDBcorrThreeProng0Pi0'),
        decayModes = cms.vstring('%i' % tauToThreeProng0PiZero)
    ),
    cms.PSet(
        payload    = cms.string('AK5tauHPSlooseCombDBcorr'),
        decayModes = cms.vstring('*')
    )
)

# switch to PFTau collection produced for fixed dR = 0.07 signal cone size
def switchToPFTauFixedCone(process,
                           pfTauLabelOld = cms.InputTag('hpsPFTauProducer'),
                           pfTauLabelNew = cms.InputTag('fixedConePFTauProducer'),
                           patTauLabel = "",
                           jecLevels = [],
                           postfix = ""):
    fixedConeIDSources = copy.copy(classicTauIDSources)
    fixedConeIDSources.extend(classicPFTauIDSources)

    fixedConeJECpayloadMapping = pfTauJECpayloadMapping

    _switchToPFTau(process, pfTauLabelOld, pfTauLabelNew, 'fixedConePFTau', fixedConeIDSources,
                   jecLevels, fixedConeJECpayloadMapping,
                   patTauLabel = patTauLabel, postfix = postfix)

# switch to PFTau collection produced for shrinking signal cone of size dR = 5.0/Et(PFTau)
def switchToPFTauShrinkingCone(process,
                               pfTauLabelOld = cms.InputTag('hpsPFTauProducer'),
                               pfTauLabelNew = cms.InputTag('shrinkingConePFTauProducer'),
                               patTauLabel = "",
                               jecLevels = [],
                               postfix = ""):
    shrinkingIDSources = copy.copy(classicTauIDSources)
    shrinkingIDSources.extend(classicPFTauIDSources)
    # Only shrinkingCone has associated TaNC discriminators, so add them here
    shrinkingIDSources.extend(tancTauIDSources)

    shrinkingConeJECpayloadMapping = pfTauJECpayloadMapping

    _switchToPFTau(process, pfTauLabelOld, pfTauLabelNew, 'shrinkingConePFTau', shrinkingIDSources,
                   jecLevels, shrinkingConeJECpayloadMapping,
                   patTauLabel = patTauLabel, postfix = postfix)

# switch to hadron-plus-strip(s) (HPS) PFTau collection
def switchToPFTauHPS(process,
                     pfTauLabelOld = cms.InputTag('hpsPFTauProducer'),
                     pfTauLabelNew = cms.InputTag('hpsPFTauProducer'),
                     patTauLabel = "",
                     jecLevels = [],
                     postfix = ""):

    hpsTauJECpayloadMapping = pfTauJECpayloadMapping

    _switchToPFTau(process, pfTauLabelOld, pfTauLabelNew, 'hpsPFTau', hpsTauIDSources,
                   jecLevels, hpsTauJECpayloadMapping,
                   patTauLabel = patTauLabel, postfix = postfix)

    ## adapt cleanPatTaus
    if hasattr(process, "cleanPatTaus" + patTauLabel + postfix):
        getattr(process, "cleanPatTaus" + patTauLabel + postfix).preselection = \
      'pt > 15 & abs(eta) < 2.3 & tauID("decayModeFinding") > 0.5 & tauID("byLooseIsolation") > 0.5' \
     + ' & tauID("againstMuonTight") > 0.5 & tauID("againstElectronLoose") > 0.5'

# switch to hadron-plus-strip(s) (HPS) PFTau collection
def switchToPFTauHPSpTaNC(process,
                          pfTauLabelOld = cms.InputTag('hpsPFTauProducer'),
                          pfTauLabelNew = cms.InputTag('hpsTancTaus'),
                          patTauLabel = "",
                          jecLevels = [],
                          postfix = ""):

    hpsTancTauJECpayloadMapping = pfTauJECpayloadMapping

    _switchToPFTau(process, pfTauLabelOld, pfTauLabelNew, 'hpsTancTaus', hpsTancTauIDSources,
                   jecLevels, hpsTancTauJECpayloadMapping,
                   patTauLabel = patTauLabel, postfix = postfix)

    ## adapt cleanPatTaus
    if hasattr(process, "cleanPatTaus" + patTauLabel + postfix):
        getattr(process, "cleanPatTaus" + patTauLabel + postfix).preselection = \
      'pt > 15 & abs(eta) < 2.3 & tauID("decayModeFinding") > 0.5 & tauID("byHPSloose") > 0.5' \
     + ' & tauID("againstMuonTight") > 0.5 & tauID("againstElectronLoose") > 0.5'

# Select switcher by string
def switchToPFTauByType(process,
                        pfTauType = None,
                        pfTauLabelNew = None,
                        pfTauLabelOld = cms.InputTag('hpsPFTauProducer'),
                        patTauLabel = "",
                        jecLevels = [],
                        postfix = "" ):
    mapping = { 'shrinkingConePFTau' : switchToPFTauShrinkingCone,
                'fixedConePFTau' : switchToPFTauFixedCone,
                'hpsPFTau' : switchToPFTauHPS,
                'caloRecoTau' : switchToCaloTau,
                'hpsTancPFTau' : switchToPFTauHPSpTaNC }
    mapping[pfTauType](process, pfTauLabelOld = pfTauLabelOld, pfTauLabelNew = pfTauLabelNew,
                       jecLevels = jecLevels,
                       patTauLabel = patTauLabel, postfix = postfix)

# switch to PFTau collection that was default in PAT production in CMSSW_3_1_x release series
def switchTo31Xdefaults(process):
    switchToPFTauFixedCone(process)
    process.cleanPatTaus.preselection = cms.string('tauID("byIsolation") > 0')

class AddTauCollection(ConfigToolBase):

    """ Add a new collection of taus. Takes the configuration from the
    already configured standard tau collection as starting point;
    replaces before calling addTauCollection will also affect the
    new tau collections
    """
    _label='addTauCollection'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters, 'tauCollection',
                          self._defaultValue, 'Input tau collection', cms.InputTag)
        self.addParameter(self._defaultParameters, 'algoLabel',
                          self._defaultValue, "label to indicate the tau algorithm (e.g.'hps')", str)
        self.addParameter(self._defaultParameters, 'typeLabel',
                          self._defaultValue, "label to indicate the type of constituents (either 'PFTau' or 'Tau')", str)
        self.addParameter(self._defaultParameters, 'doPFIsoDeposits',
                          True, "run sequence for computing particle-flow based IsoDeposits")
        ##self.addParameter(self._defaultParameters, 'jetCorrLabel',
        ##                  (pfTauJECpayloadMapping, ['L2Relative', 'L3Absolute']),
        ##                  "payload and list of new jet correction labels", tuple, acceptNoneValue = True)
        self.addParameter(self._defaultParameters, 'jetCorrLabel',
                          None, "payload and list of new jet correction labels", tuple, acceptNoneValue = True)
        self.addParameter(self._defaultParameters, 'standardAlgo',
                          "hps", "standard algorithm label of the collection from which the clones " \
                         + "for the new tau collection will be taken from " \
                         + "(note that this tau collection has to be available in the event before hand)")
        self.addParameter(self._defaultParameters, 'standardType',
                          "PFTau", "standard constituent type label of the collection from which the clones " \
                         + " for the new tau collection will be taken from "\
                         + "(note that this tau collection has to be available in the event before hand)")

        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 tauCollection      = None,
                 algoLabel          = None,
                 typeLabel          = None,
                 doPFIsoDeposits    = None,
                 jetCorrLabel       = None,
                 standardAlgo       = None,
                 standardType       = None):

        if tauCollection is None:
            tauCollection = self._defaultParameters['tauCollection'].value
        if algoLabel is None:
            algoLabel = self._defaultParameters['algoLabel'].value
        if typeLabel is None:
            typeLabel = self._defaultParameters['typeLabel'].value
        if doPFIsoDeposits is None:
            doPFIsoDeposits = self._defaultParameters['doPFIsoDeposits'].value
        if jetCorrLabel is None:
            jetCorrLabel = self._defaultParameters['jetCorrLabel'].value
        if standardAlgo is None:
            standardAlgo = self._defaultParameters['standardAlgo'].value
        if standardType is None:
            standardType = self._defaultParameters['standardType'].value

        self.setParameter('tauCollection', tauCollection)
        self.setParameter('algoLabel', algoLabel)
        self.setParameter('typeLabel', typeLabel)
        self.setParameter('doPFIsoDeposits', doPFIsoDeposits)
        self.setParameter('jetCorrLabel', jetCorrLabel)
        self.setParameter('standardAlgo', standardAlgo)
        self.setParameter('standardType', standardType)

        self.apply(process)

    def toolCode(self, process):
        tauCollection = self._parameters['tauCollection'].value
        algoLabel = self._parameters['algoLabel'].value
        typeLabel = self._parameters['typeLabel'].value
        doPFIsoDeposits = self._parameters['doPFIsoDeposits'].value
        jetCorrLabel = self._parameters['jetCorrLabel'].value
        standardAlgo = self._parameters['standardAlgo'].value
        standardType = self._parameters['standardType'].value

        ## disable computation of particle-flow based IsoDeposits
        ## in case tau is of CaloTau type
        if typeLabel == 'Tau':
#            print "NO PF Isolation will be computed for CaloTau (this could be improved later)"
            doPFIsoDeposits = False

        ## create old module label from standardAlgo
        ## and standardType and return
        def oldLabel(prefix = ''):
            if prefix == '':
                return "patTaus"
            else:
                return prefix + "PatTaus"

        ## capitalize first character of appended part
        ## when creating new module label
        ## (giving e.g. "patTausCaloRecoTau")
        def capitalize(label):
            return label[0].capitalize() + label[1:]

        ## create new module label from old module
        ## label and return
        def newLabel(oldLabel):
            newLabel = oldLabel
            if ( oldLabel.find(standardAlgo) >= 0 and oldLabel.find(standardType) >= 0 ):
                oldLabel = oldLabel.replace(standardAlgo, algoLabel).replace(standardType, typeLabel)
            else:
                oldLabel = oldLabel + capitalize(algoLabel + typeLabel)
            return oldLabel

        ## clone module and add it to the patDefaultSequence
        def addClone(hook, **replaceStatements):
            ## create a clone of the hook with corresponding
            ## parameter replacements
            newModule = getattr(process, hook).clone(**replaceStatements)
            ## add the module to the sequence
            addModuleToSequence(hook, newModule)

        ## clone module for computing particle-flow IsoDeposits
        def addPFIsoDepositClone(hook, **replaceStatements):
            newModule = getattr(process, hook).clone(**replaceStatements)
            newModuleIsoDepositExtractor = getattr(newModule, "ExtractorPSet")
            setattr(newModuleIsoDepositExtractor, "tauSource", getattr(newModule, "src"))
            addModuleToSequence(hook, newModule)

        ## add module to the patDefaultSequence
        def addModuleToSequence(hook, newModule):
            hookModule = getattr(process, hook)
            ## add the new module with standardAlgo &
            ## standardType replaced in module label
            setattr(process, newLabel(hook), newModule)
            ## add new module to default sequence
            ## just behind the hookModule
            process.patDefaultSequence.replace( hookModule, hookModule*newModule )

        ## add a clone of patTaus
        addClone(oldLabel(), tauSource = tauCollection)

        ## add a clone of selectedPatTaus
        addClone(oldLabel('selected'), src = cms.InputTag(newLabel(oldLabel())))

        ## add a clone of cleanPatTaus
        addClone(oldLabel('clean'), src=cms.InputTag(newLabel(oldLabel('selected'))))

        ## get attributes of new module
        newTaus = getattr(process, newLabel(oldLabel()))

        ## add a clone of gen tau matching
        addClone('tauMatch', src = tauCollection)
        addClone('tauGenJetMatch', src = tauCollection)

        ## add a clone of IsoDeposits computed based on particle-flow
        if doPFIsoDeposits:
            addPFIsoDepositClone('tauIsoDepositPFCandidates', src = tauCollection)
            addPFIsoDepositClone('tauIsoDepositPFChargedHadrons', src = tauCollection)
            addPFIsoDepositClone('tauIsoDepositPFNeutralHadrons', src = tauCollection)
            addPFIsoDepositClone('tauIsoDepositPFGammas', src = tauCollection)

        if jetCorrLabel:
            addClone('patTauJetCorrFactors', src = tauCollection)
            getattr(process,newLabel('patTauJetCorrFactors')).payload = jetCorrLabel[0]
            getattr(process,newLabel('patTauJetCorrFactors')).levels = jetCorrLabel[1]
            getattr(process, newLabel('patTaus')).tauJetCorrFactorsSource = cms.VInputTag(cms.InputTag(newLabel('patTauJetCorrFactors')))

        ## fix label for input tag
        def fixInputTag(x): x.setModuleLabel(newLabel(x.moduleLabel))

        ## provide patTau inputs with individual labels
        fixInputTag(newTaus.genParticleMatch)
        fixInputTag(newTaus.genJetMatch)
        fixInputTag(newTaus.isoDeposits.pfAllParticles)
        fixInputTag(newTaus.isoDeposits.pfNeutralHadron)
        fixInputTag(newTaus.isoDeposits.pfChargedHadron)
        fixInputTag(newTaus.isoDeposits.pfGamma)
        fixInputTag(newTaus.userIsolation.pfAllParticles.src)
        fixInputTag(newTaus.userIsolation.pfNeutralHadron.src)
        fixInputTag(newTaus.userIsolation.pfChargedHadron.src)
        fixInputTag(newTaus.userIsolation.pfGamma.src)

        ## set discriminators
        ## (using switchTauCollection functions)
        oldTaus = getattr(process, oldLabel())
#        if typeLabel == 'Tau':
#            switchToCaloTau(process,
#                            pfTauLabel = getattr(oldTaus, "tauSource"),
#                            caloTauLabel = getattr(newTaus, "tauSource"),
#                            patTauLabel = capitalize(algoLabel + typeLabel))
#        else:
        switchToPFTauByType(process, pfTauType = algoLabel + typeLabel,
                                pfTauLabelNew = getattr(newTaus, "tauSource"),
                                pfTauLabelOld = getattr(oldTaus, "tauSource"),
                                patTauLabel = capitalize(algoLabel + typeLabel))

addTauCollection=AddTauCollection()
