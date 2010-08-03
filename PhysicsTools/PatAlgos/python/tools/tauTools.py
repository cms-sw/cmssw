import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.coreTools import *
from PhysicsTools.PatAlgos.tools.helpers import applyPostfix 
from RecoTauTag.RecoTau.TauDiscriminatorTools import *

def redoPFTauDiscriminators(process,
                            oldPFTauLabel = cms.InputTag('shrinkingConePFTauProducer'),
                            newPFTauLabel = cms.InputTag('shrinkingConePFTauProducer'),
                            tauType='shrinkingConePFTau', postfix = ""):
    print 'Tau discriminators: ', oldPFTauLabel, '->', newPFTauLabel
    print 'Tau type: ', tauType
    tauSrc = 'PFTauProducer'

    tauDiscriminationSequence = None
    if tauType == 'hpsPFTau':
        tauDiscriminationSequence =  cloneProcessingSnippet(process, process.patHPSPFTauDiscrimination, postfix)
    elif tauType == 'fixedConePFTau':
        tauDiscriminationSequence = cloneProcessingSnippet(process, process.patFixedConePFTauDiscrimination, postfix)
    elif tauType == 'shrinkingConePFTau':
        tauDiscriminationSequence = cloneProcessingSnippet(process, process.patShrinkingConePFTauDiscrimination, postfix)
    elif tauType == 'caloTau':
        tauDiscriminationSequence = cloneProcessingSnippet(process, process.patCaloTauDiscrimination, postfix)
        tauSrc = 'CaloTauProducer'
    else:
        raise StandardError, "Unkown tauType: '%s'"%tauType

    #process.makePatTaus.replace(process.patTaus, tauDiscriminationSequence*process.patTaus)
    applyPostfix(process,"makePatTaus",postfix).replace(
        applyPostfix(process,"patTaus",postfix),
        tauDiscriminationSequence*applyPostfix(process,"patTaus",postfix)
    )

    massSearchReplaceParam(tauDiscriminationSequence, tauSrc, oldPFTauLabel, newPFTauLabel)

# switch to CaloTau collection
def switchToCaloTau(process,
                    pfTauLabel = cms.InputTag('shrinkingConePFTauProducer'),
                    caloTauLabel = cms.InputTag('caloRecoTauProducer'),
                    postfix=""):
    applyPostfix(process,"tauMatch",postfix).src       = caloTauLabel
    applyPostfix(process,"tauGenJetMatch",postfix).src = caloTauLabel
    applyPostfix(process,"patTaus",postfix).tauSource = caloTauLabel
    applyPostfix(process,"patTaus",postfix).tauIDSources = cms.PSet(
        leadingTrackFinding = cms.InputTag("caloRecoTauDiscriminationByLeadingTrackFinding" + postfix),
        leadingTrackPtCut   = cms.InputTag("caloRecoTauDiscriminationByLeadingTrackPtCut" + postfix),
        byIsolation         = cms.InputTag("caloRecoTauDiscriminationByIsolation" + postfix),
        againstElectron     = cms.InputTag("caloRecoTauDiscriminationAgainstElectron" + postfix),  
    )
    applyPostfix(process,"patTaus",postfix).addDecayMode = False
    ## Isolation is somewhat an issue, so we start just by turning it off
    print "NO PF Isolation will be computed for CaloTau (this could be improved later)"
    applyPostfix(process,"patTaus",postfix).isolation   = cms.PSet()
    applyPostfix(process,"patTaus",postfix).isoDeposits = cms.PSet()
    applyPostfix(process,"patTaus",postfix).userIsolation = cms.PSet()
    getattr(process,"patDefaultSequence"+postfix).remove(applyPostfix(process,"patPFCandidateIsoDepositSelection",postfix))
    getattr(process,"patDefaultSequence"+postfix).remove(applyPostfix(process,"patPFTauIsolation",postfix))
    ## adapt cleanPatTaus
    applyPostfix(process,"cleanPatTaus",postfix).preselection = 'tauID("leadingTrackFinding") > 0.5 & tauID("leadingTrackPtCut") > 0.5 & tauID("byIsolation") > 0.5 & tauID("againstElectron") > 0.5 & (signalTracks.size() = 1 | signalTracks.size() = 3)'

def _buildIDSourcePSet(pfTauType, idSources, postfix =""):
    """ Build a PSet defining the tau ID sources to embed into the pat::Tau """
    output = cms.PSet()
    for label, discriminator in idSources:
        setattr(output, label, cms.InputTag(pfTauType+discriminator+postfix))
    return output

def _switchToPFTau(process, pfTauLabelOld, pfTauLabelNew, pfTauType, idSources, postfix=""):
    """internal auxiliary function to switch to **any** PFTau collection"""  
    print ' Taus: ', pfTauLabelOld, '->', pfTauLabelNew
    
    applyPostfix(process,"tauMatch",postfix).src = pfTauLabelNew
    applyPostfix(process,"tauGenJetMatch",postfix).src = pfTauLabelNew
    applyPostfix(process,"tauIsoDepositPFCandidates",postfix).src = pfTauLabelNew
    applyPostfix(process,"tauIsoDepositPFCandidates",postfix).ExtractorPSet.tauSource = pfTauLabelNew
    applyPostfix(process,"tauIsoDepositPFChargedHadrons",postfix).src = pfTauLabelNew
    applyPostfix(process,"tauIsoDepositPFChargedHadrons",postfix).ExtractorPSet.tauSource = pfTauLabelNew
    applyPostfix(process,"tauIsoDepositPFNeutralHadrons",postfix).src = pfTauLabelNew
    applyPostfix(process,"tauIsoDepositPFNeutralHadrons",postfix).ExtractorPSet.tauSource = pfTauLabelNew
    applyPostfix(process,"tauIsoDepositPFGammas",postfix).src = pfTauLabelNew
    applyPostfix(process,"tauIsoDepositPFGammas",postfix).ExtractorPSet.tauSource = pfTauLabelNew
    applyPostfix(process,"patTaus",postfix).tauSource = pfTauLabelNew
    applyPostfix(process,"patTaus",postfix).tauIDSources = _buildIDSourcePSet(pfTauType, idSources, postfix)
    applyPostfix(process,"patTaus",postfix).decayModeSrc = cms.InputTag(pfTauType + "DecayModeProducer")

# Name mapping for classic tau ID sources (present for fixed and shrinkingCones)
classicTauIDSources = [
    ("leadingTrackFinding", "DiscriminationByLeadingTrackFinding"),
    ("leadingTrackPtCut", "DiscriminationByLeadingTrackPtCut"),
    ("leadingPionPtCut", "DiscriminationByLeadingPionPtCut"),
    ("trackIsolation", "DiscriminationByTrackIsolation"),
    ("trackIsolationUsingLeadingPion", "DiscriminationByTrackIsolationUsingLeadingPion"),
    ("ecalIsolation", "DiscriminationByECALIsolation"),
    ("ecalIsolationUsingLeadingPion", "DiscriminationByECALIsolationUsingLeadingPion"),
    ("byIsolation", "DiscriminationByIsolation"),
    ("byIsolationUsingLeadingPion", "DiscriminationByIsolationUsingLeadingPion"),
    ("againstElectron", "DiscriminationAgainstElectron"),
    ("againstMuon", "DiscriminationAgainstMuon") ]

# Tau Neural Classifier Discriminators
tancTauIDSources = [
    ("byTaNC", "DiscriminationByTaNC"),
    ("byTaNCfrOnePercent", "DiscriminationByTaNCfrOnePercent"),
    ("byTaNCfrHalfPercent", "DiscriminationByTaNCfrHalfPercent"),
    ("byTaNCfrQuarterPercent", "DiscriminationByTaNCfrQuarterPercent"),
    ("byTaNCfrTenthPercent", "DiscriminationByTaNCfrTenthPercent") ]
# Hadron-plus-strip(s) (HPS) Tau Discriminators
hpsTauIDSources = [
    ("leadingTrackFinding", "DiscriminationByDecayModeFinding"),
    ("byLooseIsolation", "DiscriminationByLooseIsolation"),
    ("byMediumIsolation", "DiscriminationByMediumIsolation"),
    ("byTightIsolation", "DiscriminationByTightIsolation"),
    ("againstElectron", "DiscriminationAgainstElectron"),
    ("againstMuon", "DiscriminationAgainstMuon")]

# switch to PFTau collection produced for fixed dR = 0.07 signal cone size
def switchToPFTauFixedCone(process,
                           pfTauLabelOld = cms.InputTag('shrinkingConePFTauProducer'),
                           pfTauLabelNew = cms.InputTag('fixedConePFTauProducer'),
                           postfix=""):
    _switchToPFTau(process, pfTauLabelOld, pfTauLabelNew, 'fixedConePFTau', 
                   classicTauIDSources, postfix=postfix)
    # PFTauDecayMode objects produced only for shrinking cone reco::PFTaus
    applyPostfix(process,"patTaus",postfix).addDecayMode = cms.bool(False)

# switch to hadron-plus-strip(s) (HPS) PFTau collection
def switchToPFTauHPS(process, 
                     pfTauLabelOld = cms.InputTag('shrinkingConePFTauProducer'),
                     pfTauLabelNew = cms.InputTag('hpsPFTauProducer'),
                     postfix=""):
    _switchToPFTau(process, pfTauLabelOld, pfTauLabelNew, 'hpsPFTau', hpsTauIDSources,postfix=postfix)
    # PFTauDecayMode objects produced only for shrinking cone reco::PFTaus
    applyPostfix(process,"patTaus",postfix).addDecayMode = cms.bool(False)
    ## adapt cleanPatTaus
    getattr(process, "cleanPatTaus"+postfix).preselection = 'tauID("leadingTrackFinding") > 0.5 & tauID("byMediumIsolation") > 0.5 & tauID("againstMuon") > 0.5 & tauID("againstElectron") > 0.5'

# switch to PFTau collection produced for shrinking signal cone of size dR = 5.0/Et(PFTau)
def switchToPFTauShrinkingCone(process,
                               pfTauLabelOld = cms.InputTag('shrinkingConePFTauProducer'),
                               pfTauLabelNew = cms.InputTag('shrinkingConePFTauProducer'),
                               postfix=""):
    shrinkingIDSources = copy.copy(classicTauIDSources)
    # Only shrinkingCone has associated TaNC discriminators, so add them here
    shrinkingIDSources.extend(tancTauIDSources)
    _switchToPFTau(process, pfTauLabelOld, pfTauLabelNew, 'shrinkingConePFTau', shrinkingIDSources ,postfix=postfix)

# Select switcher by string
def switchToPFTauByType(process, pfTauType=None, pfTauLabelNew=None,
                        pfTauLabelOld=cms.InputTag('shrinkingConePFTauProducer'), postfix="" ):
    mapping = { 'shrinkingConePFTau' : switchToPFTauShrinkingCone,
                'fixedConePFTau' : switchToPFTauFixedCone,
                'hpsPFTau' : switchToPFTauHPS,
                'caloTau' : switchToCaloTau }
    mapping[pfTauType](process, pfTauLabelOld=pfTauLabelOld, pfTauLabelNew=pfTauLabelNew, postfix=postfix)

# switch to PFTau collection that was default in PAT production in CMSSW_3_1_x release series
def switchTo31Xdefaults(process):
    switchToPFTauFixedCone(process)
    process.cleanPatTaus.preselection = cms.string('tauID("byIsolation") > 0')
    
