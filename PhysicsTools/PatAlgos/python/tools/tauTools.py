import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.coreTools import *

from RecoTauTag.RecoTau.TauDiscriminatorTools import *
def redoPFTauDiscriminators(process,
                            oldPFTauLabel = cms.InputTag('shrinkingConePFTauProducer'),
                            newPFTauLabel = cms.InputTag('shrinkingConePFTauProducer'),
                            tauType='shrinkingConePFTau'):
    print 'Tau discriminators: ', oldPFTauLabel, '->', newPFTauLabel
    print 'Tau type: ', tauType
    tauSrc = 'PFTauProducer'

    tauDiscriminationSequence = process.patShrinkingConePFTauDiscrimination
    if tauType == 'hpsPFTau':
        tauDiscriminationSequence = process.patHPSPFTauDiscrimination
    elif tauType == 'fixedConePFTau':
        tauDiscriminationSequence = process.patFixedConePFTauDiscrimination
    elif tauType == 'shrinkingConePFTau':
        tauDiscriminationSequence = process.patShrinkingConePFTauDiscrimination
    elif tauType == 'caloTau':
        tauDiscriminationSequence = process.patCaloTauDiscrimination
        tauSrc = 'CaloTauProducer'

    process.makePatTaus.replace(process.patTaus, tauDiscriminationSequence*process.patTaus)

    massSearchReplaceParam(tauDiscriminationSequence, tauSrc, oldPFTauLabel, newPFTauLabel)

# switch to CaloTau collection
def switchToCaloTau(process,
                    pfTauLabel = cms.InputTag('shrinkingConePFTauProducer'),
                    caloTauLabel = cms.InputTag('caloRecoTauProducer')):
    process.tauMatch.src       = caloTauLabel
    process.tauGenJetMatch.src = caloTauLabel
    process.patTaus.tauSource = caloTauLabel
    process.patTaus.tauIDSources = cms.PSet(
        leadingTrackFinding = cms.InputTag("caloRecoTauDiscriminationByLeadingTrackFinding"),
        leadingTrackPtCut   = cms.InputTag("caloRecoTauDiscriminationByLeadingTrackPtCut"),
        byIsolation         = cms.InputTag("caloRecoTauDiscriminationByIsolation"),
        againstElectron     = cms.InputTag("caloRecoTauDiscriminationAgainstElectron"),  
    )
    process.patTaus.addDecayMode = False
    ## Isolation is somewhat an issue, so we start just by turning it off
    print "NO PF Isolation will be computed for CaloTau (this could be improved later)"
    process.patTaus.isolation   = cms.PSet()
    process.patTaus.isoDeposits = cms.PSet()
    process.patTaus.userIsolation = cms.PSet()
    process.patDefaultSequence.remove(process.patPFCandidateIsoDepositSelection)
    process.patDefaultSequence.remove(process.patPFTauIsolation)
    ## adapt cleanPatTaus
    process.cleanPatTaus.preselection = 'tauID("leadingTrackFinding") > 0.5 & tauID("leadingTrackPtCut") > 0.5 & tauID("byIsolation") > 0.5 & tauID("againstElectron") > 0.5 & (signalTracks.size() = 1 | signalTracks.size() = 3)'

def _buildIDSourcePSet(pfTauType, idSources):
    """ Build a PSet defining the tau ID sources to embed into the pat::Tau """
    output = cms.PSet()
    for label, discriminator in idSources:
        setattr(output, label, cms.InputTag(pfTauType+discriminator))
    return output

# internal auxiliary function to switch to **any** PFTau collection
def _switchToPFTau(process, pfTauLabelOld, pfTauLabelNew, pfTauType, idSources):

    print ' Taus: ', pfTauLabelOld, '->', pfTauLabelNew

    process.tauMatch.src       = pfTauLabelNew
    process.tauGenJetMatch.src = pfTauLabelNew
    process.tauIsoDepositPFCandidates.src = pfTauLabelNew
    process.tauIsoDepositPFCandidates.ExtractorPSet.tauSource = pfTauLabelNew
    process.tauIsoDepositPFChargedHadrons.src = pfTauLabelNew
    process.tauIsoDepositPFChargedHadrons.ExtractorPSet.tauSource = pfTauLabelNew
    process.tauIsoDepositPFNeutralHadrons.src = pfTauLabelNew
    process.tauIsoDepositPFNeutralHadrons.ExtractorPSet.tauSource = pfTauLabelNew
    process.tauIsoDepositPFGammas.src = pfTauLabelNew
    process.tauIsoDepositPFGammas.ExtractorPSet.tauSource = pfTauLabelNew
    process.patTaus.tauSource = pfTauLabelNew
    process.patTaus.tauIDSources = _buildIDSourcePSet(pfTauType, idSources)
    process.patTaus.decayModeSrc = cms.InputTag(pfTauType + "DecayModeProducer")

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
tancIDSources = [
    ("byTaNC", "shrinkingConePFTauDiscriminationByTaNC"),
    ("byTaNCfrOnePercent", "shrinkingConePFTauDiscriminationByTaNCfrOnePercent"),
    ("byTaNCfrHalfPercent", "shrinkingConePFTauDiscriminationByTaNCfrHalfPercent"),
    ("byTaNCfrQuarterPercent", "shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent"),
    ("byTaNCfrTenthPercent", "shrinkingConePFTauDiscriminationByTaNCfrTenthPercent") ]
# Hadron-plus-strip(s) (HPS) Tau Discriminators
hpsIDSources = [
    ("leadingTrackFinding", "DiscriminationByDecayModeFinding"),
    ("byLooseIsolation", "DiscriminationByLooseIsolation"),
    ("byMediumIsolation", "DiscriminationByMediumIsolation"),
    ("byTightIsolation", "DiscriminationByTightIsolation"),
    ("againstElectron", "DiscriminationAgainstElectron"),
    ("againstMuon", "DiscriminationAgainstMuon")]

# switch to PFTau collection produced for fixed dR = 0.07 signal cone size
def switchToPFTauFixedCone(process,
                           pfTauLabelOld = cms.InputTag('shrinkingConePFTauProducer'),
                           pfTauLabelNew = cms.InputTag('fixedConePFTauProducer')):
    _switchToPFTau(process, pfTauLabelOld, pfTauLabelNew, 'fixedConePFTau', classicTauIDSources)
    # PFTauDecayMode objects produced only for shrinking cone reco::PFTaus
    process.patTaus.addDecayMode = cms.bool(False)

# switch to hadron-plus-strip(s) (HPS) PFTau collection
def switchToPFTauHPS(process, 
                     pfTauLabelOld = cms.InputTag('shrinkingConePFTauProducer'),
                     pfTauLabelNew = cms.InputTag('hpsPFTauProducer')):
    _switchToPFTau(process, pfTauLabelOld, pfTauLabelNew, 'hpsPFTau', hpsIDSources)
    # PFTauDecayMode objects produced only for shrinking cone reco::PFTaus
    process.patTaus.addDecayMode = cms.bool(False)
    ## adapt cleanPatTaus
    process.cleanPatTaus.preselection = 'tauID("leadingTrackFinding") > 0.5 & tauID("byMediumIsolation") > 0.5 & tauID("againstMuon") > 0.5 & tauID("againstElectron") > 0.5'

# switch to PFTau collection produced for shrinking signal cone of size dR = 5.0/Et(PFTau)
def switchToPFTauShrinkingCone(process,
                               pfTauLabelOld = cms.InputTag('shrinkingConePFTauProducer'),
                               pfTauLabelNew = cms.InputTag('shrinkingConePFTauProducer')):

    shrinkingIDSources = copy.copy(classicTauIDSources)
    # Only shrinkingCone has associated TaNC discriminators, so add them here
    shrinkingIDSources.extend(tauIDSources)
    _switchToPFTau(process, pfTauLabelOld, pfTauLabelNew, 'shrinkingConePFTau', shrinkingIDSources)

# Select switcher by string
def switchToPFTauByType(process, pfTauType=None, pfTauLabelNew=None,
                        pfTauLabelOld=cms.InputTag('shrinkingConePFTauProducer') ):
    mapping = { 'shrinkingConePFTau' : switchToPFTauShrinkingCone,
                'fixedConePFTau' : switchToPFTauFixedCone,
                'hpsPFTau' : switchToPFTauHPS,
                'caloTau' : switchToCaloTau }
    mapping[pfTauType](process, pfTauLabelOld=pfTauLabelOld, pfTauLabelNew=pfTauLabelNew)

# switch to PFTau collection that was default in PAT production in CMSSW_3_1_x release series
def switchTo31Xdefaults(process):
    switchToPFTauFixedCone(process)
    process.cleanPatTaus.preselection = cms.string('tauID("byIsolation") > 0')

