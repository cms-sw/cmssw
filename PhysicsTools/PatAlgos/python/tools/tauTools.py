import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.coreTools import *

from RecoTauTag.RecoTau.TauDiscriminatorTools import *
def redoPFTauDiscriminators(process,
                            oldPFTauLabel = cms.InputTag('shrinkingConePFTauProducer'),
                            newPFTauLabel = cms.InputTag('shrinkingConePFTauProducer'),
                            tauType='shrinkingConePFTau',
                            l0tauCollection=cms.InputTag('allLayer0Taus')):
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

    moduleL0 =  getattr(process,l0tauCollection.moduleLabel)
    if (l0tauCollection.moduleLabel=="allLayer0Taus"):
        process.patDefaultSequence.replace(process.patCandidates, tauDiscriminationSequence + process.patCandidates)
    if (l0tauCollection.moduleLabel=='pfLayer0Taus'):         
        process.PF2PAT.replace(moduleL0, moduleL0+tauDiscriminationSequence)

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
    ## adapt cleanPatTaus
    process.cleanPatTaus.preselection = 'tauID("leadingTrackFinding") > 0.5 & tauID("leadingTrackPtCut") > 0.5 & tauID("byIsolation") > 0.5 & tauID("againstElectron") > 0.5'

def _buildIDSourcePSet(pfTauType, idSources):
    """ Build a PSet defining the tau ID sources to embed into the pat::Tau """
    output = cms.PSet()
    for label, discriminator in idSources:
        setattr(output, label, cms.InputTag(pfTauType+discriminator))
    return output

# internal auxiliary function to switch to **any** PFTau collection
def _switchToPFTau(process, module, pfTauLabelOld, pfTauLabelNew, pfTauType, idSources):

    print ' Taus: ', pfTauLabelOld, '->', pfTauLabelNew
    mctaumatch      = getattr(process,module.genParticleMatch.moduleLabel)  
    mctaujetmatch   = getattr(process,module.genJetMatch.moduleLabel)
    tauisocand      = getattr(process,module.isoDeposits.pfAllParticles.moduleLabel)
    tauisopfch      = getattr(process,module.isoDeposits.pfChargedHadron.moduleLabel)
    tauisopfne      = getattr(process,module.isoDeposits.pfNeutralHadron.moduleLabel)
    tauisopfgam     = getattr(process,module.isoDeposits.pfGamma.moduleLabel)

    mctaumatch.src                      = pfTauLabelNew
    mctaujetmatch.src                   = pfTauLabelNew
    tauisocand.src                      = pfTauLabelNew
    tauisocand.ExtractorPSet.tauSource  = pfTauLabelNew
    tauisopfch.src                      = pfTauLabelNew
    tauisopfch.ExtractorPSet.tauSource  = pfTauLabelNew
    tauisopfne.src                      = pfTauLabelNew
    tauisopfne.ExtractorPSet.tauSource  = pfTauLabelNew
    tauisopfgam.src                     = pfTauLabelNew
    tauisopfgam.ExtractorPSet.tauSource = pfTauLabelNew
    module.tauSource = pfTauLabelNew
    # Build the tau ID source mapping
    module.tauIDSources = _buildIDSourcePSet(pfTauType, idSources)
    module.decayModeSrc = cms.InputTag(pfTauType + "DecayModeProducer")

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

# switch to PFTau collection produced for fixed dR = 0.07 signal cone size
def switchToPFTauFixedCone(process, module,
                           pfTauLabelOld = cms.InputTag('shrinkingConePFTauProducer'),
                           pfTauLabelNew = cms.InputTag('fixedConePFTauProducer')):
    _switchToPFTau(process, module, pfTauLabelOld, pfTauLabelNew, 'fixedConePFTau', classicTauIDSources)
    #
    # CV: PFTauDecayMode objects produced only for shrinking cone reco::PFTaus in
    #     RecoTauTag/Configuration global_PFTau_22X_V00-02-01 and CMSSW_3_1_x tags,
    #     so need to disable embedding of PFTauDecayMode information into pat::Tau for now...
    #
    module.addDecayMode = cms.bool(False)

# switch to PFTau collection produced for fixed dR = 0.15 signal cone size
def switchToPFTauHPS(process, module,
                     pfTauLabelOld = cms.InputTag('shrinkingConePFTauProducer'),
                     pfTauLabelNew = cms.InputTag('hpsPFTauProducer')):
    hpsIDSources = [
        ("leadingTrackFinding", "DiscriminationByDecayModeFinding"),
        ("looseIsolation", "DiscriminationByLooseIsolation"),
        ("mediumIsolation", "DiscriminationByMediumIsolation"),
        ("tightIsolation", "DiscriminationByTightIsolation"),
        ("againstElectron", "DiscriminationAgainstElectron"),
        ("againstMuon", "DiscriminationAgainstMuon")]
    _switchToPFTau(process, module, pfTauLabelOld, pfTauLabelNew, 'hpsPFTau', hpsIDSources)
    module.addDecayMode = cms.bool(False)

# switch to PFTau collection produced for shrinking signal cone of size dR = 5.0/Et(PFTau)
def switchToPFTauShrinkingCone(process,module,
                               pfTauLabelOld = cms.InputTag('shrinkingConePFTauProducer'),
                               pfTauLabelNew = cms.InputTag('shrinkingConePFTauProducer')):
    shrinkingIDSources = copy.copy(classicTauIDSources)
    # Only shrinkingCone has associated TaNC discriminators, so add them here
    shrinkingIDSources.extend(tauIDSources)
    _switchToPFTau(process, module, pfTauLabelOld, pfTauLabelNew, 'shrinkingConePFTau', shrinkingIDSources)

# Select switcher by string
def switchToPFTauByType(process, module, pfTauType=None, pfTauLabelNew=None,
                        pfTauLabelOld=cms.InputTag('shrinkingConePFTauProducer') ):
    mapping = { 'shrinkingConePFTau' : switchToPFTauShrinkingCone,
                'fixedConePFTau' : switchToPFTauFixedCone,
                'hpsPFTau' : switchToPFTauHPS,
                'caloTau' : switchToCaloTau }
    mapping[pfTauType](process, module,pfTauLabelOld=pfTauLabelOld, pfTauLabelNew=pfTauLabelNew)

# switch to PFTau collection that was default in PAT production in CMSSW_3_1_x release series
def switchTo31Xdefaults(process,module):
    switchToPFTauFixedCone(process,module)
    process.cleanPatTaus.preselection = cms.string('tauID("byIsolation") > 0')
    
# function to switch to **any** PFTau collection
# It is just to make internal function accessible externally
def switchToAnyPFTau(process,module,
                     pfTauLabelOld = cms.InputTag('shrinkingConePFTauProducer'),
                     pfTauLabelNew = cms.InputTag('shrinkingConePFTauProducer'),
                     pfTauType='shrinkingConePFTau'):
    _switchToPFTau(process,module, pfTauLabelOld, pfTauLabelNew, pfTauType)
