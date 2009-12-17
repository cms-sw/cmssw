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
    if tauType == 'fixedConeHighEffPFTau':
        tauDiscriminationSequence = process.patFixedConeHighEffPFTauDiscrimination
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

# internal auxiliary function to switch to **any** PFTau collection
def _switchToPFTau(process,module, pfTauLabelOld, pfTauLabelNew, pfTauType):

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

    module.tauIDSources = cms.PSet(
        leadingTrackFinding = cms.InputTag(pfTauType + "DiscriminationByLeadingTrackFinding"),
        leadingTrackPtCut = cms.InputTag(pfTauType + "DiscriminationByLeadingTrackPtCut"),
        leadingPionPtCut = cms.InputTag(pfTauType + "DiscriminationByLeadingPionPtCut"),
        trackIsolation = cms.InputTag(pfTauType + "DiscriminationByTrackIsolation"),
        trackIsolationUsingLeadingPion = cms.InputTag(pfTauType + "DiscriminationByTrackIsolationUsingLeadingPion"),
        ecalIsolation = cms.InputTag(pfTauType + "DiscriminationByECALIsolation"),
        ecalIsolationUsingLeadingPion = cms.InputTag(pfTauType + "DiscriminationByECALIsolationUsingLeadingPion"),
        byIsolation = cms.InputTag(pfTauType + "DiscriminationByIsolation"),
        byIsolationUsingLeadingPion = cms.InputTag(pfTauType + "DiscriminationByIsolationUsingLeadingPion"),
        againstElectron = cms.InputTag(pfTauType + "DiscriminationAgainstElectron"),
        againstMuon = cms.InputTag(pfTauType + "DiscriminationAgainstMuon")
        #
        # CV: TaNC only trained for shrinkingCone PFTaus up to now,
        #     so cannot implement switch of TaNC based discriminators
        #     generically for all kinds of PFTaus yet...
        #
        #byTaNC = cms.InputTag(pfTauType + "DiscriminationByTaNC"),
        #byTaNCfrOnePercent = cms.InputTag(pfTauType + "DiscriminationByTaNCfrOnePercent"),
        #byTaNCfrHalfPercent = cms.InputTag(pfTauType + "DiscriminationByTaNCfrHalfPercent"),
        #byTaNCfrQuarterPercent = cms.InputTag(pfTauType + "DiscriminationByTaNCfrQuarterPercent"),
        #byTaNCfrTenthPercent = cms.InputTag(pfTauType + "DiscriminationByTaNCfrTenthPercent")
    )
    module.decayModeSrc = cms.InputTag(pfTauType + "DecayModeProducer")


# switch to PFTau collection produced for fixed dR = 0.07 signal cone size
def switchToPFTauFixedCone(process, module,
                           pfTauLabelOld = cms.InputTag('shrinkingConePFTauProducer'),
                           pfTauLabelNew = cms.InputTag('fixedConePFTauProducer')):
    _switchToPFTau(process,module, pfTauLabelOld, pfTauLabelNew, 'fixedConePFTau')
    #
    # CV: PFTauDecayMode objects produced only for shrinking cone reco::PFTaus in
    #     RecoTauTag/Configuration global_PFTau_22X_V00-02-01 and CMSSW_3_1_x tags,
    #     so need to disable embedding of PFTauDecayMode information into pat::Tau for now...
    #
    module.addDecayMode = cms.bool(False)

# switch to PFTau collection produced for fixed dR = 0.15 signal cone size
def switchToPFTauFixedConeHighEff(process, module,
                                  pfTauLabelOld = cms.InputTag('shrinkingConePFTauProducer'),
                                  pfTauLabelNew = cms.InputTag('fixedConeHighEffPFTauProducer')):
    _switchToPFTau(process, module,pfTauLabelOld, pfTauLabelNew, 'fixedConeHighEffPFTau')
    #
    # CV: PFTauDecayMode objects produced only for shrinking cone reco::PFTaus in
    #     RecoTauTag/Configuration global_PFTau_22X_V00-02-01 and CMSSW_3_1_x tags,
    #     so need to disable embedding of PFTauDecayMode information into pat::Tau for now...
    #
    module.addDecayMode = cms.bool(False)

# switch to PFTau collection produced for shrinking signal cone of size dR = 5.0/Et(PFTau)
def switchToPFTauShrinkingCone(process,module,
                               pfTauLabelOld = cms.InputTag('shrinkingConePFTauProducer'),
                               pfTauLabelNew = cms.InputTag('shrinkingConePFTauProducer')):
    _switchToPFTau(process,module, pfTauLabelOld, pfTauLabelNew, 'shrinkingConePFTau')
    #
    # CV: TaNC only trained for shrinkingCone PFTaus up to now,
    #     so need to add TaNC based discriminators
    #     specifically for that case here...
    #
    module.tauIDSources = cms.PSet(
        leadingTrackFinding = cms.InputTag("shrinkingConePFTauDiscriminationByLeadingTrackFinding"),
        leadingTrackPtCut = cms.InputTag("shrinkingConePFTauDiscriminationByLeadingTrackPtCut"),
        leadingPionPtCut = cms.InputTag("shrinkingConePFTauDiscriminationByLeadingPionPtCut"),
        trackIsolation = cms.InputTag("shrinkingConePFTauDiscriminationByTrackIsolation"),
        trackIsolationUsingLeadingPion = cms.InputTag("shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion"),
        ecalIsolation = cms.InputTag("shrinkingConePFTauDiscriminationByECALIsolation"),
        ecalIsolationUsingLeadingPion = cms.InputTag("shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion"),
        byIsolation = cms.InputTag("shrinkingConePFTauDiscriminationByIsolation"),
        byIsolationUsingLeadingPion = cms.InputTag("shrinkingConePFTauDiscriminationByIsolationUsingLeadingPion"),
        againstElectron = cms.InputTag("shrinkingConePFTauDiscriminationAgainstElectron"),
        againstMuon = cms.InputTag("shrinkingConePFTauDiscriminationAgainstMuon"),
        byTaNC = cms.InputTag("shrinkingConePFTauDiscriminationByTaNC"),
        byTaNCfrOnePercent = cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrOnePercent"),
        byTaNCfrHalfPercent = cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrHalfPercent"),
        byTaNCfrQuarterPercent = cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent"),
        byTaNCfrTenthPercent = cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrTenthPercent")
    )

# Select switcher by string
def switchToPFTauByType(process,module, pfTauType=None, pfTauLabelNew=None,
                        pfTauLabelOld=cms.InputTag('shrinkingConePFTauProducer') ):
    mapping = { 'shrinkingConePFTau' : switchToPFTauShrinkingCone,
                'fixedConePFTau' : switchToPFTauFixedCone,
                'fixedConeHighEffPFTau' : switchToPFTauFixedConeHighEff,
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
