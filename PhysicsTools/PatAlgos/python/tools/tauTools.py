import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.coreTools import *

def redoPFTauDiscriminators(process,
                            oldPFTauLabel = cms.InputTag('pfRecoTauProducer'),
                            newPFTauLabel = cms.InputTag('pfRecoTauProducer'),
                            tauType='fixedConePFTau'):
    print 'Tau discriminators: ', oldPFTauLabel, '->', newPFTauLabel
    print 'Tau type: ', tauType
    tauSrc = 'PFTauProducer'
    tauDiscriminationSequence = process.patFixedConePFTauDiscrimination
    if tauType == 'fixedConeHighEffPFTau':
        tauDiscriminationSequence = process.patFixedConeHighEffPFTauDiscrimination
    elif tauType == 'shrinkingConePFTau':
        tauDiscriminationSequence = process.patShrinkingConePFTauDiscrimination
    elif tauType == 'caloTau':
        tauDiscriminationSequence = process.patCaloTauDiscrimination
        tauSrc = 'CaloTauProducer'
    process.patDefaultSequence.replace(process.allLayer1Objects,
                                       tauDiscriminationSequence +
                                       process.allLayer1Objects
                                       )
    massSearchReplaceParam(tauDiscriminationSequence, tauSrc, oldPFTauLabel, newPFTauLabel)

# switch to CaloTau collection
def switchToCaloTau(process,
                    pfTauLabel = cms.InputTag('fixedConePFTauProducer'),
                    caloTauLabel = cms.InputTag('caloRecoTauProducer')):
    process.tauMatch.src       = caloTauLabel
    process.tauGenJetMatch.src = caloTauLabel
    process.allLayer1Taus.tauSource = caloTauLabel
    process.allLayer1Taus.tauIDSources = cms.PSet(
        leadingTrackFinding = cms.InputTag("caloRecoTauDiscriminationByLeadingTrackFinding"),
        leadingTrackPtCut   = cms.InputTag("caloRecoTauDiscriminationByLeadingTrackPtCut"),
        byIsolation         = cms.InputTag("caloRecoTauDiscriminationByIsolation"),
        againstElectron     = cms.InputTag("caloRecoTauDiscriminationAgainstElectron"),  
    )
    process.allLayer1Taus.addDecayMode = False
    ## Isolation is somewhat an issue, so we start just by turning it off
    print "NO PF Isolation will be computed for CaloTau (this could be improved later)"
    process.allLayer1Taus.isolation   = cms.PSet()
    process.allLayer1Taus.isoDeposits = cms.PSet()


# internal auxiliary function to switch to **any** PFTau collection
def _switchToPFTau(process, pfTauLabelOld, pfTauLabelNew, pfTauType):

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
    process.allLayer1Taus.tauSource = pfTauLabelNew
    process.allLayer1Taus.tauIDSources = cms.PSet(
        leadingTrackFinding = cms.InputTag(pfTauType + "DiscriminationByLeadingTrackFinding"),
        leadingTrackPtCut = cms.InputTag(pfTauType + "DiscriminationByLeadingTrackPtCut"),
        trackIsolation = cms.InputTag(pfTauType + "DiscriminationByTrackIsolation"),
        ecalIsolation = cms.InputTag(pfTauType + "DiscriminationByECALIsolation"),
        byIsolation = cms.InputTag(pfTauType + "DiscriminationByIsolation"),
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
    process.allLayer1Taus.decayModeSrc = cms.InputTag(pfTauType + "DecayModeProducer")


# switch to PFTau collection produced for fixed dR = 0.07 signal cone size
def switchToPFTauFixedCone(process,
                           pfTauLabelOld = cms.InputTag('pfRecoTauProducer'),
                           pfTauLabelNew = cms.InputTag('fixedConePFTauProducer')):
    _switchToPFTau(process, pfTauLabelOld, pfTauLabelNew, 'fixedConePFTau')
    #
    # CV: PFTauDecayMode objects produced only for shrinking cone reco::PFTaus in
    #     RecoTauTag/Configuration global_PFTau_22X_V00-02-01 and CMSSW_3_1_x tags,
    #     so need to disable embedding of PFTauDecayMode information into pat::Tau for now...
    #
    process.allLayer1Taus.addDecayMode = cms.bool(False)

# switch to PFTau collection produced for fixed dR = 0.15 signal cone size
def switchToPFTauFixedConeHighEff(process,
                                  pfTauLabelOld = cms.InputTag('pfRecoTauProducer'),
                                  pfTauLabelNew = cms.InputTag('fixedConeHighEffPFTauProducer')):
    _switchToPFTau(process, pfTauLabelOld, pfTauLabelNew, 'fixedConeHighEffPFTau')
    #
    # CV: PFTauDecayMode objects produced only for shrinking cone reco::PFTaus in
    #     RecoTauTag/Configuration global_PFTau_22X_V00-02-01 and CMSSW_3_1_x tags,
    #     so need to disable embedding of PFTauDecayMode information into pat::Tau for now...
    #
    process.allLayer1Taus.addDecayMode = cms.bool(False)

# switch to PFTau collection produced for shrinking signal cone of size dR = 5.0/Et(PFTau)
def switchToPFTauShrinkingCone(process,
                               pfTauLabelOld = cms.InputTag('pfRecoTauProducer'),
                               pfTauLabelNew = cms.InputTag('shrinkingConePFTauProducer')):
    _switchToPFTau(process, pfTauLabelOld, pfTauLabelNew, 'shrinkingConePFTau')
    #
    # CV: TaNC only trained for shrinkingCone PFTaus up to now,
    #     so need to add TaNC based discriminators
    #     specifically for that case here...
    #
    process.allLayer1Taus.tauIDSources = cms.PSet(
        leadingTrackFinding = cms.InputTag("shrinkingConePFTauDiscriminationByLeadingTrackFinding"),
        leadingTrackPtCut = cms.InputTag("shrinkingConePFTauDiscriminationByLeadingTrackPtCut"),
        trackIsolation = cms.InputTag("shrinkingConePFTauDiscriminationByTrackIsolation"),
        ecalIsolation = cms.InputTag("shrinkingConePFTauDiscriminationByECALIsolation"),
        byIsolation = cms.InputTag("shrinkingConePFTauDiscriminationByIsolation"),
        againstElectron = cms.InputTag("shrinkingConePFTauDiscriminationAgainstElectron"),
        againstMuon = cms.InputTag("shrinkingConePFTauDiscriminationAgainstMuon"),
        byTaNC = cms.InputTag("shrinkingConePFTauDiscriminationByTaNC"),
        byTaNCfrOnePercent = cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrOnePercent"),
        byTaNCfrHalfPercent = cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrHalfPercent"),
        byTaNCfrQuarterPercent = cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent"),
        byTaNCfrTenthPercent = cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrTenthPercent")
    )
    
# function to switch to **any** PFTau collection
# It is just to make internal function accessible externally
def switchToAnyPFTau(process,
                     pfTauLabelOld = cms.InputTag('pfRecoTauProducer'),
                     pfTauLabelNew = cms.InputTag('fixedConePFTauProducer'),
                     pfTauType='fixedConePFTau'):
    _switchToPFTau(process, pfTauLabelOld, pfTauLabelNew, pfTauType)
