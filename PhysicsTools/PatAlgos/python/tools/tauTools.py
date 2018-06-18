import FWCore.ParameterSet.Config as cms

from FWCore.GuiBrowsers.ConfigToolBase import *
from PhysicsTools.PatAlgos.tools.helpers import cloneProcessingSnippet
from RecoTauTag.RecoTau.TauDiscriminatorTools import *
from PhysicsTools.PatAlgos.cleaningLayer1.tauCleaner_cfi import preselection

# applyPostFix function adapted to unscheduled mode
def applyPostfix(process, label, postfix):
    result = None
    if hasattr(process, label+postfix):
        result = getattr(process, label + postfix)
    else:
        raise ValueError("Error in <applyPostfix>: No module of name = %s attached to process !!" % (label + postfix))
    return result

# switch to CaloTau collection
def switchToCaloTau(process,
                    tauSource = cms.InputTag('caloRecoTauProducer'),
                    patTauLabel = "",
                    postfix = ""):
    print ' switching PAT Tau input to: ', tauSource

    applyPostfix(process, "tauMatch" + patTauLabel, postfix).src = tauSource
    applyPostfix(process, "tauGenJetMatch"+ patTauLabel, postfix).src = tauSource
    
    applyPostfix(process, "patTaus" + patTauLabel, postfix).tauSource = tauSource
    # CV: reconstruction of tau lifetime information not implemented for CaloTaus yet
    applyPostfix(process, "patTaus" + patTauLabel, postfix).tauTransverseImpactParameterSource = ""
    applyPostfix(process, "patTaus" + patTauLabel, postfix).tauIDSources = _buildIDSourcePSet('caloRecoTau', classicTauIDSources, postfix)

    ## Isolation is somewhat an issue, so we start just by turning it off
    print "NO PF Isolation will be computed for CaloTau (this could be improved later)"
    applyPostfix(process, "patTaus" + patTauLabel, postfix).isolation   = cms.PSet()
    applyPostfix(process, "patTaus" + patTauLabel, postfix).isoDeposits = cms.PSet()
    applyPostfix(process, "patTaus" + patTauLabel, postfix).userIsolation = cms.PSet()

    ## adapt cleanPatTaus
    if hasattr(process, "cleanPatTaus" + patTauLabel + postfix):
        getattr(process, "cleanPatTaus" + patTauLabel + postfix).preselection = preselection

def _buildIDSourcePSet(tauType, idSources, postfix =""):
    """ Build a PSet defining the tau ID sources to embed into the pat::Tau """
    output = cms.PSet()
    for label, discriminator in idSources:
        if ":" in discriminator:
          discr = discriminator.split(":")
          setattr(output, label, cms.InputTag(tauType + discr[0] + postfix + ":" + discr[1]))
        else:  
          setattr(output, label, cms.InputTag(tauType + discriminator + postfix))
    return output

def _switchToPFTau(process,
                   tauSource,
                   pfTauType,
                   idSources,
                   patTauLabel = "",
                   postfix = ""):
    """internal auxiliary function to switch to **any** PFTau collection"""
    print ' switching PAT Tau input to: ', tauSource

    applyPostfix(process, "tauMatch" + patTauLabel, postfix).src = tauSource
    applyPostfix(process, "tauGenJetMatch" + patTauLabel, postfix).src = tauSource
    
    applyPostfix(process, "tauIsoDepositPFCandidates" + patTauLabel, postfix).src = tauSource
    applyPostfix(process, "tauIsoDepositPFCandidates" + patTauLabel, postfix).ExtractorPSet.tauSource = tauSource
    applyPostfix(process, "tauIsoDepositPFChargedHadrons" + patTauLabel, postfix).src = tauSource
    applyPostfix(process, "tauIsoDepositPFChargedHadrons" + patTauLabel, postfix).ExtractorPSet.tauSource = tauSource
    applyPostfix(process, "tauIsoDepositPFNeutralHadrons" + patTauLabel, postfix).src = tauSource
    applyPostfix(process, "tauIsoDepositPFNeutralHadrons" + patTauLabel, postfix).ExtractorPSet.tauSource = tauSource
    applyPostfix(process, "tauIsoDepositPFGammas" + patTauLabel, postfix).src = tauSource
    applyPostfix(process, "tauIsoDepositPFGammas" + patTauLabel, postfix).ExtractorPSet.tauSource = tauSource
    
    applyPostfix(process, "patTaus" + patTauLabel, postfix).tauSource = tauSource
    # CV: reconstruction of tau lifetime information not enabled for tau collections other than 'hpsPFTauProducer' yet
    applyPostfix(process, "patTaus" + patTauLabel, postfix).tauTransverseImpactParameterSource = ""
    applyPostfix(process, "patTaus" + patTauLabel, postfix).tauIDSources = _buildIDSourcePSet(pfTauType, idSources, postfix)

    if hasattr(process, "cleanPatTaus" + patTauLabel + postfix):
        getattr(process, "cleanPatTaus" + patTauLabel + postfix).preselection = preselection

# Name mapping for classic tau ID sources (present for fixed and shrinkingCones)
classicTauIDSources = [
    ("leadingTrackFinding", "DiscriminationByLeadingTrackFinding"),
    ("leadingTrackPtCut", "DiscriminationByLeadingTrackPtCut"),
    ("trackIsolation", "DiscriminationByTrackIsolation"),
    ("ecalIsolation", "DiscriminationByECALIsolation"),
    ("byIsolation", "DiscriminationByIsolation"),
    ("againstElectron", "DiscriminationAgainstElectron"),
    ("againstMuon", "DiscriminationAgainstMuon")
]

classicPFTauIDSources = [
    ("leadingPionPtCut", "DiscriminationByLeadingPionPtCut"),
    ("trackIsolationUsingLeadingPion", "DiscriminationByTrackIsolationUsingLeadingPion"),
    ("ecalIsolationUsingLeadingPion", "DiscriminationByECALIsolationUsingLeadingPion"),
    ("byIsolationUsingLeadingPion", "DiscriminationByIsolationUsingLeadingPion")
]

# Hadron-plus-strip(s) (HPS) Tau Discriminators
hpsTauIDSources = [
    ("decayModeFindingNewDMs", "DiscriminationByDecayModeFindingNewDMs"),
    ("decayModeFinding", "DiscriminationByDecayModeFinding"), # CV: kept for backwards compatibility
    ("byLooseCombinedIsolationDeltaBetaCorr3Hits", "DiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits"),
    ("byMediumCombinedIsolationDeltaBetaCorr3Hits", "DiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits"),
    ("byTightCombinedIsolationDeltaBetaCorr3Hits", "DiscriminationByTightCombinedIsolationDBSumPtCorr3Hits"),
    ("byCombinedIsolationDeltaBetaCorrRaw3Hits", "DiscriminationByRawCombinedIsolationDBSumPtCorr3Hits"),
    ("byLooseCombinedIsolationDeltaBetaCorr3HitsdR03", "DiscriminationByLooseCombinedIsolationDBSumPtCorr3HitsdR03"),
    ("byMediumCombinedIsolationDeltaBetaCorr3HitsdR03", "DiscriminationByMediumCombinedIsolationDBSumPtCorr3HitsdR03"),
    ("byTightCombinedIsolationDeltaBetaCorr3HitsdR03", "DiscriminationByTightCombinedIsolationDBSumPtCorr3HitsdR03"),
    ("byPhotonPtSumOutsideSignalCone", "DiscriminationByPhotonPtSumOutsideSignalCone"),
    ("chargedIsoPtSum", "ChargedIsoPtSum"),
    ("neutralIsoPtSum", "NeutralIsoPtSum"),
    ("puCorrPtSum", "PUcorrPtSum"),
    ("neutralIsoPtSumWeight", "NeutralIsoPtSumWeight"),
    ("footprintCorrection", "FootprintCorrection"),
    ("photonPtSumOutsideSignalCone", "PhotonPtSumOutsideSignalCone"),
    ("againstMuonLoose3", "DiscriminationByLooseMuonRejection3"),
    ("againstMuonTight3", "DiscriminationByTightMuonRejection3"),
    ("byIsolationMVArun2v1DBoldDMwLTraw", "DiscriminationByIsolationMVArun2v1DBoldDMwLTraw"),
    ("byVLooseIsolationMVArun2v1DBoldDMwLT", "DiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT"),
    ("byLooseIsolationMVArun2v1DBoldDMwLT", "DiscriminationByLooseIsolationMVArun2v1DBoldDMwLT"),
    ("byMediumIsolationMVArun2v1DBoldDMwLT", "DiscriminationByMediumIsolationMVArun2v1DBoldDMwLT"),
    ("byTightIsolationMVArun2v1DBoldDMwLT", "DiscriminationByTightIsolationMVArun2v1DBoldDMwLT"),
    ("byVTightIsolationMVArun2v1DBoldDMwLT", "DiscriminationByVTightIsolationMVArun2v1DBoldDMwLT"),
    ("byVVTightIsolationMVArun2v1DBoldDMwLT", "DiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT"),
    ("byIsolationMVArun2v1DBnewDMwLTraw", "DiscriminationByIsolationMVArun2v1DBnewDMwLTraw"),
    ("byVLooseIsolationMVArun2v1DBnewDMwLT", "DiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT"),
    ("byLooseIsolationMVArun2v1DBnewDMwLT", "DiscriminationByLooseIsolationMVArun2v1DBnewDMwLT"),
    ("byMediumIsolationMVArun2v1DBnewDMwLT", "DiscriminationByMediumIsolationMVArun2v1DBnewDMwLT"),
    ("byTightIsolationMVArun2v1DBnewDMwLT", "DiscriminationByTightIsolationMVArun2v1DBnewDMwLT"),
    ("byVTightIsolationMVArun2v1DBnewDMwLT", "DiscriminationByVTightIsolationMVArun2v1DBnewDMwLT"),
    ("byVVTightIsolationMVArun2v1DBnewDMwLT", "DiscriminationByVVTightIsolationMVArun2v1DBnewDMwLT"),
    ("byIsolationMVArun2v1PWoldDMwLTraw", "DiscriminationByIsolationMVArun2v1PWoldDMwLTraw"),
    ("byVLooseIsolationMVArun2v1PWoldDMwLT", "DiscriminationByVLooseIsolationMVArun2v1PWoldDMwLT"),
    ("byLooseIsolationMVArun2v1PWoldDMwLT", "DiscriminationByLooseIsolationMVArun2v1PWoldDMwLT"),
    ("byMediumIsolationMVArun2v1PWoldDMwLT", "DiscriminationByMediumIsolationMVArun2v1PWoldDMwLT"),
    ("byTightIsolationMVArun2v1PWoldDMwLT", "DiscriminationByTightIsolationMVArun2v1PWoldDMwLT"),
    ("byVTightIsolationMVArun2v1PWoldDMwLT", "DiscriminationByVTightIsolationMVArun2v1PWoldDMwLT"),
    ("byVVTightIsolationMVArun2v1PWoldDMwLT", "DiscriminationByVVTightIsolationMVArun2v1PWoldDMwLT"),
    ("byIsolationMVArun2v1PWnewDMwLTraw", "DiscriminationByIsolationMVArun2v1PWnewDMwLTraw"),
    ("byVLooseIsolationMVArun2v1PWnewDMwLT", "DiscriminationByVLooseIsolationMVArun2v1PWnewDMwLT"),
    ("byLooseIsolationMVArun2v1PWnewDMwLT", "DiscriminationByLooseIsolationMVArun2v1PWnewDMwLT"),
    ("byMediumIsolationMVArun2v1PWnewDMwLT", "DiscriminationByMediumIsolationMVArun2v1PWnewDMwLT"),
    ("byTightIsolationMVArun2v1PWnewDMwLT", "DiscriminationByTightIsolationMVArun2v1PWnewDMwLT"),
    ("byVTightIsolationMVArun2v1PWnewDMwLT", "DiscriminationByVTightIsolationMVArun2v1PWnewDMwLT"),
    ("byVVTightIsolationMVArun2v1PWnewDMwLT", "DiscriminationByVVTightIsolationMVArun2v1PWnewDMwLT"),
    ("chargedIsoPtSumdR03", "ChargedIsoPtSumdR03"),
    ("neutralIsoPtSumdR03", "NeutralIsoPtSumdR03"),
    ("neutralIsoPtSumWeightdR03", "NeutralIsoPtSumWeightdR03"),
    ("footprintCorrectiondR03", "FootprintCorrectiondR03"),
    ("photonPtSumOutsideSignalConedR03", "PhotonPtSumOutsideSignalConedR03"),
    ("byIsolationMVArun2v1DBdR03oldDMwLTraw", "DiscriminationByIsolationMVArun2v1DBdR03oldDMwLTraw"),
    ("byVLooseIsolationMVArun2v1DBdR03oldDMwLT", "DiscriminationByVLooseIsolationMVArun2v1DBdR03oldDMwLT"),
    ("byLooseIsolationMVArun2v1DBdR03oldDMwLT", "DiscriminationByLooseIsolationMVArun2v1DBdR03oldDMwLT"),
    ("byMediumIsolationMVArun2v1DBdR03oldDMwLT", "DiscriminationByMediumIsolationMVArun2v1DBdR03oldDMwLT"),
    ("byTightIsolationMVArun2v1DBdR03oldDMwLT", "DiscriminationByTightIsolationMVArun2v1DBdR03oldDMwLT"),
    ("byVTightIsolationMVArun2v1DBdR03oldDMwLT", "DiscriminationByVTightIsolationMVArun2v1DBdR03oldDMwLT"),
    ("byVVTightIsolationMVArun2v1DBdR03oldDMwLT", "DiscriminationByVVTightIsolationMVArun2v1DBdR03oldDMwLT"),
    ("byIsolationMVArun2v1PWdR03oldDMwLTraw", "DiscriminationByIsolationMVArun2v1PWdR03oldDMwLTraw"),
    ("byVLooseIsolationMVArun2v1PWdR03oldDMwLT", "DiscriminationByVLooseIsolationMVArun2v1PWdR03oldDMwLT"),
    ("byLooseIsolationMVArun2v1PWdR03oldDMwLT", "DiscriminationByLooseIsolationMVArun2v1PWdR03oldDMwLT"),
    ("byMediumIsolationMVArun2v1PWdR03oldDMwLT", "DiscriminationByMediumIsolationMVArun2v1PWdR03oldDMwLT"),
    ("byTightIsolationMVArun2v1PWdR03oldDMwLT", "DiscriminationByTightIsolationMVArun2v1PWdR03oldDMwLT"),
    ("byVTightIsolationMVArun2v1PWdR03oldDMwLT", "DiscriminationByVTightIsolationMVArun2v1PWdR03oldDMwLT"),
    ("byVVTightIsolationMVArun2v1PWdR03oldDMwLT", "DiscriminationByVVTightIsolationMVArun2v1PWdR03oldDMwLT"),
    ("againstElectronMVA6Raw", "DiscriminationByMVA6rawElectronRejection"),
    ("againstElectronMVA6category", "DiscriminationByMVA6rawElectronRejection:category"),
    ("againstElectronVLooseMVA6", "DiscriminationByMVA6VLooseElectronRejection"),
    ("againstElectronLooseMVA6", "DiscriminationByMVA6LooseElectronRejection"),
    ("againstElectronMediumMVA6", "DiscriminationByMVA6MediumElectronRejection"),
    ("againstElectronTightMVA6", "DiscriminationByMVA6TightElectronRejection"),
    ("againstElectronVTightMVA6", "DiscriminationByMVA6VTightElectronRejection"),
]

# switch to PFTau collection produced for fixed dR = 0.07 signal cone size
def switchToPFTauFixedCone(process,
                           tauSource = cms.InputTag('fixedConePFTauProducer'),
                           patTauLabel = "",
                           postfix = ""):
    fixedConeIDSources = copy.copy(classicTauIDSources)
    fixedConeIDSources.extend(classicPFTauIDSources)

    _switchToPFTau(process, tauSource, 'fixedConePFTau', fixedConeIDSources,
                   patTauLabel = patTauLabel, postfix = postfix)

# switch to PFTau collection produced for shrinking signal cone of size dR = 5.0/Et(PFTau)
def switchToPFTauShrinkingCone(process,
                               tauSource = cms.InputTag('shrinkingConePFTauProducer'),
                               patTauLabel = "",
                               postfix = ""):
    shrinkingIDSources = copy.copy(classicTauIDSources)
    shrinkingIDSources.extend(classicPFTauIDSources)

    _switchToPFTau(process, tauSource, 'shrinkingConePFTau', shrinkingIDSources,
                   patTauLabel = patTauLabel, postfix = postfix)

# switch to hadron-plus-strip(s) (HPS) PFTau collection
def switchToPFTauHPS(process,
                     tauSource = cms.InputTag('hpsPFTauProducer'),
                     patTauLabel = "",
                     jecLevels = [],
                     postfix = ""):

    _switchToPFTau(process, tauSource, 'hpsPFTau', hpsTauIDSources,
                   patTauLabel = patTauLabel, postfix = postfix)

    # CV: enable tau lifetime information for HPS PFTaus
    applyPostfix(process, "patTaus" + patTauLabel, postfix).tauTransverseImpactParameterSource = tauSource.value().replace("Producer", "TransverseImpactParameters")

    ## adapt cleanPatTaus
    if hasattr(process, "cleanPatTaus" + patTauLabel + postfix):
        getattr(process, "cleanPatTaus" + patTauLabel + postfix).preselection = preselection

    from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
    from Configuration.Eras.Modifier_run2_miniAOD_94XFall17_cff import run2_miniAOD_94XFall17
    for era in [ run2_miniAOD_80XLegacy, run2_miniAOD_94XFall17]:
        _patTaus = getattr(process, "patTaus"+patTauLabel+postfix)
        _extTauIDSources = _patTaus.tauIDSources.clone()
        _extTauIDSources.byVVLooseIsolationMVArun2v1DBoldDMwLT = cms.InputTag("hpsPFTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT")
        era.toModify(_patTaus, tauIDSources = _extTauIDSources)
        
# Select switcher by string
def switchToPFTauByType(process,
                        pfTauType = None,
                        tauSource = cms.InputTag('hpsPFTauProducer'),
                        patTauLabel = "",
                        postfix = "" ):
    mapping = {
        'shrinkingConePFTau' : switchToPFTauShrinkingCone,
        'fixedConePFTau'     : switchToPFTauFixedCone,
        'hpsPFTau'           : switchToPFTauHPS,
        'caloRecoTau'        : switchToCaloTau
    }
    if not pfTauType in mapping.keys():
        raise ValueError("Error in <switchToPFTauByType>: Undefined pfTauType = %s !!" % pfTauType)
    
    mapping[pfTauType](process, tauSource = tauSource,
                       patTauLabel = patTauLabel, postfix = postfix)

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
        if standardAlgo is None:
            standardAlgo = self._defaultParameters['standardAlgo'].value
        if standardType is None:
            standardType = self._defaultParameters['standardType'].value

        self.setParameter('tauCollection', tauCollection)
        self.setParameter('algoLabel', algoLabel)
        self.setParameter('typeLabel', typeLabel)
        self.setParameter('doPFIsoDeposits', doPFIsoDeposits)
        self.setParameter('standardAlgo', standardAlgo)
        self.setParameter('standardType', standardType)

        self.apply(process)

    def toolCode(self, process):
        tauCollection = self._parameters['tauCollection'].value
        algoLabel = self._parameters['algoLabel'].value
        typeLabel = self._parameters['typeLabel'].value
        doPFIsoDeposits = self._parameters['doPFIsoDeposits'].value
        standardAlgo = self._parameters['standardAlgo'].value
        standardType = self._parameters['standardType'].value

        ## disable computation of particle-flow based IsoDeposits
        ## in case tau is of CaloTau type
        if typeLabel == 'Tau':
            print "NO PF Isolation will be computed for CaloTau (this could be improved later)"
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

        ## clone module for computing particle-flow IsoDeposits
        def addPFIsoDepositClone(hook, **replaceStatements):
            newModule = getattr(process, hook).clone(**replaceStatements)
            newModuleIsoDepositExtractor = getattr(newModule, "ExtractorPSet")
            setattr(newModuleIsoDepositExtractor, "tauSource", getattr(newModule, "src"))

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

        ## fix label for input tag
        def fixInputTag(x):
            x.setModuleLabel(newLabel(x.moduleLabel))

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
        if typeLabel == 'Tau':
            switchToCaloTau(process,
                            tauSource = getattr(newTaus, "tauSource"),
                            patTauLabel = capitalize(algoLabel + typeLabel))
        else:
            switchToPFTauByType(process, pfTauType = algoLabel + typeLabel,
                                tauSource = getattr(newTaus, "tauSource"),
                                patTauLabel = capitalize(algoLabel + typeLabel))

addTauCollection=AddTauCollection()
