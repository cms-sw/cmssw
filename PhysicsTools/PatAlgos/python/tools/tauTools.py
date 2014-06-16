import FWCore.ParameterSet.Config as cms

from FWCore.GuiBrowsers.ConfigToolBase import *
from PhysicsTools.PatAlgos.tools.helpers import cloneProcessingSnippet
from RecoTauTag.RecoTau.TauDiscriminatorTools import *

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
        getattr(process, "cleanPatTaus" + patTauLabel + postfix).preselection = \
      'tauID("leadingTrackFinding") > 0.5 & tauID("leadingTrackPtCut") > 0.5' \
     + ' & tauID("byIsolation") > 0.5 & tauID("againstElectron") > 0.5 & (signalTracks.size() = 1 | signalTracks.size() = 3)'

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

# Hadron-plus-strip(s) (HPS) Tau Discriminators
hpsTauIDSources = [
    ("decayModeFindingNewDMs", "DiscriminationByDecayModeFindingNewDMs"),
    ("decayModeFindingOldDMs", "DiscriminationByDecayModeFindingOldDMs"),
    ("decayModeFinding", "DiscriminationByDecayModeFinding"), # CV: kept for backwards compatibility
    ("byLooseIsolation", "DiscriminationByLooseIsolation"),
    ("byVLooseCombinedIsolationDeltaBetaCorr", "DiscriminationByVLooseCombinedIsolationDBSumPtCorr"),
    ("byLooseCombinedIsolationDeltaBetaCorr", "DiscriminationByLooseCombinedIsolationDBSumPtCorr"),
    ("byMediumCombinedIsolationDeltaBetaCorr", "DiscriminationByMediumCombinedIsolationDBSumPtCorr"),
    ("byTightCombinedIsolationDeltaBetaCorr", "DiscriminationByTightCombinedIsolationDBSumPtCorr"),
    ("byCombinedIsolationDeltaBetaCorrRaw", "DiscriminationByRawCombinedIsolationDBSumPtCorr"),
    ("byLooseCombinedIsolationDeltaBetaCorr3Hits", "DiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits"),
    ("byMediumCombinedIsolationDeltaBetaCorr3Hits", "DiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits"),
    ("byTightCombinedIsolationDeltaBetaCorr3Hits", "DiscriminationByTightCombinedIsolationDBSumPtCorr3Hits"),
    ("byCombinedIsolationDeltaBetaCorrRaw3Hits", "DiscriminationByRawCombinedIsolationDBSumPtCorr3Hits"),
    ("chargedIsoPtSum", "MVA3IsolationChargedIsoPtSum"),
    ("neutralIsoPtSum", "MVA3IsolationNeutralIsoPtSum"),
    ("puCorrPtSum", "MVA3IsolationPUcorrPtSum"),
    ("byIsolationMVA3oldDMwoLTraw", "DiscriminationByIsolationMVA3oldDMwoLTraw"),
    ("byVLooseIsolationMVA3oldDMwoLT", "DiscriminationByVLooseIsolationMVA3oldDMwoLT"),
    ("byLooseIsolationMVA3oldDMwoLT", "DiscriminationByLooseIsolationMVA3oldDMwoLT"),
    ("byMediumIsolationMVA3oldDMwoLT", "DiscriminationByMediumIsolationMVA3oldDMwoLT"),
    ("byTightIsolationMVA3oldDMwoLT", "DiscriminationByTightIsolationMVA3oldDMwoLT"),
    ("byVTightIsolationMVA3oldDMwoLT", "DiscriminationByVTightIsolationMVA3oldDMwoLT"),
    ("byVVTightIsolationMVA3oldDMwoLT", "DiscriminationByVVTightIsolationMVA3oldDMwoLT"),
    ("byIsolationMVA3oldDMwLTraw", "DiscriminationByIsolationMVA3oldDMwLTraw"),
    ("byVLooseIsolationMVA3oldDMwLT", "DiscriminationByVLooseIsolationMVA3oldDMwLT"),
    ("byLooseIsolationMVA3oldDMwLT", "DiscriminationByLooseIsolationMVA3oldDMwLT"),
    ("byMediumIsolationMVA3oldDMwLT", "DiscriminationByMediumIsolationMVA3oldDMwLT"),
    ("byTightIsolationMVA3oldDMwLT", "DiscriminationByTightIsolationMVA3oldDMwLT"),
    ("byVTightIsolationMVA3oldDMwLT", "DiscriminationByVTightIsolationMVA3oldDMwLT"),
    ("byVVTightIsolationMVA3oldDMwLT", "DiscriminationByVVTightIsolationMVA3oldDMwLT"),
    ("byIsolationMVA3newDMwoLTraw", "DiscriminationByIsolationMVA3newDMwoLTraw"),
    ("byVLooseIsolationMVA3newDMwoLT", "DiscriminationByVLooseIsolationMVA3newDMwoLT"),
    ("byLooseIsolationMVA3newDMwoLT", "DiscriminationByLooseIsolationMVA3newDMwoLT"),
    ("byMediumIsolationMVA3newDMwoLT", "DiscriminationByMediumIsolationMVA3newDMwoLT"),
    ("byTightIsolationMVA3newDMwoLT", "DiscriminationByTightIsolationMVA3newDMwoLT"),
    ("byVTightIsolationMVA3newDMwoLT", "DiscriminationByVTightIsolationMVA3newDMwoLT"),
    ("byVVTightIsolationMVA3newDMwoLT", "DiscriminationByVVTightIsolationMVA3newDMwoLT"),
    ("byIsolationMVA3newDMwLTraw", "DiscriminationByIsolationMVA3newDMwLTraw"),
    ("byVLooseIsolationMVA3newDMwLT", "DiscriminationByVLooseIsolationMVA3newDMwLT"),
    ("byLooseIsolationMVA3newDMwLT", "DiscriminationByLooseIsolationMVA3newDMwLT"),
    ("byMediumIsolationMVA3newDMwLT", "DiscriminationByMediumIsolationMVA3newDMwLT"),
    ("byTightIsolationMVA3newDMwLT", "DiscriminationByTightIsolationMVA3newDMwLT"),
    ("byVTightIsolationMVA3newDMwLT", "DiscriminationByVTightIsolationMVA3newDMwLT"),
    ("byVVTightIsolationMVA3newDMwLT", "DiscriminationByVVTightIsolationMVA3newDMwLT"),
    ("againstElectronLoose", "DiscriminationByLooseElectronRejection"),
    ("againstElectronMedium", "DiscriminationByMediumElectronRejection"),
    ("againstElectronTight", "DiscriminationByTightElectronRejection"),
    ("againstElectronMVA5raw", "DiscriminationByMVA5rawElectronRejection"),
    ("againstElectronMVA5category", "DiscriminationByMVA5rawElectronRejection:category"),
    ("againstElectronVLooseMVA5", "DiscriminationByMVA5VLooseElectronRejection"),
    ("againstElectronLooseMVA5", "DiscriminationByMVA5LooseElectronRejection"),
    ("againstElectronMediumMVA5", "DiscriminationByMVA5MediumElectronRejection"),
    ("againstElectronTightMVA5", "DiscriminationByMVA5TightElectronRejection"),
    ("againstElectronVTightMVA5", "DiscriminationByMVA5VTightElectronRejection"),
    ("againstElectronDeadECAL", "DiscriminationByDeadECALElectronRejection"),
    ("againstMuonLoose", "DiscriminationByLooseMuonRejection"),
    ("againstMuonMedium", "DiscriminationByMediumMuonRejection"),
    ("againstMuonTight", "DiscriminationByTightMuonRejection"),
    ("againstMuonLoose2", "DiscriminationByLooseMuonRejection2"),
    ("againstMuonMedium2", "DiscriminationByMediumMuonRejection2"),
    ("againstMuonTight2", "DiscriminationByTightMuonRejection2"),
    ("againstMuonLoose3", "DiscriminationByLooseMuonRejection3"),
    ("againstMuonTight3", "DiscriminationByTightMuonRejection3"),
    ("againstMuonMVAraw", "DiscriminationByMVArawMuonRejection"),
    ("againstMuonLooseMVA", "DiscriminationByMVALooseMuonRejection"),
    ("againstMuonMediumMVA", "DiscriminationByMVAMediumMuonRejection"),
    ("againstMuonTightMVA", "DiscriminationByMVATightMuonRejection") ]


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
        getattr(process, "cleanPatTaus" + patTauLabel + postfix).preselection = \
        'pt > 20 & abs(eta) < 2.3 & tauID("decayModeFindingOldDMs") > 0.5 & tauID("byLooseCombinedIsolationDeltaBetaCorr3Hits") > 0.5' \
        + ' & tauID("againstMuonTight3") > 0.5 & tauID("againstElectronLoose") > 0.5'

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
