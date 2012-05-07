import FWCore.ParameterSet.Config as cms

from FWCore.GuiBrowsers.ConfigToolBase import *
from PhysicsTools.PatAlgos.tools.helpers import *
from PhysicsTools.PatAlgos.tools.jetTools import *
from Configuration.AlCa.autoCond import autoCond

import os
import socket
from subprocess import *
import json
import das_client


## ------------------------------------------------------
## Deal with backweard incompatibilities of conditions
## ------------------------------------------------------

def run42xOn3yzMcInput( process
                      , l1MenuTag    = 'L1GtTriggerMenu_L1Menu_Commissioning2010_v4_mc' # L1 menu for Fall10 REDIGI (CMSSW_3_8_7)
                      ):
  """
  """
  # Use correct L1 trigger menu
  import CondCore.DBCommon.CondDBCommon_cfi
  process.l1GtTriggerMenu = cms.ESSource( "PoolDBESSource"
  , CondCore.DBCommon.CondDBCommon_cfi.CondDBCommon
  , toGet   = cms.VPSet(
      cms.PSet(
        connect = cms.untracked.string( 'frontier://FrontierProd/CMS_COND_31X_L1T' )
      , record  = cms.string( 'L1GtTriggerMenuRcd' )
      , tag     = cms.string( l1MenuTag )
      )
    )
  )
  process.preferL1GtTriggerMenu = cms.ESPrefer( "PoolDBESSource", "l1GtTriggerMenu" )


## ------------------------------------------------------
## Re-configuration of PATJetProducer
## ------------------------------------------------------

def run36xOn35xInput(process,
                     genJets = "",
                     postfix=""):
    """
    ------------------------------------------------------------------
    Reconfigure the PATJetProducer to be able to run the 36X version
    of PAT on 35X input samples.

    process : process
    ------------------------------------------------------------------
    """
    print "*********************************************************************"
    print "NOTE TO USER: when running on 35X sample with 36X s/w versions you   "
    print "              need to adapt for different event contents. The        "
    print "              adaptations need to be made:                           "
    print "                                                                     "
    print "               - re-configuration of secondary vertex tag discrimi-  "
    print "                 nator information.                                  "
    print "                                                                     "
    print "*********************************************************************"
    ## re-configure b-discriminator sources for pat jets
    process.patJets.discriminatorSources = cms.VInputTag(
        cms.InputTag("combinedSecondaryVertexBJetTags"),
        cms.InputTag("combinedSecondaryVertexMVABJetTags"),
        cms.InputTag("jetBProbabilityBJetTags"),
        cms.InputTag("jetProbabilityBJetTags"),
        cms.InputTag("simpleSecondaryVertexBJetTags"),
        cms.InputTag("softElectronByPtBJetTags"),
        cms.InputTag("softElectronByIP3dBJetTags"),
        cms.InputTag("softMuonBJetTags"),
        cms.InputTag("softMuonByPtBJetTags"),
        cms.InputTag("softMuonByIP3dBJetTags"),
        cms.InputTag("trackCountingHighEffBJetTags"),
        cms.InputTag("trackCountingHighPurBJetTags"),
    )
    if genJets != "" :
        print "*********************************************************************"
        print "NOTE TO USER: when running on 31X samples re-recoed in 3.5.x         "
        print "              with this CMSSW version of PAT                         "
        print "              it is required to re-run the GenJet production for     "
        print "              anti-kT since that is not part of the re-reco          "
        print "*********************************************************************"
        process.load("RecoJets.Configuration.GenJetParticles_cff")
        process.load("RecoJets.JetProducers." + genJets +"_cfi")
        process.patDefaultSequence.replace( getattr(process,"patCandidates"+postfix), process.genParticlesForJets+getattr(process,genJets)+getattr(process,"patCandidates"+postfix))



## ------------------------------------------------------
## Re-implementation of jetTools
## ------------------------------------------------------

class RunBTagging35X(ConfigToolBase):

    """
    ------------------------------------------------------------------
    Define sequence to run b tagging on AOD input for a given jet
    collection including a JetTracksAssociatorAtVertex module with
    name 'jetTracksAssociatorAtVertex' + 'label'.

    Return value is a pair of (sequence, labels) where 'sequence'
    is the cms.Sequence, and 'labels' is a vector with the follow-
    ing entries:

     * labels['jta']      = the name of the JetTrackAssociator module
     * labels['tagInfos'] = a list of names of the TagInfo modules
     * labels['jetTags '] = a list of names of the JetTag modules

    This is a re-implementation to run the 36X version of PAT on 35X
    input samples.
    ------------------------------------------------------------------
    """
    _label='runBTagging35X'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'jetCollection',self._defaultValue, 'input jet collection',Type=cms.InputTag)
        self.addParameter(self._defaultParameters,'label',self._defaultValue, 'postfix label to identify new sequence/modules', Type=str)
        self.addParameter(self._defaultParameters,'postfix',"", "postfix of default sequence (do not confuse with 'label')")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 jetCollection     = None,
                 label             = None,
                 postfix           = None) :
        if  jetCollection is None:
            jetCollection=self._defaultParameters['jetCollection'].value
        if  label is None:
            label=self._defaultParameters['label'].value
        if postfix  is None:
            postfix=self._defaultParameters['postfix'].value
        self.setParameter('jetCollection',jetCollection)
        self.setParameter('label',label)
        self.setParameter('postfix',postfix)

        return self.apply(process)

    def apply(self, process):
        jetCollection=self._parameters['jetCollection'].value
        label=self._parameters['label'].value
        postfix=self._parameters['postfix'].value

        if hasattr(process, "addAction"):
            process.disableRecording()

        try:
            comment=inspect.stack(2)[2][4][0].rstrip("\n")
            if comment.startswith("#"):
                self.setComment(comment.lstrip("#"))
        except:
            pass

        #############################
        ### TOOL CODE STARTS HERE ###
        #############################
        print "*********************************************************************"
        print "NOTE TO USER: when running on 35X sample with 36X s/w versions you   "
        print "              need to adapt for different event contents. The        "
        print "              adaptations need to be made:                           "
        print "                                                                     "
        print "               - re-configuration of secondary vertex tag discrimi-  "
        print "                 nator information.                                  "
        print "                                                                     "
        print "               - take out soft electron tagger information, which    "
        print "                 is not available on 35X.                            "
        print "*********************************************************************"

        if (label == ''):
        ## label is not allowed to be empty
            raise ValueError, "label for re-running b tagging is not allowed to be empty"

        ## import track associator & b tag configuration
        process.load("RecoJets.JetAssociationProducers.ak5JTA_cff")
        from RecoJets.JetAssociationProducers.ak5JTA_cff import ak5JetTracksAssociatorAtVertex
        process.load("RecoBTag.Configuration.RecoBTag_cff")
        import RecoBTag.Configuration.RecoBTag_cff as btag

        # add negative tag infos
        import PhysicsTools.PatAlgos.recoLayer0.bTagging_cff as nbtag

        ## define jetTracksAssociator; for switchJetCollection
        ## the label is 'AOD' as empty labels will lead to crashes
        ## of crab. In this case the postfix label is skiped,
        ## otherwise a postfix label is added as for the other
        ## labels
        jtaLabel = 'jetTracksAssociatorAtVertex'+postfix

        if (not label == 'AOD'):
            jtaLabel  += label
        ## define tag info labels (compare with jetProducer_cfi.py)
        ipTILabel = 'impactParameterTagInfos'     + label
        svTILabel = 'secondaryVertexTagInfos'     + label
        smTILabel = 'softMuonTagInfos'            + label

        ## produce tag infos
        setattr( process, ipTILabel, btag.impactParameterTagInfos.clone(jetTracks = cms.InputTag(jtaLabel)) )
        setattr( process, svTILabel, btag.secondaryVertexTagInfos.clone(trackIPTagInfos = cms.InputTag(ipTILabel)) )
        setattr( process, smTILabel, btag.softMuonTagInfos.clone(jets = jetCollection) )

        ## make VInputTag from strings
        def vit(*args) : return cms.VInputTag( *[ cms.InputTag(x) for x in args ] )

        ## produce btags
        setattr( process, 'jetBProbabilityBJetTags'+label, btag.jetBProbabilityBJetTags.clone(tagInfos = vit(ipTILabel)) )
        setattr( process, 'jetProbabilityBJetTags' +label, btag.jetProbabilityBJetTags.clone (tagInfos = vit(ipTILabel)) )
        setattr( process, 'trackCountingHighPurBJetTags'+label, btag.trackCountingHighPurBJetTags.clone(tagInfos = vit(ipTILabel)) )
        setattr( process, 'trackCountingHighEffBJetTags'+label, btag.trackCountingHighEffBJetTags.clone(tagInfos = vit(ipTILabel)) )
        setattr( process, 'simpleSecondaryVertexBJetTags'+label, btag.simpleSecondaryVertexBJetTags.clone(tagInfos = vit(svTILabel)) )
        setattr( process, 'combinedSecondaryVertexBJetTags'+label, btag.combinedSecondaryVertexBJetTags.clone(tagInfos = vit(ipTILabel, svTILabel)) )
        setattr( process, 'combinedSecondaryVertexMVABJetTags'+label, btag.combinedSecondaryVertexMVABJetTags.clone(tagInfos = vit(ipTILabel, svTILabel)) )
        setattr( process, 'softMuonBJetTags'+label, btag.softMuonBJetTags.clone(tagInfos = vit(smTILabel)) )
        setattr( process, 'softMuonByPtBJetTags'+label, btag.softMuonByPtBJetTags.clone(tagInfos = vit(smTILabel)) )
        setattr( process, 'softMuonByIP3dBJetTags'+label, btag.softMuonByIP3dBJetTags.clone(tagInfos = vit(smTILabel)) )

        ## define vector of (output) labels
        labels = { 'jta'      : jtaLabel,
                   'tagInfos' : (ipTILabel,svTILabel,smTILabel),
                   'jetTags'  : [ (x + label) for x in ('jetBProbabilityBJetTags',
                                                        'jetProbabilityBJetTags',
                                                        'trackCountingHighPurBJetTags',
                                                        'trackCountingHighEffBJetTags',
                                                        'simpleSecondaryVertexBJetTags',
                                                        'combinedSecondaryVertexBJetTags',
                                                        'combinedSecondaryVertexMVABJetTags',
                                                        'softMuonBJetTags',
                                                        'softMuonByPtBJetTags',
                                                        'softMuonByIP3dBJetTags'
                                                        )
                                  ]
                   }

        ## extend an existing sequence by otherLabels
        def mkseq(process, firstlabel, *otherlabels):
            seq = getattr(process, firstlabel)
            for x in otherlabels: seq += getattr(process, x)
            return cms.Sequence(seq)

        ## add tag infos to the process
        setattr( process, 'btaggingTagInfos'+label, mkseq(process, *(labels['tagInfos']) ) )
        ## add b tags to the process
        setattr( process, 'btaggingJetTags'+label,  mkseq(process, *(labels['jetTags'])  ) )
        ## add a combined sequence to the process
        seq = mkseq(process, 'btaggingTagInfos'+label, 'btaggingJetTags' + label)
        setattr( process, 'btagging'+label, seq )
        ## return the combined sequence and the labels defined above

        if hasattr(process, "addAction"):
            process.enableRecording()
            action=self.__copy__()
            process.addAction(action)
        return (seq, labels)


runBTagging35X=RunBTagging35X()


class AddJetCollection35X(ConfigToolBase):

    """
    ------------------------------------------------------------------
    Add a new collection of jets. Takes the configuration from the
    already configured standard jet collection as starting point;
    replaces before calling addJetCollection will also affect the
    new jet collections. This is a re-implementation to run the
    36X version of PAT on 35X input samples.
    ------------------------------------------------------------------
    """
    _label='addJetCollection35X'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'jetCollection',self._defaultValue,'Input jet collection', cms.InputTag)
        self.addParameter(self._defaultParameters,'algoLabel',self._defaultValue, "label to indicate the jet algorithm (e.g.'AK5')",str)
        self.addParameter(self._defaultParameters,'typeLabel',self._defaultValue, "label to indicate the type of constituents (e.g. 'Calo', 'Pflow', 'Jpt', ...)",str)
        self.addParameter(self._defaultParameters,'doJTA',True, "run b tagging sequence for new jet collection and add it to the new pat jet collection")
        self.addParameter(self._defaultParameters,'doBTagging',True, 'run JetTracksAssociation and JetCharge and add it to the new pat jet collection (will autom. be true if doBTagging is set to true)')
        self.addParameter(self._defaultParameters,'jetCorrLabel',None, "algorithm and type of JEC; use 'None' for no JEC; examples are ('AK5','Calo'), ('SC7','Calo'), ('KT4','PF')", tuple,acceptNoneValue=True)
        self.addParameter(self._defaultParameters,'doType1MET',True, "if jetCorrLabel is not 'None', set this to 'True' to redo the Type1 MET correction for the new jet colllection; at the moment it must be 'False' for non CaloJets otherwise the JetMET POG module crashes. ")
        self.addParameter(self._defaultParameters,'doL1Cleaning',True, "copy also the producer modules for cleanLayer1 will be set to 'True' automatically when doL1Counters is 'True'")
        self.addParameter(self._defaultParameters,'doL1Counters',False, "copy also the filter modules that accept/reject the event looking at the number of jets")
        self.addParameter(self._defaultParameters,'genJetCollection',cms.InputTag("ak5GenJets"), "GenJet collection to match to")
        self.addParameter(self._defaultParameters,'doJetID',True, "add jetId variables to the added jet collection?")
        self.addParameter(self._defaultParameters,'jetIdLabel',"ak5", " specify the label prefix of the xxxJetID object; in general it is the jet collection tag like ak5, kt4 sc5, aso. For more information have a look to SWGuidePATTools#add_JetCollection")
        self.addParameter(self._defaultParameters,'standardAlgo',"AK5", "standard algorithm label of the collection from which the clones for the new jet collection will be taken from (note that this jet collection has to be available in the event before hand)")
        self.addParameter(self._defaultParameters,'standardType',"Calo", "standard constituent type label of the collection from which the clones for the new jet collection will be taken from (note that this jet collection has to be available in the event before hand)")

        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 jetCollection      = None,
                 algoLabel          = None,
                 typeLabel          = None,
                 doJTA              = None,
                 doBTagging         = None,
                 jetCorrLabel       = None,
                 doType1MET         = None,
                 doL1Cleaning       = None,
                 doL1Counters       = None,
                 genJetCollection   = None,
                 doJetID            = None,
                 jetIdLabel         = None,
                 standardAlgo       = None,
                 standardType       = None):

        if jetCollection  is None:
            jetCollection=self._defaultParameters['jetCollection'].value
        if algoLabel is None:
            algoLabel=self._defaultParameters['algoLabel'].value
        if typeLabel is None:
            typeLabel=self._defaultParameters['typeLabel'].value
        if doJTA is None:
            doJTA=self._defaultParameters['doJTA'].value
        if doBTagging is None:
            doBTagging=self._defaultParameters['doBTagging'].value
        if jetCorrLabel  is None:
            jetCorrLabel=self._defaultParameters['jetCorrLabel'].value
        if doType1MET  is None:
            doType1MET=self._defaultParameters['doType1MET'].value
        if doL1Cleaning is None:
            doL1Cleaning=self._defaultParameters['doL1Cleaning'].value
        if doL1Counters  is None:
            doL1Counters=self._defaultParameters['doL1Counters'].value
        if genJetCollection  is None:
            genJetCollection=self._defaultParameters['genJetCollection'].value
        if doJetID  is None:
            doJetID=self._defaultParameters['doJetID'].value
        if jetIdLabel  is None:
            jetIdLabel=self._defaultParameters['jetIdLabel'].value
        if standardAlgo is None:
            standardAlgo=self._defaultParameters['standardAlgo'].value
        if standardType is None:
            standardType=self._defaultParameters['standardType'].value

        self.setParameter('jetCollection',jetCollection)
        self.setParameter('algoLabel',algoLabel)
        self.setParameter('typeLabel',typeLabel)
        self.setParameter('doJTA',doJTA)
        self.setParameter('doBTagging',doBTagging)
        self.setParameter('jetCorrLabel',jetCorrLabel)
        self.setParameter('doType1MET',doType1MET)
        self.setParameter('doL1Cleaning',doL1Cleaning)
        self.setParameter('doL1Counters',doL1Counters)
        self.setParameter('genJetCollection',genJetCollection)
        self.setParameter('doJetID',doJetID)
        self.setParameter('jetIdLabel',jetIdLabel)
        self.setParameter('standardAlgo',standardAlgo)
        self.setParameter('standardType',standardType)

        self.apply(process)

    def toolCode(self, process):
        jetCollection=self._parameters['jetCollection'].value
        algoLabel=self._parameters['algoLabel'].value
        typeLabel=self._parameters['typeLabel'].value
        doJTA=self._parameters['doJTA'].value
        doBTagging=self._parameters['doBTagging'].value
        jetCorrLabel=self._parameters['jetCorrLabel'].value
        doType1MET =self._parameters['doType1MET'].value
        doL1Cleaning=self._parameters['doL1Cleaning'].value
        doL1Counters=self._parameters['doL1Counters'].value
        genJetCollection=self._parameters['genJetCollection'].value
        doJetID=self._parameters['doJetID'].value
        jetIdLabel=self._parameters['jetIdLabel'].value
        standardAlgo=self._parameters['standardAlgo'].value
        standardType=self._parameters['standardType'].value

        ## define common label for pre pat jet
        ## creation steps in makePatJets
        #label=standardAlgo+standardType

        ## create old module label from standardAlgo
        ## and standardType and return
        def oldLabel(prefix=''):
            return jetCollectionString(prefix, '', '')

        ## create new module label from old module
        ## label and return
        def newLabel(oldLabel):
            newLabel=oldLabel
            if(oldLabel.find(standardAlgo)>=0 and oldLabel.find(standardType)>=0):
                oldLabel=oldLabel.replace(standardAlgo, algoLabel).replace(standardType, typeLabel)
            else:
                oldLabel=oldLabel+algoLabel+typeLabel
            return oldLabel

        ## clone module and add it to the patDefaultSequence
        def addClone(hook, **replaceStatements):
            ## create a clone of the hook with corresponding
            ## parameter replacements
            newModule = getattr(process, hook).clone(**replaceStatements)
            ## add the module to the sequence
            addModuleToSequence(hook, newModule)

        ## add module to the patDefaultSequence
        def addModuleToSequence(hook, newModule):
            hookModule = getattr(process, hook)
            ## add the new module with standardAlgo &
            ## standardType replaced in module label
            setattr( process, newLabel(hook), newModule)
            ## add new module to default sequence
            ## just behind the hookModule
            process.patDefaultSequence.replace( hookModule, hookModule*newModule )

        print "*********************************************************************"
        print "NOTE TO USER: when running on 35X sample with 36X s/w versions you   "
        print "              need to adapt for different event contents. The        "
        print "              adaptations need to be made:                           "
        print "                                                                     "
        print "               - re-configuration of secondary vertex tag discrimi-  "
        print "                 nator information.                                  "
        print "                                                                     "
        print "               - take out soft electron tagger information, which    "
        print "                 is not available on 35X.                            "
        print "*********************************************************************"

        ## add a clone of patJets
        addClone(oldLabel(), jetSource = jetCollection)
        ## add a clone of selectedPatJets
        addClone(oldLabel('selected'), src=cms.InputTag(newLabel(oldLabel())))
        ## add a clone of cleanPatJets
        if( doL1Cleaning ):
            addClone(oldLabel('clean'), src=cms.InputTag(newLabel(oldLabel('selected'))))
        ## add a clone of countPatJets
        if( doL1Counters ):
            if( doL1Cleaning ):
                addClone(oldLabel('count'), src=cms.InputTag(newLabel(oldLabel('clean'))))
            else:
                addClone(oldLabel('count'), src=cms.InputTag(newLabel(oldLabel('selected'))))

        ## get attributes of new module
        l1Jets = getattr(process, newLabel(oldLabel()))

        ## add a clone of gen jet matching
        addClone('patJetPartonMatch', src = jetCollection)
        addClone('patJetGenJetMatch', src = jetCollection, matched = genJetCollection)

        ## add a clone of parton and flavour associations
        addClone('patJetPartonAssociation', jets = jetCollection)
        addClone('patJetFlavourAssociation', srcByReference = cms.InputTag(newLabel('patJetPartonAssociation')))

        ## fix label for input tag
        def fixInputTag(x): x.setModuleLabel(newLabel(x.moduleLabel))
        ## fix label for vector of input tags
        def fixVInputTag(x): x[0].setModuleLabel(newLabel(x[0].moduleLabel))

        ## provide allLayer1Jet inputs with individual labels
        fixInputTag(l1Jets.genJetMatch)
        fixInputTag(l1Jets.genPartonMatch)
        fixInputTag(l1Jets.JetPartonMapSource)

        ## make VInputTag from strings
        def vit(*args) : return cms.VInputTag( *[ cms.InputTag(x) for x in args ] )

        if (doJTA or doBTagging):
            ## add clone of jet track association
            process.load("RecoJets.JetAssociationProducers.ak5JTA_cff")
            from RecoJets.JetAssociationProducers.ak5JTA_cff import ak5JetTracksAssociatorAtVertex
            ## add jet track association module to processes
            jtaLabel = 'jetTracksAssociatorAtVertex'+algoLabel+typeLabel
            setattr( process, jtaLabel, ak5JetTracksAssociatorAtVertex.clone(jets = jetCollection) )
            process.makePatJets.replace(process.patJetCharge, getattr(process,jtaLabel)+process.patJetCharge)
            l1Jets.trackAssociationSource = cms.InputTag(jtaLabel)
            addClone('patJetCharge', src=cms.InputTag(jtaLabel)),
            fixInputTag(l1Jets.jetChargeSource)
        else:
            ## switch embedding of track association and jet
            ## charge estimate to 'False'
            l1Jets.addAssociatedTracks = False
            l1Jets.addJetCharge = False

        if (doBTagging):
            ## define postfixLabel
            postfixLabel=algoLabel+typeLabel
            ## add b tagging sequence
            (btagSeq, btagLabels) = runBTagging35X(process, jetCollection, postfixLabel)
            ## add b tagging sequence before running the allLayer1Jets modules
            process.makePatJets.replace(getattr(process,jtaLabel), getattr(process,jtaLabel)+btagSeq)
            ## replace corresponding tags for pat jet production
            l1Jets.trackAssociationSource = cms.InputTag(btagLabels['jta'])
            l1Jets.tagInfoSources = cms.VInputTag( *[ cms.InputTag(x) for x in btagLabels['tagInfos'] ] )
            l1Jets.discriminatorSources = cms.VInputTag( *[ cms.InputTag(x) for x in btagLabels['jetTags']  ] )
        else:
            ## switch general b tagging info switch off
            l1Jets.addBTagInfo = False

        if (doJetID):
            l1Jets.addJetID = cms.bool(True)
            jetIdLabelNew = jetIdLabel + 'JetID'
            l1Jets.jetIDMap = cms.InputTag( jetIdLabelNew )
        else :
            l1Jets.addJetID = cms.bool(False)

        if (jetCorrLabel != None):
            ## add clone of jet energy corrections;
            ## catch a couple of exceptions first
            if (jetCorrLabel == False ):
                raise ValueError, "In addJetCollection 'jetCorrLabel' must be set to 'None', not 'False'"
            if (jetCorrLabel == "None"):
                raise ValueError, "In addJetCollection 'jetCorrLabel' must be set to 'None' (without quotes)"
            ## check for the correct format
            if type(jetCorrLabel) != type(('AK5','Calo')):
                raise ValueError, "In addJetCollection 'jetCorrLabel' must be 'None', or of type ('Algo','Type')"

            ## add clone of jetCorrFactors
            addClone('patJetCorrFactors', jetSource = jetCollection)
            switchJECParameters( getattr(process,newLabel('patJetCorrFactors')), jetCorrLabel[0], jetCorrLabel[1], oldAlgo='AK5',oldType='Calo' )
            fixVInputTag(l1Jets.jetCorrFactorsSource)

            ## switch type1MET corrections off for PFJets
            if( jetCollection.__str__().find('PFJets')>=0 ):
                print '================================================='
                print 'Type1MET corrections are switched off for PFJets.'
                print 'of type %s%s.' % (jetCorrLabel[0].swapcase(), jetCorrLabel[1])
                print 'Users are recommened to use pfMET together with  '
                print 'PFJets.'
                print '================================================='
                doType1MET=False

            ## add a clone of the type1MET correction for the new jet collection
            if (doType1MET):
                ## in case there is no jet correction service in the paths add it
                ## as L2L3 if possible, as combined from L2 and L3 otherwise
                if not hasattr( process, '%s%sL2L3' % (jetCorrLabel[0].swapcase(), jetCorrLabel[1]) ):
                    setattr( process, '%s%sL2L3' % (jetCorrLabel[0].swapcase(), jetCorrLabel[1]),
                             cms.ESProducer("JetCorrectionESChain",
                                          correctors = cms.vstring('%s%sL2Relative' % (jetCorrLabel[0].swapcase(), jetCorrLabel[1]),
                                                                   '%s%sL3Absolute' % (jetCorrLabel[0].swapcase(), jetCorrLabel[1])
                                                                   )
                                          )
                             )
                ## add a clone of the type1MET correction
                ## and the following muonMET correction
                addClone('metJESCorAK5CaloJet', inputUncorJetsLabel = jetCollection.value(),
                         corrector = cms.string('%s%sL2L3' % (jetCorrLabel[0].swapcase(), jetCorrLabel[1]))
                         )
                addClone('metJESCorAK5CaloJetMuons', uncorMETInputTag = cms.InputTag(newLabel('metJESCorAK5CaloJet')))
                addClone('patMETs', metSource = cms.InputTag(newLabel('metJESCorAK5CaloJetMuons')))
                l1MET = getattr(process, newLabel('patMETs'))
                ## add new met collections output to the pat summary
                process.patCandidateSummary.candidates += [ cms.InputTag(newLabel('patMETs')) ]
        else:
            ## switch jetCorrFactors off
            l1Jets.addJetCorrFactors = False



addJetCollection35X=AddJetCollection35X()


class SwitchJetCollection35X(ConfigToolBase):

    """
    ------------------------------------------------------------------
    Switch the collection of jets in PAT from the default value to a
    new jet collection. This is a re-implementation to run the 36X
    version of PAT on 35X input samples.
    ------------------------------------------------------------------
    """
    _label='switchJetCollection35X'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'jetCollection',self._defaultValue,'Input jet collection', cms.InputTag)
        self.addParameter(self._defaultParameters,'doJTA',True, "run b tagging sequence for new jet collection and add it to the new pat jet collection")
        self.addParameter(self._defaultParameters,'doBTagging',True, 'run JetTracksAssociation and JetCharge and add it to the new pat jet collection (will autom. be true if doBTagging is set to true)')
        self.addParameter(self._defaultParameters,'jetCorrLabel',None, "algorithm and type of JEC; use 'None' for no JEC; examples are ('AK5','Calo'), ('SC7','Calo'), ('KT4','PF')", tuple,acceptNoneValue=True)
        self.addParameter(self._defaultParameters,'doType1MET',True, "if jetCorrLabel is not 'None', set this to 'True' to redo the Type1 MET correction for the new jet colleection; at the moment it must be 'False' for non CaloJets otherwise the JetMET POG module crashes. ")
        self.addParameter(self._defaultParameters,'genJetCollection',cms.InputTag("ak5GenJets"), "GenJet collection to match to")
        self.addParameter(self._defaultParameters,'doJetID',True, "add jetId variables to the added jet collection")
        self.addParameter(self._defaultParameters,'jetIdLabel',"ak5", " specify the label prefix of the xxxJetID object; in general it is the jet collection tag like ak5, kt4 sc5, aso. For more information have a look to SWGuidePATTools#add_JetCollection")
        self.addParameter(self._defaultParameters,'postfix',"", "postfix of default sequence")

        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 jetCollection      = None,
                 doJTA              = None,
                 doBTagging         = None,
                 jetCorrLabel       = None,
                 doType1MET         = None,
                 genJetCollection   = None,
                 doJetID            = None,
                 jetIdLabel         = None,
                 postfix            = None):

        if jetCollection  is None:
            jetCollection=self._defaultParameters['jetCollection'].value
        if doJTA is None:
            doJTA=self._defaultParameters['doJTA'].value
        if doBTagging is None:
            doBTagging=self._defaultParameters['doBTagging'].value
        if jetCorrLabel  is None:
            jetCorrLabel=self._defaultParameters['jetCorrLabel'].value
        if doType1MET  is None:
            doType1MET=self._defaultParameters['doType1MET'].value
        if genJetCollection  is None:
            genJetCollection=self._defaultParameters['genJetCollection'].value
        if doJetID  is None:
            doJetID=self._defaultParameters['doJetID'].value
        if jetIdLabel  is None:
            jetIdLabel=self._defaultParameters['jetIdLabel'].value
        if postfix  is None:
            postfix=self._defaultParameters['postfix'].value

        self.setParameter('jetCollection',jetCollection)
        self.setParameter('doJTA',doJTA)
        self.setParameter('doBTagging',doBTagging)
        self.setParameter('jetCorrLabel',jetCorrLabel)
        self.setParameter('doType1MET',doType1MET)
        self.setParameter('genJetCollection',genJetCollection)
        self.setParameter('doJetID',doJetID)
        self.setParameter('jetIdLabel',jetIdLabel)
        self.setParameter('postfix',postfix)

        self.apply(process)

    def toolCode(self, process):
        jetCollection=self._parameters['jetCollection'].value
        doJTA=self._parameters['doJTA'].value
        doBTagging=self._parameters['doBTagging'].value
        jetCorrLabel=self._parameters['jetCorrLabel'].value
        doType1MET =self._parameters['doType1MET'].value
        genJetCollection=self._parameters['genJetCollection'].value
        doJetID=self._parameters['doJetID'].value
        jetIdLabel=self._parameters['jetIdLabel'].value
        postfix=self._parameters['postfix'].value


        ## save label of old input jet collection
        oldLabel = applyPostfix(process, "patJets", postfix).jetSource;

        ## replace input jet collection for generator matches
	applyPostfix(process, "patJetPartonMatch", postfix).src = jetCollection
	applyPostfix(process, "patJetGenJetMatch", postfix).src = jetCollection
	applyPostfix(process, "patJetGenJetMatch", postfix).matched = genJetCollection
	applyPostfix(process, "patJetPartonAssociation", postfix).jets = jetCollection
        ## replace input jet collection for pat jet production
	applyPostfix(process, "patJets", postfix).jetSource = jetCollection

        ## make VInputTag from strings
        def vit(*args) : return cms.VInputTag( *[ cms.InputTag(x) for x in args ] )

        if (doJTA or doBTagging):
            ## replace jet track association
            process.load("RecoJets.JetAssociationProducers.ak5JTA_cff")
            from RecoJets.JetAssociationProducers.ak5JTA_cff import ak5JetTracksAssociatorAtVertex
            setattr(process, "jetTracksAssociatorAtVertex"+postfix, ak5JetTracksAssociatorAtVertex.clone(jets = jetCollection))
            getattr(process, "patDefaultSequence"+postfix).replace(
                applyPostfix(process, "patJetCharge", postfix),
                getattr(process, "jetTracksAssociatorAtVertex" + postfix) #module with postfix that is not n patDefaultSequence
                + applyPostfix(process, "patJetCharge", postfix)
                )

            applyPostfix(process, "patJetCharge", postfix).src = 'jetTracksAssociatorAtVertex'+postfix
            applyPostfix(process, "patJets", postfix).trackAssociationSource = 'jetTracksAssociatorAtVertex'+postfix
        else:
            ## remove the jet track association from the std
            ## sequence
            removeIfInSequence(process,  "patJetCharge",  "patDefaultSequence", postfix)
            ## switch embedding of track association and jet
            ## charge estimate to 'False'
            applyPostfix(process, "patJets", postfix).addAssociatedTracks = False
            applyPostfix(process, "patJets", postfix).addJetCharge = False

        if (doBTagging):
            ## replace b tagging sequence; add postfix label 'AOD' as crab will
            ## crash when confronted with empy labels
            (btagSeq, btagLabels) = runBTagging35X(process, jetCollection, 'AOD',postfix)
            ## add b tagging sequence before running the allLayer1Jets modules
            getattr(process, "patDefaultSequence"+postfix).replace(
                getattr( process,"jetTracksAssociatorAtVertex"+postfix),
                getattr( process,"jetTracksAssociatorAtVertex"+postfix) + btagSeq
                )

            ## replace corresponding tags for pat jet production
            applyPostfix(process, "patJets", postfix).trackAssociationSource = btagLabels['jta']
            applyPostfix(process, "patJets", postfix).tagInfoSources = cms.VInputTag( *[ cms.InputTag(x) for x in btagLabels['tagInfos'] ] )
            applyPostfix(process, "patJets", postfix).discriminatorSources = cms.VInputTag( *[ cms.InputTag(x) for x in btagLabels['jetTags']  ] )
        else:
            ## remove b tagging from the std sequence
            removeIfInSequence(process,  "secondaryVertexNegativeTagInfos",  "patDefaultSequence", postfix)
            removeIfInSequence(process,  "simpleSecondaryVertexNegativeBJetTags",  "patDefaultSequence", postfix)

            ## switch embedding of b tagging for pat
            ## jet production to 'False'
            applyPostfix(process, "patJets", postfix).addBTagInfo = False

        if (doJetID):
            jetIdLabelNew = jetIdLabel + 'JetID'
            applyPostfix(process, "patJets", postfix).jetIDMap = cms.InputTag( jetIdLabelNew )
        else:
            applyPostfix(process, "patJets", postfix).addJetID = cms.bool(False)


        if (jetCorrLabel!=None):
            ## replace jet energy corrections; catch
            ## a couple of exceptions first
            if (jetCorrLabel == False ):
                raise ValueError, "In switchJetCollection 'jetCorrLabel' must be set to 'None', not 'False'"
            if (jetCorrLabel == "None"):
                raise ValueError, "In switchJetCollection 'jetCorrLabel' must be set to 'None' (without quotes)"
            ## check for the correct format
            if (type(jetCorrLabel)!=type(('AK5','Calo'))):
                raise ValueError, "In switchJetCollection 'jetCorrLabel' must be 'None', or of type ('Algo','Type')"

            ## switch JEC parameters to the new jet collection
            applyPostfix(process, "patJetCorrFactors", postfix).jetSource = jetCollection
            switchJECParameters(applyPostfix(process, "patJetCorrFactors", postfix), jetCorrLabel[0], jetCorrLabel[1], oldAlgo='AK5',oldType='Calo')

            ## switch type1MET corrections off for PFJets
            if( jetCollection.__str__().find('PFJets')>=0 ):
                print '================================================='
                print 'Type1MET corrections are switched off for PFJets.'
                print 'of type %s%s.' % (jetCorrLabel[0].swapcase(), jetCorrLabel[1])
                print 'Users are recommened to use pfMET together with  '
                print 'PFJets.'
                print '================================================='
                doType1MET=False

            ## redo the type1MET correction for the new jet collection
            if (doType1MET):
                ## in case there is no jet correction service in the paths add it
                ## as L2L3 if possible, as combined from L2 and L3 otherwise
                if not hasattr( process, '%s%sL2L3' % (jetCorrLabel[0].swapcase(), jetCorrLabel[1]) ):
                    setattr( process, '%s%sL2L3' % (jetCorrLabel[0].swapcase(), jetCorrLabel[1]),
                             cms.ESProducer("JetCorrectionESChain",
                                          correctors = cms.vstring('%s%sL2Relative' % (jetCorrLabel[0].swapcase(), jetCorrLabel[1]),
                                                                   '%s%sL3Absolute' % (jetCorrLabel[0].swapcase(), jetCorrLabel[1])
                                                                   )
                                          )
                             )
                ## configure the type1MET correction the following muonMET
                ## corrections have the metJESCorAK5CaloJet as input and
                ## are automatically correct
                applyPostfix(process, "metJESCorAK5CaloJet", postfix).inputUncorJetsLabel = jetCollection.value()
                applyPostfix(process, "metJESCorAK5CaloJet", postfix).corrector = '%s%sL2L3' % (jetCorrLabel[0].swapcase(), jetCorrLabel[1])
        else:
            ## remove the jetCorrFactors from the std sequence
            process.patJetMETCorrections.remove(process.patJetCorrFactors)
            ## switch embedding of jetCorrFactors off
            ## for pat jet production
            applyPostfix(process, "patJets", postfix).addJetCorrFactors = False


switchJetCollection35X=SwitchJetCollection35X()


## ------------------------------------------------------
## Automatic pick-up of RelVal input files
## ------------------------------------------------------

class PickRelValInputFiles( ConfigToolBase ):
    """  Picks up RelVal input files automatically and
  returns a vector of strings with the paths to be used in [PoolSource].fileNames
    PickRelValInputFiles( cmsswVersion, relVal, dataTier, condition, globalTag, maxVersions, skipFiles, numberOfFiles, debug )
    - useDAS       : switch to perform query in DAS rather than in DBS
                     optional; default: False
    - cmsswVersion : CMSSW release to pick up the RelVal files from
                     optional; default: the current release (determined automatically from environment)
    - formerVersion: use the last before the last valid CMSSW release to pick up the RelVal files from
                     applies also, if 'cmsswVersion' is set explicitly
                     optional; default: False
    - relVal       : RelVal sample to be used
                     optional; default: 'RelValTTbar'
    - dataTier     : data tier to be used
                     optional; default: 'GEN-SIM-RECO'
    - condition    : identifier of GlobalTag as defined in Configurations/PyReleaseValidation/python/autoCond.py
                     possibly overwritten, if 'globalTag' is set explicitly
                     optional; default: 'startup'
    - globalTag    : name of GlobalTag as it is used in the data path of the RelVals
                     optional; default: determined automatically as defined by 'condition' in Configurations/PyReleaseValidation/python/autoCond.py
      !!!            Determination is done for the release one runs in, not for the release the RelVals have been produced in.
      !!!            Example of deviation: data RelVals (CMSSW_4_1_X) might not only have the pure name of the GlobalTag 'GR_R_311_V2' in the full path,
                     but also an extension identifying the data: 'GR_R_311_V2_RelVal_wzMu2010B'
    - maxVersions  : max. versioning number of RelVal to check
                     optional; default: 9
    - skipFiles    : number of files to skip for a found RelVal sample
                     optional; default: 0
    - numberOfFiles: number of files to pick up
                     setting it to negative values, returns all found ('skipFiles' remains active though)
                     optional; default: -1
    - debug        : switch to enable enhanced messages in 'stdout'
                     optional; default: False
    """

    _label             = 'pickRelValInputFiles'
    _defaultParameters = dicttypes.SortedKeysDict()

    def getDefaultParameters( self ):
        return self._defaultParameters

    def __init__( self ):
        ConfigToolBase.__init__( self )
        self.addParameter( self._defaultParameters, 'useDAS'       , False                                                               , '' )
        self.addParameter( self._defaultParameters, 'cmsswVersion' , os.getenv( "CMSSW_VERSION" )                                        , 'auto from environment' )
        self.addParameter( self._defaultParameters, 'formerVersion', False                                                               , '' )
        self.addParameter( self._defaultParameters, 'relVal'       , 'RelValTTbar'                                                       , '' )
        self.addParameter( self._defaultParameters, 'dataTier'     , 'GEN-SIM-RECO'                                                      , '' )
        self.addParameter( self._defaultParameters, 'condition'    , 'startup'                                                           , '' )
        self.addParameter( self._defaultParameters, 'globalTag'    , autoCond[ self.getDefaultParameters()[ 'condition' ].value ][ : -5 ], 'auto from \'condition\'' )
        self.addParameter( self._defaultParameters, 'maxVersions'  , 3                                                                   , '' )
        self.addParameter( self._defaultParameters, 'skipFiles'    , 0                                                                   , '' )
        self.addParameter( self._defaultParameters, 'numberOfFiles', -1                                                                  , 'all' )
        self.addParameter( self._defaultParameters, 'debug'        , False                                                               , '' )
        self._parameters = copy.deepcopy( self._defaultParameters )
        self._comment = ""

    def __call__( self
                , useDAS        = None
                , cmsswVersion  = None
                , formerVersion = None
                , relVal        = None
                , dataTier      = None
                , condition     = None
                , globalTag     = None
                , maxVersions   = None
                , skipFiles     = None
                , numberOfFiles = None
                , debug         = None
                ):
        if useDAS is None:
            useDAS = self.getDefaultParameters()[ 'useDAS' ].value
        if cmsswVersion is None:
            cmsswVersion = self.getDefaultParameters()[ 'cmsswVersion' ].value
        if formerVersion is None:
            formerVersion = self.getDefaultParameters()[ 'formerVersion' ].value
        if relVal is None:
            relVal = self.getDefaultParameters()[ 'relVal' ].value
        if dataTier is None:
            dataTier = self.getDefaultParameters()[ 'dataTier' ].value
        if condition is None:
            condition = self.getDefaultParameters()[ 'condition' ].value
        if globalTag is None:
            globalTag = autoCond[ condition ][ : -5 ] # auto from 'condition'
        if maxVersions is None:
            maxVersions = self.getDefaultParameters()[ 'maxVersions' ].value
        if skipFiles is None:
            skipFiles = self.getDefaultParameters()[ 'skipFiles' ].value
        if numberOfFiles is None:
            numberOfFiles = self.getDefaultParameters()[ 'numberOfFiles' ].value
        if debug is None:
            debug = self.getDefaultParameters()[ 'debug' ].value
        self.setParameter( 'useDAS'       , useDAS )
        self.setParameter( 'cmsswVersion' , cmsswVersion )
        self.setParameter( 'formerVersion', formerVersion )
        self.setParameter( 'relVal'       , relVal )
        self.setParameter( 'dataTier'     , dataTier )
        self.setParameter( 'condition'    , condition )
        self.setParameter( 'globalTag'    , globalTag )
        self.setParameter( 'maxVersions'  , maxVersions )
        self.setParameter( 'skipFiles'    , skipFiles )
        self.setParameter( 'numberOfFiles', numberOfFiles )
        self.setParameter( 'debug'        , debug )
        return self.apply()

    def messageEmptyList( self ):
        print '%s DEBUG: Empty file list returned'%( self._label )
        print '    This might be overwritten by providing input files explicitly to the source module in the main configuration file.'

    def apply( self ):
        useDAS        = self._parameters[ 'useDAS'        ].value
        cmsswVersion  = self._parameters[ 'cmsswVersion'  ].value
        formerVersion = self._parameters[ 'formerVersion' ].value
        relVal        = self._parameters[ 'relVal'        ].value
        dataTier      = self._parameters[ 'dataTier'      ].value
        condition     = self._parameters[ 'condition'     ].value # only used for GT determination in initialization, if GT not explicitly given
        globalTag     = self._parameters[ 'globalTag'     ].value
        maxVersions   = self._parameters[ 'maxVersions'   ].value
        skipFiles     = self._parameters[ 'skipFiles'     ].value
        numberOfFiles = self._parameters[ 'numberOfFiles' ].value
        debug         = self._parameters[ 'debug'         ].value

        filePaths = []

        # Determine corresponding CMSSW version for RelVals
        preId      = '_pre'
        patchId    = '_patch'    # patch releases
        hltPatchId = '_hltpatch' # HLT patch releases
        dqmPatchId = '_dqmpatch' # DQM patch releases
        slhcId     = '_SLHC'     # SLHC releases
        rootId     = '_root'     # ROOT test releases
        ibId       = '_X_'       # IBs
        if patchId in cmsswVersion:
            cmsswVersion = cmsswVersion.split( patchId )[ 0 ]
        elif hltPatchId in cmsswVersion:
            cmsswVersion = cmsswVersion.split( hltPatchId )[ 0 ]
        elif dqmPatchId in cmsswVersion:
            cmsswVersion = cmsswVersion.split( dqmPatchId )[ 0 ]
        elif rootId in cmsswVersion:
            cmsswVersion = cmsswVersion.split( rootId )[ 0 ]
        elif slhcId in cmsswVersion:
            cmsswVersion = cmsswVersion.split( slhcId )[ 0 ]
        elif ibId in cmsswVersion or formerVersion:
            outputTuple = Popen( [ 'scram', 'l -c CMSSW' ], stdout = PIPE, stderr = PIPE ).communicate()
            if len( outputTuple[ 1 ] ) != 0:
                print '%s INFO : SCRAM error'%( self._label )
                if debug:
                    print '    from trying to determine last valid releases before \'%s\''%( cmsswVersion )
                    print
                    print outputTuple[ 1 ]
                    print
                    self.messageEmptyList()
                return filePaths
            versions = { 'last'      :''
                       , 'lastToLast':''
                       }
            for line in outputTuple[ 0 ].splitlines():
                version = line.split()[ 1 ]
                if cmsswVersion.split( ibId )[ 0 ] in version or cmsswVersion.rpartition( '_' )[ 0 ] in version:
                    if not ( patchId in version or hltPatchId in version or dqmPatchId in version or slhcId in version or ibId in version or rootId in version ):
                        versions[ 'lastToLast' ] = versions[ 'last' ]
                        versions[ 'last' ]       = version
                        if version == cmsswVersion:
                            break
            # FIXME: ordering of output problematic ('XYZ_pre10' before 'XYZ_pre2', no "formerVersion" for 'XYZ_pre1')
            if formerVersion:
                # Don't use pre-releases as "former version" for other releases than CMSSW_X_Y_0
                if preId in versions[ 'lastToLast' ] and not preId in versions[ 'last' ] and not versions[ 'last' ].endswith( '_0' ):
                    versions[ 'lastToLast' ] = versions[ 'lastToLast' ].split( preId )[ 0 ] # works only, if 'CMSSW_X_Y_0' esists ;-)
                # Use pre-release as "former version" for CMSSW_X_Y_0
                elif versions[ 'last' ].endswith( '_0' ) and not ( preId in versions[ 'lastToLast' ] and versions[ 'lastToLast' ].startswith( versions[ 'last' ] ) ):
                    versions[ 'lastToLast' ] = ''
                    for line in outputTuple[ 0 ].splitlines():
                        version      = line.split()[ 1 ]
                        versionParts = version.partition( preId )
                        if versionParts[ 0 ] == versions[ 'last' ] and versionParts[ 1 ] == preId:
                            versions[ 'lastToLast' ] = version
                        elif versions[ 'lastToLast' ] != '':
                            break
                # Don't use CMSSW_X_Y_0 as "former version" for pre-releases
                elif preId in versions[ 'last' ] and not preId in versions[ 'lastToLast' ] and versions[ 'lastToLast' ].endswith( '_0' ):
                    versions[ 'lastToLast' ] = '' # no alternative :-(
                cmsswVersion = versions[ 'lastToLast' ]
            else:
                cmsswVersion = versions[ 'last' ]

        # Debugging output
        if debug:
            print '%s DEBUG: Called with...'%( self._label )
            for key in self._parameters.keys():
               print '    %s:\t'%( key ),
               print self._parameters[ key ].value,
               if self._parameters[ key ].value is self.getDefaultParameters()[ key ].value:
                   print ' (default)'
               else:
                   print
               if key == 'cmsswVersion' and cmsswVersion != self._parameters[ key ].value:
                   if formerVersion:
                       print '    ==> modified to last to last valid release %s (s. \'formerVersion\' parameter)'%( cmsswVersion )
                   else:
                       print '    ==> modified to last valid release %s'%( cmsswVersion )

        # Check domain
        domain = socket.getfqdn().split( '.' )
        domainSE = ''
        if len( domain ) == 0:
            print '%s INFO : Cannot determine domain of this computer'%( self._label )
            if debug:
                self.messageEmptyList()
            return filePaths
        elif os.uname()[0] == "Darwin":
            print '%s INFO : Running on MacOSX without direct access to RelVal files.'%( self._label )
            if debug:
                self.messageEmptyList()
            return filePaths
        elif len( domain ) == 1:
            print '%s INFO : Running on local host \'%s\' without direct access to RelVal files'%( self._label, domain[ 0 ] )
            if debug:
                self.messageEmptyList()
            return filePaths
        if not ( ( domain[ -2 ] == 'cern' and domain[ -1 ] == 'ch' ) or ( domain[ -2 ] == 'fnal' and domain[ -1 ] == 'gov' ) ):
            print '%s INFO : Running on site \'%s.%s\' without direct access to RelVal files'%( self._label, domain[ -2 ], domain[ -1 ] )
            if debug:
                self.messageEmptyList()
            return filePaths
        if domain[ -2 ] == 'cern':
            domainSE = 'T2_CH_CERN'
        elif domain[ -2 ] == 'fnal':
            domainSE = 'T1_US_FNAL_MSS'
        if debug:
            print '%s DEBUG: Running at site \'%s.%s\''%( self._label, domain[ -2 ], domain[ -1 ] )
            print '%s DEBUG: Looking for SE \'%s\''%( self._label, domainSE )

        # Find files
        validVersion = 0
        dataset    = ''
        datasetAll = '/%s/%s-%s-v*/%s'%( relVal, cmsswVersion, globalTag, dataTier )
        if useDAS:
            if debug:
                print '%s DEBUG: Using DAS query'%( self._label )
            dasLimit = numberOfFiles
            if dasLimit <= 0:
                dasLimit += 1
            for version in range( maxVersions, 0, -1 ):
                filePaths    = []
                filePathsTmp = []
                fileCount    = 0
                dataset = '/%s/%s-%s-v%i/%s'%( relVal, cmsswVersion, globalTag, version, dataTier )
                dasQuery = 'file dataset=%s | grep file.name'%( dataset )
                if debug:
                    print '%s DEBUG: Querying dataset \'%s\' with'%( self._label, dataset )
                    print '    \'%s\''%( dasQuery )
                # partially stolen from das_client.py for option '--format=plain', needs filter ("grep") in the query
                dasData     = das_client.get_data( 'https://cmsweb.cern.ch', dasQuery, 0, dasLimit, False )
                jsondict    = json.loads( dasData )
                if debug:
                    print '%s DEBUG: Received DAS data:'%( self._label )
                    print '    \'%s\''%( dasData )
                    print '%s DEBUG: Determined JSON dictionary:'%( self._label )
                    print '    \'%s\''%( jsondict )
                if jsondict[ 'status' ] != 'ok':
                    print 'There was a problem while querying DAS with query \'%s\'. Server reply was:\n %s' % (dasQuery, dasData)
                    exit( 1 )
                mongo_query = jsondict[ 'mongo_query' ]
                filters     = mongo_query[ 'filters' ]
                data        = jsondict[ 'data' ]
                if debug:
                    print '%s DEBUG: Query in JSON dictionary:'%( self._label )
                    print '    \'%s\''%( mongo_query )
                    print '%s DEBUG: Filters in query:'%( self._label )
                    print '    \'%s\''%( filters )
                    print '%s DEBUG: Data in JSON dictionary:'%( self._label )
                    print '    \'%s\''%( data )
                for row in data:
                    filePath = [ r for r in das_client.get_value( row, filters ) ][ 0 ]
                    if debug:
                        print '%s DEBUG: Testing file entry \'%s\''%( self._label, filePath )
                    if len( filePath ) > 0:
                        if validVersion != version:
                            dasTest         = das_client.get_data( 'https://cmsweb.cern.ch', 'site dataset=%s | grep site.name'%( dataset ), 0, 999, False )
                            jsontestdict    = json.loads( dasTest )
                            mongo_testquery = jsontestdict[ 'mongo_query' ]
                            testfilters = mongo_testquery[ 'filters' ]
                            testdata    = jsontestdict[ 'data' ]
                            if debug:
                                print '%s DEBUG: Received DAS data (site test):'%( self._label )
                                print '    \'%s\''%( dasTest )
                                print '%s DEBUG: Determined JSON dictionary (site test):'%( self._label )
                                print '    \'%s\''%( jsontestdict )
                                print '%s DEBUG: Query in JSON dictionary (site test):'%( self._label )
                                print '    \'%s\''%( mongo_testquery )
                                print '%s DEBUG: Filters in query (site test):'%( self._label )
                                print '    \'%s\''%( testfilters )
                                print '%s DEBUG: Data in JSON dictionary (site test):'%( self._label )
                                print '    \'%s\''%( testdata )
                            foundSE = False
                            for testrow in testdata:
                                siteName = [ tr for tr in das_client.get_value( testrow, testfilters ) ][ 0 ]
                                if siteName == domainSE:
                                    foundSE = True
                                    break
                            if not foundSE:
                                if debug:
                                    print '%s DEBUG: Possible version \'v%s\' not available on SE \'%s\''%( self._label, version, domainSE )
                                break
                            validVersion = version
                            if debug:
                                print '%s DEBUG: Valid version set to \'v%i\''%( self._label, validVersion )
                        if numberOfFiles == 0:
                            break
                        # protect from double entries ( 'unique' flag in query does not work here)
                        if not filePath in filePathsTmp:
                            filePathsTmp.append( filePath )
                            if debug:
                                print '%s DEBUG: File \'%s\' found'%( self._label, filePath )
                            fileCount += 1
                            # needed, since and "limit" overrides "idx" in 'get_data' (==> "idx" set to '0' rather than "skipFiles")
                            if fileCount > skipFiles:
                                filePaths.append( filePath )
                        elif debug:
                            print '%s DEBUG: File \'%s\' found again'%( self._label, filePath )
                if validVersion > 0:
                    if numberOfFiles == 0 and debug:
                        print '%s DEBUG: No files requested'%( self._label )
                    break
        else:
            if debug:
                print '%s DEBUG: Using DBS query'%( self._label )
            for version in range( maxVersions, 0, -1 ):
                filePaths = []
                fileCount = 0
                dataset = '/%s/%s-%s-v%i/%s'%( relVal, cmsswVersion, globalTag, version, dataTier )
                dbsQuery = 'find file where dataset = %s'%( dataset )
                if debug:
                    print '%s DEBUG: Querying dataset \'%s\' with'%( self._label, dataset )
                    print '    \'%s\''%( dbsQuery )
                foundSE = False
                for line in os.popen( 'dbs search --query="%s"'%( dbsQuery ) ):
                    if line.find( '.root' ) != -1:
                        if validVersion != version:
                            if not foundSE:
                                dbsSiteQuery = 'find dataset where dataset = %s and site = %s'%( dataset, domainSE )
                                if debug:
                                    print '%s DEBUG: Querying site \'%s\' with'%( self._label, domainSE )
                                    print '    \'%s\''%( dbsSiteQuery )
                                for lineSite in os.popen( 'dbs search --query="%s"'%( dbsSiteQuery ) ):
                                    if lineSite.find( dataset ) != -1:
                                        foundSE = True
                                        break
                            if not foundSE:
                                if debug:
                                    print '%s DEBUG: Possible version \'v%s\' not available on SE \'%s\''%( self._label, version, domainSE )
                                break
                            validVersion = version
                            if debug:
                                print '%s DEBUG: Valid version set to \'v%i\''%( self._label, validVersion )
                        if numberOfFiles == 0:
                            break
                        filePath = line.replace( '\n', '' )
                        if debug:
                            print '%s DEBUG: File \'%s\' found'%( self._label, filePath )
                        fileCount += 1
                        if fileCount > skipFiles:
                            filePaths.append( filePath )
                        if not numberOfFiles < 0:
                            if numberOfFiles <= len( filePaths ):
                                break
                if validVersion > 0:
                    if numberOfFiles == 0 and debug:
                        print '%s DEBUG: No files requested'%( self._label )
                    break

        # Check output and return
        if validVersion == 0:
            print '%s INFO : No RelVal file(s) found at all in datasets \'%s*\' on SE \'%s\''%( self._label, datasetAll, domainSE )
            if debug:
                self.messageEmptyList()
        elif len( filePaths ) == 0:
            print '%s INFO : No RelVal file(s) picked up in dataset \'%s\''%( self._label, dataset )
            if debug:
                self.messageEmptyList()
        elif len( filePaths ) < numberOfFiles:
            print '%s INFO : Only %i RelVal file(s) instead of %i picked up in dataset \'%s\''%( self._label, len( filePaths ), numberOfFiles, dataset )

        if debug:
            print '%s DEBUG: returning %i file(s):\n%s'%( self._label, len( filePaths ), filePaths )
        return filePaths

pickRelValInputFiles = PickRelValInputFiles()
