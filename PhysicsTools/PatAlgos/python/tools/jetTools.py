from FWCore.GuiBrowsers.ConfigToolBase import *

from PhysicsTools.PatAlgos.tools.helpers import *


class RunBTagging(ConfigToolBase):

    """ Define sequence to run b tagging on AOD input for a given jet
    collection including a JetTracksAssociatorAtVertex module with
    name 'jetTracksAssociatorAtVertex' + 'label'
    
    Return value is a pair of (sequence, labels) where 'sequence' is
    the cms.Sequence, and 'labels' is a vector with the following
    entries:
     * labels['jta']      = the name of the JetTrackAssociator module
     * labels['tagInfos'] = a list of names of the TagInfo modules
     * labels['jetTags '] = a list of names of the JetTag modules
    """
    _label='runBTagging'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'jetCollection',self._defaultValue, 'input jet collection',Type=cms.InputTag)
        self.addParameter(self._defaultParameters,'label',self._defaultValue, 'postfix label to identify new sequence/modules', Type=str)
        self.addParameter(self._defaultParameters,'postfix',"", "postfix of default sequence (do not confuse with 'label')")
        self.addParameter(self._defaultParameters,'btagInfo',['impactParameterTagInfos','secondaryVertexTagInfos','softMuonTagInfos','secondaryVertexNegativeTagInfos'],"input btag info",allowedValues=['impactParameterTagInfos','secondaryVertexTagInfos','softMuonTagInfos','secondaryVertexNegativeTagInfos'],Type=list)
        self.addParameter(self._defaultParameters,'btagdiscriminators',['jetBProbabilityBJetTags', 'jetProbabilityBJetTags','trackCountingHighPurBJetTags','trackCountingHighEffBJetTags','simpleSecondaryVertexHighEffBJetTags','simpleSecondaryVertexHighPurBJetTags','combinedSecondaryVertexBJetTags','combinedSecondaryVertexMVABJetTags','softMuonBJetTags','softMuonByPtBJetTags','softMuonByIP3dBJetTags','simpleSecondaryVertexNegativeHighEffBJetTags','simpleSecondaryVertexNegativeHighPurBJetTags','negativeTrackCountingHighEffJetTags','negativeTrackCountingHighPurJetTags'],"input btag discriminators", allowedValues=['jetBProbabilityBJetTags', 'jetProbabilityBJetTags', 'trackCountingHighPurBJetTags', 'trackCountingHighEffBJetTags', 'simpleSecondaryVertexHighEffBJetTags','simpleSecondaryVertexHighPurBJetTags','combinedSecondaryVertexBJetTags','combinedSecondaryVertexMVABJetTags','softMuonBJetTags','softMuonByPtBJetTags','softMuonByIP3dBJetTags','simpleSecondaryVertexNegativeHighEffBJetTags','simpleSecondaryVertexNegativeHighPurBJetTags','negativeTrackCountingHighEffJetTags','negativeTrackCountingHighPurJetTags'],Type=list)
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters
 

    def __call__(self,process,
                 jetCollection      = None,
                 label              = None,
                 postfix            = None,
                 btagInfo           = None,
                 btagdiscriminators = None) :



        if  jetCollection is None:
            jetCollection=self._defaultParameters['jetCollection'].value
        if  label is None:
            label=self._defaultParameters['label'].value
        if  postfix  is None:
            postfix=self._defaultParameters['postfix'].value
        if  btagInfo  is None:
            btagInfo=self._defaultParameters['btagInfo'].value
        if  btagdiscriminators is None:
            btagdiscriminators=self._defaultParameters['btagdiscriminators'].value



        self.setParameter('jetCollection',jetCollection)
        self.setParameter('label',label)
        self.setParameter('postfix',postfix)
        self.setParameter('btagInfo',btagInfo)
        self.setParameter('btagdiscriminators',btagdiscriminators)


        return self.apply(process) 
        
    def apply(self, process):
        jetCollection=self._parameters['jetCollection'].value
        label=self._parameters['label'].value
        postfix=self._parameters['postfix'].value
        btagInfo=self._parameters['btagInfo'].value
        btagdiscriminators=self._parameters['btagdiscriminators'].value


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
        ipTILabel = 'impactParameterTagInfos'     + label + postfix
        svTILabel = 'secondaryVertexTagInfos'     + label + postfix
        nvTILabel = 'secondaryVertexNegativeTagInfos'     + label + postfix
        #seTILabel = 'softElectronTagInfos'        + label + postfix
        smTILabel = 'softMuonTagInfos'            + label + postfix
    
        ## produce tag infos
        #setattr( process, ipTILabel, btag.impactParameterTagInfos.clone(jetTracks = cms.InputTag(jtaLabel)) )
        #setattr( process, svTILabel, btag.secondaryVertexTagInfos.clone(trackIPTagInfos = cms.InputTag(ipTILabel)) )
        #setattr( process, nvTILabel, nbtag.secondaryVertexNegativeTagInfos.clone(trackIPTagInfos = cms.InputTag(ipTILabel)) )
        #setattr( process, seTILabel, btag.softElectronTagInfos.clone(jets = jetCollection) )
        #setattr( process, smTILabel, btag.softMuonTagInfos.clone(jets = jetCollection) )

        ## make VInputTag from strings
        def vit(*args) : return cms.VInputTag( *[ cms.InputTag(x) for x in args ] )
    
        ## produce btags
        print       
        print "The btaginfo below will be written to the jet collection in the PATtuple (default is all, see PatAlgos/PhysicsTools/python/tools/jetTools.py)"
        print
        for TagInfo in btagInfo:
                print TagInfo

                if TagInfo=='impactParameterTagInfos':
                        setattr( process, ipTILabel, btag.impactParameterTagInfos.clone(jetTracks = cms.InputTag(jtaLabel)) )
                if TagInfo=='secondaryVertexTagInfos':
                        setattr( process, svTILabel, btag.secondaryVertexTagInfos.clone(trackIPTagInfos = cms.InputTag(ipTILabel)) )
                if TagInfo=='softMuonTagInfos':
                        setattr( process, smTILabel, btag.softMuonTagInfos.clone(jets = jetCollection) )

                if TagInfo=='secondaryVertexNegativeTagInfos':
                        setattr( process, nvTILabel, btag.secondaryVertexNegativeTagInfos.clone(trackIPTagInfos = cms.InputTag(ipTILabel)) )

        print
        print "The bdiscriminators below will be written to the jet collection in the PATtuple (default is all, see PatAlgos/PhysicsTools/python/tools/jetTools.py)"
        for tag in btagdiscriminators:
                print  tag
                if tag == 'jetBProbabilityBJetTags' or tag == 'jetProbabilityBJetTags' or tag == 'trackCountingHighPurBJetTags' or tag == 'trackCountingHighEffBJetTags' or tag == 'negativeTrackCountingHighEffJetTags' or  tag =='negativeTrackCountingHighPurJetTags' :
                        tagInfo=getattr(btag,tag).clone(tagInfos = vit(ipTILabel))
                        setattr(process, tag+label+postfix, tagInfo)

                if tag == 'simpleSecondaryVertexHighEffBJetTags' or tag == 'simpleSecondaryVertexHighPurBJetTags':
#                       tagInfo=getattr(btag,tag).clone(tagInfos = vit(ipTILabel,svTILabel))
                        tagInfo=getattr(btag,tag).clone(tagInfos = vit(svTILabel))
                        setattr(process, tag+label+postfix, tagInfo)

                if tag == 'combinedSecondaryVertexBJetTags' or tag == 'combinedSecondaryVertexMVABJetTags':
                        tagInfo=getattr(btag,tag).clone(tagInfos = vit(ipTILabel,svTILabel))
                        setattr(process, tag+label+postfix, tagInfo)

                if tag == 'softMuonBJetTags' or tag == 'softMuonByPtBJetTags' or tag == 'softMuonByIP3dBJetTags':
                        tagInfo=getattr(btag,tag).clone(tagInfos = vit(smTILabel))
                        setattr(process, tag+label+postfix, tagInfo)


                if tag == 'simpleSecondaryVertexNegativeHighEffBJetTags' or tag == 'simpleSecondaryVertexNegativeHighPurBJetTags':
                        tagInfo=getattr(btag,tag).clone(tagInfos = vit(nvTILabel))
                        setattr(process, tag+label+postfix, tagInfo)


        ## define vector of (output) labels
        labels = { 'jta'      : jtaLabel,
                   #'tagInfos' : (ipTILabel,svTILabel,seTILabel,smTILabel),
                #'tagInfos' : (ipTILabel,svTILabel,smTILabel), 
                #'tagInfos' : [ for y in btagInfo],
                 'tagInfos' : [(y + label + postfix) for y in btagInfo],
                #'tagInfos' : (ipTILabel,svTILabel,smTILabel),
                'jetTags'  : [ (x + label+postfix) for x in btagdiscriminators]

                   }



 
        ## extend an existing sequence by otherLabels
        def mkseq(process, firstlabel, *otherlabels):
            seq = getattr(process, firstlabel)
            for x in otherlabels: seq += getattr(process, x)
            return cms.Sequence(seq)

        ## add tag infos to the process
        setattr( process, 'btaggingTagInfos'+label+postfix, mkseq(process, *(labels['tagInfos']) ) )
        ## add b tags to the process
        setattr( process, 'btaggingJetTags'+label+postfix,  mkseq(process, *(labels['jetTags'])  ) )
        ## add a combined sequence to the process
        seq = mkseq(process, 'btaggingTagInfos'+label+postfix, 'btaggingJetTags' + label + postfix)
        setattr( process, 'btagging'+label+postfix, seq )
        ## return the combined sequence and the labels defined above

        if hasattr(process, "addAction"):
            process.enableRecording()
            action=self.__copy__()
            process.addAction(action)
        return (seq, labels)

runBTagging=RunBTagging()


class AddJetCollection(ConfigToolBase):

    """ Add a new collection of jets. Takes the configuration from the
    already configured standard jet collection as starting point;
    replaces before calling addJetCollection will also affect the
    new jet collections
    """
    _label='addJetCollection'
    _defaultParameters=dicttypes.SortedKeysDict()

    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'jetCollection',self._defaultValue,'Input jet collection', cms.InputTag)
        self.addParameter(self._defaultParameters,'algoLabel',self._defaultValue, "label to indicate the jet algorithm (e.g.'AK5')",str)
        self.addParameter(self._defaultParameters,'typeLabel',self._defaultValue, "label to indicate the type of constituents (e.g. 'Calo', 'Pflow', 'Jpt', ...)",str)
        self.addParameter(self._defaultParameters,'btagInfo',['impactParameterTagInfos','secondaryVertexTagInfos','softMuonTagInfos','secondaryVertexNegativeTagInfos'],"input btag info",allowedValues=['impactParameterTagInfos','secondaryVertexTagInfos','softMuonTagInfos','secondaryVertexNegativeTagInfos'],Type=list)
        self.addParameter(self._defaultParameters,'btagdiscriminators',['jetBProbabilityBJetTags', 'jetProbabilityBJetTags','trackCountingHighPurBJetTags','trackCountingHighEffBJetTags','simpleSecondaryVertexHighEffBJetTags','simpleSecondaryVertexHighPurBJetTags','combinedSecondaryVertexBJetTags','combinedSecondaryVertexMVABJetTags','softMuonBJetTags','softMuonByPtBJetTags','softMuonByIP3dBJetTags','simpleSecondaryVertexNegativeHighEffBJetTags','simpleSecondaryVertexNegativeHighPurBJetTags','negativeTrackCountingHighEffJetTags','negativeTrackCountingHighPurJetTags'],"input btag discriminators", allowedValues=['jetBProbabilityBJetTags', 'jetProbabilityBJetTags','trackCountingHighPurBJetTags', 'trackCountingHighEffBJetTags','simpleSecondaryVertexHighEffBJetTags','simpleSecondaryVertexHighPurBJetTags','combinedSecondaryVertexBJetTags','combinedSecondaryVertexMVABJetTags','softMuonBJetTags','softMuonByPtBJetTags','softMuonByIP3dBJetTags','simpleSecondaryVertexNegativeHighEffBJetTags','simpleSecondaryVertexNegativeHighPurBJetTags','negativeTrackCountingHighEffJetTags','negativeTrackCountingHighPurJetTags'],Type=list)
        self.addParameter(self._defaultParameters,'doJTA',True, "run b tagging sequence for new jet collection and add it to the new pat jet collection")
        self.addParameter(self._defaultParameters,'doBTagging',True, 'run JetTracksAssociation and JetCharge and add it to the new pat jet collection (will autom. be true if doBTagging is set to true)')
        self.addParameter(self._defaultParameters,'jetCorrLabel',None, "payload and list of new jet correction labels, such as (\'AK5Calo\',[\'L2Relative\', \'L3Absolute\'])", tuple,acceptNoneValue=True )
        self.addParameter(self._defaultParameters,'doType1MET',True, "if jetCorrLabel is not 'None', set this to 'True' to redo the Type1 MET correction for the new jet colllection; at the moment it must be 'False' for non CaloJets otherwise the JetMET POG module crashes. ")
        self.addParameter(self._defaultParameters,'doL1Cleaning',True, "copy also the producer modules for cleanLayer1 will be set to 'True' automatically when doL1Counters is 'True'")
        self.addParameter(self._defaultParameters,'doL1Counters',False, "copy also the filter modules that accept/reject the event looking at the number of jets")
        self.addParameter(self._defaultParameters,'genJetCollection',cms.InputTag("ak5GenJets"), "GenJet collection to match to")
        self.addParameter(self._defaultParameters,'doJetID',True, "add jetId variables to the added jet collection?")
        self.addParameter(self._defaultParameters,'jetIdLabel',"ak5", " specify the label prefix of the xxxJetID object; in general it is the jet collection tag like ak5, kt4 sc5, aso. For more information have a look to SWGuidePATTools#add_JetCollection")
        self.addParameter(self._defaultParameters,'standardAlgo',"AK5", "standard algorithm label of the collection from which the clones for the new jet collection will be taken from (note that this jet collection has to be available in the event before hand)")
        self.addParameter(self._defaultParameters,'standardType',"Calo", "standard constituent type label of the collection from which the clones for the new jet collection will be taken from (note that this jet collection has to be available in the event before hand)")
        self.addParameter(self._defaultParameters, 'outputModules', ['out'], "output module labels, empty list of label indicates no output, default: ['out']")

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
                 outputModule       = None,
                 outputModules      = None,
                 btagInfo           = None,
                 btagdiscriminators = None
                                      ):

        ## stop processing if 'outputModule' exists and show the new alternative
        if  not outputModule is None:
            depricatedOptionOutputModule(self)
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
        if outputModules is None:
            outputModules=self._defaultParameters['outputModules'].value     
        if  btagInfo is None:
            btagInfo=self._defaultParameters['btagInfo'].value
        if  btagdiscriminators is None:
            btagdiscriminators=self._defaultParameters['btagdiscriminators'].value

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
        self.setParameter('outputModules',outputModules)
        self.setParameter('btagInfo',btagInfo)
        self.setParameter('btagdiscriminators',btagdiscriminators)
 
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
        outputModules=self._parameters['outputModules'].value
        btagInfo=self._parameters['btagInfo'].value
        btagdiscriminators=self._parameters['btagdiscriminators'].value


        ## create old module label from standardAlgo
        ## and standardType and return
        def oldLabel(prefix=''):        
            return jetCollectionString(prefix, '', '')

        ## create new module label from old module
        ## label and return
        def newLabel(oldLabel):
            newLabel=oldLabel
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
            process.patDefaultSequence.replace(process.patJetCharge, getattr(process,jtaLabel)+process.patJetCharge)
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
            (btagSeq, btagLabels) = runBTagging(process, jetCollection, postfixLabel,"", btagInfo,btagdiscriminators) 
            ## add b tagging sequence before running the allLayer1Jets modules
            process.patDefaultSequence.replace(getattr(process,jtaLabel), getattr(process,jtaLabel)+btagSeq)
            ## replace corresponding tags for pat jet production
            l1Jets.trackAssociationSource = cms.InputTag(btagLabels['jta'])
            l1Jets.tagInfoSources = cms.VInputTag( *[ cms.InputTag(x) for x in btagLabels['tagInfos'] ] )
            l1Jets.discriminatorSources = cms.VInputTag( *[ cms.InputTag(x) for x in btagLabels['jetTags']  ] )
        else:
            ## switch general b tagging info switch off
            l1Jets.addBTagInfo = False
            ## adjust output
            if len(outputModules) > 0:
                for outMod in outputModules:
                    if hasattr(process,outMod):
                        getattr(process,outMod).outputCommands.append("drop *_"+newLabel(oldLabel('selected'))+"_tagInfos_*")
                    else:
                        raise KeyError, "process has no OutModule named", outMod

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
            if type(jetCorrLabel) != type(('AK5Calo',['L2Relative'])): 
                raise ValueError, "In addJetCollection 'jetCorrLabel' must be 'None', or of type ('payload',['correction1', 'correction2'])"            

            ## add clone of jetCorrFactors
            addClone('patJetCorrFactors', src = jetCollection)
            switchJetCorrLevels(process, jetCorrLabel = jetCorrLabel, postfix=algoLabel+typeLabel)
            getattr(process, newLabel('patJets')).jetCorrFactorsSource = cms.VInputTag(  cms.InputTag(newLabel('patJetCorrFactors')) )
        
            ## find out type of jet collection, switch type1MET corrections off for JPTJets
            jetCollType = ''
            if   ( 'CaloJets' in jetCollection.getModuleLabel() ):
                jetCollType = 'Calo'
            elif ( 'PFJets' in jetCollection.getModuleLabel() or jetCollection.getModuleLabel().startswith('pfNo') or jetCollection.getModuleLabel() == 'particleFlow'):
                jetCollType = 'PF'
            else:
                print '============================================='
                print 'Type1MET corrections are switched off for    '
                print 'JPT Jets. Users are recommened to use tcMET  '
                print 'together with JPT jets.                      '
                print '============================================='
                doType1MET=False

            ## add a clone of the type1MET correction for the new jet collection
            if (doType1MET):
                ## create jet correctors for MET corrections
                from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import ak5PFL1Fastjet, ak5PFL1Offset, ak5PFL2Relative, ak5PFL3Absolute, ak5PFResidual
                setattr(process, jetCorrLabel[0]+'L1FastJet'   , ak5PFL1Fastjet.clone ( algorithm=jetCorrLabel[0]
                                                                                      , srcRho=cms.InputTag(newLabel('kt6'+jetCollType+'Jets'),'rho') ) )
                setattr(process, jetCorrLabel[0]+'L1Offset'    , ak5PFL1Offset.clone  ( algorithm=jetCorrLabel[0] ) )
                setattr(process, jetCorrLabel[0]+'L2Relative'  , ak5PFL2Relative.clone( algorithm=jetCorrLabel[0] ) )
                setattr(process, jetCorrLabel[0]+'L3Absolute'  , ak5PFL3Absolute.clone( algorithm=jetCorrLabel[0] ) )
                setattr(process, jetCorrLabel[0]+'L2L3Residual', ak5PFResidual.clone  ( algorithm=jetCorrLabel[0] ) )
                ## combinded corrections
                setattr(process, jetCorrLabel[0]+'CombinedCorrector', cms.ESProducer( 'JetCorrectionESChain'
                                                                                  , correctors = cms.vstring() ) )
                
                for corrLbl in jetCorrLabel[1]:
                    if corrLbl != 'L1FastJet' and corrLbl != 'L1Offset' and corrLbl != 'L2Relative' and corrLbl != 'L3Absolute' and corrLbl != 'L2L3Residual':
                        print '========================================='
                        print ' Type1MET corrections are currently only  '
                        print ' supported for the following corrections: '
                        print '   - L1FastJet'
                        print '   - L1Offset'
                        print '   - L2Relative'
                        print '   - L3Absolute'
                        print '   - L2L3Residual'
                        print ' But given was:'
                        print '   -', corrLbl
                        print '============================================='
                        raise ValueError, 'unsupported JEC for TypeI MET correction: '+corrLbl
                    else:
                        getattr(process, jetCorrLabel[0]+'CombinedCorrector').correctors.append(jetCorrLabel[0]+corrLbl)

                ## configuration of MET corrections
                if jetCollType == 'Calo':
                    from JetMETCorrections.Type1MET.caloMETCorrections_cff import caloJetMETcorr,caloType1CorrectedMet,caloType1p2CorrectedMet,produceCaloMETCorrections
                    
                    setattr(process,jetCorrLabel[0]+'JetMETcorr',   caloJetMETcorr.clone(srcMET       = "corMetGlobalMuons"))
                    setattr(process,jetCorrLabel[0]+'Type1CorMet',  caloType1CorrectedMet.clone(src   = "corMetGlobalMuons"))
                    setattr(process,jetCorrLabel[0]+'Type1p2CorMet',caloType1p2CorrectedMet.clone(src = "corMetGlobalMuons"))
                    
                    getattr(process,jetCorrLabel[0]+'JetMETcorr'   ).src          = cms.InputTag(jetCollection.getModuleLabel())
                    if ('L1FastJet' in jetCorrLabel[1] or 'L1Fastjet' in jetCorrLabel[1]):
                        getattr(process,jetCorrLabel[0]+'JetMETcorr'   ).offsetCorrLabel = cms.string(jetCorrLabel[0]+'L1FastJet')
                    elif ('L1Offset' in jetCorrLabel[1]):
                        getattr(process,jetCorrLabel[0]+'JetMETcorr'   ).offsetCorrLabel = cms.string(jetCorrLabel[0]+'L1Offset')
                    else:
                        getattr(process,jetCorrLabel[0]+'JetMETcorr'   ).offsetCorrLabel = cms.string('')
                    getattr(process,jetCorrLabel[0]+'JetMETcorr'   ).jetCorrLabel = cms.string(jetCorrLabel[0]+'CombinedCorrector')

                    getattr(process,jetCorrLabel[0]+'Type1CorMet'  ).srcType1Corrections = cms.VInputTag(
                        cms.InputTag(jetCorrLabel[0]+'JetMETcorr', 'type1')
                        )
                    
                    getattr(process,jetCorrLabel[0]+'Type1p2CorMet').srcType1Corrections = cms.VInputTag(
                        cms.InputTag(jetCorrLabel[0]+'JetMETcorr', 'type1')
                        )
                    getattr(process,jetCorrLabel[0]+'Type1p2CorMet').srcUnclEnergySums = cms.VInputTag(
                        cms.InputTag(jetCorrLabel[0]+'JetMETcorr', 'type2'),
                        cms.InputTag(jetCorrLabel[0]+'JetMETcorr', 'offset'),
                        cms.InputTag('muonCaloMETcorr')
                        )

                    ## add MET corrections to sequence
                    addClone('patMETs', metSource = cms.InputTag(jetCorrLabel[0]+'Type1CorMet'))
                    
                    setattr(process,'produce'+jetCorrLabel[0]+'METCorrections',produceCaloMETCorrections.copy())
                    getattr(process,'produce'+jetCorrLabel[0]+'METCorrections').replace(getattr(process,'caloJetMETcorr'),         getattr(process,jetCorrLabel[0]+'JetMETcorr'))
                    getattr(process,'produce'+jetCorrLabel[0]+'METCorrections').replace(getattr(process,'caloType1CorrectedMet'),  getattr(process,jetCorrLabel[0]+'Type1CorMet'))
                    getattr(process,'produce'+jetCorrLabel[0]+'METCorrections').replace(getattr(process,'caloType1p2CorrectedMet'),getattr(process,jetCorrLabel[0]+'Type1p2CorMet'))
                    process.patDefaultSequence.replace( getattr(process,newLabel('patMETs')),
                                                        getattr(process,'produce'+jetCorrLabel[0]+'METCorrections')
                                                        *getattr(process,newLabel('patMETs')) )                    
                elif jetCollType == 'PF':
                    from JetMETCorrections.Type1MET.pfMETCorrections_cff import pfCandsNotInJet,pfJetMETcorr,pfCandMETcorr,pfType1CorrectedMet,pfType1p2CorrectedMet,producePFMETCorrections
                    setattr(process,jetCorrLabel[0]+'CandsNotInJet',pfCandsNotInJet.clone(topCollection = jetCollection))
                    setattr(process,jetCorrLabel[0]+'JetMETcorr',   pfJetMETcorr.clone(src              = jetCollection))
                    setattr(process,jetCorrLabel[0]+'CandMETcorr',  pfCandMETcorr.clone(src             = cms.InputTag(jetCorrLabel[0]+'CandsNotInJet')))
                    setattr(process,jetCorrLabel[0]+'Type1CorMet',  pfType1CorrectedMet.clone())
                    setattr(process,jetCorrLabel[0]+'Type1p2CorMet',pfType1p2CorrectedMet.clone())

                    if ('L1FastJet' in jetCorrLabel[1] or 'L1Fastjet' in jetCorrLabel[1]):
                        getattr(process,jetCorrLabel[0]+'JetMETcorr'   ).offsetCorrLabel = cms.string(jetCorrLabel[0]+'L1FastJet')
                    elif ('L1Offset' in jetCorrLabel[1]):
                        getattr(process,jetCorrLabel[0]+'JetMETcorr'   ).offsetCorrLabel = cms.string(jetCorrLabel[0]+'L1Offset')
                    else:
                        getattr(process,jetCorrLabel[0]+'JetMETcorr'   ).offsetCorrLabel = cms.string('')
                    getattr(process,jetCorrLabel[0]+'JetMETcorr').jetCorrLabel    = cms.string(jetCorrLabel[0]+'CombinedCorrector')

                    getattr(process,jetCorrLabel[0]+'Type1CorMet').srcType1Corrections = cms.VInputTag(
                        cms.InputTag(jetCorrLabel[0]+'JetMETcorr', 'type1')
                        )
                    getattr(process,jetCorrLabel[0]+'Type1p2CorMet').srcType1Corrections = cms.VInputTag(
                        cms.InputTag(jetCorrLabel[0]+'JetMETcorr', 'type1')
                        )
                    getattr(process,jetCorrLabel[0]+'Type1p2CorMet').srcUnclEnergySums = cms.VInputTag(
                        cms.InputTag(jetCorrLabel[0]+'JetMETcorr', 'type2'),
                        cms.InputTag(jetCorrLabel[0]+'JetMETcorr', 'offset'),
                        cms.InputTag(jetCorrLabel[0]+'CandMETcorr')                                    
                        )

                    ## add MET corrections to sequence
                    addClone('patMETs', metSource = cms.InputTag(jetCorrLabel[0]+'Type1CorMet'), addMuonCorrections = False)
                    setattr(process,'produce'+jetCorrLabel[0]+'METCorrections',producePFMETCorrections.copy())
                    getattr(process,'produce'+jetCorrLabel[0]+'METCorrections').replace(getattr(process,'pfCandsNotInJet'),      getattr(process,jetCorrLabel[0]+'CandsNotInJet'))
                    getattr(process,'produce'+jetCorrLabel[0]+'METCorrections').replace(getattr(process,'pfJetMETcorr'),         getattr(process,jetCorrLabel[0]+'JetMETcorr'))
                    getattr(process,'produce'+jetCorrLabel[0]+'METCorrections').replace(getattr(process,'pfCandMETcorr'),        getattr(process,jetCorrLabel[0]+'CandMETcorr'))
                    getattr(process,'produce'+jetCorrLabel[0]+'METCorrections').replace(getattr(process,'pfType1CorrectedMet'),  getattr(process,jetCorrLabel[0]+'Type1CorMet'))
                    getattr(process,'produce'+jetCorrLabel[0]+'METCorrections').replace(getattr(process,'pfType1p2CorrectedMet'),getattr(process,jetCorrLabel[0]+'Type1p2CorMet'))
                    process.patDefaultSequence.replace( getattr(process,newLabel('patMETs')),
                                                        getattr(process,'produce'+jetCorrLabel[0]+'METCorrections')
                                                        *getattr(process,newLabel('patMETs')) )
        else:
            ## switch jetCorrFactors off
            l1Jets.addJetCorrFactors = False
               
addJetCollection=AddJetCollection()


class SwitchJetCollection(ConfigToolBase):

    """ Switch the collection of jets in PAT from the default value to a
    new jet collection
    """
    _label='switchJetCollection'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'jetCollection',self._defaultValue,'Input jet collection', cms.InputTag)
        self.addParameter(self._defaultParameters,'btagInfo',['impactParameterTagInfos','secondaryVertexTagInfos','softMuonTagInfos','secondaryVertexNegativeTagInfos'],"input btag info",allowedValues=['impactParameterTagInfos','secondaryVertexTagInfos','softMuonTagInfos','secondaryVertexNegativeTagInfos'],Type=list)
        self.addParameter(self._defaultParameters, 'btagdiscriminators',['jetBProbabilityBJetTags','jetProbabilityBJetTags','trackCountingHighPurBJetTags','trackCountingHighEffBJetTags','simpleSecondaryVertexHighEffBJetTags','simpleSecondaryVertexHighPurBJetTags','combinedSecondaryVertexBJetTags','combinedSecondaryVertexMVABJetTags','softMuonBJetTags','softMuonByPtBJetTags','softMuonByIP3dBJetTags','simpleSecondaryVertexNegativeHighEffBJetTags','simpleSecondaryVertexNegativeHighPurBJetTags','negativeTrackCountingHighEffJetTags','negativeTrackCountingHighPurJetTags'],"input btag discriminators", allowedValues=['jetBProbabilityBJetTags', 'jetProbabilityBJetTags','trackCountingHighPurBJetTags', 'trackCountingHighEffBJetTags','simpleSecondaryVertexHighEffBJetTags','simpleSecondaryVertexHighPurBJetTags','combinedSecondaryVertexBJetTags','combinedSecondaryVertexMVABJetTags','softMuonBJetTags','softMuonByPtBJetTags','softMuonByIP3dBJetTags','simpleSecondaryVertexNegativeHighEffBJetTags','simpleSecondaryVertexNegativeHighPurBJetTags','negativeTrackCountingHighEffJetTags','negativeTrackCountingHighPurJetTags'],Type=list)
	self.addParameter(self._defaultParameters,'doJTA',True, "run b tagging sequence for new jet collection and add it to the new pat jet collection")
        self.addParameter(self._defaultParameters,'doBTagging',True, 'run JetTracksAssociation and JetCharge and add it to the new pat jet collection (will autom. be true if doBTagging is set to true)')
        self.addParameter(self._defaultParameters,'jetCorrLabel',None, "payload and list of new jet correction labels, such as (\'AK5Calo\',[\'L2Relative\', \'L3Absolute\'])", tuple,acceptNoneValue=True )
        self.addParameter(self._defaultParameters,'doType1MET',True, "if jetCorrLabel is not 'None', set this to 'True' to redo the Type1 MET correction for the new jet colleection; at the moment it must be 'False' for non CaloJets otherwise the JetMET POG module crashes. ")
        self.addParameter(self._defaultParameters,'genJetCollection',cms.InputTag("ak5GenJets"), "GenJet collection to match to")
        self.addParameter(self._defaultParameters,'doJetID',True, "add jetId variables to the added jet collection")
        self.addParameter(self._defaultParameters,'jetIdLabel',"ak5", " specify the label prefix of the xxxJetID object; in general it is the jet collection tag like ak5, kt4 sc5, aso. For more information have a look to SWGuidePATTools#add_JetCollection")
        self.addParameter(self._defaultParameters,'postfix',"", "postfix of default sequence")
        self.addParameter(self._defaultParameters, 'outputModules', ['out'], "Output module labels, empty list of label indicates no output, default: ['out']")
        
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
                 postfix            = None,
                 outputModule       = None,
                 outputModules      = None,
                 btagInfo           = None,
                 btagdiscriminators = None
					):	
        


         
        ## stop processing if 'outputModule' exists and show the new alternative
        if  not outputModule is None:
            depricatedOptionOutputModule(self)
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
        if outputModules is None:
            outputModules=self._defaultParameters['outputModules'].value     
        if postfix  is None:
            postfix=self._defaultParameters['postfix'].value
        if  btagInfo is None:
             btagInfo=self._defaultParameters['btagInfo'].value
        if  btagdiscriminators is None:
             btagdiscriminators=self._defaultParameters['btagdiscriminators'].value




        self.setParameter('jetCollection',jetCollection)
        self.setParameter('doJTA',doJTA)
        self.setParameter('doBTagging',doBTagging)
        self.setParameter('jetCorrLabel',jetCorrLabel)
        self.setParameter('doType1MET',doType1MET)
        self.setParameter('genJetCollection',genJetCollection)
        self.setParameter('doJetID',doJetID)
        self.setParameter('jetIdLabel',jetIdLabel)
        self.setParameter('outputModules',outputModules)
        self.setParameter('postfix',postfix)
        self.setParameter('btagInfo',btagInfo)
        self.setParameter('btagdiscriminators',btagdiscriminators)
        
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
        outputModules=self._parameters['outputModules'].value
        postfix=self._parameters['postfix'].value
        btagInfo=self._parameters['btagInfo'].value
        btagdiscriminators=self._parameters['btagdiscriminators'].value

        ## save label of old input jet collection
        oldLabel = applyPostfix(process, "patJets", postfix).jetSource;
    
        ## replace input jet collection for generator matches if the
        ## genJetCollection is no empty
        if (process.patJets.addGenPartonMatch):
            applyPostfix(process, "patJetPartonMatch", postfix).src = jetCollection
        if (process.patJets.addGenJetMatch):
            applyPostfix(process, "patJetGenJetMatch", postfix).src = jetCollection
            applyPostfix(process, "patJetGenJetMatch", postfix).matched = genJetCollection
        if (process.patJets.getJetMCFlavour):
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
            ##(btagSeq, btagLabels) = runBTagging(process, jetCollection, 'AOD',postfix)
            (btagSeq, btagLabels) = runBTagging(process, jetCollection,"AOD",postfix,btagInfo,btagdiscriminators)
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
            ## adjust output
            if len(outputModules) > 0:
                for outMod in outputModules:
                    if hasattr(process,outMod):
                        getattr(process,outMod).outputCommands.append("drop *_selectedPatJets_tagInfos_*")
                    else:
                        raise KeyError, "process has no OutModule named", outMod

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
            if type(jetCorrLabel) != type(('AK5Calo',['L2Relative'])): 
                raise ValueError, "In addJetCollection 'jetCorrLabel' must be 'None', or of type ('payload',['correction1', 'correction2'])"

            ## switch JEC parameters to the new jet collection
            applyPostfix(process, "patJetCorrFactors", postfix).src = jetCollection
            switchJetCorrLevels(process, jetCorrLabel = jetCorrLabel, postfix=postfix)
            getattr( process, "patJets" + postfix).jetCorrFactorsSource = cms.VInputTag( cms.InputTag("patJetCorrFactors" + postfix ) )  

            ## find out type of jet collection, switch type1MET corrections off for JPTJets
            jetCollType = ''
            if   ( 'CaloJets' in jetCollection.getModuleLabel() ):
                jetCollType = 'Calo'
            elif ( 'PFJets' in jetCollection.getModuleLabel() or jetCollection.getModuleLabel().startswith('pfNo') or jetCollection.getModuleLabel() == 'particleFlow'):
                jetCollType = 'PF'
            else:
                print '============================================='
                print 'Type1MET corrections are switched off for    '
                print 'JPT Jets. Users are recommened to use tcMET  '
                print 'together with JPT jets.                      '
                print '============================================='
                doType1MET=False

            ## add a clone of the type1MET correction for the new jet collection
            if (doType1MET):
                ## create jet correctors for MET corrections
                from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import ak5PFL1Fastjet, ak5PFL1Offset, ak5PFL2Relative, ak5PFL3Absolute, ak5PFResidual
                setattr(process, jetCorrLabel[0]+'L1FastJet'   , ak5PFL1Fastjet.clone ( algorithm=jetCorrLabel[0]
                                                                                      , srcRho=cms.InputTag('kt6'+jetCollType+'Jets','rho') ) )
                setattr(process, jetCorrLabel[0]+'L1Offset'    , ak5PFL1Offset.clone  ( algorithm=jetCorrLabel[0] ) )
                setattr(process, jetCorrLabel[0]+'L2Relative'  , ak5PFL2Relative.clone( algorithm=jetCorrLabel[0] ) )
                setattr(process, jetCorrLabel[0]+'L3Absolute'  , ak5PFL3Absolute.clone( algorithm=jetCorrLabel[0] ) )
                setattr(process, jetCorrLabel[0]+'L2L3Residual', ak5PFResidual.clone  ( algorithm=jetCorrLabel[0] ) )
                ## combinded corrections
                setattr(process, jetCorrLabel[0]+'CombinedCorrector', cms.ESProducer( 'JetCorrectionESChain'
                                                                                  , correctors = cms.vstring() ) )
                for corrLbl in jetCorrLabel[1]:
                    if corrLbl != 'L1FastJet' and corrLbl != 'L1Offset' and corrLbl != 'L2Relative' and corrLbl != 'L3Absolute' and corrLbl != 'L2L3Residual':
                        print '========================================='
                        print ' Type1MET corrections are currently only  '
                        print ' supported for the following corrections: '
                        print '   - L1FastJet'
                        print '   - L1Offset'
                        print '   - L2Relative'
                        print '   - L3Absolute'
                        print '   - L2L3Residual'
                        print ' But given was:'
                        print '   -', corrLbl
                        print '============================================='
                        raise ValueError, 'unsupported JEC for TypeI MET correction: '+corrLbl
                    else:
                        getattr(process, jetCorrLabel[0]+'CombinedCorrector').correctors.append(jetCorrLabel[0]+corrLbl)

                ## configuration of MET corrections
                if jetCollType == 'Calo':
                    from JetMETCorrections.Type1MET.caloMETCorrections_cff import caloJetMETcorr,caloType1CorrectedMet,caloType1p2CorrectedMet,produceCaloMETCorrections
                    
                    setattr(process,'caloJetMETcorr'+         postfix, caloJetMETcorr.clone(srcMET       = "corMetGlobalMuons"))
                    setattr(process,'caloType1CorrectedMet'+  postfix, caloType1CorrectedMet.clone(src   = "corMetGlobalMuons"))
                    setattr(process,'caloType1p2CorrectedMet'+postfix, caloType1p2CorrectedMet.clone(src = "corMetGlobalMuons"))
                    
                    getattr(process,'caloJetMETcorr'+postfix).src          = cms.InputTag(jetCollection.getModuleLabel())
                    if ('L1FastJet' in jetCorrLabel[1] or 'L1Fastjet' in jetCorrLabel[1]):
                        getattr(process,'caloJetMETcorr'+postfix   ).offsetCorrLabel = cms.string(jetCorrLabel[0]+'L1FastJet')
                    elif ('L1Offset' in jetCorrLabel[1]):
                        getattr(process,'caloJetMETcorr'+postfix   ).offsetCorrLabel = cms.string(jetCorrLabel[0]+'L1Offset')
                    else:
                        getattr(process,'caloJetMETcorr'+postfix   ).offsetCorrLabel = cms.string('')
                    getattr(process,'caloJetMETcorr'+postfix   ).jetCorrLabel = cms.string(jetCorrLabel[0]+'CombinedCorrector')

                    getattr(process,'caloType1CorrectedMet'+postfix  ).srcType1Corrections = cms.VInputTag(
                        cms.InputTag('caloJetMETcorr'+postfix, 'type1')
                        )
                    
                    getattr(process,'caloType1p2CorrectedMet'+postfix).srcType1Corrections = cms.VInputTag(
                        cms.InputTag('caloJetMETcorr'+postfix, 'type1')
                        )
                    getattr(process,'caloType1p2CorrectedMet'+postfix).srcUnclEnergySums = cms.VInputTag(
                        cms.InputTag('caloJetMETcorr'+postfix, 'type2'),
                        cms.InputTag('caloJetMETcorr'+postfix, 'offset'),
                        cms.InputTag('muonCaloMETcorr')
                        )

                    ## add MET corrections to sequence
                    getattr(process, 'patMETs'+ postfix).metSource = cms.InputTag('caloType1CorrectedMet'+postfix)
                    getattr(process,'produceCaloMETCorrections'+postfix)
                    getattr(process,"patDefaultSequence"+postfix).replace( getattr(process,'patMETs'+postfix),
                                                                           getattr(process,'produceCaloMETCorrections'+postfix)
                                                                           *getattr(process,'patMETs'+postfix) )
                elif jetCollType == 'PF':
                    from JetMETCorrections.Type1MET.pfMETCorrections_cff import pfCandsNotInJet,pfJetMETcorr,pfCandMETcorr,pfType1CorrectedMet,pfType1p2CorrectedMet,producePFMETCorrections
                    setattr(process,'producePFMETCorrections'+postfix,producePFMETCorrections.copy())
                    setattr(process,'pfCandsNotInJet'        +postfix,pfCandsNotInJet.clone(topCollection = jetCollection))
                    setattr(process,'pfCandMETcorr'          +postfix,pfCandMETcorr.clone(src = cms.InputTag('pfCandsNotInJet'+postfix)))
                    setattr(process,'pfJetMETcorr'           +postfix,pfJetMETcorr.clone(src = jetCollection))

                    if ('L1FastJet' in jetCorrLabel[1] or 'L1Fastjet' in jetCorrLabel[1]):
                        getattr(process,'pfJetMETcorr' +postfix).offsetCorrLabel = cms.string(jetCorrLabel[0]+'L1FastJet')
                    elif ('L1Offset' in jetCorrLabel[1]):
                        getattr(process,'pfJetMETcorr' +postfix).offsetCorrLabel = cms.string(jetCorrLabel[0]+'L1Offset')
                    else:
                        getattr(process,'pfJetMETcorr'+postfix).offsetCorrLabel = cms.string('')
                    getattr(process,'pfJetMETcorr'+postfix).jetCorrLabel    = cms.string(jetCorrLabel[0]+'CombinedCorrector')

                    getattr(process,'pfType1CorrectedMet'+postfix).srcCHSSums = cms.VInputTag(
                        cms.InputTag("pfchsMETcorr"+postfix,"type0")
                        )
                    getattr(process,'pfType1CorrectedMet'+postfix).srcType1Corrections = cms.VInputTag(
                        cms.InputTag('pfJetMETcorr'+postfix, 'type1')
                        )

                    getattr(process,'pfType1p2CorrectedMet'+postfix).srcCHSSums = cms.VInputTag(
                        cms.InputTag("pfchsMETcorr"+postfix,"type0")
                        )
                    getattr(process,'pfType1p2CorrectedMet'+postfix).srcType1Corrections = cms.VInputTag(
                        cms.InputTag('pfJetMETcorr'+postfix, 'type1')
                        )
                    getattr(process,'pfType1p2CorrectedMet'+postfix).srcUnclEnergySums = cms.VInputTag(
                        cms.InputTag('pfJetMETcorr' +postfix, 'type2'),
                        cms.InputTag('pfJetMETcorr' +postfix, 'offset'),
                        cms.InputTag('pfCandMETcorr'+postfix)                                    
                        )

                    ## add MET corrections to sequence
                    getattr(process, 'patMETs'+ postfix).metSource = cms.InputTag('pfType1CorrectedMet'+postfix)
                    getattr(process, 'patMETs'+ postfix).addMuonCorrections = False
                    getattr(process,'producePFMETCorrections'+postfix).remove(getattr(process,'kt6PFJets'))#,            getattr(process,'kt6PFJets'            +postfix))
                    getattr(process,'producePFMETCorrections'+postfix).remove(getattr(process,'ak5PFJets'))#,            getattr(process,'ak5PFJets'            +postfix))
                    getattr(process,'producePFMETCorrections'+postfix).replace(getattr(process,'pfCandsNotInJet'),      getattr(process,'pfCandsNotInJet'      +postfix))
                    getattr(process,'producePFMETCorrections'+postfix).replace(getattr(process,'pfJetMETcorr'),         getattr(process,'pfJetMETcorr'         +postfix))
                    getattr(process,'producePFMETCorrections'+postfix).replace(getattr(process,'pfCandMETcorr'),        getattr(process,'pfCandMETcorr'        +postfix))
                    getattr(process,'producePFMETCorrections'+postfix).replace(getattr(process,'pfchsMETcorr'),         getattr(process,'pfchsMETcorr'         +postfix))
                    getattr(process,'producePFMETCorrections'+postfix).replace(getattr(process,'pfType1CorrectedMet'),  getattr(process,'pfType1CorrectedMet'  +postfix))
                    getattr(process,'producePFMETCorrections'+postfix).replace(getattr(process,'pfType1p2CorrectedMet'),getattr(process,'pfType1p2CorrectedMet'+postfix))

                    getattr(process,"patDefaultSequence"+postfix).replace( getattr(process,'patMETs'+postfix),
                                                                           getattr(process,'producePFMETCorrections'+postfix)
                                                                           *getattr(process,'patMETs'+postfix) )

            getattr(process,'patDefaultSequence'+postfix).replace(getattr(process,'patJetCorrFactors'+postfix),
                                                                  getattr(process,'kt6PFJets'+postfix)
                                                                  *getattr(process,'patJetCorrFactors'+postfix))
        else:
            ## remove the jetCorrFactors from the std sequence
            process.patJetMETCorrections.remove(process.patJetCorrFactors)
            ## switch embedding of jetCorrFactors off
            ## for pat jet production
            applyPostfix(process, "patJets", postfix).addJetCorrFactors = False
            applyPostfix(process, "patJets", postfix).jetCorrFactorsSource=[]        

        ## adjust output when switching to PFJets
        if ( 'PFJets' in jetCollection.getModuleLabel() or jetCollection.getModuleLabel().startswith("pfNo") or jetCollection.getModuleLabel() == 'particleFlow' ):
            ## in this case we can omit caloTowers and should keep pfCandidates
            if len(outputModules) > 0:
                for outMod in outputModules:
                    if hasattr(process,outMod):
                        getattr(process, outMod).outputCommands.append("keep *_selectedPatJets_pfCandidates_*")
                        getattr(process, outMod).outputCommands.append("drop *_selectedPatJets_caloTowers_*")
                    else:
                        raise KeyError, "process has no OutModule named", outMod

switchJetCollection=SwitchJetCollection()


class AddJetID(ConfigToolBase):

    """ Compute jet id for process
    """
    _label='addJetID'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'jetSrc',self._defaultValue, "", Type=cms.InputTag)
        self.addParameter(self._defaultParameters,'jetIdTag',self._defaultValue, "Tag to append to jet id map", Type=str)
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 jetSrc     = None,
                 jetIdTag    = None) :
        if  jetSrc is None:
            jetSrc=self._defaultParameters['jetSrc'].value
        if  jetIdTag is None:
            jetIdTag=self._defaultParameters['jetIdTag'].value
        self.setParameter('jetSrc',jetSrc)
        self.setParameter('jetIdTag',jetIdTag)
        self.apply(process) 
        
    def toolCode(self, process):        
        jetSrc=self._parameters['jetSrc'].value
        jetIdTag=self._parameters['jetIdTag'].value

        jetIdLabel = jetIdTag + 'JetID'
        print "Making new jet ID label with label " + jetIdTag
        
        ## replace jet id sequence
        process.load("RecoJets.JetProducers.ak5JetID_cfi")
        setattr( process, jetIdLabel, process.ak5JetID.clone(src = jetSrc))
        process.makePatJets.replace( process.patJets, getattr(process,jetIdLabel) + process.patJets )    
           
addJetID=AddJetID()


class SetTagInfos(ConfigToolBase):

    """ Replace tag infos for collection jetSrc
    """
    _label='setTagInfos'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'coll',"patJets","jet collection to set tag infos for")
        self.addParameter(self._defaultParameters,'tagInfos',cms.vstring( ), "tag infos to set")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 coll         = None,
                 tagInfos     = None) :
        if  coll is None:
            coll=self._defaultParameters['coll'].value
        if  tagInfos is None:
            tagInfos=self._defaultParameters['tagInfos'].value
        self.setParameter('coll',coll)
        self.setParameter('tagInfos',tagInfos)
        self.apply(process) 
        
    def toolCode(self, process):        
        coll=self._parameters['coll'].value
        tagInfos=self._parameters['tagInfos'].value

        found = False
        newTags = cms.VInputTag()
        iNewTags = 0
        for k in tagInfos :
            for j in getattr( process, coll ).tagInfoSources :
                vv = j.value();
                if ( vv.find(k) != -1 ):
                    found = True
                    newTags.append( j )
                    
        if not found:
            raise RuntimeError,"""
            Cannot replace tag infos in jet collection""" % (coll)
        else :
            getattr(process,coll).tagInfoSources = newTags
                        
setTagInfos=SetTagInfos()

class SwitchJetCorrLevels(ConfigToolBase):

    """ Switch from jet energy correction levels and do all necessary adjustments
    """
    _label='switchJetCorrLevels'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'jetCorrLabel',None, "payload and list of new jet correction labels, such as (\'AK5Calo\',[\'L2Relative\', \'L3Absolute\'])", tuple,acceptNoneValue=True )
        self.addParameter(self._defaultParameters,'postfix',"", "postfix of default sequence")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 jetCorrLabel = None,
                 postfix      = None) :
        if jetCorrLabel is None:
            jetCorrLabel=self._defaultParameters['jetCorrLabel'].value
        if postfix is None:
            postfix=self._defaultParameters['postfix'].value

        self.setParameter('jetCorrLabel',jetCorrLabel)
        self.setParameter('postfix',postfix)

        self.apply(process) 
        
    def toolCode(self, process):        
        jetCorrLabel=self._parameters['jetCorrLabel'].value
        postfix=self._parameters['postfix'].value

        if (jetCorrLabel!=None):
            ## replace jet energy corrections; catch
            ## a couple of exceptions first
            if (jetCorrLabel == False ):
                raise ValueError, "In switchJetCollection 'jetCorrLabel' must be set to 'None', not 'False'"
            if (jetCorrLabel == "None"):
                raise ValueError, "In switchJetCollection 'jetCorrLabel' must be set to 'None' (without quotes)"
            ## check for the correct format
            if type(jetCorrLabel) != type(('AK5Calo',['L2Relative'])): 
                raise ValueError, "In addJetCollection 'jetCorrLabel' must be 'None', or of type ('payload',['correction1', 'correction2'])"

            jetCorrFactorsModule = getattr(process, "patJetCorrFactors"+postfix)
            jetCorrFactorsModule.payload = jetCorrLabel[0]
            jetCorrFactorsModule.levels  = jetCorrLabel[1]

            ## check whether L1Offset or L1FastJet is part of levels
            error = False
            for x  in jetCorrLabel[1]:
                if x == 'L1Offset':
                    if not error:
                        jetCorrFactorsModule.useNPV = True
                        jetCorrFactorsModule.primaryVertices = 'offlinePrimaryVertices'
                        ## we set this to True now as a L1 correction type should appear only once
                        ## otherwise levels is miss configured
                        error = True
                    else:
                        print 'ERROR : you miss configured the levels parameter. A L1 correction'
                        print '        type should appear not more than once in there.'
                        print jetCorrLabel[1]

                if x == 'L1FastJet':
                    if not error:
                        ## re-run jet algo to compute rho and jetArea for the L1Fastjet corrections
                        jetType=''
                        ## find out which jetType is used (PF or Calo)
                        if jetCorrLabel[0].count('PF') > 0:
                            ## only add module if it is not already part of the process
                            ## ATTENTION: this assumes that new jet collection are always added after existing ones !!!
                            if not hasattr(process,'kt6PFJets'+postfix):
                                from RecoJets.JetProducers.kt4PFJets_cfi import kt4PFJets
                                setattr(process,'kt6PFJets'+postfix, kt4PFJets.clone(doAreaFastjet=True, doRhoFastjet=True, rParam=0.6))
                            jetType='PF'
                        elif jetCorrLabel[0].count('Calo') > 0:
                            ## only add module if it is not already part of the process
                            ## ATTENTION: this assumes that new jet collection are always added after existing ones !!!
                            if not hasattr(process,'kt6CaloJets'+postfix):
                                from RecoJets.JetProducers.kt4CaloJets_cfi import kt4CaloJets
                                setattr(process,'kt6CaloJets'+postfix, kt4CaloJets.clone(doAreaFastjet=True, doRhoFastjet=True, rParam=0.6))
                            jetType='Calo'
                        else:
                            raise TypeError, "L1FastJet corrections are currently only supported for PF and Calo jets in PAT"
                        ## check if a modified patDefaultSequence exists and add kt6Jets there (needed for example for usePF2PAT)
                        if hasattr(process,'patDefaultSequence'+postfix):
                            if not contains(getattr(process,'patDefaultSequence'+postfix), 'kt6'+jetType+'Jets'+postfix):
                                getattr(process,'patDefaultSequence'+postfix).replace(jetCorrFactorsModule, getattr(process,'kt6'+jetType+'Jets'+postfix)*jetCorrFactorsModule)
                        ## if no modified patDefaultSequence exists add kt6Jets to ordinary sequence (needed for example for addJetCollection)
                        else:
                            if not contains(getattr(process,'patDefaultSequence'), 'kt6'+jetType+'Jets'+postfix):
                                getattr(process,'patDefaultSequence').replace(jetCorrFactorsModule, getattr(process,'kt6'+jetType+'Jets'+postfix)*jetCorrFactorsModule)
                        ## configure module
                        jetCorrFactorsModule.useRho = True
                        jetCorrFactorsModule.rho = cms.InputTag('kt6'+jetType+'Jets'+postfix, 'rho')
                        ## we set this to True now as a L1 correction type should appear only once
                        ## otherwise levels is miss configured
                        error = True
                    else:
                        print 'ERROR : you miss configured the levels parameter. A L1 correction'
                        print '        type should appear not more than once in there.'
                        print jetCorrLabel[1]
                        
switchJetCorrLevels=SwitchJetCorrLevels()

def depricatedOptionOutputModule(obj):
    print "-------------------------------------------------------"
    print " Error: the option 'outputModule' is not supported"
    print "        anymore by:"
    print "                   ", obj._label
    print "        please use 'outputModules' now and specify the"
    print "        names of all needed OutModules in there"
    print "        (default: ['out'])"
    print "-------------------------------------------------------"
    raise KeyError, "unsupported option 'outputModule' used in '"+obj._label+"'"
