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
        #nvTILabel = 'secondaryVertexNegativeTagInfos'     + label + postfix
        #seTILabel = 'softElectronTagInfos'        + label + postfix
        smTILabel = 'softMuonTagInfos'            + label + postfix
    
        ## produce tag infos
        setattr( process, ipTILabel, btag.impactParameterTagInfos.clone(jetTracks = cms.InputTag(jtaLabel)) )
        setattr( process, svTILabel, btag.secondaryVertexTagInfos.clone(trackIPTagInfos = cms.InputTag(ipTILabel)) )
        #setattr( process, nvTILabel, nbtag.secondaryVertexNegativeTagInfos.clone(trackIPTagInfos = cms.InputTag(ipTILabel)) )
        #setattr( process, seTILabel, btag.softElectronTagInfos.clone(jets = jetCollection) )
        setattr( process, smTILabel, btag.softMuonTagInfos.clone(jets = jetCollection) )

        ## make VInputTag from strings
        def vit(*args) : return cms.VInputTag( *[ cms.InputTag(x) for x in args ] )
    
        ## produce btags
        setattr( process, 'jetBProbabilityBJetTags'+label+postfix, btag.jetBProbabilityBJetTags.clone(tagInfos = vit(ipTILabel)) )
        setattr( process, 'jetProbabilityBJetTags'+label+postfix, btag.jetProbabilityBJetTags.clone (tagInfos = vit(ipTILabel)) )
        setattr( process, 'trackCountingHighPurBJetTags'+label+postfix, btag.trackCountingHighPurBJetTags.clone(tagInfos = vit(ipTILabel)) )
        setattr( process, 'trackCountingHighEffBJetTags'+label+postfix, btag.trackCountingHighEffBJetTags.clone(tagInfos = vit(ipTILabel)) )
        setattr( process, 'simpleSecondaryVertexHighEffBJetTags'+label+postfix, btag.simpleSecondaryVertexHighEffBJetTags.clone(tagInfos = vit(svTILabel)) )
        setattr( process, 'simpleSecondaryVertexHighPurBJetTags'+label+postfix, btag.simpleSecondaryVertexHighPurBJetTags.clone(tagInfos = vit(svTILabel)) )
        #setattr( process, 'simpleSecondaryVertexNegativeBJetTags'+label+postfix, nbtag.simpleSecondaryVertexNegativeBJetTags.clone(tagInfos = vit(nvTILabel)) )
        setattr( process, 'combinedSecondaryVertexBJetTags'+label+postfix, btag.combinedSecondaryVertexBJetTags.clone(tagInfos = vit(ipTILabel, svTILabel)) )
        setattr( process, 'combinedSecondaryVertexMVABJetTags'+label+postfix, btag.combinedSecondaryVertexMVABJetTags.clone(tagInfos = vit(ipTILabel, svTILabel)) )
        #setattr( process, 'softElectronByPtBJetTags'+label+postfix, btag.softElectronByPtBJetTags.clone(tagInfos = vit(seTILabel)) )
        #setattr( process, 'softElectronByIP3dBJetTags'+label+postfix, btag.softElectronByIP3dBJetTags.clone(tagInfos = vit(seTILabel)) )
        setattr( process, 'softMuonBJetTags'+label+postfix, btag.softMuonBJetTags.clone(tagInfos = vit(smTILabel)) )
        setattr( process, 'softMuonByPtBJetTags'+label+postfix, btag.softMuonByPtBJetTags.clone(tagInfos = vit(smTILabel)) )
        setattr( process, 'softMuonByIP3dBJetTags'+label+postfix, btag.softMuonByIP3dBJetTags.clone(tagInfos = vit(smTILabel)) )
        
        ## define vector of (output) labels
        labels = { 'jta'      : jtaLabel, 
                   #'tagInfos' : (ipTILabel,svTILabel,seTILabel,smTILabel),
                   'tagInfos' : (ipTILabel,svTILabel,smTILabel), 
                   'jetTags'  : [ (x + label+postfix) for x in ('jetBProbabilityBJetTags',
                                                                'jetProbabilityBJetTags',
                                                                'trackCountingHighPurBJetTags',
                                                                'trackCountingHighEffBJetTags',
                                                                #'simpleSecondaryVertexNegativeBJetTags',
                                                                'simpleSecondaryVertexHighEffBJetTags',
                                                                'simpleSecondaryVertexHighPurBJetTags',
                                                                'combinedSecondaryVertexBJetTags',
                                                                'combinedSecondaryVertexMVABJetTags',
                                                                #'softElectronByPtBJetTags',
                                                                #'softElectronByIP3dBJetTags',
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
        self.addParameter(self._defaultParameters,'doJTA',True, "run b tagging sequence for new jet collection and add it to the new pat jet collection")
        self.addParameter(self._defaultParameters,'doBTagging',True, 'run JetTracksAssociation and JetCharge and add it to the new pat jet collection (will autom. be true if doBTagging is set to true)')        
        self.addParameter(self._defaultParameters,'jetCorrLabel',None, "payload and list of new jet correction labels, such as (\'AK5Calo\',[\'L2Relative\', \'L3Absolute\'])", tuple,acceptNoneValue=True )
        self.addParameter(self._defaultParameters,'doType1MET',True, "if jetCorrLabel is not 'None', set this to 'True' to redo the Type1 MET correction for the new jet colllection; at the moment it must be 'False' for non CaloJets otherwise the JetMET POG module crashes. ")
        self.addParameter(self._defaultParameters,'doL1Cleaning',True, "copy also the producer modules for cleanLayer1 will be set to 'True' automatically when doL1Counters is 'True'")
        self.addParameter(self._defaultParameters,'doL1Counters',False, "copy also the filter modules that accept/reject the event looking at the number of jets")
        self.addParameter(self._defaultParameters,'genJetCollection',cms.InputTag("ak5GenJets"), "GenJet collection to match to")
        self.addParameter(self._defaultParameters,'doJetID',True, "add jetId variables to the added jet collection?")
        self.addParameter(self._defaultParameters,'jetIdLabel',"ak5", " specify the label prefix of the xxxJetID object; in general it is the jet collection tag like ak5, kt4 sc5, aso. For more information have a look to SWGuidePATTools#add_JetCollection")
        self.addParameter(self._defaultParameters, 'outputModule', "out", "Output module label, empty label indicates no output, default: out")
        
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
                 outputModule       = None):

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
        if outputModule is None:
            outputModule=self._defaultParameters['outputModule'].value    

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
        self.setParameter('outputModule',outputModule)
   
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
        outputModule=self._parameters['outputModule'].value

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
            (btagSeq, btagLabels) = runBTagging(process, jetCollection, postfixLabel)
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
            if outputModule is not '':                
                getattr(process, outputModule).outputCommands.append("drop *_"+newLabel(oldLabel('selected'))+"_tagInfos_*")

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
            #getattr(process,newLabel('patJetCorrFactors')).payload = jetCorrLabel[0]
            #getattr(process,newLabel('patJetCorrFactors')).levels = jetCorrLabel[1]
            getattr(process, newLabel('patJets')).jetCorrFactorsSource = cms.VInputTag(  cms.InputTag(newLabel('patJetCorrFactors')) )
        
            ## switch type1MET corrections off for PFJets or JPTJets
            if ( jetCollection.getModuleLabel().find('CaloJets')<0 ):
                print '================================================='
                print 'Type1MET corrections are switched off for other  '
                print 'jet types but CaloJets. Users are recommened to  '
                print 'use pfMET together with PFJets & tcMET together  '
                print 'with JPT jets.                                   '
                print '================================================='
                doType1MET=False

            ## add a clone of the type1MET correction for the new jet collection
            if (doType1MET):
                ## in case there is no jet correction service in the paths add it
                ## as L2L3 if possible, as combined from L2 and L3 otherwise
                if not hasattr( process, '%sL2L3' % (jetCollection.getModuleLabel().replace("Jets", "")) ):
                    setattr( process, '%sL2L3' % (jetCollection.getModuleLabel().replace("Jets", "")),
                             cms.ESSource("JetCorrectionServiceChain",
                                          correctors = cms.vstring('%sL2Relative' % (jetCollection.getModuleLabel().replace("Jets", "")),
                                                                   '%sL3Absolute' % (jetCollection.getModuleLabel().replace("Jets", ""))
                                                                   )
                                           )
                             )                
                ## add a clone of the type1MET correction
                ## and the following muonMET correction
                addClone('metJESCorAK5CaloJet', inputUncorJetsLabel = jetCollection.value(),
                         corrector = cms.string('%sL2L3' % (jetCollection.getModuleLabel().replace("Jets", "")))
                         )                    
                addClone('metJESCorAK5CaloJetMuons', uncorMETInputTag = cms.InputTag(newLabel('metJESCorAK5CaloJet')))
                addClone('patMETs', metSource = cms.InputTag(newLabel('metJESCorAK5CaloJetMuons')))
                l1MET = getattr(process, newLabel('patMETs'))
                ## add new met collections output to the pat summary
                process.patCandidateSummary.candidates += [ cms.InputTag(newLabel('patMETs')) ]
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
        self.addParameter(self._defaultParameters,'doJTA',True, "run b tagging sequence for new jet collection and add it to the new pat jet collection")
        self.addParameter(self._defaultParameters,'doBTagging',True, 'run JetTracksAssociation and JetCharge and add it to the new pat jet collection (will autom. be true if doBTagging is set to true)')
        self.addParameter(self._defaultParameters,'jetCorrLabel',None, "payload and list of new jet correction labels, such as (\'AK5Calo\',[\'L2Relative\', \'L3Absolute\'])", tuple,acceptNoneValue=True )
        self.addParameter(self._defaultParameters,'doType1MET',True, "if jetCorrLabel is not 'None', set this to 'True' to redo the Type1 MET correction for the new jet colleection; at the moment it must be 'False' for non CaloJets otherwise the JetMET POG module crashes. ")
        self.addParameter(self._defaultParameters,'genJetCollection',cms.InputTag("ak5GenJets"), "GenJet collection to match to")
        self.addParameter(self._defaultParameters,'doJetID',True, "add jetId variables to the added jet collection")
        self.addParameter(self._defaultParameters,'jetIdLabel',"ak5", " specify the label prefix of the xxxJetID object; in general it is the jet collection tag like ak5, kt4 sc5, aso. For more information have a look to SWGuidePATTools#add_JetCollection")
        self.addParameter(self._defaultParameters,'postfix',"", "postfix of default sequence")
        self.addParameter(self._defaultParameters, 'outputModule', "out", "Output module label, empty label indicates no output, default: out")
        
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
                 outputModule       = None):
                 
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
        if outputModule is None:
            outputModule=self._defaultParameters['outputModule'].value     
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
        self.setParameter('outputModule',outputModule)
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
        outputModule=self._parameters['outputModule'].value
        postfix=self._parameters['postfix'].value

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
            (btagSeq, btagLabels) = runBTagging(process, jetCollection, 'AOD',postfix)
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
            if outputModule is not '':                
                getattr(process, outputModule).outputCommands.append("drop *_selectedPatJets_tagInfos_*")

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
            #getattr( process, "patJetCorrFactors" + postfix).payload = jetCorrLabel[0]
            #getattr( process, "patJetCorrFactors" + postfix).levels = jetCorrLabel[1]
            getattr( process, "patJets" + postfix).jetCorrFactorsSource = cms.VInputTag( cms.InputTag("patJetCorrFactors" + postfix ) )  

            ## switch type1MET corrections off for PFJets or JPTJets
            if ( jetCollection.getModuleLabel().find('CaloJets')<0 ):
                print '================================================='
                print 'Type1MET corrections are switched off for other  '
                print 'jet types but CaloJets. Users are recommened to  '
                print 'use pfMET together with PFJets & tcMET together  '
                print 'with JPT jets.                                   '
                print '================================================='
                doType1MET=False 

            ## redo the type1MET correction for the new jet collection
            if (doType1MET):
                ## in case there is no jet correction service in the paths add it
                ## as L2L3 if possible, as combined from L2 and L3 otherwise
                if not hasattr( process, '%sL2L3' % (jetCollection.getModuleLabel().replace("Jets", "")) ):
                    setattr( process, '%sL2L3' % (jetCollection.getModuleLabel().replace("Jets", "")),
                             cms.ESSource("JetCorrectionServiceChain",
                                          correctors = cms.vstring('%sL2Relative' % (jetCollection.getModuleLabel().replace("Jets", "")),
                                                                   '%sL3Absolute' % (jetCollection.getModuleLabel().replace("Jets", ""))
                                                                   )
                                          )
                             )                
                ## configure the type1MET correction the following muonMET
                ## corrections have the metJESCorAK5CaloJet as input and 
                ## are automatically correct  
                applyPostfix(process, "metJESCorAK5CaloJet", postfix).inputUncorJetsLabel = jetCollection.value()
                applyPostfix(process, "metJESCorAK5CaloJet", postfix).corrector = '%sL2L3' % (jetCollection.getModuleLabel().replace("Jets", ""))                
        else:
            ## remove the jetCorrFactors from the std sequence
            process.patJetMETCorrections.remove(process.patJetCorrFactors)
            ## switch embedding of jetCorrFactors off
            ## for pat jet production
            applyPostfix(process, "patJets", postfix).addJetCorrFactors = False
            applyPostfix(process, "patJets", postfix).jetCorrFactorsSource=[]        

        ## adjust output when switching to PFJets
        if (jetCollection.getModuleLabel().find('PFJets')>=0 ):
            ## in this case we can omit caloTowers and should keep pfCandidates
            if outputModule is not '':                
                getattr(process, outputModule).outputCommands.append("keep *_selectedPatJets_pfCandidates_*")
                getattr(process, outputModule).outputCommands.append("drop *_selectedPatJets_caloTowers_*")

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
        self.addParameter(self._defaultParameters,'coll',"allLayer1Jets","jet collection to set tag infos for")
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
                 jetCorrLabel       = None,
                 postfix            = None) :
        if jetCorrLabel  is None:
            jetCorrLabel=self._defaultParameters['jetCorrLabel'].value
        if postfix  is None:
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
                        primaryVertices = 'offlinePrimaryVertices'
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
                        process.load("RecoJets.JetProducers.kt4PFJets_cfi")
                        process.kt6PFJets = process.kt4PFJets.clone(doAreaFastjet=True, doRhoFastjet=True, rParam=0.6)
                        process.patDefaultSequence.replace(jetCorrFactorsModule, process.kt6PFJets*jetCorrFactorsModule)
                        ## configure module
                        jetCorrFactorsModule.useRho = True
                        jetCorrFactorsModule.rho = cms.InputTag('kt6PFJets', 'rho')
                        ## we set this to True now as a L1 correction type should appear only once
                        ## otherwise levels is miss configured
                        error = True
                    else:
                        print 'ERROR : you miss configured the levels parameter. A L1 correction'
                        print '        type should appear not more than once in there.'
                        print jetCorrLabel[1]
                        
switchJetCorrLevels=SwitchJetCorrLevels()
