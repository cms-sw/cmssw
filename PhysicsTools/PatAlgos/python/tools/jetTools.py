from FWCore.GuiBrowsers.ConfigToolBase import *
from FWCore.ParameterSet.Mixins import PrintOptions,_ParameterTypeBase,_SimpleParameterTypeBase, _Parameterizable, _ConfigureComponent, _TypedParameterizable, _Labelable,  _Unlabelable,  _ValidatingListBase
from FWCore.ParameterSet.SequenceTypes import _ModuleSequenceType, _Sequenceable
from FWCore.ParameterSet.SequenceTypes import *
from PhysicsTools.PatAlgos.tools.helpers import *
from PhysicsTools.PatAlgos.recoLayer0.bTagging_cff import *
import sys


class AddJetCollection(ConfigToolBase):
    """
    Tool to add a new jet collection to your PAT Tuple or to modify an existing one.
    """
    _label='addJetCollection'
    _defaultParameters=dicttypes.SortedKeysDict()

    def __init__(self):
        """
        Initialize elements of the class. Note that the tool needs to be derived from ConfigToolBase to be usable in the configEditor.
        """
        ## initialization of the base class
        ConfigToolBase.__init__(self)
        ## add all parameters that should be known to the class
        self.addParameter(self._defaultParameters,'labelName','', "Label name of the new patJet collection.", str)
        self.addParameter(self._defaultParameters,'postfix','', "Postfix from usePF2PAT.", str)
        self.addParameter(self._defaultParameters,'jetSource','', "Label of the input collection from which the new patJet collection should be created", cms.InputTag)
        self.addParameter(self._defaultParameters,'algo','', "Jet algorithm of the input collection from which the new patJet collection should be created")
        self.addParameter(self._defaultParameters,'jetCorrections',None, "Add all relevant information about jet energy corrections that you want to be added to your new patJet \
        collection. The format has to be given in a python tuple of type: (\'AK5Calo\',[\'L2Relative\', \'L3Absolute\'], patMet). Here the first argument corresponds to the payload \
        in the CMS Conditions database for the given jet collection; the second argument corresponds to the jet energy correction levels that you want to be embedded into your \
        new patJet collection. This should be given as a list of strings. Available values are L1Offset, L1FastJet, L1JPTOffset, L2Relative, L3Absolute, L5Falvour, L7Parton; the \
        third argument indicates whether MET(Type1/2) corrections should be applied corresponding to the new patJetCollection. If so a new patMet collection will be added to your PAT \
        Tuple in addition to the raw patMet. This new patMet collection will have the MET(Type1/2) corrections applied. The argument can have the following types: \'type-1\' for \
        type-1 corrected MET; \'type-2\' for type-1 plus type-2 corrected MET; \'\' or \'none\' if no further MET corrections should be applied to your MET. The arguments \'type-1\' \
        and \'type-2\' are not case sensitive.", tuple, acceptNoneValue=True)
        self.addParameter(self._defaultParameters,'btagDiscriminators',['None'], "If you are interested in btagging, in most cases just the labels of the btag discriminators that \
        you are interested in is all relevant information that you need for a high level analysis. Add here all btag discriminators, that you are interested in as a list of strings. \
        If this list is empty no btag discriminator information will be added to your new patJet collection.", allowedValues=supportedBtagDiscr.keys(),Type=list)
        self.addParameter(self._defaultParameters,'btagInfos',['None'], "The btagInfos objects contain all relevant information from which all discriminators of a certain \
        type have been calculated. You might be interested in keeping this information for low level tests or to re-calculate some discriminators from hand. Note that this information \
        on the one hand can be very space consuming and that it is not necessary to access the pre-calculated btag discriminator information that has been derived from it. Only in very \
        special cases the btagInfos might really be needed in your analysis. Add here all btagInfos, that you are interested in as a list of strings. If this list is empty no btagInfos \
        will be added to your new patJet collection.", allowedValues=supportedBtagInfos,Type=list)
        self.addParameter(self._defaultParameters,'jetTrackAssociation',False, "Add JetTrackAssociation and JetCharge from reconstructed tracks to your new patJet collection. This \
        switch is only of relevance if you don\'t add any btag information to your new patJet collection (btagDiscriminators or btagInfos) and still want this information added to \
        your new patJetCollection. If btag information of any form is added to the new patJet collection this information will be added automatically.")
        self.addParameter(self._defaultParameters,'outputModules',['out'],"Add a list of all output modules to which you would like the new jet collection to be added. Usually this is \
        just one single output module with name \'out\', which corresponds also the default configuration of the tool. There is cases though where you might want to add this collection \
        to more than one output module.")
        ## set defaults
        self._parameters=copy.deepcopy(self._defaultParameters)
        ## add comments
        self._comment = "This is a tool to add more patJet collectinos to your PAT Tuple or to re-configure the default collection. You can add and embed additional information like jet\
        energy correction factors, btag infomration and generator match information to the new patJet collection depending on the parameters that you pass on to this function. Consult \
        the descriptions of each parameter for more information."

    def getDefaultParameters(self):
        """
        Return default parameters of the class
        """
        return self._defaultParameters

    def __call__(self,process,labelName=None,postfix=None,jetSource=None,algo=None,jetCorrections=None,btagDiscriminators=None,btagInfos=None,jetTrackAssociation=None,outputModules=None):
        """
        Function call wrapper. This will check the parameters and call the actual implementation that
        can be found in toolCode via the base class function apply.
        """
        if labelName is None:
            labelName=self._defaultParameters['labelName'].value
        self.setParameter('labelName', labelName)
        if postfix is None:
            postfix=self._defaultParameters['postfix'].value
        self.setParameter('postfix', postfix)
        if jetSource is None:
            jetSource=self._defaultParameters['jetSource'].value
        self.setParameter('jetSource', jetSource)
        if algo is None:
            algo=self._defaultParameters['algo'].value
        self.setParameter('algo', algo)
        if jetCorrections is None:
            jetCorrections=self._defaultParameters['jetCorrections'].value
        self.setParameter('jetCorrections', jetCorrections)
        if btagDiscriminators is None:
            btagDiscriminators=self._defaultParameters['btagDiscriminators'].value
        self.setParameter('btagDiscriminators', btagDiscriminators)
        if btagInfos is None:
            btagInfos=self._defaultParameters['btagInfos'].value
        self.setParameter('btagInfos', btagInfos)
        if jetTrackAssociation is None:
            jetTrackAssociation=self._defaultParameters['jetTrackAssociation'].value
        self.setParameter('jetTrackAssociation', jetTrackAssociation)
        if outputModules is None:
            outputModules=self._defaultParameters['outputModules'].value
        self.setParameter('outputModules', outputModules)
        self.apply(process)

    def toolCode(self, process):
        """
        Tool code implementation
        """
        ## initialize parameters
        labelName='patJets'+self._parameters['labelName'].value
        postfix=self._parameters['postfix'].value
        jetSource=self._parameters['jetSource'].value
        algo=self._parameters['algo'].value
        jetCorrections=self._parameters['jetCorrections'].value
        btagDiscriminators=list(self._parameters['btagDiscriminators'].value)
        btagInfos=list(self._parameters['btagInfos'].value)
        jetTrackAssociation=self._parameters['jetTrackAssociation'].value
        outputModules=list(self._parameters['outputModules'].value)

        ## a list of all producer modules, which are already known to process
        knownModules = process.producerNames().split()
        ## determine whether btagging information is required or not
        if btagDiscriminators.count('None')>0:
            btagDiscriminators.remove('None')
        if btagInfos.count('None')>0:
            btagInfos.remove('None')
        bTagging=(len(btagDiscriminators)>0 or len(btagInfos)>0)
        ## construct postfix label for auxiliary modules; this postfix
        ## label will start with a capitalized first letter following
        ## the CMS nameing conventions and for improved readablility
        _labelName=labelName[:1].upper()+labelName[1:]
	#_labelName=labelName
        ## determine jet algorithm from jetSource; supported algo types
        ## are ak, kt, sc, ic. This loop expects that the algo type is
        ## followed by a single integer corresponding to the opening
        ## angle parameter dR times 10 (examples ak5, kt4, kt6, ...)
        _algo=algo
	#jetSource=cms.InputTag("ak5PFJets")
        for x in ["ak", "kt", "sc", "ic"]:
            if jetSource.getModuleLabel().lower().find(x)>-1:
                _algo=jetSource.getModuleLabel()[jetSource.getModuleLabel().lower().find(x):jetSource.getModuleLabel().lower().find(x)+3]
                break
	#print _algo
        ## add new patJets to process (keep instance for later further modifications)
        from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import patJets
        if labelName in knownModules :
            _newPatJets=getattr(process, labelName+postfix)
            _newPatJets.jetSource=jetSource
        else :
            #setattr(process, labelName, patJets.clone(jetSource=jetSource))
            setattr(process, labelName+postfix, patJets.clone(jetSource=jetSource))
            _newPatJets=getattr(process, labelName+postfix)
            knownModules.append(labelName+postfix)
        ## add new selectedPatJets to process
        from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import selectedPatJets
        if 'selected'+_labelName+postfix in knownModules :
            _newSelectedPatJets=getattr(process, 'selected'+_labelName+postfix)
            _newSelectedPatJets.src=labelName+postfix
        else :
            setattr(process, 'selected'+_labelName+postfix, selectedPatJets.clone(src=labelName+postfix))
            knownModules.append('selected'+_labelName+postfix)

        ## set postfix label to '' if there is no labelName given. In this case all
        ## modules should keep there names w/o postfixes. This will cover the case
        ## of switchJectCollection
        if self._parameters['labelName'].value == '' :
            _labelName = ''

	## add new patJetPartonMatch to process
        from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import patJetPartonMatch
        if 'patJetPartonMatch'+_labelName+postfix in knownModules :
            _newPatJetPartonMatch=getattr(process, 'patJetPartonMatch'+_labelName+postfix)
            _newPatJetPartonMatch.src=jetSource
        else :
            setattr(process, 'patJetPartonMatch'+_labelName+postfix, patJetPartonMatch.clone(src=jetSource))
            knownModules.append('patJetPartonMatch'+_labelName+postfix)
        ## add new patJetGenJetMatch to process
        from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import patJetGenJetMatch
        if 'patJetGenJetMatch'+_labelName+postfix in knownModules :
            _newPatJetGenJetMatch=getattr(process, 'patJetGenJetMatch'+_labelName+postfix)
            _newPatJetGenJetMatch.src=jetSource
            _newPatJetGenJetMatch.matched=_algo.lower()+'GenJets'+postfix
        else :
            setattr(process, 'patJetGenJetMatch'+_labelName+postfix, patJetGenJetMatch.clone(src=jetSource, matched=_algo+'GenJets'))
            knownModules.append('patJetGenJetMatch'+_labelName+postfix)
        ## add new patJetPartonAssociation to process
        from PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff import patJetPartonAssociation
        if 'patJetPartonAssociation'+_labelName+postfix in knownModules :
            _newPatJetPartonAssociation=getattr(process, 'patJetPartonAssociation'+_labelName+postfix)
            _newPatJetPartonAssociation.jets=jetSource
        else :
            setattr(process, 'patJetPartonAssociation'+_labelName+postfix, patJetPartonAssociation.clone(jets=jetSource))
            knownModules.append('patJetPartonAssociation'+_labelName+postfix)
        ## add new patJetPartonAssociation to process
        from PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff import patJetFlavourAssociation
        if 'patJetFlavourAssociation'+_labelName+postfix in knownModules :
            _newPatJetFlavourAssociation=getattr(process, 'patJetFlavourAssociation'+_labelName+postfix)
            _newPatJetFlavourAssociation.srcByReference='patJetPartonAssociation'+_labelName+postfix
        else:
            setattr(process, 'patJetFlavourAssociation'+_labelName+postfix, patJetFlavourAssociation.clone(srcByReference='patJetPartonAssociation'+_labelName+postfix))
            knownModules.append('patJetFlavourAssociation'+_labelName+postfix)
        ## modify new patJets collection accordingly
        _newPatJets.genJetMatch.setModuleLabel('patJetGenJetMatch'+_labelName+postfix)
        _newPatJets.genPartonMatch.setModuleLabel('patJetPartonMatch'+_labelName+postfix)
        _newPatJets.JetPartonMapSource.setModuleLabel('patJetFlavourAssociation'+_labelName+postfix)

        ## add jetTrackAssociation for btagging (or jetTracksAssociation only) if required by user
        if (jetTrackAssociation or bTagging):
            ## add new jetTracksAssociationAtVertex to process
            from RecoJets.JetAssociationProducers.ak5JTA_cff import ak5JetTracksAssociatorAtVertex
            if 'jetTracksAssociationAtVertex'+_labelName+postfix in knownModules :
                _newJetTracksAssociationAtVertex=getattr(process, 'jetTracksAssociatorAtVertex'+_labelName+postfix)
                _newJetTracksAssociationAtVertex.jets=jetSource
            else:
                setattr(process, 'jetTracksAssociatorAtVertex'+_labelName+postfix, ak5JetTracksAssociatorAtVertex.clone(jets=jetSource))
                knownModules.append('jetTracksAssociationAtVertex'+_labelName+postfix)
            ## add new patJetCharge to process
            from PhysicsTools.PatAlgos.recoLayer0.jetTracksCharge_cff import patJetCharge
            if 'patJetCharge'+_labelName+postfix in knownModules :
                _newPatJetCharge=getattr(process, 'patJetCharge'+_labelName+postfix)
                _newPatJetCharge.src='jetTracksAssociatorAtVertex'+_labelName+postfix
            else:
                setattr(process, 'patJetCharge'+_labelName+postfix, patJetCharge.clone(src = 'jetTracksAssociatorAtVertex'+_labelName+postfix))
                knownModules.append('patJetCharge'+_labelName+postfix)
            ## modify new patJets collection accordingly
            _newPatJets.addAssociatedTracks=True
            _newPatJets.trackAssociationSource=cms.InputTag('jetTracksAssociatorAtVertex'+_labelName+postfix)
            _newPatJets.addJetCharge=True
            _newPatJets.jetChargeSource=cms.InputTag('patJetCharge'+_labelName+postfix)
        else:
            ## modify new patJets collection accordingly
            _newPatJets.addAssociatedTracks=False
            _newPatJets.trackAssociationSource=''
            _newPatJets.addJetCharge=False
            _newPatJets.jetChargeSource=''
        ## run btagging if required by user
        if (bTagging):
            ## expand tagInfos to what is explicitely required by user + implicit
            ## requirements that come in from one or the other discriminator
            requiredTagInfos = list(btagInfos)
            for btagDiscr in btagDiscriminators :
                for requiredTagInfo in supportedBtagDiscr[btagDiscr] :
                    tagInfoCovered = False
                    for tagInfo in requiredTagInfos :
                        if requiredTagInfo == tagInfo :
                            tagInfoCovered = True
                            break
                    if not tagInfoCovered :
                        requiredTagInfos.append(requiredTagInfo)
            ## load sequences and setups needed fro btagging
	    ## This loads all available btagger, but the ones we need are added to the process by hand lader. Only needed to get the ESProducer. Needs improvement
            #loadWithPostFix(process,"RecoBTag.Configuration.RecoBTag_cff",postfix)
            process.load("RecoBTag.Configuration.RecoBTag_cff")
	    #addESProducers(process,'RecoBTag.Configuration.RecoBTag_cff')
            import RecoBTag.Configuration.RecoBTag_cff as btag

            ## prepare setups for simple secondary vertex infos
            setattr(process, "simpleSecondaryVertex2Trk", simpleSecondaryVertex2Trk)
            ## prepare setups for transient tracks
            setattr(process, "TransientTrackBuilderESProducer", TransientTrackBuilderESProducer)
            ## setup all required btagInfos : we give a dedicated treatment for all five different
            ## types of tagINfos here. A common treatment is possible but might require a more
            ## general approach anyway in coordination with the btaggin POG.
            acceptedTagInfos = list()
            for btagInfo in requiredTagInfos:
                if hasattr(btag,btagInfo):
                    if btagInfo == 'impactParameterTagInfos':
                        setattr(process, btagInfo+_labelName+postfix, btag.impactParameterTagInfos.clone(jetTracks = cms.InputTag('jetTracksAssociatorAtVertex'+_labelName+postfix)))
                    if btagInfo == 'secondaryVertexTagInfos':
                        setattr(process, btagInfo+_labelName+postfix, btag.secondaryVertexTagInfos.clone(trackIPTagInfos = cms.InputTag('impactParameterTagInfos'+_labelName+postfix)))
                    if btagInfo == 'inclusiveSecondaryVertexFinderTagInfos':
                        setattr(process, btagInfo+_labelName+postfix, btag.inclusiveSecondaryVertexFinderTagInfos.clone(trackIPTagInfos = cms.InputTag('impactParameterTagInfos'+_labelName+postfix)))
                    if btagInfo == 'inclusiveSecondaryVertexFinderFilteredTagInfos':
                        setattr(process, btagInfo+_labelName+postfix, btag.inclusiveSecondaryVertexFinderFilteredTagInfos.clone(trackIPTagInfos = cms.InputTag('impactParameterTagInfos'+_labelName+postfix)))
                    if btagInfo == 'secondaryVertexNegativeTagInfos':
                        setattr(process, btagInfo+_labelName+postfix, btag.secondaryVertexNegativeTagInfos.clone(trackIPTagInfos = cms.InputTag('impactParameterTagInfos'+_labelName+postfix)))
                    if btagInfo == 'softMuonTagInfos':
                        setattr(process, btagInfo+_labelName+postfix, btag.softMuonTagInfos.clone(jets = jetSource))
                    if btagInfo == 'softPFMuonsTagInfos':
                        setattr(process, btagInfo+_labelName+postfix, btag.softPFMuonsTagInfos.clone(jets = jetSource))
                    if btagInfo == 'softPFElectronsTagInfos':
                        setattr(process, btagInfo+_labelName+postfix, btag.softPFElectronsTagInfos.clone(jets = jetSource))
                    acceptedTagInfos.append(btagInfo)
                else:
                    print '  --> %s ignored, since not available via RecoBTag.Configuration.RecoBTag_cff!'%(btagInfo)
            ## setup all required btagDiscriminators
            acceptedBtagDiscriminators = list()
            for btagDiscr in btagDiscriminators :
                if hasattr(btag,btagDiscr):
                    setattr(process, btagDiscr+_labelName+postfix, getattr(btag, btagDiscr).clone(tagInfos = cms.VInputTag( *[ cms.InputTag(x+_labelName+postfix) for x in supportedBtagDiscr[btagDiscr] ] )))
                    acceptedBtagDiscriminators.append(btagDiscr)
                else:
                    print '  --> %s ignored, since not available via RecoBTag.Configuration.RecoBTag_cff!'%(btagDiscr)
            ## replace corresponding tags for pat jet production
            _newPatJets.tagInfoSources = cms.VInputTag( *[ cms.InputTag(x+_labelName+postfix) for x in acceptedTagInfos ] )
            _newPatJets.discriminatorSources = cms.VInputTag( *[ cms.InputTag(x+_labelName+postfix) for x in acceptedBtagDiscriminators ] )
            if 'inclusiveSecondaryVertexFinderTagInfos' in acceptedTagInfos:
                if not hasattr( process, 'inclusiveVertexing' ):
                    process.load( 'RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff' )
            if 'inclusiveSecondaryVertexFinderFilteredTagInfos' in acceptedTagInfos:
                if not hasattr( process, 'inclusiveVertexing' ):
                    process.load( 'RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff' )
                if not hasattr( process, 'inclusiveSecondaryVerticesFiltered' ):
                    process.load( 'RecoBTag.SecondaryVertex.secondaryVertex_cff' )
                if not hasattr( process, 'bToCharmDecayVertexMerged' ):
                    process.load( 'RecoBTag.SecondaryVertex.bToCharmDecayVertexMerger_cfi' )
        else:
            _newPatJets.addBTagInfo = False
            ## adjust output module; these collections will be empty anyhow, but we do it to stay clean
            for outputModule in outputModules:
                    if hasattr(process,outputModule):
                        getattr(process,outputModule).outputCommands.append("drop *_"+'selected'+_labelName+postfix+"_tagInfos_*")

        ## add jet correction factors if required by user
        if (jetCorrections != None):
            ## check for the correct format
            if type(jetCorrections) != type(('PAYLOAD-LABEL',['CORRECTION-LEVEL-A','CORRECTION-LEVEL-B'], 'MET-LABEL')):
                raise ValueError, "In addJetCollection: 'jetCorrections' must be 'None' (as a python value w/o quotation marks), or of type ('PAYLOAD-LABEL', ['CORRECTION-LEVEL-A', \
                'CORRECTION-LEVEL-B', ...], 'MET-LABEL'). Note that 'MET-LABEL' can be set to 'None' (as a string in quotation marks) in case you do not want to apply MET(Type1) \
                corrections."
            ## determine type of jet constituents from jetSource; supported
            ## jet constituent types are calo, pf, jpt, for pf also particleflow
            ## is aloowed as part of the jetSource label, which might be used
            ## in CommonTools.ParticleFlow
            _type="NONE"
            if jetCorrections[0].count('PF')>0:
                _type='PF'
            elif jetCorrections[0].count('Calo')>0:
                _type='Calo'
            elif jetCorrections[0].count('JPT')>0:
                _type='JPT'
            else:
                raise TypeError, "In addJetCollection: Jet energy corrections are only supported for PF, JPT and Calo jets."
            from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import patJetCorrFactors
            if 'patJetCorrFactors'+_labelName+postfix in knownModules :
                _newPatJetCorrFactors=getattr(process, 'patJetCorrFactors'+_labelName+postfix)
                _newPatJetCorrFactors.src=jetSource
            else :

                setattr(process, 'patJetCorrFactors'+_labelName+postfix, patJetCorrFactors.clone(src=jetSource))
                _newPatJetCorrFactors=getattr(process, "patJetCorrFactors"+_labelName+postfix)
            _newPatJetCorrFactors.payload=jetCorrections[0]
            _newPatJetCorrFactors.levels=jetCorrections[1]
            ## check whether L1Offset or L1FastJet is part of levels
            error=False
            for x in jetCorrections[1]:
                if x == 'L1Offset' :
                    if not error :
                        _newPatJetCorrFactors.useNPV=True
                        _newPatJetCorrFactors.primaryVertices='offlinePrimaryVertices'
                        _newPatJetCorrFactors.useRho=False
                        ## we set this to True now as a L1 correction type should appear only once
                        ## otherwise levels is miss configured
                        error=True
                    else:
                        raise ValueError, "In addJetCollection: Correction levels for jet energy corrections are miss configured. An L1 correction type should appear not more than \
                        once. Check the list of correction levels you requested to be applied: ", jetCorrections[1]
                if x == 'L1FastJet' :
                    if not error :
                        if _type == "JPT" :
                            raise TypeError, "In addJetCollection: L1FastJet corrections are only supported for PF and Calo jets."
                        ## configure module
                        _newPatJetCorrFactors.useRho=True
                        if "PF" in _type : 
                            _newPatJetCorrFactors.rho=cms.InputTag('fixedGridRhoFastjetAll')
                        else :
                            _newPatJetCorrFactors.rho=cms.InputTag('fixedGridRhoFastjetAllCalo')
                        ## we set this to True now as a L1 correction type should appear only once
                        ## otherwise levels is miss configured
                        error=True
                    else:
                        raise ValueError, "In addJetCollection: Correction levels for jet energy corrections are miss configured. An L1 correction type should appear not more than \
                        once. Check the list of correction levels you requested to be applied: ", jetCorrections[1]
            _newPatJets.jetCorrFactorsSource=cms.VInputTag(cms.InputTag('patJetCorrFactors'+_labelName+postfix))
            ## configure MET(Type1) corrections
            if jetCorrections[2].lower() != 'none' and jetCorrections[2] != '':
                if not jetCorrections[2].lower() == 'type-1' and not jetCorrections[2].lower() == 'type-2':
                    raise valueError, "In addJetCollection: Wrong choice of MET corrections for new jet collection. Possible choices are None (or empty string), Type-1, Type-2 (i.e.\
                    Type-1 and Type-2 corrections applied). This choice is not case sensitive. Your choice was: ", jetCorrections[2]
                if _type == "JPT":
                    raise ValueError, "In addJecCollection: MET(type1) corrections are not supported for JPTJets. Please set the MET-LABEL to \"None\" (as string in quatiation \
                    marks) and use raw tcMET together with JPTJets."
                ## set up jet correctors for MET corrections
                from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import ak5PFL1Fastjet
                from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import ak5PFL1Offset
                from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import ak5PFL2Relative
                from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import ak5PFL3Absolute
                from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import ak5PFResidual

                if "PF" in _type : 
                    setattr(process, jetCorrections[0]+'L1FastJet', ak5PFL1Fastjet.clone(algorithm=jetCorrections[0], srcRho=cms.InputTag('fixedGridRhoFastjetAll')))
                else :
                    setattr(process, jetCorrections[0]+'L1FastJet', ak5PFL1Fastjet.clone(algorithm=jetCorrections[0], srcRho=cms.InputTag('fixedGridRhoFastjetAllCalo')))
                setattr(process, jetCorrections[0]+'L1Offset', ak5PFL1Offset.clone(algorithm=jetCorrections[0]))
                setattr(process, jetCorrections[0]+'L2Relative', ak5PFL2Relative.clone(algorithm=jetCorrections[0]))
                setattr(process, jetCorrections[0]+'L3Absolute', ak5PFL3Absolute.clone(algorithm=jetCorrections[0]))
                setattr(process, jetCorrections[0]+'L2L3Residual', ak5PFResidual.clone(algorithm=jetCorrections[0]))
                setattr(process, jetCorrections[0]+'CombinedCorrector', cms.ESProducer( 'JetCorrectionESChain', correctors = cms.vstring()))
                for x in jetCorrections[1]:
                    if x != 'L1FastJet' and x != 'L1Offset' and x != 'L2Relative' and x != 'L3Absolute' and x != 'L2L3Residual':
                        raise ValueError, 'In addJetCollection: Unsupported JEC for MET(Type1). Currently supported jet correction levels are L1FastJet, L1Offset, L2Relative, L3Asolute, L2L3Residual. Requested was: %s'%(x)
                    else:
                        getattr(process, jetCorrections[0]+'CombinedCorrector').correctors.append(jetCorrections[0]+x)

                ## set up MET(Type1) correction modules
                if _type == 'Calo':
                    from JetMETCorrections.Type1MET.caloMETCorrections_cff import caloJetMETcorr
                    from JetMETCorrections.Type1MET.caloMETCorrections_cff import caloType1CorrectedMet
                    from JetMETCorrections.Type1MET.caloMETCorrections_cff import caloType1p2CorrectedMet
                    setattr(process,jetCorrections[0]+'JetMETcorr'+postfix, caloJetMETcorr.clone(src=jetSource,srcMET = "corMetGlobalMuons",jetCorrections = cms.string(jetCorrections[0]+'CombinedCorrector')))
                    setattr(process,jetCorrections[0]+'Type1CorMet'+postfix, caloType1CorrectedMet.clone(src = "corMetGlobalMuons",srcType1Corrections = cms.VInputTag(cms.InputTag(jetCorrections[0]+'JetMETcorr'+postfix, 'type1'))))
                    setattr(process,jetCorrections[0]+'Type1p2CorMet'+postfix,caloType1p2CorrectedMet.clone(src = "corMetGlobalMuons",srcType1Corrections = cms.VInputTag(cms.InputTag(jetCorrections[0]+'JetMETcorr'+postfix, 'type1')),srcUnclEnergySums = cms.VInputTag(cms.InputTag(jetCorrections[0]+'JetMETcorr'+postfix, 'type2'),cms.InputTag(jetCorrections[0]+'JetMETcorr'+postfix, 'offset'),cms.InputTag('muonCaloMETcorr'))))

                elif _type == 'PF':
                    from JetMETCorrections.Type1MET.pfMETCorrections_cff import pfCandsNotInJet
                    from JetMETCorrections.Type1MET.pfMETCorrections_cff import pfJetMETcorr
                    from JetMETCorrections.Type1MET.pfMETCorrections_cff import pfCandMETcorr
                    from JetMETCorrections.Type1MET.pfMETCorrections_cff import pfType1CorrectedMet
                    from JetMETCorrections.Type1MET.pfMETCorrections_cff import pfType1p2CorrectedMet
                    setattr(process,jetCorrections[0]+'CandsNotInJet'+postfix,pfCandsNotInJet.clone(topCollection = jetSource))
                    setattr(process,jetCorrections[0]+'CandMETcorr'+postfix, pfCandMETcorr.clone(src = cms.InputTag(jetCorrections[0]+'CandsNotInJet'+postfix)))
                    setattr(process,jetCorrections[0]+'JetMETcorr'+postfix, pfJetMETcorr.clone(src = jetSource))
                    setattr(process,jetCorrections[0]+'Type1CorMet'+postfix, pfType1CorrectedMet.clone(srcType1Corrections = cms.VInputTag(cms.InputTag(jetCorrections[0]+'JetMETcorr'+postfix, 'type1'))))
                    setattr(process,jetCorrections[0]+'Type1p2CorMet'+postfix, pfType1p2CorrectedMet.clone(srcType1Corrections = cms.VInputTag(cms.InputTag(jetCorrections[0]+'JetMETcorr'+postfix, 'type1')),srcUnclEnergySums = cms.VInputTag(cms.InputTag(jetCorrections[0]+'JetMETcorr'+postfix, 'type2'),cms.InputTag(jetCorrections[0]+'JetMETcorr'+postfix, 'offset'),cms.InputTag(jetCorrections[0]+'CandMETcorr'+postfix))))

                ## common configuration for Calo and PF
                if ('L1FastJet' in jetCorrections[1] or 'L1Fastjet' in jetCorrections[1]):
                    getattr(process,jetCorrections[0]+'JetMETcorr'+postfix).offsetCorrLabel = cms.string(jetCorrections[0]+'L1FastJet')
                #FIXME: What is wrong here?
                #elif ('L1Offset' in jetCorrections[1]):
                    #getattr(process,jetCorrections[0]+'JetMETcorr'+postfix).offsetCorrLabel = cms.string(jetCorrections[0]+'L1Offset')
                else:
                    getattr(process,jetCorrections[0]+'JetMETcorr'+postfix).offsetCorrLabel = cms.string('')

                from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import patMETs
                if jetCorrections[2].lower() == 'type-1':
                    setattr(process, 'patMETs'+_labelName+postfix, patMETs.clone(metSource = cms.InputTag(jetCorrections[0]+'Type1CorMet'+postfix), addMuonCorrections = False))
                elif jetCorrections[2].lower() == 'type-2':
                    setattr(process, 'patMETs'+_labelName+postfix, patMETs.clone(metSource = cms.InputTag(jetCorrections[0]+'Type1p2CorMet'+postfix), addMuonCorrections = False))
        else:
            ## switch jetCorrFactors off
            _newPatJets.addJetCorrFactors=False

addJetCollection=AddJetCollection()

class SwitchJetCollection(ConfigToolBase):
    """
    Tool to switch parameters of the PAT jet collection to your PAT Tuple.
    """
    _label='switchJetCollection'
    _defaultParameters=dicttypes.SortedKeysDict()

    def __init__(self):
        """
        Initialize elements of the class. Note that the tool needs to be derived from ConfigToolBase to be usable in the configEditor.
        """
        ## initialization of the base class
        ConfigToolBase.__init__(self)
        ## add all parameters that should be known to the class
        self.addParameter(self._defaultParameters,'jetSource','', "Label of the input collection from which the new patJet collection should be created", cms.InputTag)
        self.addParameter(self._defaultParameters,'algo','', "Jet algorithm of the input collection from which the new patJet collection should be created")
        self.addParameter(self._defaultParameters,'postfix','', "postfix from usePF2PAT")
        self.addParameter(self._defaultParameters,'jetCorrections',None, "Add all relevant information about jet energy corrections that you want to be added to your new patJet \
        collection. The format is to be passed on in a python tuple: e.g. (\'AK5Calo\',[\'L2Relative\', \'L3Absolute\'], patMet). The first argument corresponds to the payload \
        in the CMS Conditions database for the given jet collection; the second argument corresponds to the jet energy correction level that you want to be embedded into your \
        new patJet collection. This should be given as a list of strings. Available values are L1Offset, L1FastJet, L1JPTOffset, L2Relative, L3Absolute, L5Falvour, L7Parton; the \
        third argument indicates whether MET(Type1) corrections should be applied corresponding to the new patJetCollection. If so a new patMet collection will be added to your PAT \
        Tuple in addition to the raw patMet with the MET(Type1) corrections applied. The argument corresponds to the patMet collection to which the MET(Type1) corrections should be \
        applied. If you are not interested in MET(Type1) corrections to this new patJet collection pass None as third argument of the python tuple.", tuple, acceptNoneValue=True)
        self.addParameter(self._defaultParameters,'btagDiscriminators',['None'], "If you are interested in btagging in general the btag discriminators is all relevant \
        information that you need for a high level analysis. Add here all btag discriminators, that you are interested in as a list of strings. If this list is empty no btag \
        discriminator information will be added to your new patJet collection.", allowedValues=supportedBtagDiscr.keys(),Type=list)
        self.addParameter(self._defaultParameters,'btagInfos',['None'], "The btagInfos objects conatin all relevant information from which all discriminators of a certain \
        type have been calculated. Note that this information on the one hand can be very space consuming and on the other hand is not necessary to access the btag discriminator \
        information that has been derived from it. Only in very special cases the btagInfos might really be needed in your analysis. Add here all btagInfos, that you are interested \
        in as a list of strings. If this list is empty no btagInfos will be added to your new patJet collection.", allowedValues=supportedBtagInfos,Type=list)
        self.addParameter(self._defaultParameters,'jetTrackAssociation',False, "Add JetTrackAssociation and JetCharge from reconstructed tracks to your new patJet collection. This \
        switch is only of relevance if you don\'t add any btag information to your new patJet collection (btagDiscriminators or btagInfos) and still want this information added to \
        your new patJetCollection. If btag information is added to the new patJet collection this information will be added automatically.")
        self.addParameter(self._defaultParameters,'outputModules',['out'],"Output module labels. Add a list of all output modules to which you would like the new jet collection to \
        be added, in case you use more than one output module.")
        ## set defaults
        self._parameters=copy.deepcopy(self._defaultParameters)
        ## add comments
        self._comment = "This is a tool to add more patJet collectinos to your PAT Tuple. You can add and embed additional information like jet energy correction factors, btag \
        infomration and generatro match information to the new patJet collection depending on the parameters that you pass on to this function. Consult the descriptions of each \
        parameter for more information."

    def getDefaultParameters(self):
        """
        Return default parameters of the class
        """
        return self._defaultParameters

    def __call__(self,process,jetSource=None,algo=None,postfix=None,jetCorrections=None,btagDiscriminators=None,btagInfos=None,jetTrackAssociation=None,outputModules=None):
        """
        Function call wrapper. This will check the parameters and call the actual implementation that
        can be found in toolCode via the base class function apply.
        """
        if jetSource is None:
            jetSource=self._defaultParameters['jetSource'].value
        self.setParameter('jetSource', jetSource)
        if algo is None:
            algo=self._defaultParameters['algo'].value
        self.setParameter('algo', algo)
        if postfix is None:
            postfix=self._defaultParameters['postfix'].value
        self.setParameter('postfix', postfix)
        if jetCorrections is None:
            jetCorrections=self._defaultParameters['jetCorrections'].value
        self.setParameter('jetCorrections', jetCorrections)
        if btagDiscriminators is None:
            btagDiscriminators=self._defaultParameters['btagDiscriminators'].value
        self.setParameter('btagDiscriminators', btagDiscriminators)
        if btagInfos is None:
            btagInfos=self._defaultParameters['btagInfos'].value
        self.setParameter('btagInfos', btagInfos)
        if jetTrackAssociation is None:
            jetTrackAssociation=self._defaultParameters['jetTrackAssociation'].value
        self.setParameter('jetTrackAssociation', jetTrackAssociation)
        if outputModules is None:
            outputModules=self._defaultParameters['outputModules'].value
        self.setParameter('outputModules', outputModules)
        self.apply(process)

    def toolCode(self, process):
        """
        Tool code implementation
        """
        ## initialize parameters
        jetSource=self._parameters['jetSource'].value
	postfix=self._parameters['postfix'].value
        algo=self._parameters['algo'].value
        jetCorrections=self._parameters['jetCorrections'].value
        btagDiscriminators=self._parameters['btagDiscriminators'].value
        btagInfos=self._parameters['btagInfos'].value
        jetTrackAssociation=self._parameters['jetTrackAssociation'].value
        outputModules=self._parameters['outputModules'].value

        ## call addJetCollections w/o labelName; this will act on the default patJets collection
        addJetCollection(
            process,
            labelName='',
	    postfix=postfix,
            jetSource=jetSource,
	    algo=algo,
            jetCorrections=jetCorrections,
            btagDiscriminators=btagDiscriminators,
            btagInfos=btagInfos,
            jetTrackAssociation=jetTrackAssociation,
            outputModules=outputModules,
            )

switchJetCollection=SwitchJetCollection()

class AddJetID(ConfigToolBase):
    """
    Compute jet id for process
    """
    _label='addJetID'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'jetSrc','', "", Type=cms.InputTag)
        self.addParameter(self._defaultParameters,'jetIdTag','', "Tag to append to jet id map", Type=str)
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


addJetID=AddJetID()


class SetTagInfos(ConfigToolBase):
    """
    Replace tag infos for collection jetSrc
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
