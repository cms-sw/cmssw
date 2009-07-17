import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import *


def switchJECSet(process,
                 newName,
                 oldName
                 ):
    """
    ------------------------------------------------------------------
    replace tags in the JetCorrFactorsProducer for end-users:

    process : process
    newName : new name of JEC module
    oldName : old name of JEC module
    ------------------------------------------------------------------    
    """
    switchJECSet_(process.jetCorrFactors, newName, oldName)

    
def switchJECSet_(jetCorrFactors,
                  newName,
                  oldName,
                  steps=['L1Offset', 'L2Relative', 'L3Absolute', 'L4EMF', 'L5Flavor', 'L6UE', 'L7Parton']
                  ):
    """
    ------------------------------------------------------------------    
    replace tags in the JetCorrFactorsProducer (inner implementation)

    jetCorrFactors : jetCorrFactors module
    newName        : new name of JEC module
    oldName        : old name of JEC module
    steps          : correction steps in the module
    ------------------------------------------------------------------    
    """
    found = False
    for k in steps:
        ## loop jet correction steps
        vv = getattr(jetCorrFactors, k).value();
        if (vv.find(oldName) != -1):
            found = True
        if (vv != "none"):
            ## replace if the correction steps is not 'none'
            setattr(jetCorrFactors, k, vv.replace(oldName,newName))
    if not found:
        raise RuntimeError,"""
        Can't replace jet energy correction step %s with %s.
        The full configuration is %s""" % (oldName, newName, jetCorrFactors.dumpPython())

    
def switchJECParameters(jetCorrFactors,
                        newAlgo,
                        newType="Calo",
                        oldAlgo="IC5",
                        oldType="Calo"
                        ):
    """
    ------------------------------------------------------------------    
    replace tags in the JetCorrFactorsProducer

    jetCorrFactors : jetCorrFactors module
    newAlgo        : label of new jet algo [IC5,  SC5,   KT6, ...]
    newType        : label of new jet type [Calo, Pflow, Jpt, ...]
    oldAlgo        : label of old jet alog [IC5,  SC5,   KT6, ...]
    oldType        : label of old jet type [Calo, Pflow, Jpt, ...]
    ------------------------------------------------------------------    
    """    
    for k in ['L1Offset', 'L2Relative', 'L3Absolute', 'L4EMF', 'L6UE', 'L7Parton']:
        ## loop jet correction steps; the L5Flavor step
        ## is not in the list as as it is said not to
        ## dependend on the specific jet algorithm
        vv = getattr(jetCorrFactors, k).value();
        if (vv != "none"):
            ## the first replace is for '*_IC5Calo'
            ## types, the second for '*_IC5' types            
            setattr(jetCorrFactors, k, vv.replace(oldAlgo+oldType,newAlgo+newType).replace(oldAlgo,newAlgo) )


def runBTagging(process,
                jetCollection,
                label
                ) :
    """
    ------------------------------------------------------------------        
    define sequence to run b tagging on AOD input for a given jet
    collection including a JetTracksAssociatorAtVertex with name
    'jetTracksAssociatorAtVertex' + 'label'

    process       : process       
    jetCollection : input jet collection
    label         : postfix label to identify new sequence/modules

    the sequence is added to the process but not to any path; return
    value is a pair of (sequence, labels) where 'sequence' is the
    cms.Sequence, and 'labels' is a vector with the following entries:
    
     * labels['jta']      = the name of the JetTrackAssociator module
     * labels['tagInfos'] = a list of names of the TagInfo modules
     * labels['jetTags '] = a list of names of the JetTag modules
    ------------------------------------------------------------------        
    """    
    if (label == ''):
        ## label is not allowed to be empty
        raise ValueError, "label for re-running b tagging is not allowed to be empty"        

    ## import track associator & b tag configuration
    process.load("RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi")
    from RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi import ic5JetTracksAssociatorAtVertex
    process.load("RecoBTag.Configuration.RecoBTag_cff")
    import RecoBTag.Configuration.RecoBTag_cff as btag
    
    ## define tag info labels (compare with jetProducer_cfi.py)
    jtaLabel  = 'jetTracksAssociatorAtVertex' + label
    ipTILabel = 'impactParameterTagInfos'     + label
    svTILabel = 'secondaryVertexTagInfos'     + label
    seTILabel = 'softElectronTagInfos'        + label
    smTILabel = 'softMuonTagInfos'            + label
    
    ## produce tag infos
    setattr( process, ipTILabel, btag.impactParameterTagInfos.clone(jetTracks = cms.InputTag(jtaLabel)) )
    setattr( process, svTILabel, btag.secondaryVertexTagInfos.clone(trackIPTagInfos = cms.InputTag(ipTILabel)) )
    setattr( process, seTILabel, btag.softElectronTagInfos.clone(jets = jetCollection) )
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
    setattr( process, 'softElectronBJetTags'+label, btag.softElectronBJetTags.clone(tagInfos = vit(seTILabel)) )
    setattr( process, 'softElectronByIP3dBJetTags'+label, btag.softElectronByIP3dBJetTags.clone(tagInfos = vit(seTILabel)) )
    setattr( process, 'softMuonBJetTags'+label, btag.softMuonBJetTags.clone(tagInfos = vit(smTILabel)) )
    setattr( process, 'softMuonNoIPBJetTags'+label, btag.softMuonNoIPBJetTags.clone(tagInfos = vit(smTILabel)) )
    setattr( process, 'softMuonByIP3dBJetTags'+label, btag.softMuonByIP3dBJetTags.clone(tagInfos = vit(smTILabel)) )
    
    ## define vector of (output) labels
    labels = { 'jta'      : jtaLabel, 
               'tagInfos' : (ipTILabel,svTILabel,seTILabel,smTILabel), 
               'jetTags'  : [ (x + label) for x in ('jetBProbabilityBJetTags',
                                                    'jetProbabilityBJetTags',
                                                    'trackCountingHighPurBJetTags',
                                                    'trackCountingHighEffBJetTags',
                                                    'simpleSecondaryVertexBJetTags',
                                                    'combinedSecondaryVertexBJetTags',
                                                    'combinedSecondaryVertexMVABJetTags',
                                                    'softElectronBJetTags',
                                                    'softElectronByIP3dBJetTags',
                                                    'softMuonBJetTags',
                                                    'softMuonNoIPBJetTags',
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
    return (seq, labels)


def switchJetCollection(process,
                        jetCollection,
                        doJTA            = True,
                        doBTagging       = True,
                        jetCorrLabel     = None,
                        doType1MET       = True,
                        genJetCollection = cms.InputTag("iterativeCone5GenJets")
                        ):
    """
    ------------------------------------------------------------------        
    switch the collection of jets in PAT from the default value to a
    new jet collection

    process          : process
    jetCollection    : input jet collection
    doBTagging       : run b tagging sequence for new jet collection
                       and add it to the new pat jet collection
    doJTA            : run JetTracksAssociation and JetCharge and add
                       it to the new pat jet collection (will autom.
                       be true if doBTagging is set to true)
    jetCorrLabel     : algorithm and type of JEC; use 'None' for no
                       JEC; examples are ('IC5','Calo'), ('SC7',
                       'Calo'), ('KT4','PF')
    doType1MET       : if jetCorrLabel is not 'None', set this to
                       'True' to redo the Type1 MET correction for
                       the new jet colllection; at the moment it must
                       be 'False' for non CaloJets otherwise the
                       JetMET POG module crashes.
    genJetCollection : GenJet collection to match to
    ------------------------------------------------------------------        
    """
    ## save label of old jet collection
    oldLabel = process.allLayer1Jets.jetSource;
    
    ## replace input jet collection for generator matches
    process.jetPartonMatch.src        = jetCollection
    process.jetGenJetMatch.src        = jetCollection
    process.jetGenJetMatch.matched    = genJetCollection
    process.jetPartonAssociation.jets = jetCollection
    
    ## replace input jet collection for trigger matches
    ##massSearchReplaceParam(process.patTrigMatch, 'src', oldLabel, jetCollection)

    ## replace input jet collection for pat jet production
    process.allLayer1Jets.jetSource = jetCollection
    
    ## make VInputTag from strings
    def vit(*args) : return cms.VInputTag( *[ cms.InputTag(x) for x in args ] )

    if (doJTA or doBTagging):
        ## replace jet track association
        process.load("RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi")
        from RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi import ic5JetTracksAssociatorAtVertex
        process.jetTracksAssociatorAtVertex = ic5JetTracksAssociatorAtVertex.clone(jets = jetCollection)
        process.makeAllLayer1Jets.replace(process.patJetCharge, process.jetTracksAssociatorAtVertex+process.patJetCharge)
        process.patJetCharge.src = 'jetTracksAssociatorAtVertex'
        process.allLayer1Jets.trackAssociationSource = 'jetTracksAssociatorAtVertex'
    else:
        ## remove the jet track association from the std
        ## sequence
        process.makeAllLayer1Jets.remove(process.patJetCharge)
        ## switch embedding of track association and jet
        ## charge estimate to 'False'
        process.allLayer1Jets.addAssociatedTracks = False
        process.allLayer1Jets.addJetCharge = False

    if (doBTagging):
        ## replace b tagging sequence
        (btagSeq, btagLabels) = runBTagging(process, jetCollection, 'AOD')
        ## add b tagging sequence to the patAODCoreReco
        ## sequence as it is also needed by ExtraReco
        process.makeAllLayer1Jets.replace(process.jetTracksAssociatorAtVertex, process.jetTracksAssociatorAtVertex+btagSeq)

        ## replace corresponding tags for pat jet production
        process.allLayer1Jets.trackAssociationSource = btagLabels['jta']
        process.allLayer1Jets.tagInfoSources = cms.VInputTag( *[ cms.InputTag(x) for x in btagLabels['tagInfos'] ] )
        process.allLayer1Jets.discriminatorSources = cms.VInputTag( *[ cms.InputTag(x) for x in btagLabels['jetTags']  ] )
    else:
        ## switch embedding of b tagging for pat
        ## jet production to 'False'
        process.allLayer1Jets.addBTagInfo = False

    if (jetCorrLabel!=None):
        ## replace jet energy corrections; catch
        ## a couple of exceptions first
        if (jetCorrLabel == False ):
            raise ValueError, "In switchJetCollection 'jetCorrLabel' must be set to 'None', not 'False'"
        if (jetCorrLabel == "None"):
            raise ValueError, "In switchJetCollection 'jetCorrLabel' must be set to 'None' (without quotes)"
        ## check for the correct format
        if (type(jetCorrLabel)!=type(('IC5','Calo'))): 
            raise ValueError, "In switchJetCollection 'jetCorrLabel' must be 'None', or of type ('Algo','Type')"

        ## switch JEC parameters to the new jet collection
        process.jetCorrFactors.jetSource = jetCollection            
        switchJECParameters(process.jetCorrFactors, jetCorrLabel[0], jetCorrLabel[1], oldAlgo='IC5',oldType='Calo')

        ## redo the type1MET correction for the new jet collection
        if (doType1MET):
            ## in case there is no jet correction service in the paths add it
            ## as L2L3 if possible, as combined from L2 and L3 otherwise
            if (not hasattr( process, 'L2L3JetCorrector%s%s' % jetCorrLabel )):
                setattr( process, 
                         'L2L3JetCorrector%s%s' % jetCorrLabel, 
                         cms.ESSource("JetCorrectionServiceChain",
                                      correctors = cms.vstring('L2RelativeJetCorrector%s%s' % jetCorrLabel,
                                                               'L3AbsoluteJetCorrector%s%s' % jetCorrLabel),
                                      label = cms.string('L2L3JetCorrector%s%s' % jetCorrLabel)
                                      )
                         )
            ## configure the type1MET correction the following muonMET
            ## corrections have the corMetType1Icone5 as input and are
            ## automatically correct  
            process.metJESCorIC5CaloJet.inputUncorJetsLabel = jetCollection.value()
            process.metJESCorIC5CaloJet.corrector = 'L2L3JetCorrector%s%s' % jetCorrLabel
    else:
        ## remove the jetCorrFactors from the std sequence
        process.patJetMETCorrections.remove(process.jetCorrFactors)
        ## switch embedding of jetCorrFactors off
        ## for pat jet production
        process.allLayer1Jets.addJetCorrFactors = False
    

def addJetCollection(process,
                     jetCollection,
                     postfixLabel,
                     doJTA        = True,
                     doBTagging   = True,
                     jetCorrLabel = None,
                     doType1MET   = True,
                     doL1Cleaning = True,                     
                     doL1Counters = False,
                     genJetCollection=cms.InputTag("iterativeCone5GenJets")
                     ):
    """
    ------------------------------------------------------------------        
    add a new collection of jets in PAT

    process          : process
    jetCollection    : input jet collection    
    postfixLabel     : label to identify all modules that work with
                       this jet collection
    doBTagging       : run b tagging sequence for new jet collection
                       and add it to the new pat jet collection
    doJTA            : run JetTracksAssociation and JetCharge and add
                       it to the new pat jet collection (will autom.
                       be true if doBTagging is set to true)
    jetCorrLabel     : algorithm and type of JEC; use 'None' for no
                       JEC; examples are ('IC5','Calo'), ('SC7',
                       'Calo'), ('KT4','PF')
    doType1MET       : make also a new MET collection (not yet
                       implemented?)
    doL1Cleaning     : copy also the producer modules for cleanLayer1
                       will be set to 'True' automatically when
                       doL1Counters is 'True'
    doL1Counters     : copy also the filter modules that accept/reject
                       the event looking at the number of jets                       
    genJetCollection : GenJet collection to match to

    this takes the configuration from the already-configured jets as
    starting point; replaces before calling addJetCollection will
    affect also the new jets
    ------------------------------------------------------------------                     
    """    
    ## add module as process to the default sequence
    def addAlso(label, value):
        existing = getattr(process, label)
        setattr( process, label+postfixLabel, value)
        process.patDefaultSequence.replace( existing, existing*value )

    ## clone and add a module as process to the
    ## default sequence
    def addClone(label, **replaceStatements):
        new = getattr(process, label).clone(**replaceStatements)
        addAlso(label, new)
        
    ## add a clone of allLayer1Jets
    addClone('allLayer1Jets', jetSource = jetCollection)
    ## add a clone of selectedLayer1Jets    
    addClone('selectedLayer1Jets', src=cms.InputTag('allLayer1Jets'+postfixLabel))
    ## add a clone of cleanLayer1Jets    
    if (doL1Cleaning or doL1Counters):
        addClone('cleanLayer1Jets', src=cms.InputTag('selectedLayer1Jets'+postfixLabel))
    ## add a clone of countLayer1Jets    
    if (doL1Counters):
        addClone('countLayer1Jets', src=cms.InputTag('cleanLayer1Jets'+postfixLabel))

    ## attributes of allLayer1Jets
    l1Jets = getattr(process, 'allLayer1Jets'+postfixLabel)

    ## add a clone of gen jet matching
    addClone('jetPartonMatch', src = jetCollection)
    addClone('jetGenJetMatch', src = jetCollection, matched = genJetCollection)
    ## add a clone of parton and flavour associations
    addClone('jetPartonAssociation', jets = jetCollection)
    addClone('jetFlavourAssociation', srcByReference = cms.InputTag('jetPartonAssociation'+postfixLabel))

    ## fix label for input tag
    def fixInputTag(x): x.setModuleLabel(x.moduleLabel+postfixLabel)
    ## fix label for vector of input tags
    def fixVInputTag(x): x[0].setModuleLabel(x[0].moduleLabel+postfixLabel)

    ## provide allLayer1Jet inputs with individual labels
    fixInputTag(l1Jets.genJetMatch)
    fixInputTag(l1Jets.genPartonMatch)
    fixInputTag(l1Jets.JetPartonMapSource)

    ## find potential triggers for trigMatch 
    ##triggers = MassSearchParamVisitor('src', process.allLayer1Jets.jetSource)
    ##process.patTrigMatch.visit(triggers)
    ##for mod in triggers.modules():
    ##    if (doTrigMatch):
    ##        newmod = mod.clone(src = jetCollection)
    ##        setattr( process, mod.label()+postfixLabel, newmod )
    ##        process.patTrigMatch.replace( mod, mod * newmod )
    ##for it in l1Jets.trigPrimMatch.value(): fixInputTag(it)

    ## make VInputTag from strings 
    def vit(*args) : return cms.VInputTag( *[ cms.InputTag(x) for x in args ] )

    if (doJTA or doBTagging):
        ## add clone of jet track association        
        process.load("RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi")
        from RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi import ic5JetTracksAssociatorAtVertex
        ## add jet track association module to processes
        jtaLabel = 'jetTracksAssociatorAtVertex'+postfixLabel
        setattr( process, jtaLabel, ic5JetTracksAssociatorAtVertex.clone(jets = jetCollection) )
        process.makeAllLayer1Jets.replace(process.patJetCharge, getattr(process,jtaLabel)+process.patJetCharge)
        l1Jets.trackAssociationSource = cms.InputTag(jtaLabel)
        addClone('patJetCharge', src=cms.InputTag(jtaLabel)),
        fixInputTag(l1Jets.jetChargeSource)
    else:
        ## switch embedding of track association and jet
        ## charge estimate to 'False'        
        l1Jets.addAssociatedTracks = False
        l1Jets.addJetCharge = False
    
    if (doBTagging):
        ## add b tagging sequence
        (btagSeq, btagLabels) = runBTagging(process, jetCollection, postfixLabel)
        ## add b tagging sequence to the patAODCoreReco
        ## sequence as it is also needed by ExtraReco
        process.makeAllLayer1Jets.replace(getattr(process,jtaLabel), getattr(process,jtaLabel)+btagSeq)

        ## replace corresponding tags for pat jet production
        l1Jets.trackAssociationSource = cms.InputTag(btagLabels['jta'])
        l1Jets.tagInfoSources = cms.VInputTag( *[ cms.InputTag(x) for x in btagLabels['tagInfos'] ] )
        l1Jets.discriminatorSources = cms.VInputTag( *[ cms.InputTag(x) for x in btagLabels['jetTags']  ] )
    else:
        ## switch general b tagging info switch off
        l1Jets.addBTagInfo = False
        
    if (jetCorrLabel != None):
        ## add clone of jet energy corrections;
        ## catch a couple of exceptions first
        if (jetCorrLabel == False ):
            raise ValueError, "In addJetCollection 'jetCorrLabel' must be set to 'None', not 'False'"
        if (jetCorrLabel == "None"):
            raise ValueError, "In addJetCollection 'jetCorrLabel' must be set to 'None' (without quotes)"
        ## check for the correct format
        if type(jetCorrLabel) != type(('IC5','Calo')): 
            raise ValueError, "In switchJetCollection 'jetCorrLabel' must be 'None', or of type ('Algo','Type')"

        ## add clone of jetCorrFactors
        addClone('jetCorrFactors', jetSource = jetCollection)
        switchJECParameters( getattr(process,'jetCorrFactors'+postfixLabel), jetCorrLabel[0], jetCorrLabel[1], oldAlgo='IC5',oldType='Calo' )
        fixVInputTag(l1Jets.jetCorrFactorsSource)

        ## add a clone of the type1MET correction for the new jet collection
        if (doType1MET):
            ## in case there is no jet correction service in the paths add it
            ## as L2L3 if possible, as combined from L2 and L3 otherwise
            if not hasattr( process, 'L2L3JetCorrector%s%s' % jetCorrLabel ):
                setattr( process, 
                         'L2L3JetCorrector%s%s' % jetCorrLabel, 
                         cms.ESSource("JetCorrectionServiceChain",
                                      correctors = cms.vstring('L2RelativeJetCorrector%s%s' % jetCorrLabel,
                                                               'L3AbsoluteJetCorrector%s%s' % jetCorrLabel),
                                      label= cms.string('L2L3JetCorrector%s%s' % jetCorrLabel)
                                      )
                         )
            ## add a clone of the type1MET correction
            ## and the following muonMET correction  
            addClone('metJESCorIC5CaloJet', inputUncorJetsLabel = jetCollection.value(),
                     corrector = cms.string('L2L3JetCorrector%s%s' % jetCorrLabel)
                     )
            addClone('metJESCorIC5CaloJetMuons', uncorMETInputTag = cms.InputTag("metJESCorIC5CaloJet"+postfixLabel))
            addClone('layer1METs', metSource = cms.InputTag("metJESCorIC5CaloJetMuons"+postfixLabel))
            l1MET = getattr(process, 'layer1METs'+postfixLabel)

            ## find potential triggers for trigMatch             
            ##mettriggers = MassSearchParamVisitor('src', process.layer1METs.metSource)
            ##process.patTrigMatch.visit(mettriggers)
            ##for mod in mettriggers.modules():
            ##    if doTrigMatch:
            ##        newmod = mod.clone(src = l1MET.metSource)
            ##        setattr( process, mod.label()+postfixLabel, newmod )
            ##        process.patTrigMatch.replace( mod, mod * newmod )
            ##for it in l1MET.trigPrimMatch.value(): fixInputTag(it)

            ## add new met collections output to the pat summary
            process.allLayer1Summary.candidates += [ cms.InputTag('layer1METs'+postfixLabel) ]
    else:
        ## switch jetCorrFactors off
        l1Jets.addJetCorrFactors = False
