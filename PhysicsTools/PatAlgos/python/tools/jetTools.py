import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import *

def switchJECSet(process,newName,oldName):
    """Replace tags in the JetCorrFactorsProducer
       For end-users, taking the process as first argument"""
    switchJECSet_(process.jetCorrFactors,newName,oldName)
    
def switchJECSet_(jetCorrFactors,newName,oldName,
                  steps=['L1Offset', 'L2Relative', 'L3Absolute', 'L4EMF', 'L5Flavor', 'L6UE', 'L7Parton']):
    """Replace tags in the JetCorrFactorsProducer
       Inner implementation taking a jetCorrFactors object"""
    found = False
    for k in steps:
        vv = getattr(jetCorrFactors, k).value();
        if vv.find(oldName) != -1: found = True
        if (vv != "none"): 
            setattr(jetCorrFactors, k, vv.replace(oldName,newName))
            # the first replace is good for L2, L3; the last for L7 (which don't have type dependency, at least not in the name)
    if not found:
        raise RuntimeError, """None of the JEC steps uses old name %s, so I can't replace %s with %s.
                               The full configuration is %s""" % (newName,oldName,newName,jetCorrFactors.dumpPython())

def switchJECParameters(jetCorrFactors,newalgo,newtype="Calo",oldalgo="IC5",oldtype="Calo"):
    """Replace tags in the JetCorrFactorsProducer"""
    for k in ['L1Offset', 'L2Relative', 'L3Absolute', 'L4EMF', 'L5Flavor', 'L6UE', 'L7Parton']:
        vv = getattr(jetCorrFactors, k).value();
        if (vv != "none"): 
            setattr(jetCorrFactors, k, vv.replace(oldalgo+oldtype,newalgo+newtype).replace(oldalgo,newalgo) )
            # the first replace is good for L2, L3; the last for L7 (which don't have type dependency, at least not in the name)

def runBTagging(process,jetCollection,label) :
    """Define a sequence to run BTagging on AOD on top of jet collection 'jetCollection', appending 'label' to module labels.
       The sequence will be called "btaggingAOD" + 'label', and will already be added to the process (but not to any Path)
       The sequence will include a JetTracksAssociatorAtVertex with name "jetTracksAssociatorAtVertex" + 'label'
       The method will return a pair (sequence, labels) where 'sequence' is the cms.Sequence object, and 'labels' contains
         labels["jta"]      = the name of the JetTrackAssociator module
         labels["tagInfos"] = list of names of TagInfo modules
         labels["jetTags "] = list of names of JetTag modules
       these labels are meant to be used for PAT BTagging tools
       NOTE: 'label' MUST NOT BE EMPTY
     """
    if (label == ''):
        raise ValueError, "Label for re-running BTagging can't be empty, it will crash CRAB." 
    process.load("RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi")
    process.load("RecoBTag.Configuration.RecoBTag_cff")
    from RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi import ic5JetTracksAssociatorAtVertex
    import RecoBTag.Configuration.RecoBTag_cff as btag
    
    # quickly make VInputTag from strings
    def vit(*args) : return cms.VInputTag( *[ cms.InputTag(x) for x in args ] )
    
    # define labels
    jtaLabel =  'jetTracksAssociatorAtVertex' + label
    ipTILabel = 'impactParameterTagInfos'     + label
    svTILabel = 'secondaryVertexTagInfos'     + label
    seTILabel = 'softElectronTagInfos'        + label
    smTILabel = 'softMuonTagInfos'            + label
    
    # make JTA and TagInfos
    setattr( process, jtaLabel,  ic5JetTracksAssociatorAtVertex.clone(jets = cms.InputTag(jetCollection)))
    setattr( process, ipTILabel, btag.impactParameterTagInfos.clone(jetTracks = cms.InputTag(jtaLabel)) )
    setattr( process, svTILabel, btag.secondaryVertexTagInfos.clone(trackIPTagInfos = cms.InputTag(ipTILabel)) )
    setattr( process, seTILabel, btag.softElectronTagInfos.clone(jets = cms.InputTag(jetCollection)) )
    setattr( process, smTILabel, btag.softMuonTagInfos.clone(jets = cms.InputTag(jetCollection)) )
    setattr( process, 'jetBProbabilityBJetTags'+label, btag.jetBProbabilityBJetTags.clone(tagInfos = vit(ipTILabel)) )
    setattr( process, 'jetProbabilityBJetTags' +label,  btag.jetProbabilityBJetTags.clone(tagInfos = vit(ipTILabel)) )
    setattr( process, 'trackCountingHighPurBJetTags'+label, btag.trackCountingHighPurBJetTags.clone(tagInfos = vit(ipTILabel)) )
    setattr( process, 'trackCountingHighEffBJetTags'+label, btag.trackCountingHighEffBJetTags.clone(tagInfos = vit(ipTILabel)) )
    setattr( process, 'impactParameterMVABJetTags'+label, btag.impactParameterMVABJetTags.clone(tagInfos = vit(ipTILabel)) )
    setattr( process, 'simpleSecondaryVertexBJetTags'+label, btag.simpleSecondaryVertexBJetTags.clone(tagInfos = vit(svTILabel)) )
    setattr( process, 'combinedSecondaryVertexBJetTags'+label, btag.combinedSecondaryVertexBJetTags.clone(tagInfos = vit(ipTILabel, svTILabel)) )
    setattr( process, 'combinedSecondaryVertexMVABJetTags'+label, btag.combinedSecondaryVertexMVABJetTags.clone(tagInfos = vit(ipTILabel, svTILabel)) )
    setattr( process, 'softElectronBJetTags'+label, btag.softElectronBJetTags.clone(tagInfos = vit(seTILabel)) )
    setattr( process, 'softMuonBJetTags'+label, btag.softMuonBJetTags.clone(tagInfos = vit(smTILabel)) )
    setattr( process, 'softMuonNoIPBJetTags'+label, btag.softMuonNoIPBJetTags.clone(tagInfos = vit(smTILabel)) )
    
    def mkseq(process, firstlabel, *otherlabels):
       seq = getattr(process, firstlabel)
       for x in otherlabels: seq += getattr(process, x)
       return cms.Sequence(seq)
    
    labels = { 'jta' : jtaLabel, 
               'tagInfos' : (ipTILabel,svTILabel,seTILabel,smTILabel), 
               'jetTags'  : [ (x + label) for x in ('jetBProbabilityBJetTags',
                                                'jetProbabilityBJetTags',
                                                'trackCountingHighPurBJetTags',
                                                'trackCountingHighEffBJetTags',
                                                'impactParameterMVABJetTags',
                                                'simpleSecondaryVertexBJetTags',
                                                'combinedSecondaryVertexBJetTags',
                                                'combinedSecondaryVertexMVABJetTags',
                                                'softElectronBJetTags',
                                                'softMuonBJetTags',
                                                'softMuonNoIPBJetTags') ]
    }
    
    setattr( process, 'btaggingTagInfos' + label, mkseq(process, *(labels['tagInfos']) ) )
    setattr( process, 'btaggingJetTags' + label,  mkseq(process, *(labels['jetTags'])  ) )
    seq = mkseq(process, jtaLabel, 'btaggingTagInfos' + label, 'btaggingJetTags' + label) 
    setattr( process, 'btagging' + label, seq )
    return (seq, labels)

def switchJetCollection(process,jetCollection,layers=[0,1],runCleaner="CaloJet",doJTA=True,doBTagging=True,jetCorrLabel=None,doType1MET=True,
                                genJetCollection=cms.InputTag("iterativeCone5GenJets")):
    """Switch the collection of jets in PAT from the default value.
          layers      : Determine which PAT layers will be affected ([0], [0,1])   
          runCleaner  : Run the layer 0 jet cleaner. Value is the C++ type of the jet CaloJet, PFJet, BasicJet), or None.
                        The cleaner module will be always called 'allLayer0Jets'.
                        None must be written without quotes!
          doBTagging  : True to run the BTagging sequence on top of this jets, and import it into PAT.
          doJTA       : Run Jet Tracks Association and Jet Charge (will be forced to True if doBTagging is true)
          jetCorrLabel: Name of the algorithm and jet type JEC to pick corrections from, or None for no JEC 
                        Examples are ('IC5','Calo'), ('SC7','Calo'), ('KT4','PF')
                        It tries to find a 'L2L3JetCorrector' + algo + type , or otherwise to create if as a 
                        JetCorrectionServiceChain of 'L2RelativeJetCorrector' and 'L3AbsoluteJetCorrector'
          doType1MET  : If jetCorrLabel is not 'None', set this to 'True' to remake Type1 MET from these jets
                        NOTE: at the moment it must be False for non-CaloJets otherwise the JetMET POG module crashes.
          genJetCollection : GenJet collection to match to.

       Note: When turning off the cleaner, bTagging, JTA, jet corrections, MC and Trigger matching will be run directly on jetCollection
             The outputs will still be called 'layer0BTags', 'layer0JetTracksAssociatior' and so on.
       Note: Replacing only layer 1 is not a well defined task, so it's not allowed. 
             What you want is probably to replace 0+1 without any cleaning (runCleaner=None), or a simple replace of allLayer1Jets.jetSource"""
    if runCleaner == "CaloJet":
        process.allLayer0Jets.jetSource = jetCollection
    elif runCleaner == "PFJet":
        process.globalReplace('allLayer0Jets', process.allLayer0PFJets.clone(jetSource = cms.InputTag(jetCollection)))
    elif runCleaner == "BasicJet":
        from PhysicsTools.PatAlgos.cleaningLayer0.basicJetCleaner_cfi import allLayer0Jets as allLayer0BasicJets;
        process.globalReplace('allLayer0Jets', allLayer0BasicJets.clone(jetSource = cms.InputTag(jetCollection)))
    elif runCleaner == None:
        process.patLayer0.remove(process.allLayer0Jets)
        # MC match
        process.jetPartonMatch.src        = cms.InputTag(jetCollection)
        process.jetGenJetMatch.src        = cms.InputTag(jetCollection)
        process.jetGenJetMatch.match      = genJetCollection
        process.jetPartonAssociation.jets = cms.InputTag(jetCollection)
        massSearchReplaceParam(process.patTrigMatch, 'src', cms.InputTag("allLayer0Jets"), cms.InputTag(jetCollection))
        if layers.count(1) != 0:
            process.allLayer1Jets.jetSource = cms.InputTag(jetCollection)
    elif runCleaner == "None":
        raise ValueError, "In switchJetCollection, the value None for runCleaner must be written without quotes"
    else:
        raise ValueError, ("Cleaner '%s' not known" % (runCleaner,))
    if doBTagging :
          (btagSeq, btagLabels) = runBTagging(process, jetCollection, 'AOD')
          process.patLayer0.replace(process.patBeforeLevel0Reco, btagSeq + process.patBeforeLevel0Reco)
          process.patAODJetTracksAssociator.src       = jetCollection
          process.patAODJetTracksAssociator.tracks    = btagLabels['jta']
          process.patAODTagInfos.collection           = jetCollection
          process.patAODBTags.collection              = jetCollection
          process.patAODTagInfos.associations         = btagLabels['tagInfos']
          process.patAODBTags.associations            = btagLabels['jetTags']
          if runCleaner != None:
              process.layer0TagInfos.associations         = btagLabels['tagInfos']
              process.layer0BTags.associations            = btagLabels['jetTags']
          else:
              process.globalReplace('layer0JetTracksAssociator', process.patAODJetTracksAssociator.clone())
              process.globalReplace('layer0TagInfos',            process.patAODTagInfos.clone())
              process.globalReplace('layer0BTags',               process.patAODBTags.clone())
              process.patLayer0.remove(process.patAODJetTracksAssociator)
              process.patLayer0.remove(process.patAODBTags)
              process.patLayer0.remove(process.patAODTagInfos)
    else:
        process.patLayer0.remove(process.patAODBTagging)
        process.patLayer0.remove(process.patLayer0BTagging)
        if layers.count(1) != 0:  process.allLayer1Jets.addBTagInfo = False
    if doJTA or doBTagging:
        if not doBTagging:
            process.load("RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi")
            from RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi import ic5JetTracksAssociatorAtVertex
            process.jetTracksAssociatorAtVertex = ic5JetTracksAssociatorAtVertex.clone(jets = cms.InputTag(jetCollection))
            process.patLayer0.replace(process.patBeforeLevel0Reco, process.jetTracksAssociatorAtVertex + process.patBeforeLevel0Reco)
            process.patAODJetTracksAssociator.src       = jetCollection
            process.patAODJetTracksAssociator.tracks    = 'jetTracksAssociatorAtVertex'
            if runCleaner == None:
                process.globalReplace('layer0JetTracksAssociator', process.patAODJetTracksAssociator.clone())
                process.layer0JetCharge.src       = jetCollection
                process.patLayer0.remove(process.patAODJetTracksAssociator)
    else: ## no JTA
        process.patHighLevelReco_withoutPFTau.remove(process.patLayer0JetTracksCharge)
        if layers.count(1) != 0:  
            process.allLayer1Jets.addAssociatedTracks = False
            process.allLayer1Jets.addJetCharge = False
    if jetCorrLabel != None:
        if jetCorrLabel == False : raise ValueError, "In switchJetCollection 'jetCorrLabel' must be set to None, not False"
        if jetCorrLabel == "None": raise ValueError, "In switchJetCollection 'jetCorrLabel' must be set to None (without quotes), not 'None'"
        if type(jetCorrLabel) != type(('IC5','Calo')): 
            raise ValueError, "In switchJetCollection 'jetCorrLabel' must be None, or a tuple ('Algo', 'Type')"
        if not hasattr( process, 'L2L3JetCorrector%s%s' % jetCorrLabel ):
            setattr( process, 
                        'L2L3JetCorrector%s%s' % jetCorrLabel, 
                        cms.ESSource("JetCorrectionServiceChain",
                            correctors = cms.vstring('L2RelativeJetCorrector%s%s' % jetCorrLabel,
                                                     'L3AbsoluteJetCorrector%s%s' % jetCorrLabel),
                            label      = cms.string('L2L3JetCorrector%s%s' % jetCorrLabel)
                        )
                    )
        switchJECParameters(process.jetCorrFactors, jetCorrLabel[0], jetCorrLabel[1], oldalgo='IC5',oldtype='Calo')
        process.jetCorrFactors.jetSource = jetCollection
        if doType1MET:
            process.corMetType1Icone5.inputUncorJetsLabel = jetCollection
            process.corMetType1Icone5.corrector           = 'L2L3JetCorrector%s%s' % jetCorrLabel
        if runCleaner == None:
            process.globalReplace('layer0JetCorrFactors', process.jetCorrFactors.copy())
            process.patLayer0.remove(process.jetCorrFactors)
    else:
        process.patLayer0.remove(process.jetCorrFactors)
        process.patLayer0.remove(process.layer0JetCorrFactors)
        if layers.count(1) != 0:
            process.allLayer1Jets.addJetCorrFactors = False

def addJetCollection(process,jetCollection,postfixLabel,
                        layers=[0,1],runCleaner="CaloJet",doJTA=True,doBTagging=True,jetCorrLabel=None,doType1MET=True,doL1Counters=False,
                        genJetCollection=cms.InputTag("iterativeCone5GenJets")):
    """Add a new collection of jets in PAT from the default value.
          postfixLabel: Postpone this label to the name of all modules that work with these jet collection.
                        it can't be an empty string
          layers      : Determine which PAT layers will be affected ([0], [0,1])   
          runCleaner  : Run the layer 0 jet cleaner. Value is the C++ type of the jet CaloJet, PFJet, BasicJet), or None.
                        The cleaner module will be always called 'allLayer0Jets'.
                        None must be written without quotes!
          doBTagging  : True to run the BTagging sequence on top of this jets, and import it into PAT.
          doJTA       : Run Jet Tracks Association and Jet Charge (will be forced to True if doBTagging is true)
          jetCorrLabel: Name of the algorithm and jet type JEC to pick corrections from, or None for no JEC 
                        Examples are ('IC5','Calo'), ('SC7','Calo'), ('KT4','PF')
                        It tries to find a 'L2L3JetCorrector' + algo + type , or otherwise to create if as a 
                        JetCorrectionServiceChain of 'L2RelativeJetCorrector' and 'L3AbsoluteJetCorrector'
          doType1MET  : Make also a new MET (NOT IMPLEMENTED)
          doL1Counters: copy also the filter modules that accept/reject the event looking at the number of jets
          genJetCollection : GenJet collection to match to.

       Notes:
       1)  This takes the configuration from the already-configured layer 0+1 jets, so if you do 
           replaces before calling addJetCollection then they will affect also the new jets
           DON'T DON'T DON'T call this after having switched off cleaning of layer 0 jets!
    
       2)  When turning off the cleaner, bTagging, JTA, jet corrections, MC and Trigger matching will be run directly on jetCollection
             The outputs will still be called 'layer0BTags'+postfixLabel, 'layer0JetTracksAssociatior'+postfixLabel and so on."""
    def addAlso (label,value):
        existing = getattr(process, label)
        setattr( process, label + postfixLabel, value)
        process.patLayer0.replace( existing, existing * value )
        if layers.count(1) != 0 : process.patLayer1.replace( existing, existing * value )
    def addClone(label,**replaceStatements):
        new      = getattr(process, label).clone(**replaceStatements)
        addAlso(label, new)
    # --- L0 ---
    newLabel0 = 'allLayer0Jets' + postfixLabel
    if runCleaner == "CaloJet":
        addClone('allLayer0Jets', jetSource = cms.InputTag(jetCollection))
    elif runCleaner == "PFJet":
        addAlso('allLayer0Jets', process.allLayer0PFJets.clone(jetSource = cms.InputTag(jetCollection)))
    elif runCleaner == "BasicJet":
        from PhysicsTools.PatAlgos.cleaningLayer0.basicJetCleaner_cfi import allLayer0Jets as allLayer0BasicJets;
        addAlso('allLayer0Jets', allLayer0BasicJets.clone(jetSource = cms.InputTag(jetCollection)))
    elif runCleaner == None:
        pass
    elif runCleaner == "None":
        raise ValueError, "In switchJetCollection, the value None for runCleaner must be written without quotes"
    else:
        raise ValueError, ("Cleaner '%s' not known" % (runCleaner,))
    # --- L1 ---
    l1Jets = None
    if layers.count(1) != 0: 
        addClone('allLayer1Jets', jetSource = cms.InputTag(newLabel0))
        l1Jets = getattr(process, 'allLayer1Jets'+postfixLabel)
        addClone('selectedLayer1Jets', src=cms.InputTag('allLayer1Jets'+postfixLabel))
        if doL1Counters:
            addClone('minLayer1Jets',      src=cms.InputTag('selectedLayer1Jets'+postfixLabel))
            addClone('maxLayer1Jets',      src=cms.InputTag('selectedLayer1Jets'+postfixLabel))
        if runCleaner == None:
            l1Jets.jetSource = cms.InputTag(jetCollection)
    if runCleaner != None:
        addClone('jetPartonMatch',       src = cms.InputTag(newLabel0))
        addClone('jetGenJetMatch',       src = cms.InputTag(newLabel0), match = genJetCollection)
        addClone('jetPartonAssociation', jets = cms.InputTag(newLabel0))
        addClone('jetFlavourAssociation',srcByReference = cms.InputTag('jetPartonAssociation' + postfixLabel))
        triggers = MassSearchParamVisitor('src', cms.InputTag("allLayer0Jets"))
        process.patTrigMatch.visit(triggers)
        for mod in triggers.modules():
            newmod = mod.clone(src = cms.InputTag(newLabel0))
            setattr( process, mod.label() + postfixLabel, newmod )
            process.patTrigMatch.replace( mod, mod * newmod )
    else:   
        l1Jets.src = cms.InputTag(jetCollection)
        addClone('jetPartonMatch',       src = cms.InputTag(jetCollection))
        addClone('jetGenJetMatch',       src = cms.InputTag(jetCollection))
        addClone('jetPartonAssociation', jets = cms.InputTag(jetCollection))
        addClone('jetFlavourAssociation',srcByReference = cms.InputTag('jetPartonAssociation' + postfixLabel))
        triggers = MassSearchParamVisitor('src', cms.InputTag("allLayer0Jets"))
        process.patTrigMatch.visit(triggers)
        for mod in triggers.modules():
            newmod = mod.clone(src = cms.InputTag(jetCollection))
            setattr( process, mod.label() + postfixLabel, newmod )
            process.patTrigMatch.replace( mod, mod * newmod )
    def fixInputTag(x): x.setModuleLabel(x.moduleLabel+postfixLabel)
    if l1Jets != None:
        fixInputTag(l1Jets.JetPartonMapSource)
        fixInputTag(l1Jets.genJetMatch)
        fixInputTag(l1Jets.genPartonMatch)
        for it in l1Jets.trigPrimMatch.value(): fixInputTag(it)
    def vit(*args) : return cms.VInputTag( *[ cms.InputTag(x) for x in args ] )
    if doBTagging :
        (btagSeq, btagLabels) = runBTagging(process, jetCollection, postfixLabel)
        process.patLayer0.replace(process.patBeforeLevel0Reco, btagSeq + process.patBeforeLevel0Reco)
        if runCleaner != None:
            addClone('patAODJetTracksAssociator', src=cms.InputTag(jetCollection), 
                                                  tracks=cms.InputTag(btagLabels['jta']))
            addClone('patAODTagInfos'           , collection=cms.InputTag(jetCollection), 
                                                  associations=vit(*btagLabels['tagInfos']))
            addClone('patAODBTags'              , collection=cms.InputTag(jetCollection),
                                                  associations=vit(*btagLabels['jetTags']))
            addClone('layer0JetTracksAssociator', association=cms.InputTag('patAODJetTracksAssociator'+postfixLabel),
                                                  collection=cms.InputTag(newLabel0),
                                                  backrefs=cms.InputTag(newLabel0))
            addClone('layer0JetCharge'          , src=cms.InputTag(newLabel0),
                                                  jetTracksAssociation=cms.InputTag('layer0JetTracksAssociator'+postfixLabel))
            addClone('layer0TagInfos'           , commonLabel=cms.InputTag('patAODTagInfos'+postfixLabel),
                                                  associations=vit(*btagLabels['tagInfos']),
                                                  collection=cms.InputTag(newLabel0),
                                                  backrefs=cms.InputTag(newLabel0))
            addClone('layer0BTags'              , commonLabel=cms.InputTag('patAODBTags'+postfixLabel),
                                                  associations=vit(*btagLabels['jetTags']),
                                                  collection=cms.InputTag(newLabel0),
                                                  backrefs=cms.InputTag(newLabel0))
        else:
            addAlso('layer0JetTracksAssociator',  process.patAODJetTracksAssociator.clone(
                                                      src=cms.InputTag(jetCollection), 
                                                      tracks=cms.InputTag(btagLabels['jta'])))
            addClone('layer0JetCharge'          , src=cms.InputTag(jetCollection),
                                                  jetTracksAssociation=cms.InputTag('layer0JetTracksAssociator'+postfixLabel))
            addAlso('layer0TagInfos'           ,  process.patAODTagInfos.clone(
                                                    collection=cms.InputTag(jetCollection), 
                                                    associations=vit(*btagLabels['tagInfos'])))
            addAlso('layer0BTags'              , process.patAODBTags.clone(
                                                    collection=cms.InputTag(jetCollection),
                                                    associations=vit(*btagLabels['jetTags'])))
        if l1Jets != None:
            fixInputTag(l1Jets.jetChargeSource)
            fixInputTag(l1Jets.trackAssociationSource)
            fixInputTag(l1Jets.tagInfoModule)
            fixInputTag(l1Jets.discriminatorModule)
            if l1Jets.discriminatorNames != cms.vstring("*"): 
                l1Jets.discriminatorNames.setValue([x + postfixLabel for x in l1Jets.discriminatorNames.value()])
            if l1Jets.tagInfoNames != cms.vstring("*"): 
                l1Jets.tagInfoNames.setValue([x + postfixLabel for x in l1Jets.tagInfoNames.value()])
    else:
       if l1Jets != None: l1Jets.addBTagInfo = False 
    if doJTA or doBTagging:
        if not doBTagging:
            process.load("RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi")
            from RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi import ic5JetTracksAssociatorAtVertex
            jtaLabel = 'jetTracksAssociatorAtVertex' + postfixLabel
            setattr( process, jtaLabel, ic5JetTracksAssociatorAtVertex.clone(jets = cms.InputTag(jetCollection)) )
            process.patLayer0.replace(process.patBeforeLevel0Reco, getattr(process,jtaLabel) + process.patBeforeLevel0Reco)
            if runCleaner != None:
                addClone('patAODJetTracksAssociator', src=cms.InputTag(jetCollection), 
                                                      tracks=cms.InputTag(jtaLabel))
                addClone('layer0JetTracksAssociator', association=cms.InputTag('patAODJetTracksAssociator'+postfixLabel),
                                                      collection=cms.InputTag(newLabel0),
                                                      backrefs=cms.InputTag(newLabel0))
                addClone('layer0JetCharge'          , src=cms.InputTag(newLabel0),
                                                      jetTracksAssociation=cms.InputTag('layer0JetTracksAssociator'+postfixLabel))
            else:
                addAlso('layer0JetTracksAssociator',  process.patAODJetTracksAssociator.clone(
                                                          src=cms.InputTag(jetCollection), 
                                                          tracks=cms.InputTag(jtaLabel)))
                addClone('layer0JetCharge'          , src=cms.InputTag(jetCollection),
                                                      jetTracksAssociation=cms.InputTag('layer0JetTracksAssociator'+postfixLabel))
            if l1Jets != None:
                fixInputTag(l1Jets.jetChargeSource)
                fixInputTag(l1Jets.trackAssociationSource)
    else: ## no JTA
       if l1Jets != None:  
            l1Jets.addAssociatedTracks = False
            l1Jets.addJetCharge = False
    if jetCorrLabel != None:
        if jetCorrLabel == False : raise ValueError, "In addJetCollection 'jetCorrLabel' must be set to None, not False"
        if jetCorrLabel == "None": raise ValueError, "In addJetCollection 'jetCorrLabel' must be set to None (without quotes), not 'None'"
        if type(jetCorrLabel) != type(('IC5','Calo')): 
            raise ValueError, "In switchJetCollection 'jetCorrLabel' must be None, or a tuple ('Algo', 'Type')"
        if not hasattr( process, 'L2L3JetCorrector%s%s' % jetCorrLabel ):
            setattr( process, 
                        'L2L3JetCorrector%s%s' % jetCorrLabel, 
                        cms.ESSource("JetCorrectionServiceChain",
                            correctors = cms.vstring('L2RelativeJetCorrector%s%s' % jetCorrLabel,
                                                     'L3AbsoluteJetCorrector%s%s' % jetCorrLabel),
                            label      = cms.string('L2L3JetCorrector%s%s' % jetCorrLabel)
                        )
                    )
        if runCleaner != None:
            addClone('jetCorrFactors',       jetSource           = cms.InputTag(jetCollection), 
                                             defaultJetCorrector = cms.string('L2L3JetCorrector%s%s' % jetCorrLabel))
            addClone('layer0JetCorrFactors', association = cms.InputTag('jetCorrFactors'+postfixLabel),
                                             collection  = cms.InputTag(newLabel0),
                                             backrefs    = cms.InputTag(newLabel0))
            switchJECParameters( getattr(process,'jetCorrFactors'+postfixLabel), jetCorrLabel[0], jetCorrLabel[1], oldalgo='IC5',oldtype='Calo' )
        else:
            addAlso('layer0JetCorrFactors', process.jetCorrFactors.clone(
                                                jetSource           = cms.InputTag(jetCollection),
                                                defaultJetCorrector = cms.string('L2L3JetCorrector%s%s' % jetCorrLabel)))
            switchJECParameters( getattr(process,'layer0JetCorrFactors'+postfixLabel), jetCorrLabel[0], jetCorrLabel[1], oldalgo='IC5',oldtype='Calo' )
        if l1Jets != None:
            fixInputTag(l1Jets.jetCorrFactorsSource)
        if doType1MET:
            addClone('corMetType1Icone5', inputUncorJetsLabel = cms.string(jetCollection), 
                                          corrector = cms.string('L2L3JetCorrector%s%s' % jetCorrLabel))
            addClone('corMetType1Icone5Muons', uncorMETInputTag = cms.InputTag("corMetType1Icone5"+postfixLabel))
            addClone('allLayer0METs', metSource = cms.InputTag("corMetType1Icone5Muons"+postfixLabel))
            if l1Jets != None:
                addClone('allLayer1METs', metSource = cms.InputTag("allLayer0METs"+postfixLabel))
                l1MET = getattr(process, 'allLayer1METs'+postfixLabel)
                mettriggers = MassSearchParamVisitor('src', cms.InputTag("allLayer0METs"))
                process.patTrigMatch.visit(mettriggers)
                for mod in mettriggers.modules():
                    newmod = mod.clone(src = cms.InputTag("allLayer0METs"+postfixLabel))
                    setattr( process, mod.label() + postfixLabel, newmod )
                    process.patTrigMatch.replace( mod, mod * newmod )
                for it in l1MET.trigPrimMatch.value(): fixInputTag(it)
    else:
        if l1Jets != None:
            l1Jets.addJetCorrFactors = False
