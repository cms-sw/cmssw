import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import *

def switchJECParameters(jetCorrFactors,newalgo,newtype="Calo",oldalgo="IC5",oldtype="Calo"):
    """Replace input tags in the JetCorrFactorsProducer -- L5Flavor is taken out as it is said not to be dependend on the specific jet algorithm"""
    for k in ['L1Offset', 'L2Relative', 'L3Absolute', 'L4EMF', 'L6UE', 'L7Parton']:
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
    setattr( process, jtaLabel,  ic5JetTracksAssociatorAtVertex.clone(jets = jetCollection))
    setattr( process, ipTILabel, btag.impactParameterTagInfos.clone(jetTracks = cms.InputTag(jtaLabel)) )
    setattr( process, svTILabel, btag.secondaryVertexTagInfos.clone(trackIPTagInfos = cms.InputTag(ipTILabel)) )
    setattr( process, seTILabel, btag.softElectronTagInfos.clone(jets = jetCollection) )
    setattr( process, smTILabel, btag.softMuonTagInfos.clone(jets = jetCollection) )
    setattr( process, 'jetBProbabilityBJetTags'+label, btag.jetBProbabilityBJetTags.clone(tagInfos = vit(ipTILabel)) )
    setattr( process, 'jetProbabilityBJetTags' +label,  btag.jetProbabilityBJetTags.clone(tagInfos = vit(ipTILabel)) )
    setattr( process, 'trackCountingHighPurBJetTags'+label, btag.trackCountingHighPurBJetTags.clone(tagInfos = vit(ipTILabel)) )
    setattr( process, 'trackCountingHighEffBJetTags'+label, btag.trackCountingHighEffBJetTags.clone(tagInfos = vit(ipTILabel)) )
    setattr( process, 'simpleSecondaryVertexBJetTags'+label, btag.simpleSecondaryVertexBJetTags.clone(tagInfos = vit(svTILabel)) )
    setattr( process, 'combinedSecondaryVertexBJetTags'+label, btag.combinedSecondaryVertexBJetTags.clone(tagInfos = vit(ipTILabel, svTILabel)) )
    setattr( process, 'combinedSecondaryVertexMVABJetTags'+label, btag.combinedSecondaryVertexMVABJetTags.clone(tagInfos = vit(ipTILabel, svTILabel)) )
    setattr( process, 'softMuonBJetTags'+label, btag.softMuonBJetTags.clone(tagInfos = vit(smTILabel)) )
    setattr( process, 'softMuonByPtBJetTags'+label, btag.softMuonByPtBJetTags.clone(tagInfos = vit(smTILabel)) )
    setattr( process, 'softMuonByIP3dBJetTags'+label, btag.softMuonByIP3dBJetTags.clone(tagInfos = vit(smTILabel)) )
    setattr( process, 'softElectronByPtBJetTags'+label, btag.softElectronByPtBJetTags.clone(tagInfos = vit(smTILabel)) )
    setattr( process, 'softElectronByIP3dBJetTags'+label, btag.softElectronByIP3dBJetTags.clone(tagInfos = vit(smTILabel)) )
    
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
                                                'simpleSecondaryVertexBJetTags',
                                                'combinedSecondaryVertexBJetTags',
                                                'combinedSecondaryVertexMVABJetTags',
                                                'softElectronByPtBJetTags',
                                                'softElectronByIP3dBJetTags',
                                                'softMuonBJetTags',
                                                'softMuonByPtBJetTags',
                                                'softMuonByIP3dBJetTags') ]
    }
    
    setattr( process, 'btaggingTagInfos' + label, mkseq(process, *(labels['tagInfos']) ) )
    setattr( process, 'btaggingJetTags' + label,  mkseq(process, *(labels['jetTags'])  ) )
    seq = mkseq(process, jtaLabel, 'btaggingTagInfos' + label, 'btaggingJetTags' + label) 
    setattr( process, 'btagging' + label, seq )
    return (seq, labels)

def switchJetCollection(process,jetCollection,doJTA=True,doBTagging=True,jetCorrLabel=None,doType1MET=True,
                                genJetCollection=cms.InputTag("iterativeCone5GenJets")):
    """Switch the collection of jets in PAT from the default value.
          doBTagging  : True to run the BTagging sequence on top of this jets, and import it into PAT.
          doJTA       : Run Jet Tracks Association and Jet Charge (will be forced to True if doBTagging is true)
          jetCorrLabel: Name of the algorithm and jet type JEC to pick corrections from, or None for no JEC 
                        Examples are ('IC5','Calo'), ('SC7','Calo'), ('KT4','PF')
                        It tries to find a 'L2L3JetCorrector' + algo + type , or otherwise to create if as a 
                        JetCorrectionServiceChain of 'L2RelativeJetCorrector' and 'L3AbsoluteJetCorrector'
          doType1MET  : If jetCorrLabel is not 'None', set this to 'True' to remake Type1 MET from these jets
                        NOTE: at the moment it must be False for non-CaloJets otherwise the JetMET POG module crashes.
          genJetCollection : GenJet collection to match to."""
    oldLabel = process.allLayer1Jets.jetSource;
    process.jetPartonMatch.src        = jetCollection
    process.jetGenJetMatch.src        = jetCollection
    process.jetGenJetMatch.match      = genJetCollection
    process.jetPartonAssociation.jets = jetCollection
    process.allLayer1Jets.jetSource = jetCollection
    # quickly make VInputTag from strings
    def vit(*args) : return cms.VInputTag( *[ cms.InputTag(x) for x in args ] )
    if doBTagging :
        (btagSeq, btagLabels) = runBTagging(process, jetCollection, 'AOD')
        process.patAODReco.replace(process.patAODExtraReco, btagSeq + process.patAODExtraReco)
        process.patJetCharge.src                     = btagLabels['jta']
        process.allLayer1Jets.trackAssociationSource = btagLabels['jta']
        process.allLayer1Jets.tagInfoSources       = cms.VInputTag( *[ cms.InputTag(x) for x in btagLabels['tagInfos'] ] )
        process.allLayer1Jets.discriminatorSources = cms.VInputTag( *[ cms.InputTag(x) for x in btagLabels['jetTags']  ] )
    else:
        process.patAODReco.remove(process.patBTagging)
        process.allLayer1Jets.addBTagInfo = False
    if doJTA or doBTagging:
        if not doBTagging:
            process.load("RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi")
            from RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi import ic5JetTracksAssociatorAtVertex
            process.jetTracksAssociatorAtVertex = ic5JetTracksAssociatorAtVertex.clone(jets = jetCollection)
            process.patAODReco.replace(process.patJetTracksCharge, process.jetTracksAssociatorAtVertex + process.patJetTracksCharge)
            process.patJetCharge.src                     = 'jetTracksAssociatorAtVertex'
            process.allLayer1Jets.trackAssociationSource = 'jetTracksAssociatorAtVertex'
    else: ## no JTA
        process.patAODReco.remove(process.patJetTracksCharge)
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
            process.metJESCorIC5CaloJet.inputUncorJetsLabel = jetCollection.value() # FIXME it's metJESCorIC5CaloJet that's broken
            process.metJESCorIC5CaloJet.corrector           = 'L2L3JetCorrector%s%s' % jetCorrLabel
    else:
        process.patJetMETCorrections.remove(process.jetCorrFactors)
        process.allLayer1Jets.addJetCorrFactors = False
    ## Add this to the summary tables (not strictly needed, but useful)
    if oldLabel in process.aodSummary.candidates: 
        process.aodSummary.candidates[process.aodSummary.candidates.index(oldLabel)] = jetCollection
    else:
        process.aodSummary.candidates += [jetCollection]
        

def addJetCollection(process,jetCollection,postfixLabel,
                        doJTA=True,doBTagging=True,jetCorrLabel=None,doType1MET=True,doL1Counters=False,
                        genJetCollection=cms.InputTag("iterativeCone5GenJets")):
    """Add a new collection of jets in PAT from the default value.
          postfixLabel: Postpone this label to the name of all modules that work with these jet collection.
                        it can't be an empty string
          doBTagging  : True to run the BTagging sequence on top of this jets, and import it into PAT.
          doJTA       : Run Jet Tracks Association and Jet Charge (will be forced to True if doBTagging is true)
          jetCorrLabel: Name of the algorithm and jet type JEC to pick corrections from, or None for no JEC 
                        Examples are ('IC5','Calo'), ('SC7','Calo'), ('KT4','PF')
                        It tries to find a 'L2L3JetCorrector' + algo + type , or otherwise to create if as a 
                        JetCorrectionServiceChain of 'L2RelativeJetCorrector' and 'L3AbsoluteJetCorrector'
          doType1MET  : Make also a new MET (NOT IMPLEMENTED)
          doL1Counters: copy also the filter modules that accept/reject the event looking at the number of jets
          genJetCollection : GenJet collection to match to.

       Note: This takes the configuration from the already-configured jets, so if you do 
             replaces before calling addJetCollection then they will affect also the new jets
    """
    def addAlso (label,value):
        existing = getattr(process, label)
        setattr( process, label + postfixLabel, value)
        process.patDefaultSequence.replace( existing, existing * value )
    def addClone(label,**replaceStatements):
        new      = getattr(process, label).clone(**replaceStatements)
        addAlso(label, new)
    addClone('allLayer1Jets', jetSource = jetCollection)
    l1Jets = getattr(process, 'allLayer1Jets'+postfixLabel)
    addClone('selectedLayer1Jets', src=cms.InputTag('allLayer1Jets'+postfixLabel))
    addClone('cleanLayer1Jets', src=cms.InputTag('selectedLayer1Jets'+postfixLabel))
    if doL1Counters:
        addClone('countLayer1Jets', src=cms.InputTag('cleanLayer1Jets'+postfixLabel))
    addClone('jetPartonMatch',       src = jetCollection)
    addClone('jetGenJetMatch',       src = jetCollection)
    addClone('jetPartonAssociation', jets = jetCollection)
    addClone('jetFlavourAssociation',srcByReference = cms.InputTag('jetPartonAssociation' + postfixLabel))
    def fixInputTag(x): x.setModuleLabel(x.moduleLabel+postfixLabel)
    def fixVInputTag(x): x[0].setModuleLabel(x[0].moduleLabel+postfixLabel)
    fixInputTag(l1Jets.JetPartonMapSource)
    fixInputTag(l1Jets.genJetMatch)
    fixInputTag(l1Jets.genPartonMatch)
    def vit(*args) : return cms.VInputTag( *[ cms.InputTag(x) for x in args ] )
    if doBTagging :
        (btagSeq, btagLabels) = runBTagging(process, jetCollection, postfixLabel)
        process.patAODReco.replace(process.patAODExtraReco, btagSeq + process.patAODExtraReco)
        addClone('patJetCharge', src=cms.InputTag(btagLabels['jta']))
        l1Jets.trackAssociationSource = cms.InputTag(btagLabels['jta'])
        l1Jets.tagInfoSources         = cms.VInputTag( *[ cms.InputTag(x) for x in btagLabels['tagInfos'] ] )
        l1Jets.discriminatorSources   = cms.VInputTag( *[ cms.InputTag(x) for x in btagLabels['jetTags']  ] )
        fixInputTag(l1Jets.jetChargeSource)
    else:
       l1Jets.addBTagInfo = False 
    if doJTA or doBTagging:
        if not doBTagging:
            process.load("RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi")
            from RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi import ic5JetTracksAssociatorAtVertex
            jtaLabel = 'jetTracksAssociatorAtVertex' + postfixLabel
            setattr( process, jtaLabel, ic5JetTracksAssociatorAtVertex.clone(jets = jetCollection) )
            process.patAODReco.replace(process.patJetTracksCharge, getattr(process,jtaLabel) + process.patJetTracksCharge)
            l1Jets.trackAssociationSource = cms.InputTag(jtaLabel)
            addClone('patJetCharge', src=cms.InputTag(jtaLabel)),
            fixInputTag(l1Jets.jetChargeSource)
    else: ## no JTA
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
        addClone('jetCorrFactors',       jetSource           = jetCollection) 
        switchJECParameters( getattr(process,'jetCorrFactors'+postfixLabel), jetCorrLabel[0], jetCorrLabel[1], oldalgo='IC5',oldtype='Calo' )
        fixVInputTag(l1Jets.jetCorrFactorsSource)
        if doType1MET:
            addClone('metJESCorIC5CaloJet', inputUncorJetsLabel = jetCollection.value(),
                                          corrector = cms.string('L2L3JetCorrector%s%s' % jetCorrLabel))
            addClone('metJESCorIC5CaloJetMuons', uncorMETInputTag = cms.InputTag("metJESCorIC5CaloJet"+postfixLabel))
            addClone('layer1METs',              metSource = cms.InputTag("metJESCorIC5CaloJetMuons"+postfixLabel))
            l1MET = getattr(process, 'layer1METs'+postfixLabel)
            process.allLayer1Summary.candidates += [ cms.InputTag('layer1METs'+postfixLabel) ]
    else:
        l1Jets.addJetCorrFactors = False
    ## Add this to the summary tables (not strictly needed, but useful)
    if jetCollection not in process.aodSummary.candidates: 
        process.aodSummary.candidates += [ jetCollection ]
    process.allLayer1Summary.candidates      += [ cms.InputTag('allLayer1Jets'+postfixLabel) ]
    process.selectedLayer1Summary.candidates += [ cms.InputTag('selectedLayer1Jets'+postfixLabel) ]
