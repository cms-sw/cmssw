from __future__ import print_function
from PhysicsTools.PatAlgos.tools.ConfigToolBase import *
from FWCore.ParameterSet.Mixins import PrintOptions,_ParameterTypeBase,_SimpleParameterTypeBase, _Parameterizable, _ConfigureComponent, _TypedParameterizable, _Labelable,  _Unlabelable,  _ValidatingListBase
from FWCore.ParameterSet.SequenceTypes import _ModuleSequenceType, _Sequenceable
from FWCore.ParameterSet.SequenceTypes import *
from PhysicsTools.PatAlgos.tools.helpers import *
from PhysicsTools.PatAlgos.recoLayer0.bTagging_cff import *
import sys
from FWCore.ParameterSet.MassReplace import MassSearchReplaceAnyInputTagVisitor
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import pfParticleNetFromMiniAODAK4PuppiCentralTagInfos,pfParticleNetFromMiniAODAK4PuppiCentralJetTags,pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import pfParticleNetFromMiniAODAK4PuppiForwardTagInfos,pfParticleNetFromMiniAODAK4PuppiForwardJetTags,pfParticleNetFromMiniAODAK4PuppiForwardDiscriminatorsJetTags
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import pfParticleNetFromMiniAODAK4CHSCentralTagInfos,pfParticleNetFromMiniAODAK4CHSCentralJetTags,pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import pfParticleNetFromMiniAODAK4CHSForwardTagInfos,pfParticleNetFromMiniAODAK4CHSForwardJetTags,pfParticleNetFromMiniAODAK4CHSForwardDiscriminatorsJetTags
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK8_cff import pfParticleNetFromMiniAODAK8TagInfos,pfParticleNetFromMiniAODAK8JetTags,pfParticleNetFromMiniAODAK8DiscriminatorsJetTags

## dictionary with supported jet clustering algorithms
supportedJetAlgos = {
   'ak' : 'AntiKt'
 , 'ca' : 'CambridgeAachen'
 , 'kt' : 'Kt'
}

def checkJetCorrectionsFormat(jetCorrections):
    ## check for the correct format
    if not isinstance(jetCorrections, type(('PAYLOAD-LABEL',['CORRECTION-LEVEL-A','CORRECTION-LEVEL-B'], 'MET-LABEL'))):
        raise ValueError("In addJetCollection: 'jetCorrections' must be 'None' (as a python value w/o quotation marks), or of type ('PAYLOAD-LABEL', ['CORRECTION-LEVEL-A', \
        'CORRECTION-LEVEL-B', ...], 'MET-LABEL'). Note that 'MET-LABEL' can be set to 'None' (as a string in quotation marks) in case you do not want to apply MET(Type1) \
        corrections.")


def setupJetCorrections(process, knownModules, jetCorrections, jetSource, pvSource, patJets, labelName, postfix):

    task = getPatAlgosToolsTask(process)

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
        raise TypeError("In addJetCollection: Jet energy corrections are only supported for PF, JPT and Calo jets.")
    from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import patJetCorrFactors
    if 'patJetCorrFactors'+labelName+postfix in knownModules :
        _newPatJetCorrFactors=getattr(process, 'patJetCorrFactors'+labelName+postfix)
        _newPatJetCorrFactors.src=jetSource
        _newPatJetCorrFactors.primaryVertices=pvSource
    else:
        addToProcessAndTask('patJetCorrFactors'+labelName+postfix,
                            patJetCorrFactors.clone(src=jetSource, primaryVertices=pvSource),
                            process, task)
        _newPatJetCorrFactors=getattr(process, "patJetCorrFactors"+labelName+postfix)
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
                raise ValueError("In addJetCollection: Correction levels for jet energy corrections are miss configured. An L1 correction type should appear not more than \
                once. Check the list of correction levels you requested to be applied: "+ jetCorrections[1])
        if x == 'L1FastJet' :
            if not error :
                if _type == "JPT" :
                    raise TypeError("In addJetCollection: L1FastJet corrections are only supported for PF and Calo jets.")
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
                raise ValueError("In addJetCollection: Correction levels for jet energy corrections are miss configured. An L1 correction type should appear not more than \
                once. Check the list of correction levels you requested to be applied: "+ jetCorrections[1])
    patJets.jetCorrFactorsSource=cms.VInputTag(cms.InputTag('patJetCorrFactors'+labelName+postfix))
    ## configure MET(Type1) corrections
    if jetCorrections[2].lower() != 'none' and jetCorrections[2] != '':
        if not jetCorrections[2].lower() == 'type-1' and not jetCorrections[2].lower() == 'type-2':
            raise ValueError("In addJetCollection: Wrong choice of MET corrections for new jet collection. Possible choices are None (or empty string), Type-1, Type-2 (i.e.\
            Type-1 and Type-2 corrections applied). This choice is not case sensitive. Your choice was: "+ jetCorrections[2])
        if _type == "JPT":
            raise ValueError("In addJecCollection: MET(type1) corrections are not supported for JPTJets. Please set the MET-LABEL to \"None\" (as string in quatiation \
            marks) and use raw tcMET together with JPTJets.")
        ## set up jet correctors for MET corrections
        process.load( "JetMETCorrections.Configuration.JetCorrectorsAllAlgos_cff") # FIXME: This adds a lot of garbage
        # I second the FIXME comment on the last line. When I counted it, this brought in 344 EDProducers
        # to be available to run unscheduled. All jet correctors, probably some small fraction of which
        # are actually used.
        task.add(process.jetCorrectorsAllAlgosTask)
        _payloadType = jetCorrections[0].split(_type)[0].lower()+_type
        if "PF" in _type :
            addToProcessAndTask(jetCorrections[0]+'L1FastJet',
                                getattr(process, _payloadType+'L1FastjetCorrector').clone(srcRho=cms.InputTag('fixedGridRhoFastjetAll')),
                                process, task)
        else :
            addToProcessAndTask(jetCorrections[0]+'L1FastJet',
                                getattr(process, _payloadType+'L1FastjetCorrector').clone(srcRho=cms.InputTag('fixedGridRhoFastjetAllCalo')),
                                process, task)
        addToProcessAndTask(jetCorrections[0]+'L1Offset', getattr(process, _payloadType+'L1OffsetCorrector').clone(), process, task)
        addToProcessAndTask(jetCorrections[0]+'L2Relative', getattr(process, _payloadType+'L2RelativeCorrector').clone(), process, task)
        addToProcessAndTask(jetCorrections[0]+'L3Absolute', getattr(process, _payloadType+'L3AbsoluteCorrector').clone(), process, task)
        addToProcessAndTask(jetCorrections[0]+'L2L3Residual', getattr(process, _payloadType+'ResidualCorrector').clone(), process, task)
        addToProcessAndTask(jetCorrections[0]+'CombinedCorrector',
                            cms.EDProducer( 'ChainedJetCorrectorProducer', correctors = cms.VInputTag()),
                            process, task)
        for x in jetCorrections[1]:
            if x != 'L1FastJet' and x != 'L1Offset' and x != 'L2Relative' and x != 'L3Absolute' and x != 'L2L3Residual':
                raise ValueError('In addJetCollection: Unsupported JEC for MET(Type1). Currently supported jet correction levels are L1FastJet, L1Offset, L2Relative, L3Asolute, L2L3Residual. Requested was: %s'%(x))
            else:
                _corrector = _payloadType
                if x == 'L1FastJet':
                  _corrector += 'L1Fastjet'
                elif x  == 'L2L3Residual':
                  _corrector += 'Residual'
                else:
                  _corrector += x
                _corrector += 'Corrector'
                getattr(process, jetCorrections[0]+'CombinedCorrector').correctors.append(cms.InputTag(_corrector))

        ## set up MET(Type1) correction modules
        _labelCorrName = labelName
        if labelName != '':
            _labelCorrName = 'For' + labelName
        if _type == 'Calo':
            from JetMETCorrections.Type1MET.correctionTermsCaloMet_cff import corrCaloMetType1
            from JetMETCorrections.Type1MET.correctionTermsCaloMet_cff import corrCaloMetType2
            from JetMETCorrections.Type1MET.correctedMet_cff import caloMetT1
            from JetMETCorrections.Type1MET.correctedMet_cff import caloMetT1T2
            addToProcessAndTask(
                jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix,
                corrCaloMetType1.clone(src=jetSource,srcMET = "caloMetM",jetCorrLabel = cms.InputTag(jetCorrections[0]+'CombinedCorrector')),
                process, task)
            addToProcessAndTask(
                jetCorrections[0]+_labelCorrName+'JetMETcorr2'+postfix,
                corrCaloMetType2.clone(srcUnclEnergySums = cms.VInputTag(
                    cms.InputTag(jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix, 'type2'),
                    cms.InputTag(jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix, 'offset'),
                    cms.InputTag('muCaloMetCorr'))),
                process, task)
            addToProcessAndTask(
                jetCorrections[0]+_labelCorrName+'Type1CorMet'+postfix,
                caloMetT1.clone(src = "caloMetM", srcCorrections = cms.VInputTag(
                    cms.InputTag(jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix, 'type1'))),
                process, task)
            addToProcessAndTask(jetCorrections[0]+_labelCorrName+'Type1p2CorMet'+postfix,
                                caloMetT1T2.clone(src = "caloMetM", srcCorrections = cms.VInputTag(
                                    cms.InputTag(jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix, 'type1'),
                                    cms.InputTag(jetCorrections[0]+_labelCorrName+'JetMETcorr2'+postfix))),
                                process, task)
        elif _type == 'PF':
            from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import pfJetsPtrForMetCorr
            from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import pfCandsNotInJetsPtrForMetCorr
            from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import pfCandsNotInJetsForMetCorr
            from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import pfCandMETcorr
            from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import corrPfMetType1
            from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import corrPfMetType2
            from JetMETCorrections.Type1MET.correctedMet_cff import pfMetT1
            from JetMETCorrections.Type1MET.correctedMet_cff import pfMetT1T2
            addToProcessAndTask(jetCorrections[0]+_labelCorrName+'pfJetsPtrForMetCorr'+postfix,
                                pfJetsPtrForMetCorr.clone(src = jetSource), process, task)
            addToProcessAndTask(
                jetCorrections[0]+_labelCorrName+'pfCandsNotInJetsPtrForMetCorr'+postfix,
                pfCandsNotInJetsPtrForMetCorr.clone(topCollection = jetCorrections[0]+_labelCorrName+'pfJetsPtrForMetCorr'+postfix),
                process, task)
            addToProcessAndTask(
                jetCorrections[0]+_labelCorrName+'pfCandsNotInJetsForMetCorr'+postfix,
                pfCandsNotInJetsForMetCorr.clone(src = jetCorrections[0]+_labelCorrName+'pfCandsNotInJetsPtrForMetCorr'+postfix),
                process, task)
            addToProcessAndTask(
                jetCorrections[0]+_labelCorrName+'CandMETcorr'+postfix,
                pfCandMETcorr.clone(src = cms.InputTag(jetCorrections[0]+_labelCorrName+'pfCandsNotInJetsForMetCorr'+postfix)),
                process, task)
            addToProcessAndTask(
                jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix,
                corrPfMetType1.clone(src = jetSource, jetCorrLabel = cms.InputTag(jetCorrections[0]+'CombinedCorrector')),
                process, task) # FIXME: Originally w/o jet corrections?
            addToProcessAndTask(jetCorrections[0]+_labelCorrName+'corrPfMetType2'+postfix,
                                corrPfMetType2.clone(srcUnclEnergySums = cms.VInputTag(
                                    cms.InputTag(jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix, 'type2'),
                                    cms.InputTag(jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix, 'offset'),
                                    cms.InputTag(jetCorrections[0]+_labelCorrName+'CandMETcorr'+postfix))),
                                process, task)
            addToProcessAndTask(jetCorrections[0]+_labelCorrName+'Type1CorMet'+postfix,
                                pfMetT1.clone(srcCorrections = cms.VInputTag(
                                    cms.InputTag(jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix, 'type1'))),
                                process, task)
            addToProcessAndTask(jetCorrections[0]+_labelCorrName+'Type1p2CorMet'+postfix,
                                pfMetT1T2.clone(srcCorrections = cms.VInputTag(
                                    cms.InputTag(jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix, 'type1'),
                                    jetCorrections[0]+_labelCorrName+'corrPfMetType2'+postfix)),
                                process, task)
            if 'Puppi' in jetSource.value() and pfCandidates.value() == 'particleFlow':
                getattr(process,jetCorrections[0]+_labelCorrName+'CandMETcorr'+postfix).srcWeights = "puppiNoLep"

        ## common configuration for Calo and PF
        if ('L1FastJet' in jetCorrections[1] or 'L1Fastjet' in jetCorrections[1]):
            getattr(process,jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix).offsetCorrLabel = cms.InputTag(jetCorrections[0]+'L1FastJet')
        #FIXME: What is wrong here?
        #elif ('L1Offset' in jetCorrections[1]):
            #getattr(process,jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix).offsetCorrLabel = cms.InputTag(jetCorrections[0]+'L1Offset')
        else:
            getattr(process,jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix).offsetCorrLabel = cms.InputTag('')

        from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import patMETs
        if jetCorrections[2].lower() == 'type-1':
            addToProcessAndTask('patMETs'+labelName+postfix,
                                patMETs.clone(metSource = cms.InputTag(jetCorrections[0]+_labelCorrName+'Type1CorMet'+postfix),
                                              addMuonCorrections = False),
                                process, task)
        elif jetCorrections[2].lower() == 'type-2':
            addToProcessAndTask('patMETs'+labelName+postfix,
                                patMETs.clone(metSource = cms.InputTag(jetCorrections[0]+_labelCorrName+'Type1p2CorMet'+postfix),
                                              addMuonCorrections = False),
                                process, task)


def setupSVClustering(btagInfo, svClustering, algo, rParam, fatJets=cms.InputTag(''), groomedFatJets=cms.InputTag('')):
    btagInfo.useSVClustering = cms.bool(svClustering)
    btagInfo.jetAlgorithm = cms.string(algo)
    btagInfo.rParam = cms.double(rParam)
    ## if the jet is actually a subjet
    if fatJets != cms.InputTag(''):
        btagInfo.fatJets = fatJets
        if groomedFatJets != cms.InputTag(''):
            btagInfo.groomedFatJets = groomedFatJets

def setupPackedPuppi(process):
    task = getPatAlgosToolsTask(process)
    packedPuppiName = "packedpuppi"
    if not hasattr(process,packedPuppiName):
        from CommonTools.PileupAlgos.Puppi_cff import puppi
        addToProcessAndTask(packedPuppiName, puppi.clone(
            useExistingWeights = True,
            candName = 'packedPFCandidates',
            vertexName = 'offlineSlimmedPrimaryVertices') , process, task)
    return packedPuppiName

def setupBTagging(process, jetSource, pfCandidates, explicitJTA, pvSource, svSource, elSource, muSource, runIVF, tightBTagNTkHits, loadStdRecoBTag, svClustering, fatJets, groomedFatJets,
                  algo, rParam, btagDiscriminators, btagInfos, patJets, labelName, btagPrefix, postfix):

    task = getPatAlgosToolsTask(process)

    ## expand the btagDiscriminators to remove the meta taggers and substitute the equivalent sources
    discriminators = set(btagDiscriminators)
    present_metaSet = discriminators.intersection(set(supportedMetaDiscr.keys()))
    discriminators -= present_metaSet
    for meta_tagger in present_metaSet:
        for src in supportedMetaDiscr[meta_tagger]:
            discriminators.add(src)
    present_meta = sorted(present_metaSet)
    btagDiscriminators = sorted(discriminators)

    ## expand tagInfos to what is explicitly required by user + implicit
    ## requirements that come in from one or the other discriminator
    requiredTagInfos = list(btagInfos)
    for btagDiscr in btagDiscriminators :
        for tagInfoList in supportedBtagDiscr[btagDiscr] :
            for requiredTagInfo in tagInfoList :
                tagInfoCovered = False
                for tagInfo in requiredTagInfos :
                    if requiredTagInfo == tagInfo :
                        tagInfoCovered = True
                        break
                if not tagInfoCovered :
                    requiredTagInfos.append(requiredTagInfo)
    ## load sequences and setups needed for btagging
    if hasattr( process, 'candidateJetProbabilityComputer' ) == False :
        if loadStdRecoBTag: # also loading modules already run in the standard reconstruction
            process.load("RecoBTag.ImpactParameter.impactParameter_cff")
            task.add(process.impactParameterTask)
            process.load("RecoBTag.SecondaryVertex.secondaryVertex_cff")
            task.add(process.secondaryVertexTask)
            process.load("RecoBTag.SoftLepton.softLepton_cff")
            task.add(process.softLeptonTask)
            process.load("RecoBTag.Combined.combinedMVA_cff")
            task.add(process.combinedMVATask)
            process.load("RecoBTag.CTagging.cTagging_cff")
            task.add(process.cTaggingTask)
        else: # to prevent loading of modules already run in the standard reconstruction
            process.load("RecoBTag.ImpactParameter.impactParameter_EventSetup_cff")
            process.load("RecoBTag.SecondaryVertex.secondaryVertex_EventSetup_cff")
            process.load("RecoBTag.SoftLepton.softLepton_EventSetup_cff")
            process.load("RecoBTag.Combined.combinedMVA_EventSetup_cff")
            process.load("RecoBTag.CTagging.cTagging_EventSetup_cff")
    import RecoBTag.Configuration.RecoBTag_cff as btag
    import RecoJets.JetProducers.caTopTaggers_cff as toptag

    if tightBTagNTkHits:
        if not runIVF:
            sys.stderr.write("-------------------------------------------------------------------\n")
            sys.stderr.write(" Warning: For a complete switch to the legacy tight b-tag track\n")
            sys.stderr.write("          selection, please also enable the \'runIVF\' switch.\n")
            sys.stderr.write("-------------------------------------------------------------------\n")
        if btagPrefix == '':
            sys.stderr.write("-------------------------------------------------------------------\n")
            sys.stderr.write(" Warning: With the tight b-tag track selection enabled, it is\n")
            sys.stderr.write("          advisable to set \'btagPrefix\' to a non-empty string to\n")
            sys.stderr.write("          avoid unintentional modifications to the default\n")
            sys.stderr.write("          b tagging setup that might be loaded in the same job.\n")
            sys.stderr.write("-------------------------------------------------------------------\n")

    ## define c tagging CvsL SV source (for now tied to the default SV source
    ## in the first part of the module label, product instance label and process name)
    svSourceCvsL = copy.deepcopy(svSource)
    svSourceCvsL.setModuleLabel(svSource.getModuleLabel()+'CvsL')

    ## check if and under what conditions to re-run IVF
    runIVFforCTagOnly = False
    ivfcTagInfos = ['pfInclusiveSecondaryVertexFinderCvsLTagInfos', 'pfInclusiveSecondaryVertexFinderNegativeCvsLTagInfos']
    ## if MiniAOD and running c tagging
    if pvSource.getModuleLabel() == 'offlineSlimmedPrimaryVertices' and any(i in requiredTagInfos for i in ivfcTagInfos) and not runIVF:
        runIVFforCTagOnly = True
        runIVF = True
        sys.stderr.write("-------------------------------------------------------------------\n")
        sys.stderr.write(" Info: To run c tagging on MiniAOD, c-tag-specific IVF secondary\n")
        sys.stderr.write("       vertices will be remade.\n")
        sys.stderr.write("-------------------------------------------------------------------\n")
    ## adjust svSources
    if runIVF and btagPrefix != '':
        if runIVFforCTagOnly:
            svSourceCvsL.setModuleLabel(btagPrefix+svSourceCvsL.getModuleLabel())
        else:
            svSource.setModuleLabel(btagPrefix+svSource.getModuleLabel())
            svSourceCvsL.setModuleLabel(btagPrefix+svSourceCvsL.getModuleLabel())

    ## setup all required btagInfos : we give a dedicated treatment for different
    ## types of tagInfos here. A common treatment is possible but might require a more
    ## general approach anyway in coordination with the btagging POG.

    runNegativeVertexing = False
    runNegativeCvsLVertexing = False
    for btagInfo in requiredTagInfos:
        if btagInfo in (
            'pfInclusiveSecondaryVertexFinderNegativeTagInfos',
            'pfNegativeDeepFlavourTagInfos',
            'pfNegativeParticleNetAK4TagInfos',
            ):
            runNegativeVertexing = True
        if btagInfo == 'pfInclusiveSecondaryVertexFinderNegativeCvsLTagInfos':
            runNegativeCvsLVertexing = True
            
    if runNegativeVertexing or runNegativeCvsLVertexing:
        import RecoVertex.AdaptiveVertexFinder.inclusiveNegativeVertexing_cff as NegVertex

    if runNegativeVertexing:                
        addToProcessAndTask(btagPrefix+'inclusiveCandidateNegativeVertexFinder'+labelName+postfix,
                            NegVertex.inclusiveCandidateNegativeVertexFinder.clone(primaryVertices = pvSource,tracks=pfCandidates),
                            process, task)
        addToProcessAndTask(btagPrefix+'candidateNegativeVertexMerger'+labelName+postfix,
                            NegVertex.candidateNegativeVertexMerger.clone(secondaryVertices = cms.InputTag(btagPrefix+'inclusiveCandidateNegativeVertexFinder'+labelName+postfix)),
                            process, task)
        addToProcessAndTask(btagPrefix+'candidateNegativeVertexArbitrator'+labelName+postfix,
                            NegVertex.candidateNegativeVertexArbitrator.clone( secondaryVertices = cms.InputTag(btagPrefix+'candidateNegativeVertexMerger'+labelName+postfix)
                                                                               ,primaryVertices = pvSource
                                                                               ,tracks=pfCandidates),
                            process, task)
        addToProcessAndTask(btagPrefix+'inclusiveCandidateNegativeSecondaryVertices'+labelName+postfix,
                            NegVertex.inclusiveCandidateNegativeSecondaryVertices.clone(secondaryVertices = cms.InputTag(btagPrefix+'candidateNegativeVertexArbitrator'+labelName+postfix)),
                            process, task)

    if runNegativeCvsLVertexing:
        addToProcessAndTask(btagPrefix+'inclusiveCandidateNegativeVertexFinderCvsL'+labelName+postfix,
                            NegVertex.inclusiveCandidateNegativeVertexFinderCvsL.clone(primaryVertices = pvSource,tracks=pfCandidates),
                            process, task)
        addToProcessAndTask(btagPrefix+'candidateNegativeVertexMergerCvsL'+labelName+postfix,
                            NegVertex.candidateNegativeVertexMergerCvsL.clone(secondaryVertices = cms.InputTag(btagPrefix+'inclusiveCandidateNegativeVertexFinderCvsL'+labelName+postfix)),
                            process, task)
        addToProcessAndTask(btagPrefix+'candidateNegativeVertexArbitratorCvsL'+labelName+postfix,
                            NegVertex.candidateNegativeVertexArbitratorCvsL.clone( secondaryVertices = cms.InputTag(btagPrefix+'candidateNegativeVertexMergerCvsL'+labelName+postfix)
                                                                               ,primaryVertices = pvSource
                                                                               ,tracks=pfCandidates),
                            process, task)
        addToProcessAndTask(btagPrefix+'inclusiveCandidateNegativeSecondaryVerticesCvsL'+labelName+postfix,
                            NegVertex.inclusiveCandidateNegativeSecondaryVerticesCvsL.clone(secondaryVertices = cms.InputTag(btagPrefix+'candidateNegativeVertexArbitratorCvsL'+labelName+postfix)),
                            process, task)


    acceptedTagInfos = list()
    for btagInfo in requiredTagInfos:
        if hasattr(btag,btagInfo):
            if btagInfo == 'pfImpactParameterTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfImpactParameterTagInfos.clone(jets = jetSource,primaryVertex=pvSource,candidates=pfCandidates),
                                    process, task)
                if explicitJTA:
                    _btagInfo = getattr(process, btagPrefix+btagInfo+labelName+postfix)
                    _btagInfo.explicitJTA = cms.bool(explicitJTA)
                if tightBTagNTkHits:
                    _btagInfo = getattr(process, btagPrefix+btagInfo+labelName+postfix)
                    _btagInfo.minimumNumberOfPixelHits = cms.int32(2)
                    _btagInfo.minimumNumberOfHits = cms.int32(8)
            if btagInfo == 'pfImpactParameterAK8TagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfImpactParameterAK8TagInfos.clone(jets = jetSource,primaryVertex=pvSource,candidates=pfCandidates),
                                    process, task)
                if explicitJTA:
                    _btagInfo = getattr(process, btagPrefix+btagInfo+labelName+postfix)
                    _btagInfo.explicitJTA = cms.bool(explicitJTA)
                if tightBTagNTkHits:
                    _btagInfo = getattr(process, btagPrefix+btagInfo+labelName+postfix)
                    _btagInfo.minimumNumberOfPixelHits = cms.int32(2)
                    _btagInfo.minimumNumberOfHits = cms.int32(8)
            if btagInfo == 'pfImpactParameterCA15TagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfImpactParameterCA15TagInfos.clone(jets = jetSource,primaryVertex=pvSource,candidates=pfCandidates),
                                    process, task)
                if explicitJTA:
                    _btagInfo = getattr(process, btagPrefix+btagInfo+labelName+postfix)
                    _btagInfo.explicitJTA = cms.bool(explicitJTA)
                if tightBTagNTkHits:
                    _btagInfo = getattr(process, btagPrefix+btagInfo+labelName+postfix)
                    _btagInfo.minimumNumberOfPixelHits = cms.int32(2)
                    _btagInfo.minimumNumberOfHits = cms.int32(8)
            if btagInfo == 'pfSecondaryVertexTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfSecondaryVertexTagInfos.clone(
                                        trackIPTagInfos = cms.InputTag(btagPrefix+'pfImpactParameterTagInfos'+labelName+postfix)),
                                    process, task)
                if tightBTagNTkHits:
                    _btagInfo = getattr(process, btagPrefix+btagInfo+labelName+postfix)
                    _btagInfo.trackSelection.pixelHitsMin = cms.uint32(2)
                    _btagInfo.trackSelection.totalHitsMin = cms.uint32(8)
            if btagInfo == 'pfDeepCSVTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfDeepCSVTagInfos.clone(
                                        svTagInfos = cms.InputTag(btagPrefix+'pfInclusiveSecondaryVertexFinderTagInfos'+labelName+postfix)),
                                    process, task)
                if svClustering or fatJets != cms.InputTag(''):
                    setupSVClustering(getattr(process, btagPrefix+btagInfo+labelName+postfix), svClustering, algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'pfDeepCSVNegativeTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfDeepCSVNegativeTagInfos.clone(
                                        svTagInfos = cms.InputTag(btagPrefix+'pfInclusiveSecondaryVertexFinderNegativeTagInfos'+labelName+postfix)),
                                    process, task)
                if svClustering or fatJets != cms.InputTag(''):
                    setupSVClustering(getattr(process, btagPrefix+btagInfo+labelName+postfix), svClustering, algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'pfDeepCSVPositiveTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfDeepCSVPositiveTagInfos.clone(
                                        svTagInfos = cms.InputTag(btagPrefix+'pfInclusiveSecondaryVertexFinderTagInfos'+labelName+postfix)),
                                    process, task)
                if svClustering or fatJets != cms.InputTag(''):
                    setupSVClustering(getattr(process, btagPrefix+btagInfo+labelName+postfix), svClustering, algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'pfDeepCMVATagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfDeepCMVATagInfos.clone(
                                        deepNNTagInfos = cms.InputTag(btagPrefix+'pfDeepCSVTagInfos'+labelName+postfix),
                                        ipInfoSrc = cms.InputTag(btagPrefix+"pfImpactParameterTagInfos"+labelName+postfix),
                                        muInfoSrc = cms.InputTag(btagPrefix+"softPFMuonsTagInfos"+labelName+postfix),
                                        elInfoSrc = cms.InputTag(btagPrefix+"softPFElectronsTagInfos"+labelName+postfix)),
                                    process, task)
                if svClustering or fatJets != cms.InputTag(''):
                    setupSVClustering(getattr(process, btagPrefix+btagInfo+labelName+postfix), svClustering, algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'pfDeepCMVANegativeTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfDeepCMVATagInfos.clone(
                                        deepNNTagInfos = cms.InputTag(btagPrefix+'pfDeepCSVTagInfos'+labelName+postfix),
                                        ipInfoSrc = cms.InputTag(btagPrefix+"pfImpactParameterTagInfos"+labelName+postfix),
                                        muInfoSrc = cms.InputTag(btagPrefix+"softPFMuonsTagInfos"+labelName+postfix),
                                        elInfoSrc = cms.InputTag(btagPrefix+"softPFElectronsTagInfos"+labelName+postfix)),
                                    process, task)
                if svClustering or fatJets != cms.InputTag(''):
                    setupSVClustering(getattr(process, btagPrefix+btagInfo+labelName+postfix), svClustering, algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'pfDeepCMVAPositiveTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfDeepCMVATagInfos.clone(
                                        deepNNTagInfos = cms.InputTag(btagPrefix+'pfDeepCSVTagInfos'+labelName+postfix),
                                        ipInfoSrc = cms.InputTag(btagPrefix+"pfImpactParameterTagInfos"+labelName+postfix),
                                        muInfoSrc = cms.InputTag(btagPrefix+"softPFMuonsTagInfos"+labelName+postfix),
                                        elInfoSrc = cms.InputTag(btagPrefix+"softPFElectronsTagInfos"+labelName+postfix)),
                                    process, task)
                if svClustering or fatJets != cms.InputTag(''):
                    setupSVClustering(getattr(process, btagPrefix+btagInfo+labelName+postfix), svClustering, algo, rParam, fatJets, groomedFatJets)     
            if btagInfo == 'pfInclusiveSecondaryVertexFinderTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfInclusiveSecondaryVertexFinderTagInfos.clone(
                                        trackIPTagInfos = cms.InputTag(btagPrefix+'pfImpactParameterTagInfos'+labelName+postfix),
                                        extSVCollection=svSource),
                                    process, task)
                if svClustering or fatJets != cms.InputTag(''):
                    setupSVClustering(getattr(process, btagPrefix+btagInfo+labelName+postfix), svClustering, algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'pfInclusiveSecondaryVertexFinderAK8TagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfInclusiveSecondaryVertexFinderAK8TagInfos.clone(
                                        trackIPTagInfos = cms.InputTag(btagPrefix+'pfImpactParameterAK8TagInfos'+labelName+postfix),
                                        extSVCollection=svSource),
                                    process, task)
                if svClustering or fatJets != cms.InputTag(''):
                    setupSVClustering(getattr(process, btagPrefix+btagInfo+labelName+postfix), svClustering, algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'pfBoostedDoubleSVAK8TagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfBoostedDoubleSVAK8TagInfos.clone(
                                        svTagInfos = cms.InputTag(btagPrefix+'pfInclusiveSecondaryVertexFinderAK8TagInfos'+labelName+postfix)),
                                    process, task)
            if btagInfo == 'pfInclusiveSecondaryVertexFinderCA15TagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfInclusiveSecondaryVertexFinderCA15TagInfos.clone(
                                        trackIPTagInfos = cms.InputTag(btagPrefix+'pfImpactParameterCA15TagInfos'+labelName+postfix),
                                        extSVCollection=svSource),
                                    process, task)
                if svClustering or fatJets != cms.InputTag(''):
                    setupSVClustering(getattr(process, btagPrefix+btagInfo+labelName+postfix), svClustering, algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'pfBoostedDoubleSVCA15TagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfBoostedDoubleSVCA15TagInfos.clone(
                                        svTagInfos = cms.InputTag(btagPrefix+'pfInclusiveSecondaryVertexFinderCA15TagInfos'+labelName+postfix)),
                                    process, task)
            if btagInfo == 'pfInclusiveSecondaryVertexFinderCvsLTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfInclusiveSecondaryVertexFinderCvsLTagInfos.clone(
                                        trackIPTagInfos = cms.InputTag(btagPrefix+'pfImpactParameterTagInfos'+labelName+postfix),
                                        extSVCollection=svSourceCvsL),
                                    process, task)
                if svClustering or fatJets != cms.InputTag(''):
                    setupSVClustering(getattr(process, btagPrefix+btagInfo+labelName+postfix), svClustering, algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'pfInclusiveSecondaryVertexFinderNegativeCvsLTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfInclusiveSecondaryVertexFinderNegativeCvsLTagInfos.clone(
                                        trackIPTagInfos = cms.InputTag(btagPrefix+'pfImpactParameterTagInfos'+labelName+postfix),
                                        extSVCollection = btagPrefix+'inclusiveCandidateNegativeSecondaryVerticesCvsL'+labelName+postfix),
                                    process, task)
                if svClustering or fatJets != cms.InputTag(''):
                    setupSVClustering(getattr(process, btagPrefix+btagInfo+labelName+postfix), svClustering, algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'pfGhostTrackVertexTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfGhostTrackVertexTagInfos.clone(
                                        trackIPTagInfos = cms.InputTag(btagPrefix+'pfImpactParameterTagInfos'+labelName+postfix)),
                                    process, task)
            if btagInfo == 'pfSecondaryVertexNegativeTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfSecondaryVertexNegativeTagInfos.clone(
                                        trackIPTagInfos = cms.InputTag(btagPrefix+'pfImpactParameterTagInfos'+labelName+postfix)),
                                    process, task)
                if tightBTagNTkHits:
                    _btagInfo = getattr(process, btagPrefix+btagInfo+labelName+postfix)
                    _btagInfo.trackSelection.pixelHitsMin = cms.uint32(2)
                    _btagInfo.trackSelection.totalHitsMin = cms.uint32(8)
            if btagInfo == 'pfInclusiveSecondaryVertexFinderNegativeTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfInclusiveSecondaryVertexFinderNegativeTagInfos.clone(
                                        trackIPTagInfos = cms.InputTag(btagPrefix+'pfImpactParameterTagInfos'+labelName+postfix),
                                        extSVCollection=cms.InputTag(btagPrefix+'inclusiveCandidateNegativeSecondaryVertices'+labelName+postfix)),
                                    process, task)
                if svClustering or fatJets != cms.InputTag(''):
                    setupSVClustering(getattr(process, btagPrefix+btagInfo+labelName+postfix), svClustering, algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'impactParameterTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix, 
                                    btag.impactParameterTagInfos.clone(
                                        jetTracks = cms.InputTag('jetTracksAssociatorAtVertex'+labelName+postfix),
                                        primaryVertex=pvSource),
                                    process, task)
            if btagInfo == 'secondaryVertexTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.secondaryVertexTagInfos.clone(
                                        trackIPTagInfos = cms.InputTag(btagPrefix+'impactParameterTagInfos'+labelName+postfix)),
                                    process, task)
            if btagInfo == 'inclusiveSecondaryVertexFinderTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.inclusiveSecondaryVertexFinderTagInfos.clone(
                                        trackIPTagInfos = cms.InputTag(btagPrefix+'impactParameterTagInfos'+labelName+postfix)),
                                    process, task)
                if svClustering or fatJets != cms.InputTag(''):
                    setupSVClustering(getattr(process, btagPrefix+btagInfo+labelName+postfix), svClustering, algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'inclusiveSecondaryVertexFinderFilteredTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.inclusiveSecondaryVertexFinderFilteredTagInfos.clone(
                                        trackIPTagInfos = cms.InputTag(btagPrefix+'impactParameterTagInfos'+labelName+postfix)),
                                    process, task)
                if svClustering or fatJets != cms.InputTag(''):
                    setupSVClustering(getattr(process, btagPrefix+btagInfo+labelName+postfix), svClustering, algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'secondaryVertexNegativeTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.secondaryVertexNegativeTagInfos.clone(
                                        trackIPTagInfos = cms.InputTag(btagPrefix+'impactParameterTagInfos'+labelName+postfix)),
                                    process, task)
            if btagInfo == 'inclusiveSecondaryVertexFinderNegativeTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.inclusiveSecondaryVertexFinderNegativeTagInfos.clone(
                                        trackIPTagInfos = cms.InputTag(btagPrefix+'impactParameterTagInfos'+labelName+postfix)),
                                    process, task)
                if svClustering or fatJets != cms.InputTag(''):
                    setupSVClustering(getattr(process, btagPrefix+btagInfo+labelName+postfix), svClustering, algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'inclusiveSecondaryVertexFinderFilteredNegativeTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.inclusiveSecondaryVertexFinderFilteredNegativeTagInfos.clone(
                                        trackIPTagInfos = cms.InputTag(btagPrefix+'impactParameterTagInfos'+labelName+postfix)),
                                    process, task)
                if svClustering or fatJets != cms.InputTag(''):
                    setupSVClustering(getattr(process, btagPrefix+btagInfo+labelName+postfix), svClustering, algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'softMuonTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.softMuonTagInfos.clone(jets = jetSource, primaryVertex=pvSource),
                                    process, task)
            if btagInfo == 'softPFMuonsTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.softPFMuonsTagInfos.clone(jets = jetSource, primaryVertex=pvSource, muons=muSource),
                                    process, task)
            if btagInfo == 'softPFElectronsTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.softPFElectronsTagInfos.clone(jets = jetSource, primaryVertex=pvSource, electrons=elSource),
                                    process, task)
            if btagInfo == 'pixelClusterTagInfos':
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pixelClusterTagInfos.clone(jets = jetSource, vertices=pvSource),
                                    process, task)



            if 'pfBoostedDouble' in btagInfo or 'SecondaryVertex' in btagInfo:
              _btagInfo = getattr(process, btagPrefix+btagInfo+labelName+postfix)
              if pfCandidates.value() == 'packedPFCandidates':
                packedPuppiName = setupPackedPuppi(process)
                _btagInfo.weights = cms.InputTag(packedPuppiName)
              else:
                _btagInfo.weights = cms.InputTag("puppi")

            if 'DeepFlavourTagInfos' in btagInfo:
                svUsed = svSource
                if btagInfo == 'pfNegativeDeepFlavourTagInfos':
                    deep_csv_tag_infos = 'pfDeepCSVNegativeTagInfos'
                    svUsed = cms.InputTag(btagPrefix+'inclusiveCandidateNegativeSecondaryVertices'+labelName+postfix)
                    flip = True 
                else:
                    deep_csv_tag_infos = 'pfDeepCSVTagInfos' 
                    flip = False

                # use right input tags when running with RECO PF candidates, which actually
                # depens of whether jets use "particleFlow"
                if pfCandidates.value() == 'packedPFCandidates':
                    puppi_value_map = setupPackedPuppi(process)
                    vertex_associator = cms.InputTag("")
                else:
                    puppi_value_map = cms.InputTag("puppi")
                    vertex_associator = cms.InputTag("primaryVertexAssociation","original")

                # If this jet is a puppi jet, then set is_weighted_jet to true.
                is_weighted_jet = False
                if ('puppi' in jetSource.value().lower()):
                    is_weighted_jet = True
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfDeepFlavourTagInfos.clone(
                                      jets = jetSource,
                                      vertices=pvSource,
                                      secondary_vertices=svUsed,
                                      shallow_tag_infos = cms.InputTag(btagPrefix+deep_csv_tag_infos+labelName+postfix),
                                      puppi_value_map = puppi_value_map,
                                      vertex_associator = vertex_associator,
                                      is_weighted_jet = is_weighted_jet,
                                      flip = flip),
                                    process, task)

            if 'ParticleTransformerAK4TagInfos' in btagInfo:
                svUsed = svSource
                if btagInfo == 'pfNegativeParticleTransformerAK4TagInfos':
                    svUsed = cms.InputTag(btagPrefix+'inclusiveCandidateNegativeSecondaryVertices'+labelName+postfix)
                    flip = True 
                else:
                    flip = False
                # use right input tags when running with RECO PF candidates, which actually
                # depends of whether jets use "particleFlow"
                if pfCandidates.value() == 'packedPFCandidates':
                    puppi_value_map = setupPackedPuppi(process)
                    vertex_associator = cms.InputTag("")
                else:
                    puppi_value_map = cms.InputTag("puppi")
                    vertex_associator = cms.InputTag("primaryVertexAssociation","original")

                # If this jet is a puppi jet, then set is_weighted_jet to true.
                is_weighted_jet = False
                if ('puppi' in jetSource.value().lower()):
                    is_weighted_jet = True
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfParticleTransformerAK4TagInfos.clone(
                                      jets = jetSource,
                                      vertices=pvSource,
                                      secondary_vertices=svUsed,
                                      puppi_value_map = puppi_value_map,
                                      vertex_associator = vertex_associator,
                                      is_weighted_jet = is_weighted_jet,
                                      flip = flip),
                                    process, task)

            if btagInfo == 'pfDeepDoubleXTagInfos':
                # can only run on PAT jets, so the updater needs to be used
                if 'updated' not in jetSource.value().lower():
                    raise ValueError("Invalid jet collection: %s. pfDeepDoubleXTagInfos only supports running via updateJetCollection." % jetSource.value())
                packedPuppiName = setupPackedPuppi(process)
                puppi_value_map = cms.InputTag(packedPuppiName)
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfDeepDoubleXTagInfos.clone(
                                      jets = jetSource,
                                      vertices=pvSource,
                                      secondary_vertices=svSource,
                                      shallow_tag_infos = cms.InputTag(btagPrefix+'pfBoostedDoubleSVAK8TagInfos'+labelName+postfix),
                                      puppi_value_map = puppi_value_map,
                                      ),
                                    process, task)

            if btagInfo == 'pfHiggsInteractionNetTagInfos':
                packedPuppiName = setupPackedPuppi(process)
                puppi_value_map = cms.InputTag(packedPuppiName)
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfHiggsInteractionNetTagInfos.clone(
                                      jets = jetSource,
                                      vertices = pvSource,
                                      secondary_vertices = svSource,
                                      pf_candidates = pfCandidates,
                                      puppi_value_map = puppi_value_map
                                      ),
                                    process, task)

            if btagInfo == 'pfDeepBoostedJetTagInfos':
                if pfCandidates.value() == 'packedPFCandidates':
                    # case 1: running over jets whose daughters are PackedCandidates (only via updateJetCollection for now)
                    if 'updated' not in jetSource.value().lower():
                        raise ValueError("Invalid jet collection: %s. pfDeepBoostedJetTagInfos only supports running via updateJetCollection." % jetSource.value())
                    puppi_value_map = setupPackedPuppi(process)
                    vertex_associator = ""
                elif pfCandidates.value() == 'particleFlow':
                    raise ValueError("Running pfDeepBoostedJetTagInfos with reco::PFCandidates is currently not supported.")
                    # case 2: running on new jet collection whose daughters are PFCandidates (e.g., cluster jets in RECO/AOD)
                    # daughters are the particles used in jet clustering, so already scaled by their puppi weights
                    # Uncomment the lines below after running pfDeepBoostedJetTagInfos with reco::PFCandidates becomes supported
#                     puppi_value_map = "puppi"
#                     vertex_associator = "primaryVertexAssociation:original"
                else:
                    raise ValueError("Invalid pfCandidates collection: %s." % pfCandidates.value())
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfDeepBoostedJetTagInfos.clone(
                                      jets = jetSource,
                                      vertices = pvSource,
                                      secondary_vertices = svSource,
                                      pf_candidates = pfCandidates,
                                      puppi_value_map = puppi_value_map,
                                      vertex_associator = vertex_associator,
                                      ),
                                    process, task)

            if btagInfo == 'pfParticleNetTagInfos':
                if pfCandidates.value() == 'packedPFCandidates':
                    # case 1: running over jets whose daughters are PackedCandidates (only via updateJetCollection for now)
                    puppi_value_map = setupPackedPuppi(process)
                    vertex_associator = ""
                elif pfCandidates.value() == 'particleFlow':
                    raise ValueError("Running pfDeepBoostedJetTagInfos with reco::PFCandidates is currently not supported.")
                    # case 2: running on new jet collection whose daughters are PFCandidates (e.g., cluster jets in RECO/AOD)
                    puppi_value_map = "puppi"
                    vertex_associator = "primaryVertexAssociation:original"
                else:
                    raise ValueError("Invalid pfCandidates collection: %s." % pfCandidates.value())
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfParticleNetTagInfos.clone(
                                      jets = jetSource,
                                      vertices = pvSource,
                                      secondary_vertices = svSource,
                                      pf_candidates = pfCandidates,
                                      puppi_value_map = puppi_value_map,
                                      vertex_associator = vertex_associator,
                                      ),
                                    process, task)

            if 'ParticleNetAK4TagInfos' in btagInfo:
                if btagInfo == 'pfNegativeParticleNetAK4TagInfos':
                    secondary_vertices = btagPrefix + \
                        'inclusiveCandidateNegativeSecondaryVertices' + labelName + postfix
                    flip_ip_sign = True
                    sip3dSigMax = 10
                else:
                    secondary_vertices = svSource
                    flip_ip_sign = False
                    sip3dSigMax = -1
                if pfCandidates.value() == 'packedPFCandidates':
                    # case 1: running over jets whose daughters are PackedCandidates (only via updateJetCollection for now)
                    puppi_value_map = setupPackedPuppi(process)
                    vertex_associator = ""
                elif pfCandidates.value() == 'particleFlow':
                    raise ValueError("Running pfDeepBoostedJetTagInfos with reco::PFCandidates is currently not supported.")
                    # case 2: running on new jet collection whose daughters are PFCandidates (e.g., cluster jets in RECO/AOD)
                    puppi_value_map = "puppi"
                    vertex_associator = "primaryVertexAssociation:original"
                else:
                    raise ValueError("Invalid pfCandidates collection: %s." % pfCandidates.value())
                # If this jet is a Puppi jet, use puppi-weighted p4.
                use_puppiP4 = False
                if "puppi" in jetSource.value().lower():
                    use_puppiP4 = True
                addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                    btag.pfParticleNetAK4TagInfos.clone(
                                      jets = jetSource,
                                      vertices = pvSource,
                                      secondary_vertices = secondary_vertices,
                                      pf_candidates = pfCandidates,
                                      puppi_value_map = puppi_value_map,
                                      vertex_associator = vertex_associator,
                                      flip_ip_sign = flip_ip_sign,
                                      sip3dSigMax = sip3dSigMax,
                                      use_puppiP4 = use_puppiP4
                                      ),
                                    process, task)

            acceptedTagInfos.append(btagInfo)
        elif hasattr(toptag, btagInfo) :
            acceptedTagInfos.append(btagInfo)
        elif btagInfo == 'pfParticleNetFromMiniAODAK4PuppiCentralTagInfos':
            # ParticleNetFromMiniAOD cannot be run on RECO inputs, so need a workaround
            if pfCandidates.value() != 'packedPFCandidates':
                raise ValueError("Invalid pfCandidates collection: %s." % pfCandidates.value())
            addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                pfParticleNetFromMiniAODAK4PuppiCentralTagInfos.clone(
                                  jets = jetSource,
                                  vertices = pvSource,
                                  secondary_vertices = svSource,
                                  pf_candidates = pfCandidates,
                                  ),
                                process, task)
            acceptedTagInfos.append(btagInfo)
        elif btagInfo == 'pfParticleNetFromMiniAODAK4PuppiForwardTagInfos':
            # ParticleNetFromMiniAOD cannot be run on RECO inputs, so need a workaround
            if pfCandidates.value() != 'packedPFCandidates':
                raise ValueError("Invalid pfCandidates collection: %s." % pfCandidates.value())
            addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                pfParticleNetFromMiniAODAK4PuppiForwardTagInfos.clone(
                                  jets = jetSource,
                                  vertices = pvSource,
                                  secondary_vertices = svSource,
                                  pf_candidates = pfCandidates,
                                  ),
                                process, task)
            acceptedTagInfos.append(btagInfo)
        elif btagInfo == 'pfParticleNetFromMiniAODAK4CHSCentralTagInfos':
            # ParticleNetFromMiniAOD cannot be run on RECO inputs, so need a workaround
            if pfCandidates.value() != 'packedPFCandidates':
                raise ValueError("Invalid pfCandidates collection: %s." % pfCandidates.value())
            addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                pfParticleNetFromMiniAODAK4CHSCentralTagInfos.clone(
                                  jets = jetSource,
                                  vertices = pvSource,
                                  secondary_vertices = svSource,
                                  pf_candidates = pfCandidates,
                                  ),
                                process, task)
            acceptedTagInfos.append(btagInfo)
        elif btagInfo == 'pfParticleNetFromMiniAODAK4CHSForwardTagInfos':
            # ParticleNetFromMiniAOD cannot be run on RECO inputs, so need a workaround
            if pfCandidates.value() != 'packedPFCandidates':
                raise ValueError("Invalid pfCandidates collection: %s." % pfCandidates.value())
            addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                pfParticleNetFromMiniAODAK4CHSForwardTagInfos.clone(
                                  jets = jetSource,
                                  vertices = pvSource,
                                  secondary_vertices = svSource,
                                  pf_candidates = pfCandidates,
                                  ),
                                process, task)
            acceptedTagInfos.append(btagInfo)
        elif btagInfo == 'pfParticleNetFromMiniAODAK8TagInfos':
            # ParticleNetFromMiniAOD cannot be run on RECO inputs, so need a workaround
            if pfCandidates.value() != 'packedPFCandidates':
                raise ValueError("Invalid pfCandidates collection: %s." % pfCandidates.value())
            addToProcessAndTask(btagPrefix+btagInfo+labelName+postfix,
                                pfParticleNetFromMiniAODAK8TagInfos.clone(
                                  jets = jetSource,
                                  vertices = pvSource,
                                  secondary_vertices = svSource,
                                  pf_candidates = pfCandidates,
                                  ),
                                process, task)
            acceptedTagInfos.append(btagInfo)
        else:
            print('  --> %s ignored, since not available via RecoBTag.Configuration.RecoBTag_cff!'%(btagInfo))
    # setup all required btagDiscriminators
    acceptedBtagDiscriminators = list()
    for discriminator_name in btagDiscriminators :			
        btagDiscr = discriminator_name.split(':')[0] #split input tag to get the producer label
        #print discriminator_name, '-->', btagDiscr
        newDiscr = btagPrefix+btagDiscr+labelName+postfix #new discriminator name
        if hasattr(btag,btagDiscr): 
            if hasattr(process, newDiscr):
                pass 
            elif hasattr(getattr(btag, btagDiscr), 'tagInfos'):
                addToProcessAndTask(
                    newDiscr,
                    getattr(btag, btagDiscr).clone(
                        tagInfos = cms.VInputTag(
                            *[ cms.InputTag(btagPrefix+x+labelName+postfix) \
                            for x in supportedBtagDiscr[discriminator_name][0] ]
                        )
                    ),
                    process,
                    task
                )
            elif hasattr(getattr(btag, btagDiscr), 'src'):
                addToProcessAndTask(
                    newDiscr,
                    getattr(btag, btagDiscr).clone(
                        src = cms.InputTag(btagPrefix+supportedBtagDiscr[discriminator_name][0][0]+labelName+postfix)
                    ),
                    process,
                    task
                )
            else:
                raise ValueError('I do not know how to update %s it does not have neither "tagInfos" nor "src" attributes' % btagDiscr)
            acceptedBtagDiscriminators.append(discriminator_name)
        elif btagDiscr=='pfParticleNetFromMiniAODAK4PuppiCentralJetTags':
            if hasattr(process, newDiscr):
                pass
            addToProcessAndTask(
                newDiscr,
                pfParticleNetFromMiniAODAK4PuppiCentralJetTags.clone(
                    src = cms.InputTag(btagPrefix+supportedBtagDiscr[discriminator_name][0][0]+labelName+postfix)
                ),
                process,
                task
            )
            acceptedBtagDiscriminators.append(discriminator_name)
        elif btagDiscr=='pfParticleNetFromMiniAODAK4PuppiForwardJetTags':
            if hasattr(process, newDiscr):
                pass
            addToProcessAndTask(
                newDiscr,
                pfParticleNetFromMiniAODAK4PuppiForwardJetTags.clone(
                    src = cms.InputTag(btagPrefix+supportedBtagDiscr[discriminator_name][0][0]+labelName+postfix)
                ),
                process,
                task
            )
            acceptedBtagDiscriminators.append(discriminator_name)
        elif btagDiscr=='pfParticleNetFromMiniAODAK4CHSCentralJetTags':
            if hasattr(process, newDiscr):
                pass
            addToProcessAndTask(
                newDiscr,
                pfParticleNetFromMiniAODAK4CHSCentralJetTags.clone(
                    src = cms.InputTag(btagPrefix+supportedBtagDiscr[discriminator_name][0][0]+labelName+postfix)
                ),
                process,
                task
            )
            acceptedBtagDiscriminators.append(discriminator_name)
        elif btagDiscr=='pfParticleNetFromMiniAODAK4CHSForwardJetTags':
            if hasattr(process, newDiscr):
                pass
            addToProcessAndTask(
                newDiscr,
                pfParticleNetFromMiniAODAK4CHSForwardJetTags.clone(
                    src = cms.InputTag(btagPrefix+supportedBtagDiscr[discriminator_name][0][0]+labelName+postfix)
                ),
                process,
                task
            )
            acceptedBtagDiscriminators.append(discriminator_name)
        elif btagDiscr=='pfParticleNetFromMiniAODAK8JetTags':
            if hasattr(process, newDiscr):
                pass
            addToProcessAndTask(
                newDiscr,
                pfParticleNetFromMiniAODAK8JetTags.clone(
                    src = cms.InputTag(btagPrefix+supportedBtagDiscr[discriminator_name][0][0]+labelName+postfix)
                ),
                process,
                task
            )
            acceptedBtagDiscriminators.append(discriminator_name)
        else:
            print('  --> %s ignored, since not available via RecoBTag.Configuration.RecoBTag_cff!'%(btagDiscr))
            
    #update meta-taggers, if any
    for meta_tagger in present_meta:
        btagDiscr = meta_tagger.split(':')[0] #split input tag to get the producer label
        #print discriminator_name, '-->', btagDiscr
        newDiscr = btagPrefix+btagDiscr+labelName+postfix #new discriminator name
        if hasattr(btag,btagDiscr): 
            if hasattr(process, newDiscr):
                pass 
            else:
                addToProcessAndTask(
                    newDiscr,
                    getattr(btag, btagDiscr).clone(),
                    process,
                    task
                )
                for dependency in supportedMetaDiscr[meta_tagger]:
                    if ':' in dependency:
                        new_dep = btagPrefix+dependency.split(':')[0]+labelName+postfix+':'+dependency.split(':')[1]
                    else:
                        new_dep = btagPrefix+dependency+labelName+postfix
                    replace = MassSearchReplaceAnyInputTagVisitor(dependency, new_dep)
                    replace.doIt(getattr(process, newDiscr), newDiscr)
            acceptedBtagDiscriminators.append(meta_tagger)
        elif btagDiscr=='pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags':
            if hasattr(process, newDiscr):
                pass 
            else:
                addToProcessAndTask(
                    newDiscr,
                    pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags.clone(),
                    process,
                    task
                )
                for dependency in supportedMetaDiscr[meta_tagger]:
                    if ':' in dependency:
                        new_dep = btagPrefix+dependency.split(':')[0]+labelName+postfix+':'+dependency.split(':')[1]
                    else:
                        new_dep = btagPrefix+dependency+labelName+postfix
                    replace = MassSearchReplaceAnyInputTagVisitor(dependency, new_dep)
                    replace.doIt(getattr(process, newDiscr), newDiscr)
            acceptedBtagDiscriminators.append(meta_tagger)
        elif btagDiscr=='pfParticleNetFromMiniAODAK4PuppiForwardDiscriminatorsJetTags':
            if hasattr(process, newDiscr):
                pass 
            else:
                addToProcessAndTask(
                    newDiscr,
                    pfParticleNetFromMiniAODAK4PuppiForwardDiscriminatorsJetTags.clone(),
                    process,
                    task
                )
                for dependency in supportedMetaDiscr[meta_tagger]:
                    if ':' in dependency:
                        new_dep = btagPrefix+dependency.split(':')[0]+labelName+postfix+':'+dependency.split(':')[1]
                    else:
                        new_dep = btagPrefix+dependency+labelName+postfix
                    replace = MassSearchReplaceAnyInputTagVisitor(dependency, new_dep)
                    replace.doIt(getattr(process, newDiscr), newDiscr)
            acceptedBtagDiscriminators.append(meta_tagger)
        elif btagDiscr=='pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags':
            if hasattr(process, newDiscr):
                pass 
            else:
                addToProcessAndTask(
                    newDiscr,
                    pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags.clone(),
                    process,
                    task
                )
                for dependency in supportedMetaDiscr[meta_tagger]:
                    if ':' in dependency:
                        new_dep = btagPrefix+dependency.split(':')[0]+labelName+postfix+':'+dependency.split(':')[1]
                    else:
                        new_dep = btagPrefix+dependency+labelName+postfix
                    replace = MassSearchReplaceAnyInputTagVisitor(dependency, new_dep)
                    replace.doIt(getattr(process, newDiscr), newDiscr)
            acceptedBtagDiscriminators.append(meta_tagger)
        elif btagDiscr=='pfParticleNetFromMiniAODAK4CHSForwardDiscriminatorsJetTags':
            if hasattr(process, newDiscr):
                pass 
            else:
                addToProcessAndTask(
                    newDiscr,
                    pfParticleNetFromMiniAODAK4CHSForwardDiscriminatorsJetTags.clone(),
                    process,
                    task
                )
                for dependency in supportedMetaDiscr[meta_tagger]:
                    if ':' in dependency:
                        new_dep = btagPrefix+dependency.split(':')[0]+labelName+postfix+':'+dependency.split(':')[1]
                    else:
                        new_dep = btagPrefix+dependency+labelName+postfix
                    replace = MassSearchReplaceAnyInputTagVisitor(dependency, new_dep)
                    replace.doIt(getattr(process, newDiscr), newDiscr)
            acceptedBtagDiscriminators.append(meta_tagger)
        elif btagDiscr=='pfParticleNetFromMiniAODAK8DiscriminatorsJetTags':
            if hasattr(process, newDiscr):
                pass 
            else:
                addToProcessAndTask(
                    newDiscr,
                    pfParticleNetFromMiniAODAK8DiscriminatorsJetTags.clone(),
                    process,
                    task
                )
                for dependency in supportedMetaDiscr[meta_tagger]:
                    if ':' in dependency:
                        new_dep = btagPrefix+dependency.split(':')[0]+labelName+postfix+':'+dependency.split(':')[1]
                    else:
                        new_dep = btagPrefix+dependency+labelName+postfix
                    replace = MassSearchReplaceAnyInputTagVisitor(dependency, new_dep)
                    replace.doIt(getattr(process, newDiscr), newDiscr)
            acceptedBtagDiscriminators.append(meta_tagger)
                        
        else:
            print('  --> %s ignored, since not available via RecoBTag.Configuration.RecoBTag_cff!'%(btagDiscr))
        
    ## replace corresponding tags for pat jet production
    patJets.tagInfoSources = cms.VInputTag( *[ cms.InputTag(btagPrefix+x+labelName+postfix) for x in acceptedTagInfos ] )
    patJets.discriminatorSources = cms.VInputTag(*[ 
        cms.InputTag(btagPrefix+x+labelName+postfix) \
          if ':' not in x else \
          cms.InputTag(btagPrefix+x.split(':')[0]+labelName+postfix+':'+x.split(':')[1]) \
          for x in acceptedBtagDiscriminators 
        ])
    if len(acceptedBtagDiscriminators) > 0 :
        patJets.addBTagInfo = True
    ## if re-running IVF
    if runIVF:
        if not tightBTagNTkHits:
            if pvSource.getModuleLabel() == 'offlineSlimmedPrimaryVertices': ## MiniAOD case
                if not runIVFforCTagOnly: rerunningIVFMiniAOD()
            else:
                rerunningIVF()
        from PhysicsTools.PatAlgos.tools.helpers import loadWithPrefix
        ivfbTagInfos = ['pfInclusiveSecondaryVertexFinderTagInfos', 'pfInclusiveSecondaryVertexFinderAK8TagInfos', 'pfInclusiveSecondaryVertexFinderCA15TagInfos']
        if any(i in acceptedTagInfos for i in ivfbTagInfos) and not runIVFforCTagOnly:
            if not hasattr( process, btagPrefix+'inclusiveCandidateVertexFinder' ):
                loadWithPrefix(process, 'RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff', btagPrefix, task.label())
            if tightBTagNTkHits:
                if hasattr( process, btagPrefix+'inclusiveCandidateVertexFinder' ):
                    _temp = getattr(process, btagPrefix+'inclusiveCandidateVertexFinder')
                    _temp.minHits = cms.uint32(8)
            ## MiniAOD case
            if pvSource.getModuleLabel() == 'offlineSlimmedPrimaryVertices':
                if hasattr( process, btagPrefix+'inclusiveCandidateVertexFinder' ):
                    _temp = getattr(process, btagPrefix+'inclusiveCandidateVertexFinder')
                    _temp.primaryVertices = pvSource
                    _temp.tracks = pfCandidates
                if hasattr( process, btagPrefix+'candidateVertexArbitrator' ):
                    _temp = getattr(process, btagPrefix+'candidateVertexArbitrator')
                    _temp.primaryVertices = pvSource
                    _temp.tracks = pfCandidates
                if hasattr( process, btagPrefix+'inclusiveCandidateSecondaryVertices' ) and not hasattr( process, svSource.getModuleLabel() ):
                    addToProcessAndTask(svSource.getModuleLabel(),
                                        getattr(process, btagPrefix+'inclusiveCandidateSecondaryVertices').clone(),
                                        process, task)
        if any(i in acceptedTagInfos for i in ivfcTagInfos):
            if not hasattr( process, btagPrefix+'inclusiveCandidateVertexFinderCvsL' ):
                loadWithPrefix(process, 'RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff', btagPrefix, task.label())
            if tightBTagNTkHits:
                if hasattr( process, btagPrefix+'inclusiveCandidateVertexFinderCvsL' ):
                    _temp = getattr(process, btagPrefix+'inclusiveCandidateVertexFinderCvsL')
                    _temp.minHits = cms.uint32(8)
            ## MiniAOD case
            if pvSource.getModuleLabel() == 'offlineSlimmedPrimaryVertices':
                if hasattr( process, btagPrefix+'inclusiveCandidateVertexFinderCvsL' ):
                    _temp = getattr(process, btagPrefix+'inclusiveCandidateVertexFinderCvsL')
                    _temp.primaryVertices = pvSource
                    _temp.tracks = pfCandidates
                if hasattr( process, btagPrefix+'candidateVertexArbitratorCvsL' ):
                    _temp = getattr(process, btagPrefix+'candidateVertexArbitratorCvsL')
                    _temp.primaryVertices = pvSource
                    _temp.tracks = pfCandidates
                if hasattr( process, btagPrefix+'inclusiveCandidateSecondaryVerticesCvsL' ) and not hasattr( process, svSourceCvsL.getModuleLabel() ):
                    addToProcessAndTask(svSourceCvsL.getModuleLabel(),
                                        getattr(process, btagPrefix+'inclusiveCandidateSecondaryVerticesCvsL').clone(),
                                        process, task)
        if 'inclusiveSecondaryVertexFinderTagInfos' in acceptedTagInfos:
            if not hasattr( process, 'inclusiveVertexing' ):
                process.load( 'RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff' )
                task.add(process.inclusiveVertexingTask)
                task.add(process.inclusiveCandidateVertexingTask)
                task.add(process.inclusiveCandidateVertexingCvsLTask)
        if 'inclusiveSecondaryVertexFinderFilteredTagInfos' in acceptedTagInfos:
            if not hasattr( process, 'inclusiveVertexing' ):
                process.load( 'RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff' )
                task.add(process.inclusiveVertexingTask)
                task.add(process.inclusiveCandidateVertexingTask)
                task.add(process.inclusiveCandidateVertexingCvsLTask)
    if 'inclusiveSecondaryVertexFinderFilteredTagInfos' in acceptedTagInfos:
        if not hasattr( process, 'inclusiveSecondaryVerticesFiltered' ):
            process.load( 'RecoBTag.SecondaryVertex.inclusiveSecondaryVerticesFiltered_cfi' )
            task.add(process.inclusiveSecondaryVerticesFiltered)
            task.add(process.bVertexFilter)
        if not hasattr( process, 'bToCharmDecayVertexMerged' ):
            process.load( 'RecoBTag.SecondaryVertex.bToCharmDecayVertexMerger_cfi' )
            task.add(process.bToCharmDecayVertexMerged)
    if 'caTopTagInfos' in acceptedTagInfos :
        patJets.addTagInfos = True
        if not hasattr( process, 'caTopTagInfos' ) and not hasattr( process, 'caTopTagInfosAK8' ):
            process.load( 'RecoJets.JetProducers.caTopTaggers_cff' )
            task.add(process.caTopTaggersTask)

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
        self.addParameter(self._defaultParameters,'labelName', 'UNDEFINED', "Label name of the new patJet collection.", str)
        self.addParameter(self._defaultParameters,'postfix','', "Postfix from usePF2PAT.", str)
        self.addParameter(self._defaultParameters,'btagPrefix','', "Prefix to be added to b-tag discriminator and TagInfo names", str)
        self.addParameter(self._defaultParameters,'jetSource','', "Label of the input collection from which the new patJet collection should be created", cms.InputTag)
        self.addParameter(self._defaultParameters,'pfCandidates',cms.InputTag('particleFlow'), "Label of the input collection for candidatecandidatese used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'explicitJTA', False, "Use explicit jet-track association")
        self.addParameter(self._defaultParameters,'pvSource',cms.InputTag('offlinePrimaryVertices'), "Label of the input collection for primary vertices used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'svSource',cms.InputTag('inclusiveCandidateSecondaryVertices'), "Label of the input collection for IVF vertices used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'elSource',cms.InputTag('gedGsfElectrons'), "Label of the input collection for electrons used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'muSource',cms.InputTag('muons'), "Label of the input collection for muons used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'runIVF', False, "Re-run IVF secondary vertex reconstruction")
        self.addParameter(self._defaultParameters,'tightBTagNTkHits', False, "Enable legacy tight b-tag track selection")
        self.addParameter(self._defaultParameters,'loadStdRecoBTag', False, "Load the standard reconstruction b-tagging modules")
        self.addParameter(self._defaultParameters,'svClustering', False, "Secondary vertices ghost-associated to jets using jet clustering (mostly intended for subjets)")
        self.addParameter(self._defaultParameters,'fatJets', cms.InputTag(''), "Fat jet collection used for secondary vertex clustering", cms.InputTag)
        self.addParameter(self._defaultParameters,'groomedFatJets', cms.InputTag(''), "Groomed fat jet collection used for secondary vertex clustering", cms.InputTag)
        self.addParameter(self._defaultParameters,'algo', 'AK', "Jet algorithm of the input collection from which the new patJet collection should be created")
        self.addParameter(self._defaultParameters,'rParam', 0.4, "Jet size (distance parameter R used in jet clustering)")
        self.addParameter(self._defaultParameters,'getJetMCFlavour', True, "Get jet MC truth flavour")
        self.addParameter(self._defaultParameters,'genJetCollection', cms.InputTag("ak4GenJets"), "GenJet collection to match to", cms.InputTag)
        self.addParameter(self._defaultParameters,'genParticles', cms.InputTag("genParticles"), "GenParticle collection to be used", cms.InputTag)
        self.addParameter(self._defaultParameters,'jetCorrections',None, "Add all relevant information about jet energy corrections that you want to be added to your new patJet \
        collection. The format has to be given in a python tuple of type: (\'AK4Calo\',[\'L2Relative\', \'L3Absolute\'], patMet). Here the first argument corresponds to the payload \
        in the CMS Conditions database for the given jet collection; the second argument corresponds to the jet energy correction levels that you want to be embedded into your \
        new patJet collection. This should be given as a list of strings. Available values are L1Offset, L1FastJet, L1JPTOffset, L2Relative, L3Absolute, L5Falvour, L7Parton; the \
        third argument indicates whether MET(Type1/2) corrections should be applied corresponding to the new patJetCollection. If so a new patMet collection will be added to your PAT \
        Tuple in addition to the raw patMet. This new patMet collection will have the MET(Type1/2) corrections applied. The argument can have the following types: \'type-1\' for \
        type-1 corrected MET; \'type-2\' for type-1 plus type-2 corrected MET; \'\' or \'none\' if no further MET corrections should be applied to your MET. The arguments \'type-1\' \
        and \'type-2\' are not case sensitive.", tuple, acceptNoneValue=True)
        self.addParameter(self._defaultParameters,'btagDiscriminators',['None'], "If you are interested in btagging, in most cases just the labels of the btag discriminators that \
        you are interested in is all relevant information that you need for a high level analysis. Add here all btag discriminators, that you are interested in as a list of strings. \
        If this list is empty no btag discriminator information will be added to your new patJet collection.", allowedValues=(list(set().union(supportedBtagDiscr.keys(),supportedMetaDiscr.keys()))),Type=list)
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
        energy correction factors, btag information and generator match information to the new patJet collection depending on the parameters that you pass on to this function. Consult \
        the descriptions of each parameter for more information."

    def getDefaultParameters(self):
        """
        Return default parameters of the class
        """
        return self._defaultParameters

    def __call__(self,process,labelName=None,postfix=None,btagPrefix=None,jetSource=None,pfCandidates=None,explicitJTA=None,pvSource=None,svSource=None,elSource=None,muSource=None,runIVF=None,tightBTagNTkHits=None,loadStdRecoBTag=None,svClustering=None,fatJets=None,groomedFatJets=None,algo=None,rParam=None,getJetMCFlavour=None,genJetCollection=None,genParticles=None,jetCorrections=None,btagDiscriminators=None,btagInfos=None,jetTrackAssociation=None,outputModules=None):
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
        if btagPrefix is None:
            btagPrefix=self._defaultParameters['btagPrefix'].value
        self.setParameter('btagPrefix', btagPrefix)
        if jetSource is None:
            jetSource=self._defaultParameters['jetSource'].value
        self.setParameter('jetSource', jetSource)
        if pfCandidates is None:
            pfCandidates=self._defaultParameters['pfCandidates'].value
        self.setParameter('pfCandidates', pfCandidates)
        if explicitJTA is None:
            explicitJTA=self._defaultParameters['explicitJTA'].value
        self.setParameter('explicitJTA', explicitJTA)
        if pvSource is None:
            pvSource=self._defaultParameters['pvSource'].value
        self.setParameter('pvSource', pvSource)
        if svSource is None:
            svSource=self._defaultParameters['svSource'].value
        self.setParameter('svSource', svSource)
        if elSource is None:
            elSource=self._defaultParameters['elSource'].value
        self.setParameter('elSource', elSource)
        if muSource is None:
            muSource=self._defaultParameters['muSource'].value
        self.setParameter('muSource', muSource)
        if runIVF is None:
            runIVF=self._defaultParameters['runIVF'].value
        self.setParameter('runIVF', runIVF)
        if tightBTagNTkHits is None:
            tightBTagNTkHits=self._defaultParameters['tightBTagNTkHits'].value
        self.setParameter('tightBTagNTkHits', tightBTagNTkHits)
        if loadStdRecoBTag is None:
            loadStdRecoBTag=self._defaultParameters['loadStdRecoBTag'].value
        self.setParameter('loadStdRecoBTag', loadStdRecoBTag)
        if svClustering is None:
            svClustering=self._defaultParameters['svClustering'].value
        self.setParameter('svClustering', svClustering)
        if fatJets is None:
            fatJets=self._defaultParameters['fatJets'].value
        self.setParameter('fatJets', fatJets)
        if groomedFatJets is None:
            groomedFatJets=self._defaultParameters['groomedFatJets'].value
        self.setParameter('groomedFatJets', groomedFatJets)
        if algo is None:
            algo=self._defaultParameters['algo'].value
        self.setParameter('algo', algo)
        if rParam is None:
            rParam=self._defaultParameters['rParam'].value
        self.setParameter('rParam', rParam)
        if getJetMCFlavour is None:
            getJetMCFlavour=self._defaultParameters['getJetMCFlavour'].value
        self.setParameter('getJetMCFlavour', getJetMCFlavour)
        if genJetCollection is None:
            genJetCollection=self._defaultParameters['genJetCollection'].value
        self.setParameter('genJetCollection', genJetCollection)
        if genParticles is None:
            genParticles=self._defaultParameters['genParticles'].value
        self.setParameter('genParticles', genParticles)
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
        task = getPatAlgosToolsTask(process)

        ## initialize parameters
        labelName=self._parameters['labelName'].value
        postfix=self._parameters['postfix'].value
        btagPrefix=self._parameters['btagPrefix'].value
        jetSource=self._parameters['jetSource'].value
        pfCandidates=self._parameters['pfCandidates'].value
        explicitJTA=self._parameters['explicitJTA'].value
        pvSource=self._parameters['pvSource'].value
        svSource=self._parameters['svSource'].value
        elSource=self._parameters['elSource'].value
        muSource=self._parameters['muSource'].value
        runIVF=self._parameters['runIVF'].value
        tightBTagNTkHits=self._parameters['tightBTagNTkHits'].value
        loadStdRecoBTag=self._parameters['loadStdRecoBTag'].value
        svClustering=self._parameters['svClustering'].value
        fatJets=self._parameters['fatJets'].value
        groomedFatJets=self._parameters['groomedFatJets'].value
        algo=self._parameters['algo'].value
        rParam=self._parameters['rParam'].value
        getJetMCFlavour=self._parameters['getJetMCFlavour'].value
        genJetCollection=self._parameters['genJetCollection'].value
        genParticles=self._parameters['genParticles'].value
        jetCorrections=self._parameters['jetCorrections'].value
        btagDiscriminators=list(self._parameters['btagDiscriminators'].value)
        btagInfos=list(self._parameters['btagInfos'].value)
        jetTrackAssociation=self._parameters['jetTrackAssociation'].value
        outputModules=list(self._parameters['outputModules'].value)

        ## added jets must have a defined 'labelName'
        if labelName=='UNDEFINED':
            undefinedLabelName(self)

        ## a list of all producer modules, which are already known to process
        knownModules = process.producerNames().split()
        ## determine whether btagging information is required or not
        if btagDiscriminators.count('None')>0:
            btagDiscriminators.remove('None')
        if btagInfos.count('None')>0:
            btagInfos.remove('None')
        bTagging=(len(btagDiscriminators)>0 or len(btagInfos)>0)
        ## check if any legacy btag discriminators are being used
        infos = 0
        for info in btagInfos:
            if info.startswith('pf'): infos = infos + 1
            if 'softpf' in info.lower(): infos = infos + 1
        tags = 0
        for tag in btagDiscriminators:
            if tag.startswith('pf'): tags = tags + 1
            if 'softpf' in tag.lower(): tags = tags + 1
        bTaggingLegacy=(len(btagDiscriminators)>tags or len(btagInfos)>infos)
        ## construct postfix label for auxiliary modules; this postfix
        ## label will start with a capitalized first letter following
        ## the CMS naming conventions and for improved readablility
        _labelName=labelName[:1].upper()+labelName[1:]

        ## supported algo types are ak, ca, and kt
        _algo=''
        for x in ["ak", "ca", "kt"]:
            if x in algo.lower():
                _algo=supportedJetAlgos[x]
                break
        if _algo=='':
            unsupportedJetAlgorithm(self)
        ## add new patJets to process (keep instance for later further modifications)
        from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import _patJets as patJets
        if 'patJets'+_labelName+postfix in knownModules :
            _newPatJets=getattr(process, 'patJets'+_labelName+postfix)
            _newPatJets.jetSource=jetSource
        else :
            addToProcessAndTask('patJets'+_labelName+postfix, patJets.clone(jetSource=jetSource), process, task)
            _newPatJets=getattr(process, 'patJets'+_labelName+postfix)
            knownModules.append('patJets'+_labelName+postfix)
        ## add new selectedPatJets to process
        from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import selectedPatJets
        if 'selectedPatJets'+_labelName+postfix in knownModules :
            _newSelectedPatJets=getattr(process, 'selectedPatJets'+_labelName+postfix)
            _newSelectedPatJets.src='patJets'+_labelName+postfix
        else :
            addToProcessAndTask('selectedPatJets'+_labelName+postfix,
                                selectedPatJets.clone(src='patJets'+_labelName+postfix),
                                process, task)
            knownModules.append('selectedPatJets'+_labelName+postfix)

        ## add new patJetPartonMatch to process
        from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import patJetPartonMatch
        if 'patJetPartonMatch'+_labelName+postfix in knownModules :
            _newPatJetPartonMatch=getattr(process, 'patJetPartonMatch'+_labelName+postfix)
            _newPatJetPartonMatch.src=jetSource
            _newPatJetPartonMatch.matched=genParticles
        else :
            addToProcessAndTask('patJetPartonMatch'+_labelName+postfix,
                                patJetPartonMatch.clone(src=jetSource, matched=genParticles),
                                process, task)
            knownModules.append('patJetPartonMatch'+_labelName+postfix)
        ## add new patJetGenJetMatch to process
        from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import patJetGenJetMatch
        if 'patJetGenJetMatch'+_labelName+postfix in knownModules :
            _newPatJetGenJetMatch=getattr(process, 'patJetGenJetMatch'+_labelName+postfix)
            _newPatJetGenJetMatch.src=jetSource
            _newPatJetGenJetMatch.maxDeltaR=rParam
            _newPatJetGenJetMatch.matched=genJetCollection
        else :
            addToProcessAndTask('patJetGenJetMatch'+_labelName+postfix,
                                patJetGenJetMatch.clone(src=jetSource, maxDeltaR=rParam, matched=genJetCollection),
                                process, task)
            knownModules.append('patJetGenJetMatch'+_labelName+postfix)
        ## modify new patJets collection accordingly
        _newPatJets.genJetMatch.setModuleLabel('patJetGenJetMatch'+_labelName+postfix)
        _newPatJets.genPartonMatch.setModuleLabel('patJetPartonMatch'+_labelName+postfix)
        ## get jet MC truth flavour if required by user
        if (getJetMCFlavour):
            ## legacy jet flavour (see https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideBTagMCTools)
            ## add new patJetPartonsLegacy to process
            from PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff import patJetPartonsLegacy
            if 'patJetPartonsLegacy'+postfix not in knownModules :
                addToProcessAndTask('patJetPartonsLegacy'+postfix, patJetPartonsLegacy.clone(src=genParticles),
                                    process, task)
                knownModules.append('patJetPartonsLegacy'+postfix)
            else:
                getattr(process, 'patJetPartonsLegacy'+postfix).src=genParticles
            ## add new patJetPartonAssociationLegacy to process
            from PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff import patJetPartonAssociationLegacy
            if 'patJetPartonAssociationLegacy'+_labelName+postfix in knownModules :
                _newPatJetPartonAssociation=getattr(process, 'patJetPartonAssociationLegacy'+_labelName+postfix)
                _newPatJetPartonAssociation.jets=jetSource
            else :
                addToProcessAndTask('patJetPartonAssociationLegacy'+_labelName+postfix,
                                    patJetPartonAssociationLegacy.clone(jets=jetSource), process, task)
                knownModules.append('patJetPartonAssociationLegacy'+_labelName+postfix)
            ## add new patJetPartonAssociationLegacy to process
            from PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff import patJetFlavourAssociationLegacy
            if 'patJetFlavourAssociationLegacy'+_labelName+postfix in knownModules :
                _newPatJetFlavourAssociation=getattr(process, 'patJetFlavourAssociationLegacy'+_labelName+postfix)
                _newPatJetFlavourAssociation.srcByReference='patJetPartonAssociationLegacy'+_labelName+postfix
            else:
                addToProcessAndTask('patJetFlavourAssociationLegacy'+_labelName+postfix,
                                    patJetFlavourAssociationLegacy.clone(
                                        srcByReference='patJetPartonAssociationLegacy'+_labelName+postfix),
                                    process, task)
                knownModules.append('patJetFlavourAssociationLegacy'+_labelName+postfix)
            ## modify new patJets collection accordingly
            _newPatJets.JetPartonMapSource.setModuleLabel('patJetFlavourAssociationLegacy'+_labelName+postfix)
            ## new jet flavour (see https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideBTagMCTools)
            ## add new patJetPartons to process
            from PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff import patJetPartons
            if 'patJetPartons'+postfix not in knownModules :
                addToProcessAndTask('patJetPartons'+postfix, patJetPartons.clone(particles=genParticles), process, task)
                knownModules.append('patJetPartons'+postfix)
            else:
                getattr(process, 'patJetPartons'+postfix).particles=genParticles
            ## add new patJetFlavourAssociation to process
            from PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff import patJetFlavourAssociation
            if 'patJetFlavourAssociation'+_labelName+postfix in knownModules :
                _newPatJetFlavourAssociation=getattr(process, 'patJetFlavourAssociation'+_labelName+postfix)
                _newPatJetFlavourAssociation.jets=jetSource
                _newPatJetFlavourAssociation.jetAlgorithm=_algo
                _newPatJetFlavourAssociation.rParam=rParam
                _newPatJetFlavourAssociation.bHadrons=cms.InputTag("patJetPartons"+postfix,"bHadrons")
                _newPatJetFlavourAssociation.cHadrons=cms.InputTag("patJetPartons"+postfix,"cHadrons")
                _newPatJetFlavourAssociation.partons=cms.InputTag("patJetPartons"+postfix,"physicsPartons")
                _newPatJetFlavourAssociation.leptons=cms.InputTag("patJetPartons"+postfix,"leptons")
            else :
                addToProcessAndTask('patJetFlavourAssociation'+_labelName+postfix,
                                    patJetFlavourAssociation.clone(
                                        jets=jetSource,
                                        jetAlgorithm=_algo,
                                        rParam=rParam,
                                        bHadrons = cms.InputTag("patJetPartons"+postfix,"bHadrons"),
                                        cHadrons = cms.InputTag("patJetPartons"+postfix,"cHadrons"),
                                        partons = cms.InputTag("patJetPartons"+postfix,"physicsPartons"),
                                        leptons = cms.InputTag("patJetPartons"+postfix,"leptons")),
                                    process, task)

                knownModules.append('patJetFlavourAssociation'+_labelName+postfix)
            if 'Puppi' in jetSource.value() and pfCandidates.value() == 'particleFlow':
                _newPatJetFlavourAssociation=getattr(process, 'patJetFlavourAssociation'+_labelName+postfix)
                _newPatJetFlavourAssociation.weights = cms.InputTag("puppi")
            ## modify new patJets collection accordingly
            _newPatJets.JetFlavourInfoSource.setModuleLabel('patJetFlavourAssociation'+_labelName+postfix)
            ## if the jets is actually a subjet
            if fatJets != cms.InputTag('') and groomedFatJets != cms.InputTag(''):
                _newPatJetFlavourAssociation=getattr(process, 'patJetFlavourAssociation'+_labelName+postfix)
                _newPatJetFlavourAssociation.jets=fatJets
                _newPatJetFlavourAssociation.groomedJets=groomedFatJets
                _newPatJetFlavourAssociation.subjets=jetSource
                _newPatJets.JetFlavourInfoSource=cms.InputTag('patJetFlavourAssociation'+_labelName+postfix,'SubJets')
        else:
            _newPatJets.getJetMCFlavour = False
            _newPatJets.addJetFlavourInfo = False

        ## add jetTrackAssociation for legacy btagging (or jetTracksAssociation only) if required by user
        if (jetTrackAssociation or bTaggingLegacy):
            ## add new jetTracksAssociationAtVertex to process
            from RecoJets.JetAssociationProducers.ak4JTA_cff import ak4JetTracksAssociatorAtVertex, ak4JetTracksAssociatorExplicit
            if 'jetTracksAssociationAtVertex'+_labelName+postfix in knownModules :
                _newJetTracksAssociationAtVertex=getattr(process, 'jetTracksAssociatorAtVertex'+_labelName+postfix)
                _newJetTracksAssociationAtVertex.jets=jetSource
                _newJetTracksAssociationAtVertex.pvSrc=pvSource
            else:
                jetTracksAssociator=ak4JetTracksAssociatorAtVertex
                if explicitJTA:
                    jetTracksAssociator=ak4JetTracksAssociatorExplicit
                addToProcessAndTask('jetTracksAssociatorAtVertex'+_labelName+postfix,
                                    jetTracksAssociator.clone(jets=jetSource,pvSrc=pvSource),
                                    process, task)
                knownModules.append('jetTracksAssociationAtVertex'+_labelName+postfix)
            ## add new patJetCharge to process
            from PhysicsTools.PatAlgos.recoLayer0.jetTracksCharge_cff import patJetCharge
            if 'patJetCharge'+_labelName+postfix in knownModules :
                _newPatJetCharge=getattr(process, 'patJetCharge'+_labelName+postfix)
                _newPatJetCharge.src='jetTracksAssociatorAtVertex'+_labelName+postfix
            else:
                addToProcessAndTask('patJetCharge'+_labelName+postfix,
                                    patJetCharge.clone(src = 'jetTracksAssociatorAtVertex'+_labelName+postfix),
                                    process, task)
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
            setupBTagging(process, jetSource, pfCandidates, explicitJTA, pvSource, svSource, elSource, muSource, runIVF, tightBTagNTkHits, loadStdRecoBTag, svClustering, fatJets, groomedFatJets,
                          _algo, rParam, btagDiscriminators, btagInfos, _newPatJets, _labelName, btagPrefix, postfix)
        else:
            _newPatJets.addBTagInfo = False
            _newPatJets.addTagInfos = False
            ## adjust output module; these collections will be empty anyhow, but we do it to stay clean
            for outputModule in outputModules:
                    if hasattr(process,outputModule):
                        getattr(process,outputModule).outputCommands.append("drop *_"+'selected'+_labelName+postfix+"_tagInfos_*")

        ## add jet correction factors if required by user
        if (jetCorrections != None):
            ## check the jet corrections format
            checkJetCorrectionsFormat(jetCorrections)
            ## setup jet energy corrections and MET corrections
            setupJetCorrections(process, knownModules, jetCorrections, jetSource, pvSource, _newPatJets, _labelName, postfix)
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
        self.addParameter(self._defaultParameters,'postfix','', "postfix from usePF2PAT")
        self.addParameter(self._defaultParameters,'btagPrefix','', "Prefix to be added to b-tag discriminator and TagInfo names", str)
        self.addParameter(self._defaultParameters,'jetSource','', "Label of the input collection from which the new patJet collection should be created", cms.InputTag)
        self.addParameter(self._defaultParameters,'pfCandidates',cms.InputTag('particleFlow'), "Label of the input collection for candidatecandidatese used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'explicitJTA', False, "Use explicit jet-track association")
        self.addParameter(self._defaultParameters,'pvSource',cms.InputTag('offlinePrimaryVertices'), "Label of the input collection for primary vertices used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'svSource',cms.InputTag('inclusiveCandidateSecondaryVertices'), "Label of the input collection for IVF vertices used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'elSource',cms.InputTag('gedGsfElectrons'), "Label of the input collection for electrons used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'muSource',cms.InputTag('muons'), "Label of the input collection for muons used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'runIVF', False, "Re-run IVF secondary vertex reconstruction")
        self.addParameter(self._defaultParameters,'tightBTagNTkHits', False, "Enable legacy tight b-tag track selection")
        self.addParameter(self._defaultParameters,'loadStdRecoBTag', False, "Load the standard reconstruction b-tagging modules")
        self.addParameter(self._defaultParameters,'svClustering', False, "Secondary vertices ghost-associated to jets using jet clustering (mostly intended for subjets)")
        self.addParameter(self._defaultParameters,'fatJets', cms.InputTag(''), "Fat jet collection used for secondary vertex clustering", cms.InputTag)
        self.addParameter(self._defaultParameters,'groomedFatJets', cms.InputTag(''), "Groomed fat jet collection used for secondary vertex clustering", cms.InputTag)
        self.addParameter(self._defaultParameters,'algo', 'AK', "Jet algorithm of the input collection from which the new patJet collection should be created")
        self.addParameter(self._defaultParameters,'rParam', 0.4, "Jet size (distance parameter R used in jet clustering)")
        self.addParameter(self._defaultParameters,'getJetMCFlavour', True, "Get jet MC truth flavour")
        self.addParameter(self._defaultParameters,'genJetCollection', cms.InputTag("ak4GenJets"), "GenJet collection to match to")
        self.addParameter(self._defaultParameters,'genParticles', cms.InputTag("genParticles"), "GenParticle collection to be used", cms.InputTag)
        self.addParameter(self._defaultParameters,'jetCorrections',None, "Add all relevant information about jet energy corrections that you want to be added to your new patJet \
        collection. The format is to be passed on in a python tuple: e.g. (\'AK4Calo\',[\'L2Relative\', \'L3Absolute\'], patMet). The first argument corresponds to the payload \
        in the CMS Conditions database for the given jet collection; the second argument corresponds to the jet energy correction level that you want to be embedded into your \
        new patJet collection. This should be given as a list of strings. Available values are L1Offset, L1FastJet, L1JPTOffset, L2Relative, L3Absolute, L5Falvour, L7Parton; the \
        third argument indicates whether MET(Type1) corrections should be applied corresponding to the new patJetCollection. If so a new patMet collection will be added to your PAT \
        Tuple in addition to the raw patMet with the MET(Type1) corrections applied. The argument corresponds to the patMet collection to which the MET(Type1) corrections should be \
        applied. If you are not interested in MET(Type1) corrections to this new patJet collection pass None as third argument of the python tuple.", tuple, acceptNoneValue=True)
        self.addParameter(self._defaultParameters,'btagDiscriminators',['None'], "If you are interested in btagging in general the btag discriminators is all relevant \
        information that you need for a high level analysis. Add here all btag discriminators, that you are interested in as a list of strings. If this list is empty no btag \
        discriminator information will be added to your new patJet collection.", allowedValues=(list(set().union(supportedBtagDiscr.keys(),supportedMetaDiscr.keys()))),Type=list)
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
        information and generator match information to the new patJet collection depending on the parameters that you pass on to this function. Consult the descriptions of each \
        parameter for more information."

    def getDefaultParameters(self):
        """
        Return default parameters of the class
        """
        return self._defaultParameters

    def __call__(self,process,postfix=None,btagPrefix=None,jetSource=None,pfCandidates=None,explicitJTA=None,pvSource=None,svSource=None,elSource=None,muSource=None,runIVF=None,tightBTagNTkHits=None,loadStdRecoBTag=None,svClustering=None,fatJets=None,groomedFatJets=None,algo=None,rParam=None,getJetMCFlavour=None,genJetCollection=None,genParticles=None,jetCorrections=None,btagDiscriminators=None,btagInfos=None,jetTrackAssociation=None,outputModules=None):
        """
        Function call wrapper. This will check the parameters and call the actual implementation that
        can be found in toolCode via the base class function apply.
        """
        if postfix is None:
            postfix=self._defaultParameters['postfix'].value
        self.setParameter('postfix', postfix)
        if btagPrefix is None:
            btagPrefix=self._defaultParameters['btagPrefix'].value
        self.setParameter('btagPrefix', btagPrefix)
        if jetSource is None:
            jetSource=self._defaultParameters['jetSource'].value
        self.setParameter('jetSource', jetSource)
        if pfCandidates is None:
            pfCandidates=self._defaultParameters['pfCandidates'].value
        self.setParameter('pfCandidates', pfCandidates)
        if explicitJTA is None:
            explicitJTA=self._defaultParameters['explicitJTA'].value
        self.setParameter('explicitJTA', explicitJTA)
        if pvSource is None:
            pvSource=self._defaultParameters['pvSource'].value
        self.setParameter('pvSource', pvSource)
        if svSource is None:
            svSource=self._defaultParameters['svSource'].value
        self.setParameter('svSource', svSource)
        if elSource is None:
            elSource=self._defaultParameters['elSource'].value
        self.setParameter('elSource', elSource)
        if muSource is None:
            muSource=self._defaultParameters['muSource'].value
        self.setParameter('muSource', muSource)
        if runIVF is None:
            runIVF=self._defaultParameters['runIVF'].value
        self.setParameter('runIVF', runIVF)
        if tightBTagNTkHits is None:
            tightBTagNTkHits=self._defaultParameters['tightBTagNTkHits'].value
        self.setParameter('tightBTagNTkHits', tightBTagNTkHits)
        if loadStdRecoBTag is None:
            loadStdRecoBTag=self._defaultParameters['loadStdRecoBTag'].value
        self.setParameter('loadStdRecoBTag', loadStdRecoBTag)
        if svClustering is None:
            svClustering=self._defaultParameters['svClustering'].value
        self.setParameter('svClustering', svClustering)
        if fatJets is None:
            fatJets=self._defaultParameters['fatJets'].value
        self.setParameter('fatJets', fatJets)
        if groomedFatJets is None:
            groomedFatJets=self._defaultParameters['groomedFatJets'].value
        self.setParameter('groomedFatJets', groomedFatJets)
        if algo is None:
            algo=self._defaultParameters['algo'].value
        self.setParameter('algo', algo)
        if rParam is None:
            rParam=self._defaultParameters['rParam'].value
        self.setParameter('rParam', rParam)
        if getJetMCFlavour is None:
            getJetMCFlavour=self._defaultParameters['getJetMCFlavour'].value
        self.setParameter('getJetMCFlavour', getJetMCFlavour)
        if genJetCollection is None:
            genJetCollection=self._defaultParameters['genJetCollection'].value
        self.setParameter('genJetCollection', genJetCollection)
        if genParticles is None:
            genParticles=self._defaultParameters['genParticles'].value
        self.setParameter('genParticles', genParticles)
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
        postfix=self._parameters['postfix'].value
        btagPrefix=self._parameters['btagPrefix'].value
        jetSource=self._parameters['jetSource'].value
        pfCandidates=self._parameters['pfCandidates'].value
        explicitJTA=self._parameters['explicitJTA'].value
        pvSource=self._parameters['pvSource'].value
        svSource=self._parameters['svSource'].value
        elSource=self._parameters['elSource'].value
        muSource=self._parameters['muSource'].value
        runIVF=self._parameters['runIVF'].value
        tightBTagNTkHits=self._parameters['tightBTagNTkHits'].value
        loadStdRecoBTag=self._parameters['loadStdRecoBTag'].value
        svClustering=self._parameters['svClustering'].value
        fatJets=self._parameters['fatJets'].value
        groomedFatJets=self._parameters['groomedFatJets'].value
        algo=self._parameters['algo'].value
        rParam=self._parameters['rParam'].value
        getJetMCFlavour=self._parameters['getJetMCFlavour'].value
        genJetCollection=self._parameters['genJetCollection'].value
        genParticles=self._parameters['genParticles'].value
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
            btagPrefix=btagPrefix,
            jetSource=jetSource,
            pfCandidates=pfCandidates,
            explicitJTA=explicitJTA,
            pvSource=pvSource,
            svSource=svSource,
            elSource=elSource,
            muSource=muSource,
            runIVF=runIVF,
            tightBTagNTkHits=tightBTagNTkHits,
            loadStdRecoBTag=loadStdRecoBTag,
            svClustering=svClustering,
            fatJets=fatJets,
            groomedFatJets=groomedFatJets,
            algo=algo,
            rParam=rParam,
            getJetMCFlavour=getJetMCFlavour,
            genJetCollection=genJetCollection,
            genParticles=genParticles,
            jetCorrections=jetCorrections,
            btagDiscriminators=btagDiscriminators,
            btagInfos=btagInfos,
            jetTrackAssociation=jetTrackAssociation,
            outputModules=outputModules,
            )

switchJetCollection=SwitchJetCollection()


class UpdateJetCollection(ConfigToolBase):
    """
    Tool to update a jet collection in your PAT Tuple (primarily intended for MiniAOD for which the default input argument values have been set).
    """
    _label='updateJetCollection'
    _defaultParameters=dicttypes.SortedKeysDict()

    def __init__(self):
        """
        Initialize elements of the class. Note that the tool needs to be derived from ConfigToolBase to be usable in the configEditor.
        """
        ## initialization of the base class
        ConfigToolBase.__init__(self)
        ## add all parameters that should be known to the class
        self.addParameter(self._defaultParameters,'labelName', '', "Label name of the new patJet collection.", str)
        self.addParameter(self._defaultParameters,'postfix','', "Postfix from usePF2PAT.", str)
        self.addParameter(self._defaultParameters,'btagPrefix','', "Prefix to be added to b-tag discriminator and TagInfo names", str)
        self.addParameter(self._defaultParameters,'jetSource','', "Label of the input collection from which the new patJet collection should be created", cms.InputTag)
        self.addParameter(self._defaultParameters,'pfCandidates',cms.InputTag('packedPFCandidates'), "Label of the input collection for candidatecandidatese used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'explicitJTA', False, "Use explicit jet-track association")
        self.addParameter(self._defaultParameters,'pvSource',cms.InputTag('offlineSlimmedPrimaryVertices'), "Label of the input collection for primary vertices used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'svSource',cms.InputTag('slimmedSecondaryVertices'), "Label of the input collection for IVF vertices used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'elSource',cms.InputTag('slimmedElectrons'), "Label of the input collection for electrons used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'muSource',cms.InputTag('slimmedMuons'), "Label of the input collection for muons used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'runIVF', False, "Re-run IVF secondary vertex reconstruction")
        self.addParameter(self._defaultParameters,'tightBTagNTkHits', False, "Enable legacy tight b-tag track selection")
        self.addParameter(self._defaultParameters,'loadStdRecoBTag', False, "Load the standard reconstruction b-tagging modules")
        self.addParameter(self._defaultParameters,'svClustering', False, "Secondary vertices ghost-associated to jets using jet clustering (mostly intended for subjets)")
        self.addParameter(self._defaultParameters,'fatJets', cms.InputTag(''), "Fat jet collection used for secondary vertex clustering", cms.InputTag)
        self.addParameter(self._defaultParameters,'groomedFatJets', cms.InputTag(''), "Groomed fat jet collection used for secondary vertex clustering", cms.InputTag)
        self.addParameter(self._defaultParameters,'algo', 'AK', "Jet algorithm of the input collection from which the new patJet collection should be created")
        self.addParameter(self._defaultParameters,'rParam', 0.4, "Jet size (distance parameter R used in jet clustering)")
        self.addParameter(self._defaultParameters,'sortByPt', True, "Set to False to not modify incoming jet order")
        self.addParameter(self._defaultParameters,'printWarning', True, "To be use as False in production to reduce log size")
        self.addParameter(self._defaultParameters,'jetCorrections',None, "Add all relevant information about jet energy corrections that you want to be added to your new patJet \
        collection. The format has to be given in a python tuple of type: (\'AK4Calo\',[\'L2Relative\', \'L3Absolute\'], patMet). Here the first argument corresponds to the payload \
        in the CMS Conditions database for the given jet collection; the second argument corresponds to the jet energy correction levels that you want to be embedded into your \
        new patJet collection. This should be given as a list of strings. Available values are L1Offset, L1FastJet, L1JPTOffset, L2Relative, L3Absolute, L5Falvour, L7Parton; the \
        third argument indicates whether MET(Type1/2) corrections should be applied corresponding to the new patJetCollection. If so a new patMet collection will be added to your PAT \
        Tuple in addition to the raw patMet. This new patMet collection will have the MET(Type1/2) corrections applied. The argument can have the following types: \'type-1\' for \
        type-1 corrected MET; \'type-2\' for type-1 plus type-2 corrected MET; \'\' or \'none\' if no further MET corrections should be applied to your MET. The arguments \'type-1\' \
        and \'type-2\' are not case sensitive.", tuple, acceptNoneValue=True)
        self.addParameter(self._defaultParameters,'btagDiscriminators',['None'], "If you are interested in btagging, in most cases just the labels of the btag discriminators that \
        you are interested in is all relevant information that you need for a high level analysis. Add here all btag discriminators, that you are interested in as a list of strings. \
        If this list is empty no btag discriminator information will be added to your new patJet collection.", allowedValues=(list(set().union(supportedBtagDiscr.keys(),supportedMetaDiscr.keys()))),Type=list)
        self.addParameter(self._defaultParameters,'btagInfos',['None'], "The btagInfos objects contain all relevant information from which all discriminators of a certain \
        type have been calculated. You might be interested in keeping this information for low level tests or to re-calculate some discriminators from hand. Note that this information \
        on the one hand can be very space consuming and that it is not necessary to access the pre-calculated btag discriminator information that has been derived from it. Only in very \
        special cases the btagInfos might really be needed in your analysis. Add here all btagInfos, that you are interested in as a list of strings. If this list is empty no btagInfos \
        will be added to your new patJet collection.", allowedValues=supportedBtagInfos,Type=list)
        self.addParameter(self._defaultParameters,'outputModules',['out'],"Add a list of all output modules to which you would like the new jet collection to be added. Usually this is \
        just one single output module with name \'out\', which corresponds also the default configuration of the tool. There is cases though where you might want to add this collection \
        to more than one output module.")
        ## set defaults
        self._parameters=copy.deepcopy(self._defaultParameters)
        ## add comments
        self._comment = "This is a tool to add more patJet collectinos to your PAT Tuple or to re-configure the default collection. You can add and embed additional information like jet\
        energy correction factors, btag information and generator match information to the new patJet collection depending on the parameters that you pass on to this function. Consult \
        the descriptions of each parameter for more information."

    def getDefaultParameters(self):
        """
        Return default parameters of the class
        """
        return self._defaultParameters

    def __call__(self,process,labelName=None,postfix=None,btagPrefix=None,jetSource=None,pfCandidates=None,explicitJTA=None,pvSource=None,svSource=None,elSource=None,muSource=None,runIVF=None,tightBTagNTkHits=None,loadStdRecoBTag=None,svClustering=None,fatJets=None,groomedFatJets=None,algo=None,rParam=None,sortByPt=None,printWarning=None,jetCorrections=None,btagDiscriminators=None,btagInfos=None):
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
        if btagPrefix is None:
            btagPrefix=self._defaultParameters['btagPrefix'].value
        self.setParameter('btagPrefix', btagPrefix)
        if jetSource is None:
            jetSource=self._defaultParameters['jetSource'].value
        self.setParameter('jetSource', jetSource)
        if pfCandidates is None:
            pfCandidates=self._defaultParameters['pfCandidates'].value
        self.setParameter('pfCandidates', pfCandidates)
        if explicitJTA is None:
            explicitJTA=self._defaultParameters['explicitJTA'].value
        self.setParameter('explicitJTA', explicitJTA)
        if pvSource is None:
            pvSource=self._defaultParameters['pvSource'].value
        self.setParameter('pvSource', pvSource)
        if svSource is None:
            svSource=self._defaultParameters['svSource'].value
        self.setParameter('svSource', svSource)
        if elSource is None:
            elSource=self._defaultParameters['elSource'].value
        self.setParameter('elSource', elSource)
        if muSource is None:
            muSource=self._defaultParameters['muSource'].value
        self.setParameter('muSource', muSource)
        if runIVF is None:
            runIVF=self._defaultParameters['runIVF'].value
        self.setParameter('runIVF', runIVF)
        if tightBTagNTkHits is None:
            tightBTagNTkHits=self._defaultParameters['tightBTagNTkHits'].value
        self.setParameter('tightBTagNTkHits', tightBTagNTkHits)
        if loadStdRecoBTag is None:
            loadStdRecoBTag=self._defaultParameters['loadStdRecoBTag'].value
        self.setParameter('loadStdRecoBTag', loadStdRecoBTag)
        if svClustering is None:
            svClustering=self._defaultParameters['svClustering'].value
        self.setParameter('svClustering', svClustering)
        if fatJets is None:
            fatJets=self._defaultParameters['fatJets'].value
        self.setParameter('fatJets', fatJets)
        if groomedFatJets is None:
            groomedFatJets=self._defaultParameters['groomedFatJets'].value
        self.setParameter('groomedFatJets', groomedFatJets)
        if algo is None:
            algo=self._defaultParameters['algo'].value
        self.setParameter('algo', algo)
        if rParam is None:
            rParam=self._defaultParameters['rParam'].value
        self.setParameter('rParam', rParam)
        if sortByPt is None:
            sortByPt=self._defaultParameters['sortByPt'].value
        self.setParameter('sortByPt', sortByPt)
        if printWarning is None:
            printWarning=self._defaultParameters['printWarning'].value
        self.setParameter('printWarning', printWarning)
        if jetCorrections is None:
            jetCorrections=self._defaultParameters['jetCorrections'].value
        self.setParameter('jetCorrections', jetCorrections)
        if btagDiscriminators is None:
            btagDiscriminators=self._defaultParameters['btagDiscriminators'].value
        self.setParameter('btagDiscriminators', btagDiscriminators)
        if btagInfos is None:
            btagInfos=self._defaultParameters['btagInfos'].value
        self.setParameter('btagInfos', btagInfos)
        self.apply(process)

    def toolCode(self, process):
        """
        Tool code implementation
        """
        ## initialize parameters
        labelName=self._parameters['labelName'].value
        postfix=self._parameters['postfix'].value
        btagPrefix=self._parameters['btagPrefix'].value
        jetSource=self._parameters['jetSource'].value
        pfCandidates=self._parameters['pfCandidates'].value
        explicitJTA=self._parameters['explicitJTA'].value
        pvSource=self._parameters['pvSource'].value
        svSource=self._parameters['svSource'].value
        elSource=self._parameters['elSource'].value
        muSource=self._parameters['muSource'].value
        runIVF=self._parameters['runIVF'].value
        tightBTagNTkHits=self._parameters['tightBTagNTkHits'].value
        loadStdRecoBTag=self._parameters['loadStdRecoBTag'].value
        svClustering=self._parameters['svClustering'].value
        fatJets=self._parameters['fatJets'].value
        groomedFatJets=self._parameters['groomedFatJets'].value
        algo=self._parameters['algo'].value
        rParam=self._parameters['rParam'].value
        sortByPt=self._parameters['sortByPt'].value
        printWarning=self._parameters['printWarning'].value
        jetCorrections=self._parameters['jetCorrections'].value
        btagDiscriminators=list(self._parameters['btagDiscriminators'].value)
        btagInfos=list(self._parameters['btagInfos'].value)

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
        ## the CMS naming conventions and for improved readablility
        _labelName=labelName[:1].upper()+labelName[1:]

        ## supported algo types are ak, ca, and kt
        _algo=''
        for x in ["ak", "ca", "kt"]:
            if x in algo.lower():
                _algo=supportedJetAlgos[x]
                break
        if _algo=='':
            unsupportedJetAlgorithm(self)

        task = getPatAlgosToolsTask(process)

        ## add new updatedPatJets to process (keep instance for later further modifications)
        from PhysicsTools.PatAlgos.producersLayer1.jetUpdater_cfi import updatedPatJets
        if not sortByPt: # default is True
            updatedPatJets.sort = cms.bool(False)
        if 'updatedPatJets'+_labelName+postfix in knownModules :
            _newPatJets=getattr(process, 'updatedPatJets'+_labelName+postfix)
            _newPatJets.jetSource=jetSource
        else :
            addToProcessAndTask('updatedPatJets'+_labelName+postfix,
                                updatedPatJets.clone(jetSource=jetSource,
                                printWarning=printWarning), process, task)
            _newPatJets=getattr(process, 'updatedPatJets'+_labelName+postfix)
            knownModules.append('updatedPatJets'+_labelName+postfix)
        ## add new selectedUpdatedPatJets to process
        from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import selectedPatJets
        if 'selectedUpdatedPatJets'+_labelName+postfix in knownModules :
            _newSelectedPatJets=getattr(process, 'selectedUpdatedPatJets'+_labelName+postfix)
            _newSelectedPatJets.src='updatedPatJets'+_labelName+postfix
        else :
            addToProcessAndTask('selectedUpdatedPatJets'+_labelName+postfix,
                                selectedPatJets.clone(src='updatedPatJets'+_labelName+postfix),
                                process, task)
            knownModules.append('selectedUpdatedPatJets'+_labelName+postfix)

        ## run btagging if required by user
        if (bTagging):
            if printWarning:
               sys.stderr.write("**************************************************************\n")
               sys.stderr.write("b tagging needs to be run on uncorrected jets. Hence, the JECs\n")
               sys.stderr.write("will first be undone for 'updatedPatJets%s' and then applied to\n" % (_labelName+postfix) )
               sys.stderr.write("'updatedPatJetsTransientCorrected%s'.\n" % (_labelName+postfix) )
               sys.stderr.write("**************************************************************\n")
            _jetSource = cms.InputTag('updatedPatJets'+_labelName+postfix)
            ## insert new jet collection with jet corrections applied and btag info added
            self(
                process,
                labelName = ('TransientCorrected'+_labelName),
                jetSource = _jetSource,
                pfCandidates=pfCandidates,
                explicitJTA=explicitJTA,
                pvSource=pvSource,
                svSource=svSource,
                elSource=elSource,
                muSource=muSource,
                runIVF=runIVF,
                tightBTagNTkHits=tightBTagNTkHits,
                loadStdRecoBTag=loadStdRecoBTag,
                svClustering=svClustering,
                fatJets=fatJets,
                groomedFatJets=groomedFatJets,
                algo=algo,
                rParam=rParam,
                jetCorrections = jetCorrections,
                btagPrefix = btagPrefix,
                postfix = postfix
            )
            ## setup btagging
            _patJets=getattr(process, 'updatedPatJetsTransientCorrected'+_labelName+postfix)
            setupBTagging(process, _jetSource, pfCandidates, explicitJTA, pvSource, svSource, elSource, muSource, runIVF, tightBTagNTkHits, loadStdRecoBTag, svClustering, fatJets, groomedFatJets,
                          _algo, rParam, btagDiscriminators, btagInfos, _patJets, _labelName, btagPrefix, postfix)
            ## update final selected jets
            _newSelectedPatJets=getattr(process, 'selectedUpdatedPatJets'+_labelName+postfix)
            _newSelectedPatJets.src='updatedPatJetsTransientCorrected'+_labelName+postfix
            ## remove automatically added but redundant 'TransientCorrected' selected jets
            delattr(process, 'selectedUpdatedPatJetsTransientCorrected'+_labelName+postfix)
        else:
            _newPatJets.addBTagInfo = False
            _newPatJets.addTagInfos = False

        ## add jet correction factors if required by user
        if (jetCorrections is not None or bTagging):
            ## check the jet corrections format
            if jetCorrections is None and bTagging:
                raise ValueError("Passing jetCorrections = None while running bTagging is likely not intended.")
            else:
                checkJetCorrectionsFormat(jetCorrections)
            ## reset MET corrrection
            if jetCorrections[2].lower() != 'none' and jetCorrections[2] != '':
                sys.stderr.write("-------------------------------------------------------------------\n")
                sys.stderr.write(" Warning: MET correction was set to " + jetCorrections[2] + " but\n")
                sys.stderr.write("          will be ignored. Please set it to \"None\" to avoid\n")
                sys.stderr.write("          getting this warning.\n")
                sys.stderr.write("-------------------------------------------------------------------\n")
                jetCorrectionsList = list(jetCorrections)
                jetCorrectionsList[2] = 'None'
                jetCorrections = tuple(jetCorrectionsList)
            ## if running b tagging, need to use uncorrected jets
            if (bTagging):
                jetCorrections = ('AK4PFchs', cms.vstring([]), 'None')
            ## setup jet energy corrections
            setupJetCorrections(process, knownModules, jetCorrections, jetSource, pvSource, _newPatJets, _labelName, postfix)
        else:
            ## switch jetCorrFactors off
            _newPatJets.addJetCorrFactors=False

updateJetCollection=UpdateJetCollection()


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
        sys.stderr.write("Making new jet ID label with label " + jetIdTag + "\n")

        ## replace jet id sequence
        task = getPatAlgosToolsTask(process)
        process.load("RecoJets.JetProducers.ak4JetID_cfi")
        task.add(process.ak4JetID)
        addToProcessAndTask(jetIdLabel, process.ak4JetID.clone(src = jetSrc),
                            process, task)


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
            raise RuntimeError("""
            Cannot replace tag infos in jet collection""" % (coll))
        else :
            getattr(process,coll).tagInfoSources = newTags

setTagInfos=SetTagInfos()

def deprecatedOptionOutputModule(obj):
    sys.stderr.write("-------------------------------------------------------\n")
    sys.stderr.write(" Error: the option 'outputModule' is not supported\n")
    sys.stderr.write("        anymore by:\n")
    sys.stderr.write("                     " + obj._label + "\n")
    sys.stderr.write("        please use 'outputModules' now and specify the\n")
    sys.stderr.write("        names of all needed OutModules in there\n")
    sys.stderr.write("        (default: ['out'])\n")
    sys.stderr.write("-------------------------------------------------------\n")
    raise KeyError("Unsupported option 'outputModule' used in '"+obj._label+"'")

def undefinedLabelName(obj):
    sys.stderr.write("-------------------------------------------------------\n")
    sys.stderr.write(" Error: the jet 'labelName' is not defined.\n")
    sys.stderr.write("        All added jets must have 'labelName' defined.\n")
    sys.stderr.write("-------------------------------------------------------\n")
    raise KeyError("Undefined jet 'labelName' used in '"+obj._label+"'")

def unsupportedJetAlgorithm(obj):
    sys.stderr.write("-------------------------------------------------------\n")
    sys.stderr.write(" Error: Unsupported jet algorithm detected.\n")
    sys.stderr.write("        The supported algorithms are:\n")
    for key in supportedJetAlgos.keys():
        sys.stderr.write("        " + key.upper() + ", " + key.lower() + ": " + supportedJetAlgos[key] + "\n")
    sys.stderr.write("-------------------------------------------------------\n")
    raise KeyError("Unsupported jet algorithm used in '"+obj._label+"'")

def rerunningIVF():
    sys.stderr.write("-------------------------------------------------------------------\n")
    sys.stderr.write(" Warning: You are attempting to remake the IVF secondary vertices\n")
    sys.stderr.write("          already produced by the standard reconstruction. This\n")
    sys.stderr.write("          option is not enabled by default so please use it only if\n")
    sys.stderr.write("          you know what you are doing.\n")
    sys.stderr.write("-------------------------------------------------------------------\n")

def rerunningIVFMiniAOD():
    sys.stderr.write("-------------------------------------------------------------------\n")
    sys.stderr.write(" Warning: You are attempting to remake IVF secondary vertices from\n")
    sys.stderr.write("          MiniAOD. If that was your intention, note that secondary\n")
    sys.stderr.write("          vertices remade from MiniAOD will have somewhat degraded\n")
    sys.stderr.write("          performance compared to those remade from RECO/AOD.\n")
    sys.stderr.write("-------------------------------------------------------------------\n")
