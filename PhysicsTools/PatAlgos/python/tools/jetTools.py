from FWCore.GuiBrowsers.ConfigToolBase import *
from FWCore.ParameterSet.Mixins import PrintOptions,_ParameterTypeBase,_SimpleParameterTypeBase, _Parameterizable, _ConfigureComponent, _TypedParameterizable, _Labelable,  _Unlabelable,  _ValidatingListBase
from FWCore.ParameterSet.SequenceTypes import _ModuleSequenceType, _Sequenceable
from FWCore.ParameterSet.SequenceTypes import *
from PhysicsTools.PatAlgos.tools.helpers import *
from PhysicsTools.PatAlgos.recoLayer0.bTagging_cff import *
import sys

## dictionary with supported jet clustering algorithms
supportedJetAlgos = {
   'ak' : 'AntiKt'
 , 'ca' : 'CambridgeAachen'
 , 'kt' : 'Kt'
}

def checkJetCorrectionsFormat(jetCorrections):
    ## check for the correct format
    if type(jetCorrections) != type(('PAYLOAD-LABEL',['CORRECTION-LEVEL-A','CORRECTION-LEVEL-B'], 'MET-LABEL')):
        raise ValueError, "In addJetCollection: 'jetCorrections' must be 'None' (as a python value w/o quotation marks), or of type ('PAYLOAD-LABEL', ['CORRECTION-LEVEL-A', \
        'CORRECTION-LEVEL-B', ...], 'MET-LABEL'). Note that 'MET-LABEL' can be set to 'None' (as a string in quotation marks) in case you do not want to apply MET(Type1) \
        corrections."


def setupJetCorrections(process, knownModules, jetCorrections, jetSource, pvSource, patJets, labelName, postfix):
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
    if 'patJetCorrFactors'+labelName+postfix in knownModules :
        _newPatJetCorrFactors=getattr(process, 'patJetCorrFactors'+labelName+postfix)
        _newPatJetCorrFactors.src=jetSource
        _newPatJetCorrFactors.primaryVertices=pvSource
    else:
        setattr(process, 'patJetCorrFactors'+labelName+postfix, patJetCorrFactors.clone(src=jetSource, primaryVertices=pvSource))
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
    patJets.jetCorrFactorsSource=cms.VInputTag(cms.InputTag('patJetCorrFactors'+labelName+postfix))
    ## configure MET(Type1) corrections
    if jetCorrections[2].lower() != 'none' and jetCorrections[2] != '':
        if not jetCorrections[2].lower() == 'type-1' and not jetCorrections[2].lower() == 'type-2':
            raise valueError, "In addJetCollection: Wrong choice of MET corrections for new jet collection. Possible choices are None (or empty string), Type-1, Type-2 (i.e.\
            Type-1 and Type-2 corrections applied). This choice is not case sensitive. Your choice was: ", jetCorrections[2]
        if _type == "JPT":
            raise ValueError, "In addJecCollection: MET(type1) corrections are not supported for JPTJets. Please set the MET-LABEL to \"None\" (as string in quatiation \
            marks) and use raw tcMET together with JPTJets."
        ## set up jet correctors for MET corrections
        process.load( "JetMETCorrections.Configuration.JetCorrectorsAllAlgos_cff") # FIXME: This adds a lot of garbage

        _payloadType = jetCorrections[0].split(_type)[0].lower()+_type
        if "PF" in _type :
            setattr(process, jetCorrections[0]+'L1FastJet', getattr(process, _payloadType+'L1FastjetCorrector').clone(srcRho=cms.InputTag('fixedGridRhoFastjetAll')))
        else :
            setattr(process, jetCorrections[0]+'L1FastJet', getattr(process, _payloadType+'L1FastjetCorrector').clone(srcRho=cms.InputTag('fixedGridRhoFastjetAllCalo')))
        setattr(process, jetCorrections[0]+'L1Offset', getattr(process, _payloadType+'L1OffsetCorrector').clone())
        setattr(process, jetCorrections[0]+'L2Relative', getattr(process, _payloadType+'L2RelativeCorrector').clone())
        setattr(process, jetCorrections[0]+'L3Absolute', getattr(process, _payloadType+'L3AbsoluteCorrector').clone())
        setattr(process, jetCorrections[0]+'L2L3Residual', getattr(process, _payloadType+'ResidualCorrector').clone())
        setattr(process, jetCorrections[0]+'CombinedCorrector', cms.EDProducer( 'ChainedJetCorrectorProducer', correctors = cms.VInputTag()))
        for x in jetCorrections[1]:
            if x != 'L1FastJet' and x != 'L1Offset' and x != 'L2Relative' and x != 'L3Absolute' and x != 'L2L3Residual':
                raise ValueError, 'In addJetCollection: Unsupported JEC for MET(Type1). Currently supported jet correction levels are L1FastJet, L1Offset, L2Relative, L3Asolute, L2L3Residual. Requested was: %s'%(x)
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
            setattr(process,jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix, corrCaloMetType1.clone(src=jetSource,srcMET = "caloMetM",jetCorrLabel = cms.InputTag(jetCorrections[0]+'CombinedCorrector')))
            setattr(process,jetCorrections[0]+_labelCorrName+'JetMETcorr2'+postfix, corrCaloMetType2.clone(srcUnclEnergySums = cms.VInputTag(cms.InputTag(jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix, 'type2'),cms.InputTag(jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix, 'offset'),cms.InputTag('muCaloMetCorr'))))
            setattr(process,jetCorrections[0]+_labelCorrName+'Type1CorMet'+postfix, caloMetT1.clone(src = "caloMetM", srcCorrections = cms.VInputTag(cms.InputTag(jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix, 'type1'))))
            setattr(process,jetCorrections[0]+_labelCorrName+'Type1p2CorMet'+postfix, caloMetT1T2.clone(src = "caloMetM", srcCorrections = cms.VInputTag(cms.InputTag(jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix, 'type1'), cms.InputTag(jetCorrections[0]+_labelCorrName+'JetMETcorr2'+postfix))))

        elif _type == 'PF':
            from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import pfJetsPtrForMetCorr
            from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import pfCandsNotInJetsPtrForMetCorr
            from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import pfCandsNotInJetsForMetCorr
            from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import pfCandMETcorr
            from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import corrPfMetType1
            from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import corrPfMetType2
            from JetMETCorrections.Type1MET.correctedMet_cff import pfMetT1
            from JetMETCorrections.Type1MET.correctedMet_cff import pfMetT1T2
            setattr(process,jetCorrections[0]+_labelCorrName+'pfJetsPtrForMetCorr'+postfix,pfJetsPtrForMetCorr.clone(src = jetSource))
            setattr(process,jetCorrections[0]+_labelCorrName+'pfCandsNotInJetsPtrForMetCorr'+postfix,pfCandsNotInJetsPtrForMetCorr.clone(topCollection = jetCorrections[0]+_labelCorrName+'pfJetsPtrForMetCorr'+postfix))
            setattr(process,jetCorrections[0]+_labelCorrName+'pfCandsNotInJetsForMetCorr'+postfix,pfCandsNotInJetsForMetCorr.clone(src = jetCorrections[0]+_labelCorrName+'pfCandsNotInJetsPtrForMetCorr'+postfix))
            setattr(process,jetCorrections[0]+_labelCorrName+'CandMETcorr'+postfix, pfCandMETcorr.clone(src = cms.InputTag(jetCorrections[0]+_labelCorrName+'pfCandsNotInJetsForMetCorr'+postfix)))
            setattr(process,jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix, corrPfMetType1.clone(src = jetSource, jetCorrLabel = cms.InputTag(jetCorrections[0]+'CombinedCorrector'))) # FIXME: Originally w/o jet corrections?
            setattr(process,jetCorrections[0]+_labelCorrName+'corrPfMetType2'+postfix, corrPfMetType2.clone(srcUnclEnergySums = cms.VInputTag(cms.InputTag(jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix, 'type2'),cms.InputTag(jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix, 'offset'),cms.InputTag(jetCorrections[0]+_labelCorrName+'CandMETcorr'+postfix))))
            setattr(process,jetCorrections[0]+_labelCorrName+'Type1CorMet'+postfix, pfMetT1.clone(srcCorrections = cms.VInputTag(cms.InputTag(jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix, 'type1'))))
            setattr(process,jetCorrections[0]+_labelCorrName+'Type1p2CorMet'+postfix, pfMetT1T2.clone(srcCorrections = cms.VInputTag(cms.InputTag(jetCorrections[0]+_labelCorrName+'JetMETcorr'+postfix, 'type1'), jetCorrections[0]+_labelCorrName+'corrPfMetType2'+postfix)))

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
            setattr(process, 'patMETs'+labelName+postfix, patMETs.clone(metSource = cms.InputTag(jetCorrections[0]+_labelCorrName+'Type1CorMet'+postfix), addMuonCorrections = False))
        elif jetCorrections[2].lower() == 'type-2':
            setattr(process, 'patMETs'+labelName+postfix, patMETs.clone(metSource = cms.InputTag(jetCorrections[0]+_labelCorrName+'Type1p2CorMet'+postfix), addMuonCorrections = False))


def setupSVClustering(btagInfo, algo, rParam, fatJets=cms.InputTag(''), groomedFatJets=cms.InputTag('')):
    btagInfo.useSVClustering = cms.bool(True)
    btagInfo.jetAlgorithm    = cms.string(algo)
    btagInfo.rParam          = cms.double(rParam)
    ## if the jets is actually a subjet
    if fatJets != cms.InputTag('') and groomedFatJets != cms.InputTag(''):
        btagInfo.fatJets        = fatJets
        btagInfo.groomedFatJets = groomedFatJets


def setupBTagging(process, jetSource, pfCandidates, explicitJTA, pvSource, svSource, elSource, muSource, runIVF, svClustering, fatJets, groomedFatJets,
                  algo, rParam, btagDiscriminators, btagInfos, patJets, labelName, postfix):
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
    ## load sequences and setups needed for btagging
    ## This loads all available btagger, but the ones we need are added to the process by hand later. Only needed to get the ESProducer. Needs improvement
    if hasattr( process, 'candidateJetProbabilityComputer' ) == False :
        #process.load("RecoBTag.Configuration.RecoBTag_cff") # commented out to prevent loading of IVF modules already run in the standard reconstruction. Instead, loading individual cffs from RecoBTag_cff
        process.load("RecoBTag.ImpactParameter.impactParameter_cff")
        process.load("RecoBTag.SecondaryVertex.secondaryVertex_cff")
        process.load("RecoBTag.SoftLepton.softLepton_cff")
        process.load("RecoBTag.Combined.combinedMVA_cff")
        process.load("RecoBTag.CTagging.RecoCTagging_cff")
    #addESProducers(process,'RecoBTag.Configuration.RecoBTag_cff')
    import RecoBTag.Configuration.RecoBTag_cff as btag
    import RecoJets.JetProducers.caTopTaggers_cff as toptag

    ## setup all required btagInfos : we give a dedicated treatment for different
    ## types of tagInfos here. A common treatment is possible but might require a more
    ## general approach anyway in coordination with the btagging POG.
    acceptedTagInfos = list()
    for btagInfo in requiredTagInfos:
        if hasattr(btag,btagInfo):
            if btagInfo == 'pfImpactParameterTagInfos':
                setattr(process, btagInfo+labelName+postfix, btag.pfImpactParameterTagInfos.clone(jets = jetSource,primaryVertex=pvSource,candidates=pfCandidates))
                if explicitJTA:
                    _btagInfo = getattr(process, btagInfo+labelName+postfix)
                    _btagInfo.explicitJTA = cms.bool(explicitJTA)
            if btagInfo == 'pfImpactParameterAK8TagInfos':
                setattr(process, btagInfo+labelName+postfix, btag.pfImpactParameterAK8TagInfos.clone(jets = jetSource,primaryVertex=pvSource,candidates=pfCandidates))
                if explicitJTA:
                    _btagInfo = getattr(process, btagInfo+labelName+postfix)
                    _btagInfo.explicitJTA = cms.bool(explicitJTA)
            if btagInfo == 'pfImpactParameterCA15TagInfos':
                setattr(process, btagInfo+labelName+postfix, btag.pfImpactParameterCA15TagInfos.clone(jets = jetSource,primaryVertex=pvSource,candidates=pfCandidates))
                if explicitJTA:
                    _btagInfo = getattr(process, btagInfo+labelName+postfix)
                    _btagInfo.explicitJTA = cms.bool(explicitJTA)
            if btagInfo == 'pfSecondaryVertexTagInfos':
                setattr(process, btagInfo+labelName+postfix, btag.pfSecondaryVertexTagInfos.clone(trackIPTagInfos = cms.InputTag('pfImpactParameterTagInfos'+labelName+postfix)))
            if btagInfo == 'pfInclusiveSecondaryVertexFinderTagInfos':
                setattr(process, btagInfo+labelName+postfix, btag.pfInclusiveSecondaryVertexFinderTagInfos.clone(trackIPTagInfos = cms.InputTag('pfImpactParameterTagInfos'+labelName+postfix), extSVCollection=svSource))
                if svClustering:
                    setupSVClustering(getattr(process, btagInfo+labelName+postfix), algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'pfInclusiveSecondaryVertexFinderAK8TagInfos':
                setattr(process, btagInfo+labelName+postfix, btag.pfInclusiveSecondaryVertexFinderAK8TagInfos.clone(trackIPTagInfos = cms.InputTag('pfImpactParameterAK8TagInfos'+labelName+postfix), extSVCollection=svSource))
                if svClustering:
                    setupSVClustering(getattr(process, btagInfo+labelName+postfix), algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'pfInclusiveSecondaryVertexFinderCA15TagInfos':
                setattr(process, btagInfo+labelName+postfix, btag.pfInclusiveSecondaryVertexFinderCA15TagInfos.clone(trackIPTagInfos = cms.InputTag('pfImpactParameterCA15TagInfos'+labelName+postfix), extSVCollection=svSource))
                if svClustering:
                    setupSVClustering(getattr(process, btagInfo+labelName+postfix), algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'pfInclusiveSecondaryVertexFinderCvsLTagInfos':
                setattr(
                    process, 
                    btagInfo+labelName+postfix, 
                    btag.pfInclusiveSecondaryVertexFinderCvsLTagInfos.clone(
                        trackIPTagInfos = cms.InputTag('pfImpactParameterTagInfos'+labelName+postfix), 
                        extSVCollection=svSource
                        )
                    )
                if svClustering:
                    setupSVClustering(getattr(process, btagInfo+labelName+postfix), algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'pfInclusiveSecondaryVertexFinderCvsBTagInfos':
                setattr(
                    process, 
                    btagInfo+labelName+postfix, 
                    btag.pfInclusiveSecondaryVertexFinderCvsBTagInfos.clone(
                        trackIPTagInfos = cms.InputTag('pfImpactParameterTagInfos'+labelName+postfix), 
                        extSVCollection=svSource
                        )
                    )
                if svClustering:
                    setupSVClustering(getattr(process, btagInfo+labelName+postfix), algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'pfSecondaryVertexNegativeTagInfos':
                setattr(process, btagInfo+labelName+postfix, btag.pfSecondaryVertexNegativeTagInfos.clone(trackIPTagInfos = cms.InputTag('pfImpactParameterTagInfos'+labelName+postfix)))
            if btagInfo == 'pfInclusiveSecondaryVertexFinderNegativeTagInfos':
                setattr(process, btagInfo+labelName+postfix, btag.pfInclusiveSecondaryVertexFinderNegativeTagInfos.clone(trackIPTagInfos = cms.InputTag('pfImpactParameterTagInfos'+labelName+postfix), extSVCollection=svSource))
                if svClustering:
                    setupSVClustering(getattr(process, btagInfo+labelName+postfix), algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'impactParameterTagInfos':
                setattr(process, btagInfo+labelName+postfix, btag.impactParameterTagInfos.clone(jetTracks = cms.InputTag('jetTracksAssociatorAtVertex'+labelName+postfix), primaryVertex=pvSource))
            if btagInfo == 'secondaryVertexTagInfos':
                setattr(process, btagInfo+labelName+postfix, btag.secondaryVertexTagInfos.clone(trackIPTagInfos = cms.InputTag('impactParameterTagInfos'+labelName+postfix)))
            if btagInfo == 'inclusiveSecondaryVertexFinderTagInfos':
                setattr(process, btagInfo+labelName+postfix, btag.inclusiveSecondaryVertexFinderTagInfos.clone(trackIPTagInfos = cms.InputTag('impactParameterTagInfos'+labelName+postfix)))
                if svClustering:
                    setupSVClustering(getattr(process, btagInfo+labelName+postfix), algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'inclusiveSecondaryVertexFinderFilteredTagInfos':
                setattr(process, btagInfo+labelName+postfix, btag.inclusiveSecondaryVertexFinderFilteredTagInfos.clone(trackIPTagInfos = cms.InputTag('impactParameterTagInfos'+labelName+postfix)))
                if svClustering:
                    setupSVClustering(getattr(process, btagInfo+labelName+postfix), algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'secondaryVertexNegativeTagInfos':
                setattr(process, btagInfo+labelName+postfix, btag.secondaryVertexNegativeTagInfos.clone(trackIPTagInfos = cms.InputTag('impactParameterTagInfos'+labelName+postfix)))
            if btagInfo == 'inclusiveSecondaryVertexFinderNegativeTagInfos':
                setattr(process, btagInfo+labelName+postfix, btag.inclusiveSecondaryVertexFinderNegativeTagInfos.clone(trackIPTagInfos = cms.InputTag('impactParameterTagInfos'+labelName+postfix)))
                if svClustering:
                    setupSVClustering(getattr(process, btagInfo+labelName+postfix), algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'inclusiveSecondaryVertexFinderFilteredNegativeTagInfos':
                setattr(process, btagInfo+labelName+postfix, btag.inclusiveSecondaryVertexFinderFilteredNegativeTagInfos.clone(trackIPTagInfos = cms.InputTag('impactParameterTagInfos'+labelName+postfix)))
                if svClustering:
                    setupSVClustering(getattr(process, btagInfo+labelName+postfix), algo, rParam, fatJets, groomedFatJets)
            if btagInfo == 'softMuonTagInfos':
                setattr(process, btagInfo+labelName+postfix, btag.softMuonTagInfos.clone(jets = jetSource, primaryVertex=pvSource))
            if btagInfo == 'softPFMuonsTagInfos':
                setattr(process, btagInfo+labelName+postfix, btag.softPFMuonsTagInfos.clone(jets = jetSource, primaryVertex=pvSource, muons=muSource))
            if btagInfo == 'softPFElectronsTagInfos':
                setattr(process, btagInfo+labelName+postfix, btag.softPFElectronsTagInfos.clone(jets = jetSource, primaryVertex=pvSource, electrons=elSource))
            acceptedTagInfos.append(btagInfo)
        elif hasattr(toptag, btagInfo) :
            acceptedTagInfos.append(btagInfo)
        else:
            print '  --> %s ignored, since not available via RecoBTag.Configuration.RecoBTag_cff!'%(btagInfo)
    ## setup all required btagDiscriminators
    acceptedBtagDiscriminators = list()
    for btagDiscr in btagDiscriminators :
        if hasattr(btag,btagDiscr):
            setattr(process, btagDiscr+labelName+postfix, getattr(btag, btagDiscr).clone(tagInfos = cms.VInputTag( *[ cms.InputTag(x+labelName+postfix) for x in supportedBtagDiscr[btagDiscr] ] )))
            acceptedBtagDiscriminators.append(btagDiscr)
        else:
            print '  --> %s ignored, since not available via RecoBTag.Configuration.RecoBTag_cff!'%(btagDiscr)
    ## replace corresponding tags for pat jet production
    patJets.tagInfoSources = cms.VInputTag( *[ cms.InputTag(x+labelName+postfix) for x in acceptedTagInfos ] )
    patJets.discriminatorSources = cms.VInputTag( *[ cms.InputTag(x+labelName+postfix) for x in acceptedBtagDiscriminators ] )
    if len(acceptedBtagDiscriminators) > 0 :
        patJets.addBTagInfo = True
    if runIVF:
        rerunningIVF()
        if 'pfInclusiveSecondaryVertexFinderTagInfos' in acceptedTagInfos:
            if not hasattr( process, 'inclusiveCandidateVertexing' ):
                process.load( 'RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff' )
        if 'inclusiveSecondaryVertexFinderTagInfos' in acceptedTagInfos:
            if not hasattr( process, 'inclusiveVertexing' ):
                process.load( 'RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff' )
        if 'inclusiveSecondaryVertexFinderFilteredTagInfos' in acceptedTagInfos:
            if not hasattr( process, 'inclusiveVertexing' ):
                process.load( 'RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff' )
    if 'inclusiveSecondaryVertexFinderFilteredTagInfos' in acceptedTagInfos:
        if not hasattr( process, 'inclusiveSecondaryVerticesFiltered' ):
            process.load( 'RecoBTag.SecondaryVertex.secondaryVertex_cff' )
        if not hasattr( process, 'bToCharmDecayVertexMerged' ):
            process.load( 'RecoBTag.SecondaryVertex.bToCharmDecayVertexMerger_cfi' )
    if 'caTopTagInfos' in acceptedTagInfos :
        patJets.addTagInfos = True
        if not hasattr( process, 'caTopTagInfos' ) and not hasattr( process, 'caTopTagInfosAK8' ):
            process.load( 'RecoJets.JetProducers.caTopTaggers_cff' )


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
        self.addParameter(self._defaultParameters,'jetSource','', "Label of the input collection from which the new patJet collection should be created", cms.InputTag)
        self.addParameter(self._defaultParameters,'pfCandidates',cms.InputTag('particleFlow'), "Label of the input collection for candidatecandidatese used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'explicitJTA', False, "Use explicit jet-track association")
        self.addParameter(self._defaultParameters,'pvSource',cms.InputTag('offlinePrimaryVertices'), "Label of the input collection for primary vertices used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'svSource',cms.InputTag('inclusiveCandidateSecondaryVertices'), "Label of the input collection for IVF vertices used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'elSource',cms.InputTag('gedGsfElectrons'), "Label of the input collection for electrons used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'muSource',cms.InputTag('muons'), "Label of the input collection for muons used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'runIVF', False, "Re-run IVF secondary vertex reconstruction")
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

    def __call__(self,process,labelName=None,postfix=None,jetSource=None,pfCandidates=None,explicitJTA=None,pvSource=None,svSource=None,elSource=None,muSource=None,runIVF=None,svClustering=None,fatJets=None,groomedFatJets=None,algo=None,rParam=None,getJetMCFlavour=None,genJetCollection=None,genParticles=None,jetCorrections=None,btagDiscriminators=None,btagInfos=None,jetTrackAssociation=None,outputModules=None):
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
        labelName=self._parameters['labelName'].value
        postfix=self._parameters['postfix'].value
        jetSource=self._parameters['jetSource'].value
        pfCandidates=self._parameters['pfCandidates'].value
        explicitJTA=self._parameters['explicitJTA'].value
        pvSource=self._parameters['pvSource'].value
        svSource=self._parameters['svSource'].value
        elSource=self._parameters['elSource'].value
        muSource=self._parameters['muSource'].value
        runIVF=self._parameters['runIVF'].value
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
        from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import patJets
        if 'patJets'+_labelName+postfix in knownModules :
            _newPatJets=getattr(process, 'patJets'+_labelName+postfix)
            _newPatJets.jetSource=jetSource
        else :
            setattr(process, 'patJets'+_labelName+postfix, patJets.clone(jetSource=jetSource))
            _newPatJets=getattr(process, 'patJets'+_labelName+postfix)
            knownModules.append('patJets'+_labelName+postfix)
        ## add new selectedPatJets to process
        from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import selectedPatJets
        if 'selectedPatJets'+_labelName+postfix in knownModules :
            _newSelectedPatJets=getattr(process, 'selectedPatJets'+_labelName+postfix)
            _newSelectedPatJets.src='patJets'+_labelName+postfix
        else :
            setattr(process, 'selectedPatJets'+_labelName+postfix, selectedPatJets.clone(src='patJets'+_labelName+postfix))
            knownModules.append('selectedPatJets'+_labelName+postfix)

        ## add new patJetPartonMatch to process
        from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import patJetPartonMatch
        if 'patJetPartonMatch'+_labelName+postfix in knownModules :
            _newPatJetPartonMatch=getattr(process, 'patJetPartonMatch'+_labelName+postfix)
            _newPatJetPartonMatch.src=jetSource
            _newPatJetPartonMatch.matched=genParticles
        else :
            setattr(process, 'patJetPartonMatch'+_labelName+postfix, patJetPartonMatch.clone(src=jetSource, matched=genParticles))
            knownModules.append('patJetPartonMatch'+_labelName+postfix)
        ## add new patJetGenJetMatch to process
        from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import patJetGenJetMatch
        if 'patJetGenJetMatch'+_labelName+postfix in knownModules :
            _newPatJetGenJetMatch=getattr(process, 'patJetGenJetMatch'+_labelName+postfix)
            _newPatJetGenJetMatch.src=jetSource
            _newPatJetGenJetMatch.maxDeltaR=rParam
            _newPatJetGenJetMatch.matched=genJetCollection
        else :
            setattr(process, 'patJetGenJetMatch'+_labelName+postfix, patJetGenJetMatch.clone(src=jetSource, maxDeltaR=rParam, matched=genJetCollection))
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
                setattr(process, 'patJetPartonsLegacy'+postfix, patJetPartonsLegacy.clone(src=genParticles))
                knownModules.append('patJetPartonsLegacy'+postfix)
            else:
                getattr(process, 'patJetPartonsLegacy'+postfix).src=genParticles
            ## add new patJetPartonAssociationLegacy to process
            from PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff import patJetPartonAssociationLegacy
            if 'patJetPartonAssociationLegacy'+_labelName+postfix in knownModules :
                _newPatJetPartonAssociation=getattr(process, 'patJetPartonAssociationLegacy'+_labelName+postfix)
                _newPatJetPartonAssociation.jets=jetSource
            else :
                setattr(process, 'patJetPartonAssociationLegacy'+_labelName+postfix, patJetPartonAssociationLegacy.clone(jets=jetSource))
                knownModules.append('patJetPartonAssociationLegacy'+_labelName+postfix)
            ## add new patJetPartonAssociationLegacy to process
            from PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff import patJetFlavourAssociationLegacy
            if 'patJetFlavourAssociationLegacy'+_labelName+postfix in knownModules :
                _newPatJetFlavourAssociation=getattr(process, 'patJetFlavourAssociationLegacy'+_labelName+postfix)
                _newPatJetFlavourAssociation.srcByReference='patJetPartonAssociationLegacy'+_labelName+postfix
            else:
                setattr(process, 'patJetFlavourAssociationLegacy'+_labelName+postfix, patJetFlavourAssociationLegacy.clone(srcByReference='patJetPartonAssociationLegacy'+_labelName+postfix))
                knownModules.append('patJetFlavourAssociationLegacy'+_labelName+postfix)
            ## modify new patJets collection accordingly
            _newPatJets.JetPartonMapSource.setModuleLabel('patJetFlavourAssociationLegacy'+_labelName+postfix)
            ## new jet flavour (see https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideBTagMCTools)
            ## add new patJetPartons to process
            from PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff import patJetPartons
            if 'patJetPartons'+postfix not in knownModules :
                setattr(process, 'patJetPartons'+postfix, patJetPartons.clone(particles=genParticles))
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
                _newPatJetFlavourAssociation.partons=cms.InputTag("patJetPartons"+postfix,"algorithmicPartons")
                _newPatJetFlavourAssociation.leptons=cms.InputTag("patJetPartons"+postfix,"leptons")
            else :
                setattr(process, 'patJetFlavourAssociation'+_labelName+postfix,
                        patJetFlavourAssociation.clone(
                            jets=jetSource,
                            jetAlgorithm=_algo,
                            rParam=rParam,
                            bHadrons = cms.InputTag("patJetPartons"+postfix,"bHadrons"),
                            cHadrons = cms.InputTag("patJetPartons"+postfix,"cHadrons"),
                            partons = cms.InputTag("patJetPartons"+postfix,"algorithmicPartons"),
                            leptons = cms.InputTag("patJetPartons"+postfix,"leptons")
                        )
                )
                knownModules.append('patJetFlavourAssociation'+_labelName+postfix)
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
                setattr(process, 'jetTracksAssociatorAtVertex'+_labelName+postfix, jetTracksAssociator.clone(jets=jetSource,pvSrc=pvSource))
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
            setupBTagging(process, jetSource, pfCandidates, explicitJTA, pvSource, svSource, elSource, muSource, runIVF, svClustering, fatJets, groomedFatJets,
                          _algo, rParam, btagDiscriminators, btagInfos, _newPatJets, _labelName, postfix)
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
        self.addParameter(self._defaultParameters,'jetSource','', "Label of the input collection from which the new patJet collection should be created", cms.InputTag)
        self.addParameter(self._defaultParameters,'pfCandidates',cms.InputTag('particleFlow'), "Label of the input collection for candidatecandidatese used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'explicitJTA', False, "Use explicit jet-track association")
        self.addParameter(self._defaultParameters,'pvSource',cms.InputTag('offlinePrimaryVertices'), "Label of the input collection for primary vertices used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'svSource',cms.InputTag('inclusiveCandidateSecondaryVertices'), "Label of the input collection for IVF vertices used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'elSource',cms.InputTag('gedGsfElectrons'), "Label of the input collection for electrons used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'muSource',cms.InputTag('muons'), "Label of the input collection for muons used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'runIVF', False, "Re-run IVF secondary vertex reconstruction")
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

    def __call__(self,process,postfix=None,jetSource=None,pfCandidates=None,explicitJTA=None,pvSource=None,svSource=None,elSource=None,muSource=None,runIVF=None,svClustering=None,fatJets=None,groomedFatJets=None,algo=None,rParam=None,getJetMCFlavour=None,genJetCollection=None,genParticles=None,jetCorrections=None,btagDiscriminators=None,btagInfos=None,jetTrackAssociation=None,outputModules=None):
        """
        Function call wrapper. This will check the parameters and call the actual implementation that
        can be found in toolCode via the base class function apply.
        """
        if postfix is None:
            postfix=self._defaultParameters['postfix'].value
        self.setParameter('postfix', postfix)
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
        jetSource=self._parameters['jetSource'].value
        pfCandidates=self._parameters['pfCandidates'].value
        explicitJTA=self._parameters['explicitJTA'].value
        pvSource=self._parameters['pvSource'].value
        svSource=self._parameters['svSource'].value
        elSource=self._parameters['elSource'].value
        muSource=self._parameters['muSource'].value
        runIVF=self._parameters['runIVF'].value
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
            jetSource=jetSource,
            pfCandidates=pfCandidates,
            explicitJTA=explicitJTA,
            pvSource=pvSource,
            svSource=svSource,
            elSource=elSource,
            muSource=muSource,
            runIVF=runIVF,
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
        self.addParameter(self._defaultParameters,'jetSource','', "Label of the input collection from which the new patJet collection should be created", cms.InputTag)
        self.addParameter(self._defaultParameters,'pfCandidates',cms.InputTag('packedPFCandidates'), "Label of the input collection for candidatecandidatese used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'explicitJTA', False, "Use explicit jet-track association")
        self.addParameter(self._defaultParameters,'pvSource',cms.InputTag('offlineSlimmedPrimaryVertices'), "Label of the input collection for primary vertices used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'svSource',cms.InputTag('slimmedSecondaryVertices'), "Label of the input collection for IVF vertices used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'elSource',cms.InputTag('slimmedElectrons'), "Label of the input collection for electrons used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'muSource',cms.InputTag('slimmedMuons'), "Label of the input collection for muons used in b-tagging", cms.InputTag)
        self.addParameter(self._defaultParameters,'runIVF', False, "Re-run IVF secondary vertex reconstruction")
        self.addParameter(self._defaultParameters,'svClustering', False, "Secondary vertices ghost-associated to jets using jet clustering (mostly intended for subjets)")
        self.addParameter(self._defaultParameters,'fatJets', cms.InputTag(''), "Fat jet collection used for secondary vertex clustering", cms.InputTag)
        self.addParameter(self._defaultParameters,'groomedFatJets', cms.InputTag(''), "Groomed fat jet collection used for secondary vertex clustering", cms.InputTag)
        self.addParameter(self._defaultParameters,'algo', 'AK', "Jet algorithm of the input collection from which the new patJet collection should be created")
        self.addParameter(self._defaultParameters,'rParam', 0.4, "Jet size (distance parameter R used in jet clustering)")
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
        If this list is empty no btag discriminator information will be added to your new patJet collection.", allowedValues=supportedBtagDiscr.keys(),Type=list)
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
        energy correction factors, btag infomration and generator match information to the new patJet collection depending on the parameters that you pass on to this function. Consult \
        the descriptions of each parameter for more information."

    def getDefaultParameters(self):
        """
        Return default parameters of the class
        """
        return self._defaultParameters

    def __call__(self,process,labelName=None,postfix=None,jetSource=None,pfCandidates=None,explicitJTA=None,pvSource=None,svSource=None,elSource=None,muSource=None,runIVF=None,svClustering=None,fatJets=None,groomedFatJets=None,algo=None,rParam=None,jetCorrections=None,btagDiscriminators=None,btagInfos=None):
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
        jetSource=self._parameters['jetSource'].value
        pfCandidates=self._parameters['pfCandidates'].value
        explicitJTA=self._parameters['explicitJTA'].value
        pvSource=self._parameters['pvSource'].value
        svSource=self._parameters['svSource'].value
        elSource=self._parameters['elSource'].value
        muSource=self._parameters['muSource'].value
        runIVF=self._parameters['runIVF'].value
        svClustering=self._parameters['svClustering'].value
        fatJets=self._parameters['fatJets'].value
        groomedFatJets=self._parameters['groomedFatJets'].value
        algo=self._parameters['algo'].value
        rParam=self._parameters['rParam'].value
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
        ## add new updatedPatJets to process (keep instance for later further modifications)
        from PhysicsTools.PatAlgos.producersLayer1.jetUpdater_cfi import updatedPatJets
        if 'updatedPatJets'+_labelName+postfix in knownModules :
            _newPatJets=getattr(process, 'updatedPatJets'+_labelName+postfix)
            _newPatJets.jetSource=jetSource
        else :
            setattr(process, 'updatedPatJets'+_labelName+postfix, updatedPatJets.clone(jetSource=jetSource))
            _newPatJets=getattr(process, 'updatedPatJets'+_labelName+postfix)
            knownModules.append('updatedPatJets'+_labelName+postfix)
        ## add new selectedUpdatedPatJets to process
        from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import selectedPatJets
        if 'selectedUpdatedPatJets'+_labelName+postfix in knownModules :
            _newSelectedPatJets=getattr(process, 'selectedUpdatedPatJets'+_labelName+postfix)
            _newSelectedPatJets.src='updatedPatJets'+_labelName+postfix
        else :
            setattr(process, 'selectedUpdatedPatJets'+_labelName+postfix, selectedPatJets.clone(src='updatedPatJets'+_labelName+postfix))
            knownModules.append('selectedUpdatedPatJets'+_labelName+postfix)

        ## run btagging if required by user
        if (bTagging):
            print "**************************************************************"
            print "b tagging needs to be run on uncorrected jets. Hence, the JECs"
            print "will first be undone for 'updatedPatJets%s' and then applied to"%(_labelName+postfix)
            print "'updatedPatJetsTransientCorrected%s'."%(_labelName+postfix)
            print "**************************************************************"
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
                svClustering=svClustering,
                fatJets=fatJets,
                groomedFatJets=groomedFatJets,
                algo=algo,
                rParam=rParam,
                jetCorrections = jetCorrections,
                postfix = postfix
            )
            ## setup btagging
            _patJets=getattr(process, 'updatedPatJetsTransientCorrected'+_labelName+postfix)
            setupBTagging(process, _jetSource, pfCandidates, explicitJTA, pvSource, svSource, elSource, muSource, runIVF, svClustering, fatJets, groomedFatJets,
                          _algo, rParam, btagDiscriminators, btagInfos, _patJets, _labelName, postfix)
            ## update final selected jets
            _newSelectedPatJets=getattr(process, 'selectedUpdatedPatJets'+_labelName+postfix)
            _newSelectedPatJets.src='updatedPatJetsTransientCorrected'+_labelName+postfix
            ## remove automatically added but redundant 'TransientCorrected' selected jets
            delattr(process, 'selectedUpdatedPatJetsTransientCorrected'+_labelName+postfix)
        else:
            _newPatJets.addBTagInfo = False
            _newPatJets.addTagInfos = False

        ## add jet correction factors if required by user
        if (jetCorrections != None or bTagging):
            ## check the jet corrections format
            checkJetCorrectionsFormat(jetCorrections)
            ## reset MET corrrection
            if jetCorrections[2].lower() != 'none' and jetCorrections[2] != '':
                print "-------------------------------------------------------------------"
                print " Warning: MET correction was set to " + jetCorrections[2] + " but"
                print "          will be ignored. Please set it to \"None\" to avoid"
                print "          getting this warning."
                print "-------------------------------------------------------------------"
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
        print "Making new jet ID label with label " + jetIdTag

        ## replace jet id sequence
        process.load("RecoJets.JetProducers.ak4JetID_cfi")
        setattr( process, jetIdLabel, process.ak4JetID.clone(src = jetSrc))


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

def deprecatedOptionOutputModule(obj):
    print "-------------------------------------------------------"
    print " Error: the option 'outputModule' is not supported"
    print "        anymore by:"
    print "                   ", obj._label
    print "        please use 'outputModules' now and specify the"
    print "        names of all needed OutModules in there"
    print "        (default: ['out'])"
    print "-------------------------------------------------------"
    raise KeyError, "Unsupported option 'outputModule' used in '"+obj._label+"'"

def undefinedLabelName(obj):
    print "-------------------------------------------------------"
    print " Error: the jet 'labelName' is not defined."
    print "        All added jets must have 'labelName' defined."
    print "-------------------------------------------------------"
    raise KeyError, "Undefined jet 'labelName' used in '"+obj._label+"'"

def unsupportedJetAlgorithm(obj):
    print "-------------------------------------------------------"
    print " Error: Unsupported jet algorithm detected."
    print "        The supported algorithms are:"
    for key in supportedJetAlgos.keys():
        print "        " + key.upper() + ", " + key.lower() + ": " + supportedJetAlgos[key]
    print "-------------------------------------------------------"
    raise KeyError, "Unsupported jet algorithm used in '"+obj._label+"'"

def rerunningIVF():
    print "-------------------------------------------------------------------"
    print " Warning: You are attempting to remake the IVF secondary vertices"
    print "          already produced by the standard reconstruction. If that"
    print "          was your intention, note that they should be remade only"
    print "          from RECO and AOD, i.e., MiniAOD should not be used."
    print "-------------------------------------------------------------------"
