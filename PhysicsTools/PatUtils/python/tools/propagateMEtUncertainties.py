import FWCore.ParameterSet.Config as cms
import PhysicsTools.PatAlgos.tools.helpers as configtools

def propagateMEtUncertainties(process, particleCollection, particleType,
                              shiftType, particleCollectionShiftUp, 
                              particleCollectionShiftDown, metProducer,
                              metType, sequence, postfix):

    # produce MET correction objects
    # (sum of differences in four-momentum between original and up/down shifted particle collection)

    moduleMETcorrShiftUp = cms.EDProducer("ShiftedParticleMETcorrInputProducer",
        srcOriginal = cms.InputTag(particleCollection),
        srcShifted = cms.InputTag(particleCollectionShiftUp)
    )
    moduleMETcorrShiftUpName = "pat%sMETcorr%s%sUp" % (metType,particleType, shiftType)
    moduleMETcorrShiftUpName += postfix
    setattr(process, moduleMETcorrShiftUpName, moduleMETcorrShiftUp)
    sequence += moduleMETcorrShiftUp
    moduleMETcorrShiftDown = moduleMETcorrShiftUp.clone(
        srcShifted = cms.InputTag(particleCollectionShiftDown)
    )
    moduleMETcorrShiftDownName = "pat%sMETcorr%s%sDown" % (metType,particleType, shiftType)
    moduleMETcorrShiftDownName += postfix
    setattr(process, moduleMETcorrShiftDownName, moduleMETcorrShiftDown)
    sequence += moduleMETcorrShiftDown
    
    # propagate effects of up/down shifts to MET
    moduleMETshiftUp = cms.EDProducer("CorrectedPATMETProducer",
            applyType1Corrections = cms.bool(True),
            applyType2Corrections = cms.bool(False),
            src = cms.InputTag(metProducer.label()),
            srcType1Corrections = cms.VInputTag(cms.InputTag(moduleMETcorrShiftUpName))
    )
    metProducerLabel = metProducer.label()
    if postfix != "":
        if metProducerLabel[-len(postfix):] == postfix:
            metProducerLabel = metProducerLabel[0:-len(postfix)]
        else:
            raise StandardError("Tried to remove postfix %s from label %s, but it wasn't there" % (postfix, metProducerLabel))
    moduleMETshiftUpName = "%s%s%sUp" % (metProducerLabel, particleType, shiftType)
    moduleMETshiftUpName += postfix
    setattr(process, moduleMETshiftUpName, moduleMETshiftUp)
    sequence += moduleMETshiftUp
    moduleMETshiftDown = moduleMETshiftUp.clone(
        srcType1Corrections = cms.VInputTag(
            cms.InputTag(moduleMETcorrShiftDownName)
        )
    )
    moduleMETshiftDownName = "%s%s%sDown" % (metProducerLabel, particleType, shiftType)
    moduleMETshiftDownName += postfix
    setattr(process, moduleMETshiftDownName, moduleMETshiftDown)
    sequence += moduleMETshiftDown
    
    metCollectionsUp_Down = [
        moduleMETshiftUpName,
        moduleMETshiftDownName
    ]
    
    return metCollectionsUp_Down




def propagateShiftedSingleParticles(process, shiftedParticleCollections, metProducers,
                                   metType, sequence, postfix):

    metCollectionsUpDown = []

    #looping over existing single particles collections, rejecting jets and Up or Down variations

    for shiftedCol in shiftedParticleCollections.keys():
        if shiftedCol.find('jet')!=-1 or shiftedCol.find('Jet')!=-1 or shiftedCol.find('Up')!=-1 or shiftedCol.find('Down')!=-1 :
            continue

        particleType=""
        if shiftedCol.find('electron')!=-1:
            particleType="Electron"
        if shiftedCol.find('photon')!=-1:
            particleType="Photon"
        if shiftedCol.find('muon')!=-1:
            particleType="Muon"
        if shiftedCol.find('tau')!=-1:     
            particleType="Tau"

        for metProducer in metProducers:
            tmpMetCollections = propagateMEtUncertainties(process, 
                                  shiftedParticleCollections[shiftedCol],
                                  particleType, "En",
                                  shiftedParticleCollections[shiftedCol+'EnUp'],
                                  shiftedParticleCollections[shiftedCol+'EnDown'],
                                  metProducer, metType, sequence, postfix)

            metCollectionsUpDown.extend( tmpMetCollections )

    return metCollectionsUpDown




#def propagateESShiftedJets(process, shiftedParticleCollections, metProducers,
#                           metType, sequence, postfix):
#
# metCollectionsUp_DownForRawMEt = \
#            propagateMEtUncertainties(
#              process, shiftedParticleCollections['lastJetCollection'], "Jet", "En",
#              shiftedParticleCollections['jetCollectionEnUpForRawMEt'], shiftedParticleCollections['jetCollectionEnDownForRawMEt'],
#              getattr(process, "patPFMet" + postfix), "PF", metUncertaintySequence, postfix)
#        collectionsToKeep.extend(metCollectionsUp_DownForRawMEt)




def propagateERShiftedJets(process, shiftedParticleCollections, metProducers,
                           metType, sequence, postfix):
    
    metCollectionsUpDown = []

    for metProducer in metProducers:

        tmpMetCollections = propagateMEtUncertainties(process,
            shiftedParticleCollections['lastJetCollection'], "Jet", "Res",
            shiftedParticleCollections['jetCollectionResUp'],
            shiftedParticleCollections['jetCollectionResDown'],
            metProducer, metType, sequence, postfix)
         
        metCollectionsUpDown.extend( tmpMetCollections )

    return metCollectionsUpDown





def createPatMETModules(process, metType, metPatSequence, applyT1Cor=False, 
                        applyT2Cor=False, applyT0pcCor=False, applyXYShiftCor=False, 
                        applyUncEnCalib=False,sysShiftCorrParameter=cms.VPSet(), postfix=""):

    ##FIXME: postfix is set to null as the whoelsequence receive it later
    postfix=""

    if applyUncEnCalib :
        applyT2Cor = True

    ## standard naming convention
    metModName = "pat"+metType+"Met"
    metModNameT1=metModName
    metModNameT1T2=metModName
    if applyT0pcCor :
        metModNameT1 += "T0pc"
        metModNameT1T2 += "T0pc"
    metModNameT1 += "T1"
    metModNameT1T2 += "T1T2"
    if applyXYShiftCor :
        metModNameT1 += "Txy"
        metModNameT1T2 += "Txy"
    # disabled spec. name for the moment as it just modifies the type2 MET
  #  if applyUncEnCalib :
  #      metModNameT1 += "UEC"
  #      metModNameT1T2 += "UEC"


    #plug the MET modules in to the sequence
    setattr(process, metModName,  getattr(process, metModName ) )
    if applyT1Cor :
        setattr(process, metModNameT1+postfix, getattr(process, metModNameT1 ).clone(
                src = cms.InputTag(metModName + postfix) 
                ))
        metPatSequence += getattr(process, metModNameT1+postfix)
    if applyT2Cor :
        setattr(process, metModNameT1T2+postfix, getattr(process, metModNameT1T2 ).clone(
                src = cms.InputTag(metModName + postfix) 
                ) )
        metPatSequence += getattr(process, metModNameT1T2+postfix)

    patMetCorrectionsCentralValue = []


    #Type0 for pfT1 and pfT1T2 MET
    if metType == "PF":
        patMetCorrectionsCentralValue = [ cms.InputTag('patPFJetMETtype1p2Corr' + postfix, 'type1') ]
        if applyT0pcCor :
            patMetCorrectionsCentralValue.extend([ cms.InputTag('patPFMETtype0Corr' + postfix) ])


    # compute XY shift correction if asked, and configure the tool accordingly
    if applyXYShiftCor :
        if not hasattr(process, 'pfMEtSysShiftCorrSequence'):
            process.load("JetMETCorrections.Type1MET.pfMETsysShiftCorrections_cfi")
        if postfix != "":
            configtools.cloneProcessingSnippet(process, process.pfMEtSysShiftCorrSequence, postfix)

        getattr(process, "pfMEtSysShiftCorr" + postfix).parameter = sysShiftCorrParameter
        metPatSequence += getattr(process, "pfMEtSysShiftCorrSequence" + postfix)

        patMetCorrectionsCentralValue.extend([ cms.InputTag('pfMEtSysShiftCorr' + postfix) ])
        

    #finalize T1/T2 correction process
    if applyT1Cor :
        getattr(process, metModNameT1 + postfix).srcType1Corrections = cms.VInputTag(patMetCorrectionsCentralValue)
    if applyT2Cor :
        getattr(process, metModNameT1T2 + postfix).srcType1Corrections = cms.VInputTag(patMetCorrectionsCentralValue)


    # Apply unclustered energy calibration on pfMET T1T2 if asked -> discard type2 and replace it with 
    # calibration computed with the jet residual correction
    if metType == "PF":
        if applyUncEnCalib:
            applyUnclEnergyCalibrationOnPfT1T2Met(process, postfix)
            patPFMetT1T2 = getattr(process, metModNameT1T2)
            patPFMetT1T2.applyType2Corrections = cms.bool(True)
            patPFMetT1T2.srcUnclEnergySums = cms.VInputTag(
                cms.InputTag('pfCandMETresidualCorr' + postfix),
                cms.InputTag("patPFJetMETtype1p2Corr" + postfix, "type2")
                )
            patPFMetT1T2.type2CorrFormula = cms.string("A")
            patPFMetT1T2.type2CorrParameter = cms.PSet(A = cms.double(2.))


    collectionsToKeep = [ 'patPFMet' + postfix ]
    if applyT1Cor:
        collectionsToKeep.append( metModNameT1 + postfix )
    if applyT2Cor:
        collectionsToKeep.append( metModNameT1T2 + postfix )

    return (metModName, metModNameT1, metModNameT1T2, collectionsToKeep)




def applyUnclEnergyCalibrationOnPfT1T2Met(process, postfix):

    patPFJetMETtype1p2Corr = getattr(process, "patPFJetMETtype1p2Corr" + postfix)
    patPFJetMETtype1p2Corr.type2ResidualCorrLabel = cms.string("")
    patPFJetMETtype1p2Corr.type2ResidualCorrEtaMax = cms.double(9.9)
    patPFJetMETtype1p2Corr.type2ResidualCorrOffset = cms.double(1.)
    patPFJetMETtype1p2Corr.type2ExtraCorrFactor = cms.double(1.)
    patPFJetMETtype1p2Corr.isMC = cms.bool(True)
    patPFJetMETtype1p2Corr.srcGenPileUpSummary = cms.InputTag('addPileupInfo')
    patPFJetMETtype1p2Corr.type2ResidualCorrVsNumPileUp = cms.PSet(
        data = cms.PSet(
            offset = cms.FileInPath('JetMETCorrections/Type1MET/data/unclEnResidualCorr_Data_runs190456to208686_pfCands_offset.txt'),
            slope = cms.FileInPath('JetMETCorrections/Type1MET/data/unclEnResidualCorr_Data_runs190456to208686_pfCands_slope.txt')
            ),
        mc = cms.PSet(
            offset = cms.FileInPath('JetMETCorrections/Type1MET/data/unclEnResidualCorr_ZplusJets_madgraph_pfCands_offset.txt'),
            slope = cms.FileInPath('JetMETCorrections/Type1MET/data/unclEnResidualCorr_ZplusJets_madgraph_pfCands_slope.txt')
            )
        )
    patPFJetMETtype1p2Corr.verbosity = cms.int32(0)
    pfCandMETcorr = getattr(process, "pfCandMETcorr" + postfix)
    pfCandMETresidualCorr = pfCandMETcorr.clone(
        residualCorrLabel = cms.string(""),
        residualCorrEtaMax = cms.double(9.9),
        residualCorrOffset = cms.double(1.),
        extraCorrFactor = cms.double(1.),
        isMC = cms.bool(True),
        srcGenPileUpSummary = cms.InputTag('addPileupInfo'),
        residualCorrVsNumPileUp = cms.PSet(
            data = cms.PSet(
                offset = cms.FileInPath('JetMETCorrections/Type1MET/data/unclEnResidualCorr_Data_runs190456to208686_pfCands_offset.txt'),
                slope = cms.FileInPath('JetMETCorrections/Type1MET/data/unclEnResidualCorr_Data_runs190456to208686_pfCands_slope.txt')
                ),
            mc = cms.PSet(
                offset = cms.FileInPath('JetMETCorrections/Type1MET/data/unclEnResidualCorr_ZplusJets_madgraph_pfCands_offset.txt'),
                slope = cms.FileInPath('JetMETCorrections/Type1MET/data/unclEnResidualCorr_ZplusJets_madgraph_pfCands_slope.txt')
                )
            ),
        verbosity = cms.int32(0)  
        )
    setattr(process, "pfCandMETresidualCorr" + postfix, pfCandMETresidualCorr)
    getattr(process, "producePatPFMETCorrectionsUnc" + postfix).replace(pfCandMETcorr, pfCandMETcorr + pfCandMETresidualCorr)
    
