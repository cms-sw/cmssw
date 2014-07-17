import FWCore.ParameterSet.Config as cms

def propagateMEtUncertainties(process,
                              particleCollection, particleType, shiftType, particleCollectionShiftUp, particleCollectionShiftDown,
                              metProducer, sequence, postfix):
    
    # produce MET correction objects
    # (sum of differences in four-momentum between original and up/down shifted particle collection)
    moduleMETcorrShiftUp = cms.EDProducer("ShiftedParticleMETcorrInputProducer",
        srcOriginal = cms.InputTag(particleCollection),
        srcShifted = cms.InputTag(particleCollectionShiftUp)
    )
    moduleMETcorrShiftUpName = "patPFMETcorr%s%sUp" % (particleType, shiftType)
    moduleMETcorrShiftUpName += postfix
    setattr(process, moduleMETcorrShiftUpName, moduleMETcorrShiftUp)
    sequence += moduleMETcorrShiftUp
    moduleMETcorrShiftDown = moduleMETcorrShiftUp.clone(
        srcShifted = cms.InputTag(particleCollectionShiftDown)
    )
    moduleMETcorrShiftDownName = "patPFMETcorr%s%sDown" % (particleType, shiftType)
    moduleMETcorrShiftDownName += postfix
    setattr(process, moduleMETcorrShiftDownName, moduleMETcorrShiftDown)
    sequence += moduleMETcorrShiftDown
    
    # propagate effects of up/down shifts to MET
    moduleMETshiftUp = metProducer.clone(
        src = cms.InputTag(metProducer.label()),
        srcType1Corrections = cms.VInputTag(
            cms.InputTag(moduleMETcorrShiftUpName)
        )
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

