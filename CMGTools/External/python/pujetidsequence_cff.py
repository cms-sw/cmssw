import FWCore.ParameterSet.Config as cms

from CMGTools.External.pujetidproducer_cfi import pileupJetIdProducer, stdalgos_4x, stdalgos_5x, stdalgos, cutbased, chsalgos_4x, chsalgos_5x, chsalgos
# from RecoJets.JetProducers.PileupJetID_cfi import pileupJetIdProducer, _stdalgos_4x, _stdalgos_5x, _stdalgos, cutbased, _chsalgos_4x, _chsalgos_5x, _chsalgos

#
# Standard pfJets
#
puJetId = pileupJetIdProducer.clone(
    produceJetIds = cms.bool(True),
    jetids = cms.InputTag(""),
    runMvas = cms.bool(False),
    jets = cms.InputTag("selectedPatJets"),
    vertexes = cms.InputTag("offlinePrimaryVertices"),
    algos = cms.VPSet(cutbased)
    )

# 
puJetMva = pileupJetIdProducer.clone(
    produceJetIds = cms.bool(False),
    jetids = cms.InputTag("puJetId"),
    runMvas = cms.bool(True),
    jets = cms.InputTag("selectedPatJets"),
    vertexes = cms.InputTag("offlinePrimaryVertices"),
    algos = stdalgos
    )

# 
puJetIdSqeuence = cms.Sequence(puJetId*puJetMva)

#
# Charged Hadron Subtraction
#
puJetIdChs = pileupJetIdProducer.clone(
    produceJetIds = cms.bool(True),
    jetids = cms.InputTag(""),
    runMvas = cms.bool(False),
    jets = cms.InputTag("selectedPatJets"),
    vertexes = cms.InputTag("offlinePrimaryVertices"),
    algos = cms.VPSet(cutbased)
    )

# 
puJetMvaChs = pileupJetIdProducer.clone(
    produceJetIds = cms.bool(False),
    jetids = cms.InputTag("puJetIdChs"),
    runMvas = cms.bool(True),
    jets = cms.InputTag("selectedPatJets"),
    vertexes = cms.InputTag("offlinePrimaryVertices"),
    algos = chsalgos
    )

# 
puJetIdSqeuenceChs = cms.Sequence(puJetIdChs*puJetMvaChs)

## utility function to build jet is sequence
def loadPujetId(process,collection,mvaOnly=False,isChs=False,release="44X"):

    ## FIXME 52X and CHS options need to be properly filled
    if release.startswith("4"):
        if isChs:
            algos = stdalgos_4x
        else:
            algos = stdalgos_4x
    elif release.startswith("5"):
        if isChs:
            algos = chsalgos_5x
        else:
            algos = stdalgos_5x

    if not mvaOnly:
        setattr(process,
                "%s%s" % ("puJetId",collection), 
                pileupJetIdProducer.clone(
            produceJetIds = cms.bool(True),
            jetids = cms.InputTag(""),
            runMvas = cms.bool(False),
            jets = cms.InputTag(collection),
            vertexes = cms.InputTag("offlinePrimaryVertices"),
            algos = cms.VPSet(algos[0])
            )
                )
    setattr(process,
            "%s%s" % ("puJetMva",collection), 
            pileupJetIdProducer.clone(
        produceJetIds = cms.bool(False),
        jetids = cms.InputTag("%s%s" % ("puJetId",collection)),
        runMvas = cms.bool(True),
        jets = cms.InputTag(collection),
        vertexes = cms.InputTag("offlinePrimaryVertices"),
        algos = cms.VPSet(*algos)
        )
            )
    
    if mvaOnly:
        setattr(process, "%s%s" % ("puJetIdSequence",collection),
                getattr(process,"%s%s" % ("puJetMva",collection)) )
    else:
        setattr(process, "%s%s" % ("puJetIdSequence",collection),
                cms.Sequence(getattr(process,"%s%s" % ("puJetId",collection)) * getattr(process,"%s%s" % ("puJetMva",collection)) )
                )

    seqeuence = getattr(process, "%s%s" % ("puJetIdSequence",collection))
    inputsTag = cms.InputTag("%s%s" % ("puJetId",collection))
    mvaTags  = {}
    idTags   = {}
    for a in algos:
        mvaTags[a.label.value()] = cms.InputTag("%s%s" % ("puJetMva",collection), "%sDiscriminant" % a.label.value())
        idTags[a.label.value()] = cms.InputTag("%s%s" % ("puJetMva",collection), "%sId"           % a.label.value())
    outputCommands = [ "keep *_%s%s_*_*" % ("puJetId",collection),  "keep *_%s%s_*_*" % ("puJetMva",collection) ]
    
    return seqeuence,inputsTag,mvaTags,idTags,outputCommands
