import FWCore.ParameterSet.Config as cms

def customiseGenPUProtons(process):

    process.mix.mixObjects.mixHepMC.makeCrossingFrame = True

    process.genPUProtons = cms.EDProducer("GenPUProtonProducer",
	mix = cms.string("mix"),
	bunchCrossingList = cms.vint32(),
	minPz = cms.double( 4800. )
    )
    process.genPUProtonsBx0 = process.genPUProtons.clone()
    process.genPUProtonsBx0.bunchCrossingList = [0]
    process.genPUProtonsBxm1 = process.genPUProtons.clone()
    process.genPUProtonsBxm1.bunchCrossingList = [-1]
    process.genPUProtonsBxp1 = process.genPUProtons.clone()
    process.genPUProtonsBxp1.bunchCrossingList = [1]
    process.genPUProtons_seq = cms.Sequence(process.genPUProtonsBx0+process.genPUProtonsBxm1+process.genPUProtonsBxp1)

    # Path and EndPath definitions
    process.digitisation_step.replace( process.pdigi, process.pdigi*process.genPUProtons_seq)
     
    return process
