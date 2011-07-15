import FWCore.ParameterSet.Config as cms

def customise(process):
    REDIGIInputEventSkimming= cms.PSet(
        inputCommands=cms.untracked.vstring('drop *')
        )

    GeneratorInterfaceRAWNoGenParticles = process.GeneratorInterfaceRAW.outputCommands
    for item in GeneratorInterfaceRAWNoGenParticles:
      if 'genParticles' in item:
        GeneratorInterfaceRAWNoGenParticles.remove(item) 

    REDIGIInputEventSkimming.inputCommands.extend(process.SimG4CoreRAW.outputCommands) 
    REDIGIInputEventSkimming.inputCommands.extend(GeneratorInterfaceRAWNoGenParticles) 
    REDIGIInputEventSkimming.inputCommands.extend(process.IOMCRAW.outputCommands) 

    process.source.inputCommands = REDIGIInputEventSkimming.inputCommands
    process.source.dropDescendantsOfDroppedBranches=cms.untracked.bool(False)
    
    process.RandomNumberGeneratorService.restoreStateLabel = cms.untracked.string('randomEngineStateProducer')

    # Remove the old RNGState product on output
    RNGStateCleaning= cms.PSet(
        outputCommands=cms.untracked.vstring('drop RandomEngineStates_*_*_*',
                                             'keep RandomEngineStates_*_*_'+process.name_())
        )

    for item in process.outputModules_().values():
        item.outputCommands.extend(RNGStateCleaning.outputCommands)

    # REDO the GenJets etc. in case labels have been changed
    process.load('Configuration/StandardSequences/Generator_cff')
    process.fixGenInfo = cms.Path(process.GeneInfo * process.genJetMET)
    process.schedule.append(process.fixGenInfo)
    
    return(process)
