import FWCore.ParameterSet.Config as cms

def customiseForRunningOnMC(process, pathName) :

    if hasattr(process,"muNtupleProducer") :
        print("[customiseForRunningOnMC]: updating ntuple input tags")
        
        process.muNtupleTwinMuxInFiller.dtTpTag = "none"
        process.muNtupleTwinMuxOutFiller.dtTpTag = "none"
        process.muNtupleTwinMuxInThFiller.dtTpTag = "none"

    return process

def customiseForMuonWorkflow(process) :
    print("[customiseForMuonWorkflow]: adding VarParsing")

    import FWCore.ParameterSet.VarParsing as VarParsing
    options = VarParsing.VarParsing()

    options.register('globalTag',
                     '130X_dataRun3_Express_v1', #default value
                      VarParsing.VarParsing.multiplicity.singleton,
                      VarParsing.VarParsing.varType.string,
                      "Global Tag")

    options.register('nEvents',
                      100, #default value
                      VarParsing.VarParsing.multiplicity.singleton,
                      VarParsing.VarParsing.varType.int,
                      "Maximum number of processed events")

    options.parseArguments()

    process.GlobalTag.globaltag = cms.string(options.globalTag)
    process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.nEvents))

    return process