import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

def ecal_phisym_workflow(process, 
                         produce_by_run : bool=False, 
                         save_edm : bool=False,
                         save_edmnano : bool=False,
                         save_flatnano : bool=True):
    """
    Customize the process to include the entire EcalPhiSym workflow:
    - ECAL local reco
    - PhiSymRecHit producer
    - EDM output (standard, EDMNANO, FlatNANO)
    """

    reco = ecal_phisym_reco_sequence(process, produce_by_run=produce_by_run)
    tables = ecal_phisym_flattables(process, produce_by_run=produce_by_run)
    outputs = ecal_phisym_output(process, 
                                 save_edm=save_edm, 
                                 save_edmnano=save_edmnano, 
                                 save_flatnano=save_flatnano)

    process.path = cms.Path(reco*tables)
    process.output_step = cms.EndPath()
    for out in outputs:
        process.output_step += out
    process.schedule = cms.Schedule(process.path, process.output_step)

def ecal_phisym_reco_sequence(process, produce_by_run : bool=False):
    """
    Customize process to include the EcalPhiSym standard reco sequence
    """

    process.load('RecoLuminosity.LumiProducer.bunchSpacingProducer_cfi')
    process.load('RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cfi')
    process.load('RecoLocalCalo.EcalRecProducers.ecalUncalibRecHit_cfi')
    process.load('RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi')

    #ecalMultiFitUncalibRecHit
    process.ecalMultiFitUncalibRecHit.EBdigiCollection = cms.InputTag("hltEcalPhiSymFilter","phiSymEcalDigisEB")
    process.ecalMultiFitUncalibRecHit.EEdigiCollection = cms.InputTag("hltEcalPhiSymFilter","phiSymEcalDigisEE")

    #ecalRecHit (no ricovery)
    process.ecalRecHit.killDeadChannels = cms.bool( False )
    process.ecalRecHit.recoverEBVFE = cms.bool( False )
    process.ecalRecHit.recoverEEVFE = cms.bool( False )
    process.ecalRecHit.recoverEBFE = cms.bool( False )
    process.ecalRecHit.recoverEEFE = cms.bool( False )
    process.ecalRecHit.recoverEEIsolatedChannels = cms.bool( False )
    process.ecalRecHit.recoverEBIsolatedChannels = cms.bool( False )

    # PHISYM producer
    process.load('Calibration.EcalCalibAlgos.EcalPhiSymRecHitProducers_cfi')

    # SCHEDULE
    reconstruction_step = cms.Sequence( process.bunchSpacingProducer * (process.ecalMultiFitUncalibRecHit + process.ecalRecHit) )
    reconstruction_step *= process.EcalPhiSymRecHitProducerRun if produce_by_run else process.EcalPhiSymRecHitProducerLumi

    return reconstruction_step

def ecal_phisym_flattables(process, produce_by_run : bool=False):
    """
    Add the NanoAOD flat table producers.
    This functions adjust also the output columns.
    Should be called once nMisCalib has been set in the EcalPhiSymRecHitProducer
    """

    process.load('Calibration.EcalCalibAlgos.EcalPhiSymFlatTableProducers_cfi')
    
    nmis = process.EcalPhiSymRecHitProducerRun.nMisCalib.value()
    for imis in range(1, nmis+1):
        # get the naming and indexing right.
        if imis<nmis/2+1:
            var_name = 'sumEt_m'+str(abs(int(imis-(nmis/2)-1)))
            var = Var(f'sumEt({imis})', float, doc='ECAL PhiSym rechits: '+str(imis-(nmis/2)-1)+'*miscalib et', precision=23)
        else:
            var_name = 'sumEt_p'+str(int(imis-(nmis/2)))
            var = Var(f'sumEt({imis})', float, doc='ECAL PhiSym rechits: '+str(imis-(nmis/2))+'*miscalib et', precision=23)
        
        if produce_by_run:
            setattr(process.ecalPhiSymRecHitRunTableEB.variables, var_name, var)
            setattr(process.ecalPhiSymRecHitRunTableEE.variables, var_name, var)
            flattable_sequence = cms.Sequence( process.ecalPhiSymRecHitRunTableEB + 
                                               process.ecalPhiSymRecHitRunTableEE +
                                               process.ecalPhiSymInfoRunTable )
        else:
            setattr(process.ecalPhiSymRecHitLumiTableEB.variables, var_name, var)
            setattr(process.ecalPhiSymRecHitLumiTableEE.variables, var_name, var)
            flattable_sequence = cms.Sequence( process.ecalPhiSymRecHitLumiTableEB + 
                                               process.ecalPhiSymRecHitLumiTableEE + 
                                               process.ecalPhiSymInfoLumiTable
            )

    return flattable_sequence

def ecal_phisym_output(process, 
                       save_edm : bool=False,
                       save_edmnano : bool=False,
                       save_flatnano : bool=True):
    """
    Customize EcalPhiSym output
    """

    outputs = []

    if save_flatnano or save_edmnano:
        NanoAODEcalPhiSymEventContent = cms.PSet(
            outputCommands = cms.untracked.vstring(
                'drop *',
                "keep nanoaod*_*_*_*",     # event data
                "keep nanoaodMergeableCounterTable_*Table_*_*", # accumulated per/run or per/lumi data
                "keep nanoaodUniqueString_nanoMetadata_*_*",   # basic metadata
            )
        )        
                       
    if save_flatnano:
        process.nanoout = cms.OutputModule("NanoAODOutputModule",
                                           fileName = cms.untracked.string('ecal_phisym_nano.root'),
                                           outputCommands = NanoAODEcalPhiSymEventContent.outputCommands,
                                           compressionLevel = cms.untracked.int32(9),
                                           compressionAlgorithm = cms.untracked.string("LZMA"),
                                       )
        outputs.append(process.nanoout)

    if save_edmnano:
        process.nanooutedm = cms.OutputModule("PoolOutputModule",
                                              fileName = cms.untracked.string('ecal_phisym_edmnano.root'),
                                              outputCommands = NanoAODEcalPhiSymEventContent.outputCommands,
                                          )
        outputs.append(process.nanooutedm)

    if save_edm:
        ECALPHISYM_output_commands = cms.untracked.vstring(
            "drop *",
            "keep *_PhiSymProducerRun_*_*")

        process.EcalPhiSymOutput = cms.OutputModule("PoolOutputModule",
                                                    splitLevel = cms.untracked.int32(2),
                                                    compressionLevel = cms.untracked.int32(5),
                                                    compressionAlgorithm = cms.untracked.string('LZMA'),
                                                    outputCommands = ECALPHISYM_output_commands,
                                                    fileName = cms.untracked.string('ecal_phisym_reco.root')
                                                )
        outputs.append(process.EcalPhiSymOutput)

    return outputs

def customise(process):
    """
    Function to customize the process produced by cmsDriver.
    The customisation works for a process that satisfies the following conditions:
    - Run on /AlCaPhiSym/*/RAW data
    - Run the following sequence (-s option of cmsDriver): 
    RECO:bunchSpacingProducer+ecalMultiFitUncalibRecHitTask+ecalCalibratedRecHitTask,ALCA:EcalPhiSymByRun (or EcalPhiSymByLumi)
    """
    
    # Change input collection for the /AlCaPhiSym/*/RAW stream dataformat
    process.ecalMultiFitUncalibRecHit.cpu.EBdigiCollection = cms.InputTag("hltEcalPhiSymFilter", "phiSymEcalDigisEB")
    process.ecalMultiFitUncalibRecHit.cpu.EEdigiCollection = cms.InputTag("hltEcalPhiSymFilter", "phiSymEcalDigisEE")
    process.ecalRecHit.cpu.killDeadChannels = cms.bool( False )
    process.ecalRecHit.cpu.recoverEBVFE = cms.bool( False )
    process.ecalRecHit.cpu.recoverEEVFE = cms.bool( False )
    process.ecalRecHit.cpu.recoverEBFE = cms.bool( False )
    process.ecalRecHit.cpu.recoverEEFE = cms.bool( False )
    process.ecalRecHit.cpu.recoverEEIsolatedChannels = cms.bool( False )
    process.ecalRecHit.cpu.recoverEBIsolatedChannels = cms.bool( False )

    if "ALCARECOStreamEcalPhiSymByRunOutPath" in process.pathNames():
        process.schedule.remove(process.ALCARECOStreamEcalPhiSymByRunOutPath)
    if "ALCARECOStreamEcalPhiSymByLumiOutPath" in process.pathNames():
        process.schedule.remove(process.ALCARECOStreamEcalPhiSymByLumiOutPath)

    return process
