import FWCore.ParameterSet.Config as cms
import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam
from L1Trigger.L1THGCal.l1tHGCalConcentratorProducer_cfi import threshold_conc_proc, best_conc_proc, supertc_conc_proc, coarsetc_onebitfraction_proc, coarsetc_equalshare_proc, bestchoice_ndata_decentralized, custom_conc_proc, autoEncoder_conc_proc

def custom_triggercellselect_supertriggercell(process,
                                              stcSize=supertc_conc_proc.stcSize,
                                              type_energy_division=supertc_conc_proc.type_energy_division,
                                              fixedDataSizePerHGCROC=supertc_conc_proc.fixedDataSizePerHGCROC
                                              ):
    parameters = supertc_conc_proc.clone(stcSize = stcSize,
                                         type_energy_division = type_energy_division,
                                         fixedDataSizePerHGCROC = fixedDataSizePerHGCROC  
                                         )
    process.l1tHGCalConcentratorProducer.ProcessorParameters = parameters
    return process


def custom_triggercellselect_threshold(process,
                                       threshold_silicon=threshold_conc_proc.threshold_silicon,  # in mipT
                                       threshold_scintillator=threshold_conc_proc.threshold_scintillator,  # in mipT
                                       coarsenTriggerCells=threshold_conc_proc.coarsenTriggerCells  
                                       ):
    parameters = threshold_conc_proc.clone(
            threshold_silicon = threshold_silicon,
            threshold_scintillator = threshold_scintillator,
            coarsenTriggerCells = coarsenTriggerCells  
            )
    process.l1tHGCalConcentratorProducer.ProcessorParameters = parameters
    return process


def custom_triggercellselect_bestchoice(process,
                                        triggercells=best_conc_proc.NData
                                        ):
    parameters = best_conc_proc.clone(NData = triggercells)
    process.l1tHGCalConcentratorProducer.ProcessorParameters = parameters
    return process


def custom_triggercellselect_bestchoice_decentralized(process):
    return custom_triggercellselect_bestchoice(process, triggercells=bestchoice_ndata_decentralized)


def custom_coarsetc_onebitfraction(process,
                                   stcSize=coarsetc_onebitfraction_proc.stcSize,
                                   fixedDataSizePerHGCROC=coarsetc_onebitfraction_proc.fixedDataSizePerHGCROC,
                                   oneBitFractionThreshold = coarsetc_onebitfraction_proc.oneBitFractionThreshold,
                                   oneBitFractionLowValue = coarsetc_onebitfraction_proc.oneBitFractionLowValue,
                                   oneBitFractionHighValue = coarsetc_onebitfraction_proc.oneBitFractionHighValue,
                                   ):
    parameters = coarsetc_onebitfraction_proc.clone(
        stcSize = stcSize,    
        fixedDataSizePerHGCROC = fixedDataSizePerHGCROC, 
        oneBitFractionThreshold = oneBitFractionThreshold,
        oneBitFractionLowValue = oneBitFractionLowValue,
        oneBitFractionHighValue = oneBitFractionHighValue,
    )
    process.l1tHGCalConcentratorProducer.ProcessorParameters = parameters
    return process




def custom_coarsetc_equalshare(process,
                               stcSize=coarsetc_equalshare_proc.stcSize,
                               fixedDataSizePerHGCROC=coarsetc_equalshare_proc.fixedDataSizePerHGCROC,
                               ):
    parameters = coarsetc_equalshare_proc.clone(
        stcSize = stcSize,    
        fixedDataSizePerHGCROC = fixedDataSizePerHGCROC, 
    )
    process.l1tHGCalConcentratorProducer.ProcessorParameters = parameters
    return process
    
def custom_triggercellselect_mixedBestChoiceSuperTriggerCell(process,
                                              stcSize=custom_conc_proc.stcSize,
                                              type_energy_division=custom_conc_proc.type_energy_division,
                                              fixedDataSizePerHGCROC=custom_conc_proc.fixedDataSizePerHGCROC,
                                              triggercells=custom_conc_proc.NData
                                              ):
    parameters = custom_conc_proc.clone(stcSize = stcSize,
                                        type_energy_division = type_energy_division,
                                        fixedDataSizePerHGCROC = fixedDataSizePerHGCROC,
                                        NData=triggercells,
                                        Method = cms.vstring('bestChoiceSelect','superTriggerCellSelect','superTriggerCellSelect'),        
    )
    process.l1tHGCalConcentratorProducer.ProcessorParameters = parameters
    return process

def custom_triggercellselect_mixedBestChoiceSuperTriggerCell_decentralized(process):
    return custom_triggercellselect_mixedBestChoiceSuperTriggerCell(process, triggercells=bestchoice_ndata_decentralized)



def custom_triggercellselect_autoencoder(process,
        cellRemap = autoEncoder_conc_proc.cellRemap,
        nBitsPerInput = autoEncoder_conc_proc.nBitsPerInput,
        maxBitsPerOutput = autoEncoder_conc_proc.maxBitsPerOutput,
        bitsPerLink = autoEncoder_conc_proc.bitsPerLink,
        modelFiles = autoEncoder_conc_proc.modelFiles,
        linkToGraphMap = autoEncoder_conc_proc.linkToGraphMap,
        zeroSuppresionThreshold = autoEncoder_conc_proc.zeroSuppresionThreshold,
        saveEncodedValues = autoEncoder_conc_proc.saveEncodedValues,
        preserveModuleSum = autoEncoder_conc_proc.preserveModuleSum,
        scintillatorMethod = 'thresholdSelect',
        ):
    parameters = autoEncoder_conc_proc.clone(
            cellRemap = cellRemap,
            nBitsPerInput = nBitsPerInput,
            maxBitsPerOutput = maxBitsPerOutput,
            bitsPerLink = bitsPerLink,
            modelFiles = modelFiles,
            linkToGraphMap = linkToGraphMap,
            zeroSuppresionThreshold = zeroSuppresionThreshold,
            saveEncodedValues = saveEncodedValues,
            preserveModuleSum = preserveModuleSum,
            Method = cms.vstring(['autoEncoder','autoEncoder', scintillatorMethod]),
            )
    process.l1tHGCalConcentratorProducer.ProcessorParameters = parameters
    return process
