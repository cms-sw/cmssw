import FWCore.ParameterSet.Config as cms
import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam

# Digitization parameters
adcSaturationBH_MIP = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcSaturation_fC
adcNbitsBH = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcNbits


threshold_conc_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                               Method = cms.string('thresholdSelect'),
                               threshold_silicon = cms.double(2.), # MipT
                               threshold_scintillator = cms.double(2.), # MipT
                               )


best_conc_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                          Method = cms.string('bestChoiceSelect'),
                          # Column is Nlinks, Row is NWafers
                          # Requested size = 8(links)x8(wafers)
                          # Values taken from https://indico.cern.ch/event/747610/contributions/3155360/, slide 13
                          # For motherboards larger than 3, it is split in two
                          NData = cms.vuint32(
                              13, 42, 75,  0,   0,   0, 0, 0,
                              13, 40, 74, 80, 114, 148, 0, 0,
                              12, 39, 72, 82, 116, 146, 0, 0,
                              12, 26, 53, 80, 114, 148, 0, 0,
                              12, 25, 52, 79, 112, 146, 0, 0,
                               0, 24, 51, 78, 111, 144, 0, 0,
                               0,  0,  0,  0,   0,   0, 0, 0,
                               0,  0,  0,  0,   0,   0, 0, 0,
                              ),
                          )


supertc_conc_proc = cms.PSet(ProcessorName  = cms.string('HGCalConcentratorProcessorSelection'),
                             Method = cms.string('superTriggerCellSelect'),
                             stcSize = cms.vuint32(4,4,4)
                             )


from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9
# V9 samples have a different defintiion of the dEdx calibrations. To account for it
# we reascale the thresholds of the FE selection
# (see https://indico.cern.ch/event/806845/contributions/3359859/attachments/1815187/2966402/19-03-20_EGPerf_HGCBE.pdf
# for more details)
phase2_hgcalV9.toModify(threshold_conc_proc,
                        threshold_silicon=1.35,  # MipT
                        threshold_scintillator=1.35,  # MipT
                        )


hgcalConcentratorProducer = cms.EDProducer(
    "HGCalConcentratorProducer",
    InputTriggerCells = cms.InputTag('hgcalVFEProducer:HGCalVFEProcessorSums'),
    InputTriggerSums = cms.InputTag('hgcalVFEProducer:HGCalVFEProcessorSums'),
    ProcessorParameters = threshold_conc_proc.clone()
    )
