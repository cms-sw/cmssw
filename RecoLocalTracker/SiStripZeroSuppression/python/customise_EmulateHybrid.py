import FWCore.ParameterSet.Config as cms

## Add the ZS algorithm (in hybrid emulation mode) before repacking, to produce emulated hybrid samples with
##   cmsDriver --step RAW2DIGI,REPACK:DigiToZS10RawRepack --customiseRecoLocalTracker/SiStripZeroSuppression/customise_EmulateHybrid.py ...
def customise(process):
    process.load("RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi")
    zs = process.siStripZeroSuppression
    zs.produceRawDigis = False
    zs.produceHybridFormat = True
    zs.Algorithms.APVInspectMode = "HybridEmulation"
    zs.Algorithms.APVRestoreMode = ""
    zs.Algorithms.CommonModeNoiseSubtractionMode = 'Median'
    zs.Algorithms.MeanCM = 512
    zs.Algorithms.DeltaCMThreshold = 20
    zs.Algorithms.Use10bitsTruncation = True
    zs.RawDigiProducersList = cms.VInputTag(cms.InputTag("siStripDigis","VirginRaw"))

    process.DigiToZS10RawRepack.insert(0, zs) ## insert before repacking
    return process
