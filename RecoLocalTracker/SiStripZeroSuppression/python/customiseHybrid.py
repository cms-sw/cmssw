import FWCore.ParameterSet.Config as cms

## Adjust the local reco sequence for running on hybrid zero-suppressed data
def runOnHybridZS(process):
    zsInputs = process.siStripZeroSuppression.RawDigiProducersList
    clusInputs = process.siStripClusters.DigiProducersList
    unpackedZS = cms.InputTag("siStripDigis", "ZeroSuppressed")
    zsInputs.append(unpackedZS)
    clusInputs.remove(unpackedZS)
    clusInputs.append(cms.InputTag("siStripZeroSuppression","ZeroSuppressed"))
    # for on-demand clusterizer
    from FWCore.ParameterSet.MassReplace import massReplaceParameter
    massReplaceParameter(process, "HybridZeroSuppressed", cms.bool(False), cms.bool(True))
    return process

## Add the ZS algorithm (in hybrid emulation mode) before repacking, to produce emulated hybrid samples with
##   cmsDriver --step RAW2DIGI,REPACK:DigiToZS10RawRepack --customiseRecoLocalTracker/SiStripZeroSuppression/customiseHybrid.addHybridEmulationBeforeRepackZS10 ...
def addHybridEmulationBeforeRepackZS10(process):
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
    zs.RawDigiProducersList = cms.VInputTag(cms.InputTag("siStripDigis", "VirginRaw"))

    process.DigiToZS10RawRepack.insert(0, zs) ## insert before repacking
    process.rawDataRepacker.RawCollectionList = cms.VInputTag(
            cms.InputTag("SiStripDigiToZS10Raw"),
            cms.InputTag('source'),
            cms.InputTag('rawDataCollector'))

    return process
