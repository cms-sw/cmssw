import FWCore.ParameterSet.Config as cms

# Customization for running on HCAL local run data
#   - Call from cmsDriver.py with: `--customise DPGAnalysis/HcalNanoAOD/customise_hcalLocal_cff.customiseHcalLocal`
def customiseHcalLocal(process):
    input_files = process.source.fileNames
    max_events = process.maxEvents.input
    process.source = cms.Source("HcalTBSource",
        fileNames = input_files,
        maxEvents = max_events,
        firstLuminosityBlockForEachRun = cms.untracked.VLuminosityBlockID([]),
    )
    process.hcalDigis.InputLabel = cms.InputTag('source')
    #process.hcalDigis.saveQIE10DataNSamples = cms.untracked.vint32(10) 
    #process.hcalDigis.saveQIE10DataTags = cms.untracked.vstring("ZDC")

    if hasattr(process, "hcalDigiSortedTableTask"):
        process.hcalDigiSortedTable.nTS_HB = cms.untracked.uint32(8)
        process.hcalDigiSortedTable.nTS_HE = cms.untracked.uint32(8)
        process.hcalDigiSortedTable.nTS_HF = cms.untracked.uint32(6)
        process.hcalDigiSortedTable.nTS_HO = cms.untracked.uint32(10)

    process.load("DPGAnalysis.HcalNanoAOD.hcalUMNioTable_cff")
    if hasattr(process, "hcalNanoTask"):
        process.hcalNanoTask.add(process.uMNioTable)
    if hasattr(process, "hcalNanoDigiTask"):
        process.hcalNanoDigiTask.add(process.uMNioTable)

    return process
