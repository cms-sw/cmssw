import FWCore.ParameterSet.Config as cms

EcalEleCalibMon = DQMStep1Module('DQMSourceEleCalib',
    # product to monitor
    AlCaStreamEBTag = cms.InputTag("alCaIsolatedElectrons","alcaBarrelHits"),
#    SaveToFile = cms.untracked.bool(True),
    AlCaStreamEETag = cms.InputTag("alCaIsolatedElectrons","alcaEndcapHits"),
    electronCollection = cms.InputTag("electronFilter"),
    # DQM folder to write to
    FolderName = cms.untracked.string('AlCaReco/EcalSingleEle')
)

