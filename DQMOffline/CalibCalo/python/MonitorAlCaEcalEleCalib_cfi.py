import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
EcalEleCalibMon = DQMEDAnalyzer('DQMSourceEleCalib',
    # product to monitor
    AlCaStreamEBTag = cms.InputTag("alCaIsolatedElectrons","alcaBarrelHits"),
#    SaveToFile = cms.untracked.bool(True),
    AlCaStreamEETag = cms.InputTag("alCaIsolatedElectrons","alcaEndcapHits"),
    electronCollection = cms.InputTag("electronFilter"),
    # DQM folder to write to
    FolderName = cms.untracked.string('AlCaReco/EcalSingleEle')
)

