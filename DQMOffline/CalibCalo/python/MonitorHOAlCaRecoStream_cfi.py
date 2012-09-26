import FWCore.ParameterSet.Config as cms

MonitorHOAlCaRecoStream = cms.EDAnalyzer("DQMHOAlCaRecoStream",
    RootFileName = cms.untracked.string('test.root'),
    folderName = cms.untracked.string('AlCaReco/HcalHO'),
    sigmaval =  cms.untracked.double(0.2),
    lowradposinmuch =  cms.untracked.double(400.0),
    highradposinmuch =  cms.untracked.double(480),
                                   
    lowedge =  cms.untracked.double(-2.0),
    highedge =  cms.untracked.double(10.0),
    nbins =  cms.untracked.int32(100),
    saveToFile = cms.untracked.bool(False),
    hoCalibVariableCollectionTag = cms.InputTag('hoCalibProducer', 'HOCalibVariableCollection')
)
