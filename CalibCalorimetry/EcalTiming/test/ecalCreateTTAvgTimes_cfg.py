import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalCreateTTAvgTimes")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

#process.source = cms.Source("PoolSource",
#    # replace 'myfile.root' with the source file you want to use
#    fileNames = cms.untracked.vstring(
#        'file:myfile.root'
#    )
#)

process.source = cms.Source("EmptySource")

process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
process.load("Geometry.EcalCommonData.EcalOnly_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")


process.createAvgs = cms.EDAnalyzer('EcalCreateTTAvgTimes',
    #subtractTowerAvgForOfflineCalibs = cms.untracked.bool(False)

#SIC SPLASH 2009 Data
    #timingCalibFile = cms.untracked.string('splash09_run120020_CalibsEB.calibs.txt')
    #timingCalibFile = cms.untracked.string('splash09_run120020_CalibsEE.calibs.txt')

# FIXED RATIO PRODUCER SPLASH 2009
#timingCalibFile = cms.untracked.string('newFirstSplash09_calibs/timingCalibsEE.calibs.txt')
#timingCalibFile = cms.untracked.string('newFirstSplash09_calibs/timingCalibsEB.calibs.txt')

# SAME AS ABOVE BUT NO FILTERING
#timingCalibFile = cms.untracked.string('newFirstSplash09_calibs_noFilter/timingCalibsEE.calibs.txt')
#timingCalibFile = cms.untracked.string('newFirstSplash09_calibs_noFilter/timingCalibsEB.calibs.txt')

)

process.p = cms.Path(process.createAvgs)
