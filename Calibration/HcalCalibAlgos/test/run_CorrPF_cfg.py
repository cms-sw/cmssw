import FWCore.ParameterSet.Config as cms

process = cms.Process("HcalPFCorrsCulculation")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Calibration.HcalCalibAlgos.HcalCorrPFCalculation_cfi")
process.load("Configuration.StandardSequences.GeometryECALHCAL_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(200)

process.PoolSource.fileNames = ['rfio:/castor/cern.ch/user/a/abdullin/pi50_fullproduction_312/pi50_1.root']

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.hcalRecoAnalyzer = cms.EDFilter("HcalCorrPFCalculation",
    outputFile = cms.untracked.string('HcalCorrPF.root'),
    eventype = cms.untracked.string('single'),
    mc = cms.untracked.string('yes'),
    sign = cms.untracked.string('*'),
    hcalselector = cms.untracked.string('all'),
#    RespcorrAdd = cms.untracked.bool(True),
#    PFcorrAdd = cms.untracked.bool(True),
    ConeRadiusCm = cms.untracked.double(30.),
    ecalselector = cms.untracked.string('yes')
)

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_31X_V5::All')
#process.GlobalTag.globaltag = cms.string('STARTUP31X_V1::All')
process.prefer("GlobalTag")

process.es_ascii2 = cms.ESSource("HcalTextCalibrations",
                        appendToDataLabel = cms.string('recalibrate'),
                         input = cms.VPSet(
                            cms.PSet(
                                object = cms.string('RespCorrs'),
                                file =
                                cms.FileInPath('Calibration/HcalCalibAlgos/data/calibConst_IsoTrk_testCone_26.3cm.txt')
                            ),
                            cms.PSet(
                                object = cms.string('PFCorrs'),
                                file =
                                cms.FileInPath('Calibration/HcalCalibAlgos/data/HcalPFCorrs_v1.03_mc.txt')
                            )
                         )
)

process.p = cms.Path(process.hcalRecoAnalyzer)


