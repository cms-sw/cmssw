import FWCore.ParameterSet.Config as cms

process = cms.Process("HcalPFCorrsCulculation")

process.load("Configuration.StandardSequences.VtxSmearedBetafuncEarlyCollision_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(2000))
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(5000)

process.load("Calibration.HcalCalibAlgos.pfCorrs_cfi")
process.hcalPFcorrs.calibrationConeSize = cms.double(35.)

#process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'MC_31X_V5::All'
process.GlobalTag.globaltag = 'DESIGN_3X_V24::All'
process.prefer("GlobalTag")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

'/store/user/andrey/SinglePions_50GeV_Rel352_v3/SinglePions_50GeV_Rel352_v3/791ecbb28bc75b5af691fc4b56276304/SinglePionMinus_50_1.root',

#'rfio:/castor/cern.ch/user/a/abdullin/pi50_fullproduction_312/pi50_3.root',
#'/store/user/andrey/SinglePion_50GeV_314/SinglePion_50GeV_314/0d8aafd1bbf7b6158b7a4e52f0fb00b6/SinglePion_50GeV_314_9.root',

    )
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('HcalCorrPF.root')
)


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

process.p = cms.Path(process.hcalPFcorrs)


