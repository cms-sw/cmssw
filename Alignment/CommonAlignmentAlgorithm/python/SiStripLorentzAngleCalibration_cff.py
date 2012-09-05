import FWCore.ParameterSet.Config as cms

SiStripLorentzAngleCalibration_peak = cms.PSet(
    # Name that is bound to the LorentzAngleCalibration, defined by 
    # the DEFINE_EDM_PLUGIN macro in LorentzAngleCalibration.cc:
    calibrationName = cms.string('LorentzAngleCalibration'),

    # Configuration parameters of LorentzAngleCalibration
    readoutMode = cms.string('peak'), # peak or deconvolution
    treeFile = cms.string('treeFile.root'), # to store Lorentz angle values (in-&output)
    mergeTreeFiles = cms.vstring(), # files with input/output from various parallel jobs
    saveToDB = cms.bool(False), # save result in poolDBOutputService
    # If we save to DB, the recordNameDBwrite must match what is specified
    # as 'record' in the PoolDBOutputService:
    recordNameDBwrite = cms.string('SiStripLorentzAngle_peak'),
    )

SiStripLorentzAngleCalibration_deco = SiStripLorentzAngleCalibration_peak.clone(
    readoutMode    = 'deconvolution',
    recordNameDBwrite = cms.string('SiStripLorentzAngle_deco'),
    )
