import FWCore.ParameterSet.Config as cms

SiPixelLorentzAngleCalibration = cms.PSet(
    # Name that is bound to the SiPixelLorentzAngleCalibration, defined by 
    # the DEFINE_EDM_PLUGIN macro in SiPixelLorentzAngleCalibration.cc:
    calibrationName = cms.string('SiPixelLorentzAngleCalibration'),

    # Configuration parameters of SiPixelLorentzAngleCalibration
    treeFile = cms.string('treeFile.root'), # to store Lorentz angle values (in-&output)
    mergeTreeFiles = cms.vstring(), # files with input/output from various parallel jobs
    saveToDB = cms.bool(False), # save result in poolDBOutputService
    # If we save to DB, the recordNameDBwrite must match what is specified
    # as 'record' in the PoolDBOutputService:
    recordNameDBwrite = cms.string('SiPixelLorentzAngleRcd'),
    
    # Configuration of the granularity for the Lorentz angle calibration
    LorentzAngleModuleGroups = cms.PSet(),
    )
