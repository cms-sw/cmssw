import FWCore.ParameterSet.Config as cms

SiStripBackplaneCalibration = cms.PSet(
    # Name that is bound to the SiStripBackplaneCalibration, defined by 
    # the DEFINE_EDM_PLUGIN macro in SiStripBackplaneCalibration.cc:
    calibrationName = cms.string('SiStripBackplaneCalibration'),

    # Configuration parameters of BackplaneCalibration
    readoutMode = cms.string('deconvolution'), # 'peak' is reference, so do not change this
    treeFile = cms.string('treeFile.root'), # to store backplane correction values (in-&output)
    mergeTreeFiles = cms.vstring(), # files with input/output from various parallel jobs
    saveToDB = cms.bool(False), # save result in poolDBOutputService
    # If we save to DB, the recordNameDBwrite must match what is specified
    # as 'record' in the PoolDBOutputService:
    recordNameDBwrite = cms.string('SiStripBackPlaneCorrectionRcd'),

    # Configuration of the granularity for the backplane correction determination
    BackplaneModuleGroups = cms.PSet(),
    )
