import FWCore.ParameterSet.Config as cms

process = cms.Process("electrons")
process.load("RecoEgamma.EgammaElectronProducers.siStripSeeds_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cff")
process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff")
process.load("RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(1),
    debugFlag = cms.untracked.bool(True),
    fileNames = 
cms.untracked.vstring('/store/relval/2008/7/13/RelVal-RelValSingleElectronPt35-1215820444-IDEAL_V5-2nd/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre8-RelVal-1215820444-IDEAL_V5-2nd-IDEAL_V5-unmerged/0000/14AD4148-F850-DD11-A295-000423D98DD4.root','/store/relval/2008/7/13/RelVal-RelValSingleElectronPt35-1215820444-IDEAL_V5-2nd/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre8-RelVal-1215820444-IDEAL_V5-2nd-IDEAL_V5-unmerged/0000/F082A8E5-F650-DD11-B6DA-0019DB2F3F9B.root')

)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *',
        'keep *_*_*_electrons'),
    fileName = cms.untracked.string('electrons.root')
)

process.p = cms.Path(process.siStripSeeds)
process.outpath = cms.EndPath(process.out)
process.GlobalTag.globaltag = 'IDEAL_V5::All'