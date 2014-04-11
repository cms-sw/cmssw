import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelFakeGenErrorDBSourceReader")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("CalibTracker.SiPixelESProducers.SiPixelFakeGenErrorDBObjectESSource_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_2_1_9/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/2A00EECC-A185-DD11-93A9-000423D9517C.root'
    )
)

process.reader = cms.EDAnalyzer('SiPixelFakeGenErrorDBSourceReader'
)


process.p = cms.Path(process.reader)
