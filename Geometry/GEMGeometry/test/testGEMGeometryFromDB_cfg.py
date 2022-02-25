import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process("Demo",eras.run3_GEM)



process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.XMLFromDBSource.label = cms.string('Extended')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2021', '')

from Configuration.AlCa.autoCond import autoCond
process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource") 

process.source = cms.Source("EmptySource")

process.test = cms.EDAnalyzer("GEMGeometryAnalyzer")

process.p = cms.Path(process.test)

### TO ACTIVATE LogTrace NEED TO COMPILE IT WITH:
### -----------------------------------------------------------
### --> scram b -j8 USER_CXXFLAGS="-DEDM_ML_DEBUG"
### Make sure that you first cleaned your CMSSW version:
### --> scram b clean
### before issuing the scram command above
###############################################################
process.load("FWCore.MessageLogger.MessageLogger_cfi")
# 
# 
process.MessageLogger.debugModules = cms.untracked.vstring("*")
process.MessageLogger.cerr.enable = False
process.MessageLogger.files.junk = dict()
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("DEBUG"),
    default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
    FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # GEMGeometryBuilderFromDDD   = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # GEMNumberingScheme            = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
)
