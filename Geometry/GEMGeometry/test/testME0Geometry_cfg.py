import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
process.load("Configuration.Geometry.GeometryExtended2023D21_cff")
process.load("Configuration.Geometry.GeometryExtended2023D21Reco_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

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
# 
process.MessageLogger.debugModules = cms.untracked.vstring("*")
process.MessageLogger.cerr.enable = False
process.MessageLogger.files.junk = dict()
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("DEBUG"),
    default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
    FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # ME0GeometryESModule           = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # ME0GeometryBuilder   = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # ME0NumberingScheme          = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.test = cms.EDAnalyzer("ME0GeometryAnalyzer")

process.p = cms.Path(process.test)

