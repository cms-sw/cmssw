import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep

process = cms.Process("Demo", eras.run3_GEM, Run3_dd4hep)

process.load('Configuration.Geometry.GeometryDD4hepExtended2021Reco_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
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
