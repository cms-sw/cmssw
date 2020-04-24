import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")


process.load('Configuration.Geometry.GeometryExtended2023D4Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D4_cff')


#process.load('Configuration.Geometry.GeometryExtended2023D1_cff')
#process.load('Configuration.Geometry.GeometryExtended2023D1Reco_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    categories = cms.untracked.vstring('GEMGeometryBuilderFromDDD'),
                                    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            ),
        GEMGeometryBuilderFromDDD = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            )
        )
                                    )

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
# process.MessageLogger.categories.append("GEMGeometryBuilderFromDDD")
# process.MessageLogger.categories.append("GEMNumberingScheme")
process.MessageLogger.debugModules = cms.untracked.vstring("*")
process.MessageLogger.destinations = cms.untracked.vstring("cout","junk")
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string("DEBUG"),
    default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
    FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # GEMGeometryBuilderFromDDD   = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # GEMNumberingScheme            = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
)                                                                                                                                                                 
