import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
"rfio:/castor/cern.ch/cms/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/065/1C27DDD3-84DD-DE11-AFA2-001617C3B66C.root"
),
   secondaryFileNames = cms.untracked.vstring(
)


#        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/578/085EFED4-E5AB-DD11-9ACA-001617C3B6FE.root')
)




process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/GoodVertex_cfg.py,v $'),
    annotation = cms.untracked.string('At least two general track or one pixel track or one pixelLess track')
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))


process.primaryVertexFilter = cms.EDFilter("GoodVertexFilter",
                                                      vertexCollection = cms.InputTag('offlinePrimaryVertices'),
                                                      minimumNumberOfTracks = cms.uint32(3) ,
 						      maxAbsZ = cms.double(15),	
 						      maxd0 = cms.double(2)	
                                                      )

process.primaryVertexPath = cms.Path(process.primaryVertexFilter)


process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('primaryVertexPath')),
                               dataset = cms.untracked.PSet(
			                 dataTier = cms.untracked.string('RAW-RECO'),
                                         filterName = cms.untracked.string('GoodPrimaryVertex')),
                               fileName = cms.untracked.string('withvertex.root')
                               )

process.this_is_the_end = cms.EndPath(process.out)
