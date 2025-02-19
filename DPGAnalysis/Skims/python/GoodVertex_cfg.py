import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")
process.source = cms.Source("PoolSource",
			
                            fileNames = cms.untracked.vstring(
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/30F7C9C1-39E2-DE11-99EF-003048D2C108.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/307D51EE-54E2-DE11-8A62-001617E30D4A.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/3079CB3D-3FE2-DE11-A578-0019DB29C5FC.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/2CAE01BA-5BE2-DE11-B4E9-000423D98634.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/2A773003-3BE2-DE11-AFA7-003048D3750A.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/2A1BA28D-45E2-DE11-8150-003048D2BE08.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/28DAE662-37E2-DE11-93BC-001617DBD224.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/2836CEBD-39E2-DE11-8472-0016177CA7A0.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/26AB7E83-55E2-DE11-8049-001D09F2424A.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/24D24155-37E2-DE11-9B23-0030486780B8.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/223BC3ED-3DE2-DE11-AEA6-001D09F24FEC.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/2085F48F-36E2-DE11-B843-000423D9863C.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/202B3CF2-54E2-DE11-B542-000423D996C8.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/1ECC7C87-44E2-DE11-819F-000423D33970.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/1EBC4591-36E2-DE11-A654-000423D6BA18.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/1E68E9DE-34E2-DE11-B26A-000423D99AAA.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/1C681C0A-42E2-DE11-99EE-000423D99BF2.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/1C3CABBE-48E2-DE11-8F9D-001D09F252DA.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/1A80C207-42E2-DE11-9380-001D09F29538.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/1A1AC218-47E2-DE11-B7BC-003048D3756A.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/18C5820E-47E2-DE11-9E37-001D09F251FE.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/168D79DB-49E2-DE11-9E66-001D09F2906A.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/1284873F-3FE2-DE11-9E93-000423D99F1E.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/10447316-3CE2-DE11-A868-000423D991F0.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/0E6D4C2D-43E2-DE11-B115-0030487A1FEC.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/0A85BE90-45E2-DE11-8EE1-001D09F2514F.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/0A38D516-5AE2-DE11-BAB0-001D09F28755.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/084F0FCB-47E2-DE11-BFFD-0016177CA778.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/08314929-41E2-DE11-BDC1-000423D8FA38.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/0297DC23-58E2-DE11-AEC2-001D09F232B9.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/021BEBFC-5AE2-DE11-B1FF-000423D991D4.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/596/00D8AF8D-36E2-DE11-BEC4-001D09F24303.root"
),
   secondaryFileNames = cms.untracked.vstring(
)


#        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/578/085EFED4-E5AB-DD11-9ACA-001617C3B6FE.root')
)




process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.5 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DPGAnalysis/Skims/python/GoodVertex_cfg.py,v $'),
    annotation = cms.untracked.string('At least two general track or one pixel track or one pixelLess track')
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))


process.primaryVertexFilter = cms.EDFilter("GoodVertexFilter",
                                                      vertexCollection = cms.InputTag('offlinePrimaryVertices'),
                                                      minimumNDOF = cms.uint32(4) ,
 						      maxAbsZ = cms.double(24),	
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
