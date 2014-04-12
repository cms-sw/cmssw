import FWCore.ParameterSet.Config as cms

process = cms.Process("DUMP")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'DESIGN42_V17::All'
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.XMLFromDBSource.label=''
process.GlobalTag.toGet = cms.VPSet(
        cms.PSet(record = cms.string("GeometryFileRcd"),
                 tag = cms.string("XMLFILE_Geometry_428SLHCYV0_Phase1_R30F12_HCal_Ideal_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_42X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string('IdealGeometryRecord'),
                 tag = cms.string('TKRECO_Geometry_428SLHCYV0'),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_42X_GEOMETRY")),
        cms.PSet(record = cms.string('PGeometricDetExtraRcd'),
                 tag = cms.string('TKExtra_Geometry_428SLHCYV0'),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_42X_GEOMETRY")),        
        )

process.add_(cms.ESProducer("FWRecoGeometryESProducer"))

#Adding Timing service:
process.Timing = cms.Service("Timing")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )
process.dump = cms.EDAnalyzer("DumpFWRecoGeometry",
                              level = cms.untracked.int32(1)
                              )

process.p = cms.Path(process.dump)
