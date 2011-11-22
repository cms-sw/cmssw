import FWCore.ParameterSet.Config as cms

def customise(process):
    process.XMLFromDBSource.label='Extended'
    process.GlobalTag.toGet = cms.VPSet(
        cms.PSet(record = cms.string("GeometryFileRcd"),
                 tag = cms.string("XMLFILE_Geometry_380V3_Extended_mc"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY"),
                 label = cms.untracked.string("Extended")
                 )
        )
    return (process)
