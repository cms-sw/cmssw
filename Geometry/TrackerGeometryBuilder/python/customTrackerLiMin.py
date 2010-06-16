import FWCore.ParameterSet.Config as cms
def customise(process):
    process.XMLFromDBSource.label='ExtendedLiMin'
    process.GlobalTag.toGet = cms.VPSet(
        cms.PSet(record = cms.string("GeometryFileRcd"),
                 tag = cms.string("XMLFILE_Geometry_ExtendedLiMin_38YV0"),
                 connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_GEOMETRY"),
                 label = cms.untracked.string("ExtendedLiMin")
                 )
        )
    return (process)
