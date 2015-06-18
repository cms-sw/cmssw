import FWCore.ParameterSet.Config as cms
def customise(process):
    process.GlobalTag.toGet = cms.VPSet(cms.PSet(record = cms.string('PTrackerParametersRcd'),
                                                 tag = cms.string('TKParameters_Geometry_Run2_Test02'),
                                                 connect = cms.untracked.string("sqlite_file:../myfilerun2.db")
                                                 )
                                        )
    return (process)
