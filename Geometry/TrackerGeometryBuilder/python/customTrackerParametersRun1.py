import FWCore.ParameterSet.Config as cms
def customise(process):
    process.GlobalTag.toGet = cms.VPSet(cms.PSet(record = cms.string('PTrackerParametersRcd'),
                                                 tag = cms.string('TKParameters_Geometry_Test01'),
                                                 connect = cms.untracked.string("sqlite_file:../myfile.db")
                                                 )
                                        )
    return (process)
