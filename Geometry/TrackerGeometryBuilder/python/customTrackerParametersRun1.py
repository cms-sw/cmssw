import FWCore.ParameterSet.Config as cms
def customise(process):
    process.GlobalTag.toGet = cms.VPSet(cms.PSet(record = cms.string('PTrackerParametersRcd'),
                                                 tag = cms.string('TK_Parameters_Test02'),
                                                 connect = cms.untracked.string("sqlite_file:../myfile.db")
                                                 )
                                        )
    return (process)
