import FWCore.ParameterSet.Config as cms

def addPoolDBESSource(process,moduleName,record,tag,label='',connect='sqlite_file:'):
    from CondCore.CondDB.CondDB_cfi import CondDB
    calibDB = cms.ESSource("PoolDBESSource",
                           CondDB,
                           toGet = cms.VPSet(cms.PSet(
                               record = cms.string(record),
                               tag = cms.string(tag),
                               label = cms.untracked.string(label)
                           ))
                           )
    calibDB.connect = cms.string(connect)
    if connect.find('oracle:') != -1: calibDB.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
    setattr(process,moduleName,calibDB)
    setattr(process,"es_prefer_" + moduleName,cms.ESPrefer('PoolDBESSource',moduleName))
