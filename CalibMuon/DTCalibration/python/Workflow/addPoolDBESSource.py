import FWCore.ParameterSet.Config as cms

def addPoolDBESSource(process,moduleName,record,tag,label='',connect='sqlite_file:'):
    from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup 
    calibDB = cms.ESSource("PoolDBESSource",
                           CondDBSetup,
                           timetype = cms.string('runnumber'),
                           toGet = cms.VPSet(cms.PSet(
                               record = cms.string(record),
                               tag = cms.string(tag),
                               label = cms.untracked.string(label)
                           )),
                           connect = cms.string(connect),
                           authenticationMethod = cms.untracked.uint32(0))
    #if authPath: calibDB.DBParameters.authenticationPath = authPath
    if connect.find('oracle:') != -1: calibDB.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
    setattr(process,moduleName,calibDB)
    setattr(process,"es_prefer_" + moduleName,cms.ESPrefer('PoolDBESSource',moduleName))
