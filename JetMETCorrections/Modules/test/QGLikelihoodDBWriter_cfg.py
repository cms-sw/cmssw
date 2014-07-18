import FWCore.ParameterSet.Config as cms 
process = cms.Process('qgldb')

process.MessageLogger = cms.Service("MessageLogger",
            destinations = cms.untracked.vstring(
                    'cout'
            ),
            cout = cms.untracked.PSet(
                    threshold = cms.untracked.string( 'INFO' )
            ),
)

qgDatabaseVersion = 'v0-test'

process.load('CondCore.DBCommon.CondDBCommon_cfi')
process.CondDBCommon.connect = 'sqlite_file:QGL_'+qgDatabaseVersion+'.db'
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.source = cms.Source('EmptySource')
process.PoolDBOutputService = cms.Service('PoolDBOutputService',
   process.CondDBCommon,
   toPut = cms.VPSet(
      cms.PSet(
         record = cms.string('QGL_AK5PF'),
         tag    = cms.string('QGLikelihoodObject_'+qgDatabaseVersion+'_AK5PF'),
         label  = cms.string('QGL_AK5PF')
      ),
      cms.PSet(
         record = cms.string('QGL_AK5PFchs'),
         tag    = cms.string('QGLikelihoodObject_'+qgDatabaseVersion+'_AK5PFchs'),
         label  = cms.string('QGL_AK5PFchs')
      ),
      cms.PSet(
         record = cms.string('QGL_Syst_Pythia'),
         tag    = cms.string('QGLikelihoodSystematicsObject_'+qgDatabaseVersion+'_Pythia'),
         label  = cms.string('QGL_Syst_Pythia')
      ),
      cms.PSet(
         record = cms.string('QGL_Syst_Herwig++'),
         tag    = cms.string('QGLikelihoodSystematicsObject_'+qgDatabaseVersion+'_Herwig++'),
         label  = cms.string('QGL_Syst_Herwig++')
      ),
   )
)

srcDirectory = 'temp/'
process.dbWriterAK5PF = cms.EDAnalyzer('QGLikelihoodDBWriter',
   src    = cms.string(srcDirectory + 'ReducedHisto_2012.root'),
   payload= cms.string('QGL_AK5PF')
) 
process.dbWriterAK5PFchs = cms.EDAnalyzer('QGLikelihoodDBWriter',
   src    = cms.string(srcDirectory + 'ReducedHisto_2012_CHS.root'),
   payload= cms.string('QGL_AK5PFchs')
) 
process.dbWriterSystPythia = cms.EDAnalyzer('QGLikelihoodSystematicsDBWriter',
   src    = cms.string(srcDirectory + 'SystDatabase.txt'),
   payload= cms.string('QGL_Syst_Pythia')
)
process.dbWriterSystHerwigpp = cms.EDAnalyzer('QGLikelihoodSystematicsDBWriter',
   src    = cms.string(srcDirectory + 'SystDatabase_Hpp.txt'),
   payload= cms.string('QGL_Syst_Herwig++')
)

process.p = cms.Path(process.dbWriterAK5PF * process.dbWriterAK5PFchs * process.dbWriterSystPythia * process.dbWriterSystHerwigpp)
