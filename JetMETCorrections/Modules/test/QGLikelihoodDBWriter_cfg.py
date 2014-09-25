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

qgDatabaseVersion = 'v1.1'

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
         record = cms.string('QGL_AK4PF'),
         tag    = cms.string('QGLikelihoodObject_'+qgDatabaseVersion+'_AK4PF'),
         label  = cms.string('QGL_AK4PF')
      ),
      cms.PSet(
         record = cms.string('QGL_AK4PFchs'),
         tag    = cms.string('QGLikelihoodObject_'+qgDatabaseVersion+'_AK4PFchs'),
         label  = cms.string('QGL_AK4PFchs')
      ),
# ONLY AFTER FIRST DATA
#     cms.PSet(
#        record = cms.string('QGL_Syst_Pythia'),
#        tag    = cms.string('QGLikelihoodSystematicsObject_'+qgDatabaseVersion+'_Pythia'),
#        label  = cms.string('QGL_Syst_Pythia')
#     ),
#     cms.PSet(
#        record = cms.string('QGL_Syst_Herwig++'),
#        tag    = cms.string('QGLikelihoodSystematicsObject_'+qgDatabaseVersion+'_Herwig++'),
#        label  = cms.string('QGL_Syst_Herwig++')
#     ),
   )
)

srcDirectory = 'temp/'
process.dbWriterAK4PF = cms.EDAnalyzer('QGLikelihoodDBWriter',
   src    = cms.string(srcDirectory + 'pdfQG_AK4_13TeV.root'),
   payload= cms.string('QGL_AK4PF')
) 
process.dbWriterAK4PFchs = cms.EDAnalyzer('QGLikelihoodDBWriter',
   src    = cms.string(srcDirectory + 'pdfQG_AK4chs_13TeV.root'),
   payload= cms.string('QGL_AK4PFchs')
) 
process.dbWriterAK5PF = cms.EDAnalyzer('QGLikelihoodDBWriter',
   src    = cms.string(srcDirectory + 'pdfQG_AK5_13TeV.root'),
   payload= cms.string('QGL_AK5PF')
) 
process.dbWriterAK5PFchs = cms.EDAnalyzer('QGLikelihoodDBWriter',
   src    = cms.string(srcDirectory + 'pdfQG_AK5chs_13TeV.root'),
   payload= cms.string('QGL_AK5PFchs')
)

# ONLY AFTER FIRST DATA:
#process.dbWriterSystPythia = cms.EDAnalyzer('QGLikelihoodSystematicsDBWriter',
#   src    = cms.string(srcDirectory + 'SystDatabase.txt'),
#   payload= cms.string('QGL_Syst_Pythia')
#)
#process.dbWriterSystHerwigpp = cms.EDAnalyzer('QGLikelihoodSystematicsDBWriter',
#   src    = cms.string(srcDirectory + 'SystDatabase_Hpp.txt'),
#   payload= cms.string('QGL_Syst_Herwig++')
#)
 

process.p = cms.Path(process.dbWriterAK4PF*process.dbWriterAK5PF*process.dbWriterAK4PFchs*process.dbWriterAK5PFchs)
