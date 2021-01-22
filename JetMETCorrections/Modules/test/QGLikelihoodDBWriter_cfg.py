import FWCore.ParameterSet.Config as cms 
process = cms.Process('qgldb')

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)

qgDatabaseVersion = 'v1'

process.load('CondCore.DBCommon.CondDBCommon_cfi')
process.CondDBCommon.connect = 'sqlite_file:QGL_'+qgDatabaseVersion+'.db'
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.source = cms.Source('EmptySource')
process.PoolDBOutputService = cms.Service('PoolDBOutputService',
   process.CondDBCommon,
   toPut = cms.VPSet(
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
      cms.PSet(
         record = cms.string('QGL_AK4PF_antib'),
         tag    = cms.string('QGLikelihoodObject_'+qgDatabaseVersion+'_AK4PF_antib'),
         label  = cms.string('QGL_AK4PF_antib')
      ),
      cms.PSet(
         record = cms.string('QGL_AK4PFchs_antib'),
         tag    = cms.string('QGLikelihoodObject_'+qgDatabaseVersion+'_AK4PFchs_antib'),
         label  = cms.string('QGL_AK4PFchs_antib')
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

# The ROOT files of v1 are at /afs/cern.ch/user/t/tomc/public/qgTagger/QGLikelihoodDBFiles/QGL_v1
srcDirectory = 'JetMETCorrections/Modules/test/' + qgDatabaseVersion + '/'
process.dbWriterAK4PF = cms.EDAnalyzer('QGLikelihoodDBWriter',
   src    = cms.string(srcDirectory + 'pdfQG_AK4_13TeV_' + qgDatabaseVersion + '.root'),
   payload= cms.string('QGL_AK4PF')
)
process.dbWriterAK4PFchs = cms.EDAnalyzer('QGLikelihoodDBWriter',
   src    = cms.string(srcDirectory + 'pdfQG_AK4chs_13TeV_' + qgDatabaseVersion + '.root'),
   payload= cms.string('QGL_AK4PFchs')
)
process.dbWriterAK4PF_antib = cms.EDAnalyzer('QGLikelihoodDBWriter',
   src    = cms.string(srcDirectory + 'pdfQG_AK4_antib_13TeV_' + qgDatabaseVersion + '.root'),
   payload= cms.string('QGL_AK4PF_antib')
)
process.dbWriterAK4PFchs_antib = cms.EDAnalyzer('QGLikelihoodDBWriter',
   src    = cms.string(srcDirectory + 'pdfQG_AK4chs_antib_13TeV_' + qgDatabaseVersion + '.root'),
   payload= cms.string('QGL_AK4PFchs_antib')
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
 

process.p = cms.Path(process.dbWriterAK4PF * process.dbWriterAK4PFchs * process.dbWriterAK4PF_antib * process.dbWriterAK4PFchs_antib)
