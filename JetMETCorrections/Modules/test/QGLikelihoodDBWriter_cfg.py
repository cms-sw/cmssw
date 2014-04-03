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


process.load('CondCore.DBCommon.CondDBCommon_cfi') 
process.CondDBCommon.connect = 'sqlite_file:QGL_V1.db' 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 
process.source = cms.Source('EmptySource') 
process.PoolDBOutputService = cms.Service('PoolDBOutputService', 
   process.CondDBCommon, 
   toPut = cms.VPSet( 
      cms.PSet(
         record = cms.string('QGL_AK5PF'), 
         tag    = cms.string('QGLikelihoodObject_V1_AK5PF'), 
         label  = cms.string('QGL_AK5PF') 
      ),
      cms.PSet(
         record = cms.string('QGL_AK5PFchs'), 
         tag    = cms.string('QGLikelihoodObject_V1_AK5PFchs'), 
         label  = cms.string('QGL_AK5PFchs') 
      ),
   ) 
) 

process.dbWriterAK5PF = cms.EDAnalyzer('QGLikelihoodDBWriter', 
   src    = cms.string('CondFormats/JetMETObjects/data/ReducedHisto_2012.root'),  
   payload= cms.string('QGL_AK5PF') 
) 
process.dbWriterAK5PFchs = cms.EDAnalyzer('QGLikelihoodDBWriter', 
   src    = cms.string('CondFormats/JetMETObjects/data/ReducedHisto_2012_CHS.root'),  
   payload= cms.string('QGL_AK5PFchs') 
) 


process.p = cms.Path( 
process.dbWriterAK5PF *
process.dbWriterAK5PFchs
) 
