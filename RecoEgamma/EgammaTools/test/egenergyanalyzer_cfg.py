import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

#process.source = cms.Source("PoolSource",
    ## replace 'myfile.root' with the source file you want to use
    #fileNames = cms.untracked.vstring(
        #'file:myfile.root'
    #)
#)




process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring('/store/relval/CMSSW_5_0_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START50_V7-v1/0070/903C7DE3-7722-E111-A53A-003048678B86.root'),
)

        

process.egenergyanalyzer = cms.EDAnalyzer('EGEnergyAnalyzer'
)


process.GlobalTag.globaltag = 'START50_V7::All'    

            
process.load("CondCore.DBCommon.CondDBCommon_cfi")
# input database (in this case local sqlite file)
process.CondDBCommon.connect = 'sqlite_file:GBRWrapper.db'

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    DumpStat=cms.untracked.bool(True),
    toGet = cms.VPSet(
      cms.PSet(
        record = cms.string('GBRWrapperRcd'),
        tag = cms.string('wgbrph_EBCorrection'),
        label = cms.untracked.string('wgbrph_EBCorrection')
      ),
      cms.PSet(
        record = cms.string('GBRWrapperRcd'),
        tag = cms.string('wgbrph_EBUncertainty'),
        label = cms.untracked.string('wgbrph_EBUncertainty')
      ),    
      cms.PSet(
        record = cms.string('GBRWrapperRcd'),
        tag = cms.string('wgbrph_EECorrection'),
        label = cms.untracked.string('wgbrph_EECorrection')
      ),
      cms.PSet(
        record = cms.string('GBRWrapperRcd'),
        tag = cms.string('wgbrph_EEUncertainty'),
        label = cms.untracked.string('wgbrph_EEUncertainty')
      ),        
    )
)
            
                
                    
process.p = cms.Path(process.egenergyanalyzer)
