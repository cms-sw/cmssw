##############
# Loading...
##############
import FWCore.ParameterSet.Config as cms
process = cms.Process("DQM")


##############
# DQM
##############
process.load("DQMServices.Core.DQM_cfg")
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQMOffline.Trigger.HLTInclusiveVBFSource_cfi")
process.load("DQMOffline.Trigger.HLTInclusiveVBFClient_cfi")
#
process.DQMStore.verbose = 0 #0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online' #Online
process.dqmSaver.saveByRun = 1 #0
process.dqmSaver.saveAtJobEnd = True


##############
# Other statements
##############
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff') 
process.GlobalTag.globaltag = 'GR_R_52_V7::All'


##############
# Source
##############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
                                
)
process.source = cms.Source("PoolSource",fileNames = cms.untracked.vstring())
process.source.fileNames.extend(['/store/data/Run2012A/MET/AOD/PromptReco-v1/000/193/621/1A96DF37-C49A-E111-BFB2-E0CB4E553673.root',
                                 #'/store/data/Run2012A/MET/AOD/PromptReco-v1/000/193/621/1E2FD882-989A-E111-926B-0025901D5D78.root',
                                 #'/store/data/Run2012A/MET/AOD/PromptReco-v1/000/193/621/2649DC14-C29A-E111-BD4F-001D09F29597.root',
                                 #'/store/data/Run2012A/MET/AOD/PromptReco-v1/000/193/621/42039B67-C19A-E111-8767-0025901D6272.root',
                                 #'/store/data/Run2012A/MET/AOD/PromptReco-v1/000/193/621/50667401-419B-E111-BCC4-001D09F252E9.root',
                                 #'/store/data/Run2012A/MET/AOD/PromptReco-v1/000/193/621/5ABCFC54-BF9A-E111-86A0-001D09F29597.root',
                                 #'/store/data/Run2012A/MET/AOD/PromptReco-v1/000/193/621/5AEA525D-B59A-E111-8EBD-003048F024F6.root',
                                 #'/store/data/Run2012A/MET/AOD/PromptReco-v1/000/193/621/6441AF26-AC9A-E111-A573-003048678110.root',
                                 #'/store/data/Run2012A/MET/AOD/PromptReco-v1/000/193/621/6A34C06C-D49A-E111-8497-0025B32034EA.root',
                                 #'/store/data/Run2012A/MET/AOD/PromptReco-v1/000/193/621/6CF943B6-C79A-E111-9A95-002481E0D7D8.root',
                                 #'/store/data/Run2012A/MET/AOD/PromptReco-v1/000/193/621/70C3AAF6-A29A-E111-AAE8-003048673374.root',
                                 #'/store/data/Run2012A/MET/AOD/PromptReco-v1/000/193/621/AA9A5C2D-B89A-E111-B74E-001D09F291D2.root',
                                 #'/store/data/Run2012A/MET/AOD/PromptReco-v1/000/193/621/CE1AD180-BF9A-E111-B29C-002481E0DEC6.root',
                                 #'/store/data/Run2012A/MET/AOD/PromptReco-v1/000/193/621/D8F7ED66-C19A-E111-9D33-BCAEC5364C93.root',
                                 #'/store/data/Run2012A/MET/AOD/PromptReco-v1/000/193/621/E4AE6098-B99A-E111-A6CB-0025B32036E2.root',
                                 #'/store/data/Run2012A/MET/AOD/PromptReco-v1/000/193/621/F2830385-A19A-E111-AEA2-BCAEC5364C93.root',
                                 #'/store/data/Run2012A/MET/AOD/PromptReco-v1/000/193/621/FEFBA30F-B69A-E111-BF2E-001D09F295A1.root',
                                 ])

##############
# Output
##############
#process.saveInclusiveVBFSave = cms.EDAnalyzer("DQMSimpleFileSaver",
#    outputFileName = cms.string("hist_InclusiveVBF_Parking.root")
#)


##############
# Logger
##############
process.MessageLogger = cms.Service("MessageLogger",
    detailedInfo = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    critical = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('detailedInfo', 
        'critical', 
        'cout')
)

process.myHltInclusiveVBFSource = process.hltInclusiveVBFSource.clone()
process.myHltInclusiveVBFSource.debug = cms.untracked.bool(False)

##############
# Let's it runs
##############
process.psource = cms.Path(process.myHltInclusiveVBFSource*process.hltInclusiveVBFClient)
#process.p = cms.EndPath(process.saveInclusiveVBFSave)
process.p = cms.EndPath(process.dqmSaver)
