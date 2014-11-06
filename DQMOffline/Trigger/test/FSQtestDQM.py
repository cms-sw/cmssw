import FWCore.ParameterSet.Config as cms

process = cms.Process("DQMPathChecker")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.load("DQMServices.Core.DQM_cfg")

#import DQMServices.Components.DQMEnvironment_cfi
#process.dqmEnvHLT= DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
#process.dqmEnvHLT.subSystemFolder = 'HLT'


process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmEnv.subSystemFolder = 'HLT'



process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.globaltag = 'MCRUN2_72_V1::All'

f='/nfs/dust/cms/user/fruboest/2014.11.HLTJec721p1/CMSSW_7_2_1_patch1/src/outputFULL.root'
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:'+f
    )
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


fsqdirname = "HLT/FSQ/"
process.ttt = cms.EDAnalyzer("FSQDiJetAve",
    dirname = cms.untracked.string("HLT/FSQ/DiJETAve/"),
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","TTT"),
    triggerResultsLabel = cms.InputTag("TriggerResults","","TTT"),
    useGenWeight = cms.bool(False),
    #useGenWeight = cms.bool(True),
    todo = cms.VPSet(
        cms.PSet(
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string("HLT_DiPFJetAve30_HFJEC_"),
            partialFilterName  = cms.string("ForHFJECBase"),
            dqmhistolabel  = cms.string("hltcalo"),
            mainDQMDirname = cms.untracked.string(fsqdirname),
            singleObjectsPreselection = cms.string("1==1"),
            combinedObjectSelection =  cms.string("abs(at(0).eta)<1.4 || abs(at(0).eta) > 2.7 "),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            drawables =  cms.VPSet(
                cms.PSet (
                    name = cms.string("pt"),
                    expression = cms.string("at(0).pt"),
                    bins = cms.int32(100),
                    min = cms.double(0),
                    max = cms.double(100)
                )
            )
        ),


        cms.PSet(
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string("HLT_DiPFJetAve60_HFJEC_"),
            partialFilterName  = cms.string("ForHFJECBase"),
            dqmhistolabel  = cms.string("hltcalo"),
            mainDQMDirname = cms.untracked.string(fsqdirname),
            singleObjectsPreselection = cms.string("1==1"),
            combinedObjectSelection =  cms.string("abs(at(0).eta)<1.4 || abs(at(0).eta) > 2.7 "),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            drawables =  cms.VPSet(
                cms.PSet (
                    name = cms.string("pt"),
                    expression = cms.string("at(0).pt"),
                    bins = cms.int32(100),
                    min = cms.double(0),
                    max = cms.double(100)
                )
            )
        ),


    )
)

process.load('DQMServices.Components.DQMFileSaver_cfi')
process.dqmSaver.workflow = "/HLT/FSQ/All"

process.p = cms.Path(process.ttt*process.dqmEnv*process.dqmSaver)
#process.MessageLogger.threshold = cms.untracked.string( "INFO" )
#process.MessageLogger.categories.append("FSQDiJetAve")

