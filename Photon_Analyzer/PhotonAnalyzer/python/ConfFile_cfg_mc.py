import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")


process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use                                                                              
    fileNames = cms.untracked.vstring(
        'file:/hdfs/store/mc/RunIIFall17MiniAODv2/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v3/70000/4CCA5300-7959-E911-95CE-0025901D0C50.root '
    )

)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('/afs/hep.wisc.edu/home/bsahu/Photon_analyzer_2/CMSSW_9_4_9/src/Photon_Analyzer/PhotonAnalyzer/test/Output_Ntuple_mc.root')
    )


process.demo = cms.EDAnalyzer('PhotonAnalyzer',
                              photonToken                          =  cms.InputTag("slimmedPhotons")
)


process.p = cms.Path(process.demo)

