import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
                        

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        'file:/hdfs/store/mc/RunIIFall17MiniAODv2/ZNuNuGJets_MonoPhoton_PtG-130_TuneCP5_13TeV-madgraph/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/70000/40A54173-E490-E911-8F5E-0025905B85DE.root '
    )

)

process.TFileService = cms.Service("TFileService", 
    fileName = cms.string('/afs/hep.wisc.edu/home/bsahu/Photon_analyzer_2/CMSSW_9_4_9/src/Photon_Analyzer/PhotonAnalyzer/test/Signal_Ntuple/Output_Ntuple_Signal_70000_2.root')
    )


process.demo = cms.EDAnalyzer('PhotonAnalyzer',
                              photonToken                          =  cms.InputTag("slimmedPhotons")
)


process.p = cms.Path(process.demo)
