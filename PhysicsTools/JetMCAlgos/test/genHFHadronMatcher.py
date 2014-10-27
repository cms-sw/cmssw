import FWCore.ParameterSet.Config as cms

process = cms.Process("Analyzer")


## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.FwkReport.reportEvery = 100

## define input
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(    
    ## add your favourite file here (5000+ events in each file defined below)
    '/store/mc/Spring14dr/TTJets_MSDecaysCKM_central_Tune4C_13TeV-madgraph-tauola/AODSIM/PU_S14_POSTLS170_V6-v1/00000/00120F7A-84F5-E311-9FBE-002618943910.root',
    '/store/mc/Spring14dr/TT_Tune4C_13TeV-pythia8-tauola/AODSIM/Flat20to50_POSTLS170_V5-v1/00000/023E1847-ADDC-E311-91A2-003048FFD754.root',
    '/store/mc/Spring14dr/TTbarH_HToBB_M-125_13TeV_pythia6/AODSIM/PU20bx25_POSTLS170_V5-v1/00000/1CAB7E58-0BD0-E311-B688-00266CFFBC3C.root',
    '/store/mc/Spring14dr/TTbarH_M-125_13TeV_amcatnlo-pythia8-tauola/AODSIM/PU20bx25_POSTLS170_V5-v1/00000/0E3D08A9-C610-E411-A862-0025B3E0657E.root',
    ),
    skipEvents = cms.untracked.uint32(0)
 )

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5000)
)

# get sequence for B-hadron matching
process.load("PhysicsTools.JetMCAlgos.sequences.GenHFHadronMatching_cff")



process.p1 = cms.Path(
    process.genBCHadronMatchingSequence
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('genHFHadronMatcher_out.root'),
    outputCommands = cms.untracked.vstring('drop *', 'keep *_matchGen*_*_*')
    )
process.outpath = cms.EndPath(process.out)
