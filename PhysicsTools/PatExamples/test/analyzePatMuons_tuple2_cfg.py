## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

## ---
## This is an example of the use of the plain edm::Tuple dumper to analyze pat::Muons
process.patMuonAnalyzer = cms.EDProducer(
    "CandViewNtpProducer", 
    src = cms.InputTag("cleanPatMuons"),
    lazyParser = cms.untracked.bool(True),
    prefix = cms.untracked.string(""),
    eventInfo = cms.untracked.bool(True),
    variables = cms.VPSet(
        cms.PSet(
            tag = cms.untracked.string("pt"),
            quantity = cms.untracked.string("pt")
        ),
        cms.PSet(
            tag = cms.untracked.string("eta"),
            quantity = cms.untracked.string("eta")
        ),
        cms.PSet(
            tag = cms.untracked.string("phi"),
            quantity = cms.untracked.string("phi")
        ),
    )  
)

## let it run
process.p = cms.Path(
    process.patDefaultSequence
  * process.patMuonAnalyzer
    )

process.out.fileName = "edmTuple.root"
process.out.outputCommands = ['drop *', 'keep *_patMuonAnalyzer_*_*']
