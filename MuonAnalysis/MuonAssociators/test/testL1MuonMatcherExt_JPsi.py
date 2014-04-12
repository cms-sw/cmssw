import FWCore.ParameterSet.Config as cms

process = cms.Process("TestL1MatcherExtended")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'GR_R_36X_V12::All'

process.MessageLogger.cerr.FwkReport.reportEvery = 100 
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
        'root://pcmssd12.cern.ch//data/gpetrucc/7TeV/jpsi/CS_Onia-Jun14thSkim_v1_RAW-RECO_run136082_443584C2-B27E-DF11-9E13-0017A477001C.root'
    )
)

process.load("MuonAnalysis.MuonAssociators.muonL1MatchExtended_cfi")
import PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi
process.patMuons = PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi.patMuons.clone(
    muonSource = 'muons',
    embedTrack          = True,
    embedCombinedMuon   = True,
    embedStandAloneMuon = True,
    embedPickyMuon = False,
    embedTpfmsMuon = False, 
    userIsolation = cms.PSet(), # no extra isolation
    isoDeposits = cms.PSet(),   # no isodeposits
    addGenMatch = False,        # no mc
)

from MuonAnalysis.MuonAssociators.muonL1MatchExtended_cfi import addUserData as addMuonL1MatchExtended
addMuonL1MatchExtended(process.patMuons)

## Good Muons (very simplified selection)
process.mu4j = cms.EDFilter("PATMuonRefSelector", 
    src = cms.InputTag("patMuons"),
    cut = cms.string("muonID('TMLastStationAngTight')")
)
## Trigger matched
process.mu4jt  = process.mu4j.clone(cut = process.mu4j.cut.value() + " && userInt('muonL1MatchExtended') > 0");
## Trigger matched, geometrically
process.mu4jtg = process.mu4j.clone(cut = process.mu4j.cut.value() + " && userInt('muonL1MatchExtended') >= 10");

## Di-muons, all
process.dimu = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("mu4j@+ mu4j@-"),
    cut   = cms.string("mass > 2"),
)
## Di-muons, both trigger matched
process.dimutt  = process.dimu.clone(decay = "mu4jt@+ mu4jt@-")
## Di-muons, both trigger matched geometrically
process.dimuttg = process.dimu.clone(decay = "mu4jtg@+ mu4jtg@-")
process.any = cms.EDFilter("CandViewCountFilter", src = cms.InputTag("dimu"),    minNumber = cms.uint32(1))
process.tt  = cms.EDFilter("CandViewCountFilter", src = cms.InputTag("dimutt"),  minNumber = cms.uint32(1))
process.ttg = cms.EDFilter("CandViewCountFilter", src = cms.InputTag("dimuttg"), minNumber = cms.uint32(1))

## Test trigger bit
from HLTrigger.HLTfilters.hltHighLevelDev_cfi import hltHighLevelDev
process.trigBit = hltHighLevelDev.clone(HLTPaths = ['HLT_L1DoubleMuOpen'], HLTPathsPrescales = [1])

process.s = cms.Sequence(
    process.csctfDigis *
    process.muonL1MatchExtended *
    process.patMuons *
    ( process.mu4j + process.mu4jt + process.mu4jtg   ) *
    ( process.dimu + process.dimutt + process.dimuttg )
)

process.pAny = cms.Path(
    process.trigBit +
    process.s +
    process.any
)

process.pMatch = cms.Path(
    process.trigBit +
    process.s +
    process.tt
)

process.pMatchGeom = cms.Path(
    process.trigBit +
    process.s +
    process.ttg
)

#process.o = cms.OutputModule("PoolOutputModule",
#    fileName = cms.untracked.string("patMuons_L1MatcherExtended.root"),
#    outputCommands = cms.untracked.vstring("drop *", "keep *_patMuons__*", "keep l1extraL1MuonParticles_l1extraParticles__*")
#)
#process.e = cms.EndPath(process.o)

