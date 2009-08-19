
import FWCore.ParameterSet.Config as cms

process = cms.Process('SkimHWWFakeRates')

# Complete Preselection Sequence for 2e2mu analysis

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 10000

process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'MC_31X_V3::All'


# Complete Skim analysis
process.load('HiggsAnalysis/Skimming/higgsToWW2Leptons_FakeRatesSequences_cff')


process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.hToWWTo2leptonsFakeRatesSkimPath = cms.Path(
process.higgsToWW2LeptonsFakeRatesFilter) #+process.eca)


# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('hToWWTo2leptons_Skim.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('higgsToWW2LeptonsFakeRatesSequence ')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('hToWWTo2leptonsFakeRatesSkimPath')
    )                              
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.source = cms.Source("PoolSource",
                            debugFlag = cms.untracked.bool(True),
                            debugVebosity = cms.untracked.uint32(10),
                            fileNames = cms.untracked.vstring('/store/mc/Summer09/ZeeJet_Pt300toInf/GEN-SIM-RECO/MC_31X_V3-v1/0004/F41C0D61-727E-DE11-9EF2-001CC4BF8694.root')
                           )


# Endpath
process.o = cms.EndPath ( process.output )
