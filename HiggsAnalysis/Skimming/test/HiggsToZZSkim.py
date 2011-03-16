
import FWCore.ParameterSet.Config as cms

process = cms.Process('SkimHZZ')

# Complete Preselection Sequence for 2e2mu analysis

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')

process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START39_V8::All'

# Complete Skim analysis
process.load('HiggsAnalysis/Skimming/higgsToZZ4Leptons_Sequences_cff')

process.hTozzTo4leptonsSkimPath = cms.Path(process.higgsToZZ4LeptonsSequence)


# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('hTozzTo4leptons_Skim.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('higgsToZZ4LeptonsSequence')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('hTozzTo4leptonsSkimPath')
    )
                               
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:98D927A6-D526-E011-B68D-0025B3E05D46.root'
                             )
                           )


# Endpath
process.o = cms.EndPath ( process.output )
