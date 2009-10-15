
import FWCore.ParameterSet.Config as cms

process = cms.Process('SkimHZZ')

# Complete Preselection Sequence for 2e2mu analysis

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')

process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'MC_31X_V8::All'

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

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.source = cms.Source("PoolSource",
                            debugFlag = cms.untracked.bool(True),
                            debugVebosity = cms.untracked.uint32(10),
                            # fileNames = cms.untracked.vstring('file:/home/llr/cms/ndefilip/RAW2DIGI_RECO_IDEAL_21_2e2mu.root'                            
                            fileNames = cms.untracked.vstring('file:/data3/data.polcms/FileMoverData/store/relval/CMSSW_3_2_6/RelValZmumuJets_Pt_20_300_GEN/GEN-SIM-RECO/MC_31X_V8_LowLumiPileUp-v1/0014/FE698689-5C9B-DE11-BFB7-001D09F23A07.root'
                             )
                           )


# Endpath
process.o = cms.EndPath ( process.output )
