
import FWCore.ParameterSet.Config as cms

process = cms.Process('SkimHZZ')

# Complete Preselection Sequence for 2e2mu analysis

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')

process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'IDEAL_V9::All'

# Complete Skim analysis
process.load('HiggsAnalysis/Skimming/higgsToZZ4Leptons_Sequences_cff')

process.eca = cms.EDAnalyzer("EventContentAnalyzer")
process.hTozzTo4leptonsSkimPath = cms.Path(process.higgsToZZ4LeptonsSequence+process.eca)


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
                            fileNames = cms.untracked.vstring('/store/user/ndefilip/comphep-bbll/CMSSW_2_2_3-bkg_RAW2DIGI_RECO_IDEAL/6e3420323cbcc78f83bfe627dc999a04/RAW2DIGI_RECO_IDEAL_998.root'
                             )
                           )


# Endpath
process.o = cms.EndPath ( process.output )
