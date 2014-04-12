import FWCore.ParameterSet.Config as cms

# process
process = cms.Process('L1DTTFGlobalTagTest')

# number of events and source
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source('EmptySource')

# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
 
process.GlobalTag.globaltag = 'IDEAL_30X::All'

# paths to be run
process.DTExtLutTester = cms.EDAnalyzer("DTExtLutTester")
process.DTPhiLutTester = cms.EDAnalyzer("DTPhiLutTester")
process.DTPtaLutTester = cms.EDAnalyzer("DTPtaLutTester")
process.DTEtaPatternLutTester = cms.EDAnalyzer("DTEtaPatternLutTester")
process.DTQualPatternLutTester = cms.EDAnalyzer("DTQualPatternLutTester")
process.DTTFParametersTester = cms.EDAnalyzer("DTTFParametersTester")
process.DTTFMasksTester = cms.EDAnalyzer("DTTFMasksTester")

process.p = cms.Path(process.DTExtLutTester
                    *process.DTPhiLutTester
                    *process.DTPtaLutTester
                    *process.DTEtaPatternLutTester
                    *process.DTQualPatternLutTester
                    *process.DTTFParametersTester
                    *process.DTTFMasksTester
)
