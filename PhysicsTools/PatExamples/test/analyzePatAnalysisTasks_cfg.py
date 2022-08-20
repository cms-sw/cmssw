import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    "file:patTuple.root"
  )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.MessageLogger = cms.Service("MessageLogger")

#################
#               #
# EXERCISE 1    #
#               #
#################

process.jecAnalyzer = cms.EDAnalyzer("WrappedEDAnalysisTasksAnalyzerJEC",
  Jets = cms.InputTag("cleanPatJets"), 
  jecLevel=cms.string("SomeWrongName"), 
#  jecLevel=cms.string("L3Absolute"),
  patJetCorrFactors= cms.string('SomeWrongName'),
#  patJetCorrFactors= cms.string('patJetCorrFactors'),
  help=cms.bool(False)
)

process.p = cms.Path(process.jecAnalyzer)


#################
#               #
# EXERCISE 2    #
#               #
#################

process.TFileService = cms.Service("TFileService",
  fileName = cms.string('TFileServiceOutput.root')
)


process.jecAnalyzerRel=process.jecAnalyzer.clone(jecLevel="L2Relative")
process.jecAnalyzerNon=process.jecAnalyzer.clone(jecLevel="Uncorrected")
process.p.replace(process.jecAnalyzer, process.jecAnalyzer * process.jecAnalyzerRel * process.jecAnalyzerNon)


#################
#               #
# EXERCISE 3.2  #
#               #
#################
## Geometry and Detector Conditions (needed for a Scaling up and down the JEC)
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = cms.string( autoCond[ 'startup' ] )
process.load("Configuration.StandardSequences.MagneticField_cff")

# load the standard PAT config
process.load("PhysicsTools.PatAlgos.patSequences_cff")
from PhysicsTools.PatUtils.tools.metUncertaintyTools import runMEtUncertainties
runMEtUncertainties(process)

#process.shiftedPatJets=process.shiftedPatJetsEnUp.clone(src = cms.InputTag("cleanPatJets"),shiftBy=cms.double(1))

#process.jecAnalyzerEnUp=process.jecAnalyzer.clone(Jets = cms.InputTag("shiftedPatJets"))
#process.p.replace(process.jecAnalyzer,  process.shiftedPatJets * process.jecAnalyzerEnUp)

#################
#               #
# EXERCISE 3.3  #
#               #
#################

#process.btagAnalyzer = cms.EDAnalyzer("WrappedEDAnalysisTasksAnalyzerBTag",
#  Jets = cms.InputTag("cleanPatJets"),    
#   bTagAlgo=cms.string('trackCountingHighEffBJetTags'),
#   bins=cms.uint32(100),
#   lowerbin=cms.double(0.),
#   upperbin=cms.double(10.)
#)

#process.btagAnalyzerTCHP=process.btagAnalyzer.clone(bTagAlgo="trackCountingHighPurBJetTags")

#process.p.replace(process.jecAnalyzer, process.jecAnalyzer * process.btagAnalyzer * process.btagAnalyzerTCHP)




