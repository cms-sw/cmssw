import FWCore.ParameterSet.Config as cms

process = cms.Process("MVAJetTagsSQLiteSave")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_31X_V5::All'

#process.prefer("BTauMVAJetTagComputerRecord")

process.jetTagMVATrainerSave = cms.EDAnalyzer("JetTagMVATrainerFileSave",
	trained = cms.untracked.bool(False),
#	ImpactParameterMVA = cms.string('ImpactParameterMVA.mva'), 
	CombinedMVA = cms.string('CombinedMVA.mva'), 
	CombinedSVRecoVertex = cms.string('CombinedSVRecoVertex.mva'), 
	CombinedSVPseudoVertex = cms.string('CombinedSVPseudoVertex.mva'), 
	CombinedSVNoVertex = cms.string('CombinedSVNoVertex.mva'), 
	CombinedSVMVARecoVertex = cms.string('CombinedSVMVARecoVertex.mva'), 
	CombinedSVMVAPseudoVertex = cms.string('CombinedSVMVAPseudoVertex.mva'), 
	CombinedSVMVANoVertex = cms.string('CombinedSVMVANoVertex.mva')
)

process.outpath = cms.EndPath(process.jetTagMVATrainerSave)
