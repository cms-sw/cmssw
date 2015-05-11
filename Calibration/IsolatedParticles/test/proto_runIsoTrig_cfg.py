import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.coreTools import *

process = cms.Process("Demo")

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff')
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Calibration.IsolatedParticles.isoTrig_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run1_data']
#process.GlobalTag.globaltag = 'FT_R_53_V6::All' ## July15ReReco Run2012A & B
#process.GlobalTag.globaltag = 'FT_R_53_V10::All' ## Aug24ReReco Run2012C
#process.GlobalTag.globaltag = 'GR_P_V41_AN2::All'  ##Prompt Reco 2012C v2

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100000)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
#process.options.wantSummary = True

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/data/Run2012A/Commissioning/RECO/PromptReco-v1/000/191/247/08F60E33-3B88-E111-A812-5404A6388699.root'
    )
)

process.IsoTrigHB.Verbosity = 0
process.IsoTrigHE = process.IsoTrigHB.clone(Det = cms.string("HE"))

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('IsoTrig.root')
                                   )

process.p = cms.Path(process.IsoTrigHB+process.IsoTrigHE)
