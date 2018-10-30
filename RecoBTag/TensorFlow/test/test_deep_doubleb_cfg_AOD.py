from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask
import six

process = cms.Process("PAT")

## MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

## Options and Output Report
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

## Source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)
## Maximal Number of Events
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

## Geometry and Detector Conditions (needed for a few patTuple production steps)
process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')
process.load("Configuration.StandardSequences.MagneticField_cff")

## Output Module Configuration (expects a path 'p')
from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoCleaning
process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('patTuple.root'),
                               ## save only events passing the full path
                               #SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
                               ## save PAT output; you need a '*' to unpack the list of commands
                               ## 'patEventContent'
                               outputCommands = cms.untracked.vstring('drop *', *patEventContentNoCleaning )
                               )

patAlgosToolsTask = getPatAlgosToolsTask(process)
process.outpath = cms.EndPath(process.out, patAlgosToolsTask)

## and add them to the event content
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection,addJetCollection

## Puppi
jetSeq = cms.Sequence()
process.load("CommonTools.PileupAlgos.Puppi_cff")
#process.puppi.candName = cms.InputTag( 'particleFlow' ) 
jetSeq += process.puppi

from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJetsPuppi
process.ak8PFJetsPuppi = ak4PFJetsPuppi.clone( src = cms.InputTag( 'puppi' ),
                                               doAreaFastjet = True, 
                                               rParam = 0.8, 
                                               jetAlgorithm = 'AntiKt' )
jetSeq += process.ak8PFJetsPuppi

process.jetSequence = jetSeq

addJetCollection(
   process,
   #labelName = 'AK8PFCHS',
   labelName = 'AK8PFPuppi',
   #jetSource = cms.InputTag('ak8PFJetsCHS'),
   jetSource = cms.InputTag('ak8PFJetsPuppi'),
   #pvSource = cms.InputTag('offlinePrimaryVertices'),
   #svSource = cms.InputTag('inclusiveCandidateSecondaryVertices'),
   algo = 'AK',
   rParam = 0.8,
   #jetCorrections = ('AK8PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-2'),
   jetCorrections = ('AK8PFPuppi', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-2'),
   btagDiscriminators = [
      'pfBoostedDoubleSecondaryVertexAK8BJetTags',
      'pfDeepDoubleBvLJetTags:probQCD', 
      'pfDeepDoubleBvLJetTags:probHbb', 
      ]
   )

#process.patJetsAK8PFCHS.addTagInfos = True
process.patJetsAK8PFPuppi.addTagInfos = True

from pdb import set_trace

from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarPileUpMINIAODSIM

process.source.fileNames = filesRelValTTbarPileUpMINIAODSIM
#process.source.fileNames = cms.untracked.vstring('root://cmsxrootd.fnal.gov//store/mc/PhaseIFall16MiniAOD/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8/MINIAODSIM/PhaseIFall16PUFlat20to50_PhaseIFall16_81X_upgrade2017_realistic_v26-v1/50000/08358A47-61E3-E611-8B77-001E677928AE.root')

process.source.fileNames = cms.untracked.vstring('root://cmsxrootd.fnal.gov//store/relval/CMSSW_10_1_0_pre2/RelValTTbar_13/GEN-SIM-DIGI-RECO/100X_mcRun2_asymptotic_v2_FastSim-v1/20000/228EFC7D-351F-E811-AB67-0025905A612A.root',
                                                 'root://cmsxrootd.fnal.gov//store/relval/CMSSW_10_1_0_pre2/RelValTTbar_13/GEN-SIM-DIGI-RECO/100X_mcRun2_asymptotic_v2_FastSim-v1/20000/7E149C80-291F-E811-BA0C-0CC47A745250.root',
                                                 'root://cmsxrootd.fnal.gov//store/relval/CMSSW_10_1_0_pre2/RelValTTbar_13/GEN-SIM-DIGI-RECO/100X_mcRun2_asymptotic_v2_FastSim-v1/20000/8AE4CC80-F41E-E811-892E-0025905AA9CC.root',
                                                 'root://cmsxrootd.fnal.gov//store/relval/CMSSW_10_1_0_pre2/RelValTTbar_13/GEN-SIM-DIGI-RECO/100X_mcRun2_asymptotic_v2_FastSim-v1/20000/D418B3BA-B51E-E811-B2A8-0CC47A4C8E98.root',)
#'root://cmsxrootd.fnal.gov//store/mc/RunIISummer16DR80Premix/BulkGravTohhTohbbhbb_narrow_M-2500_13TeV-madgraph/AODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/120000/0EA8CDD5-47B2-E611-B64D-A0369F7FC608.root',)
#                                                 'root://cmsxrootd.fnal.gov//store/mc/RunIISummer16DR80Premix/BulkGravTohhTohbbhbb_narrow_M-2500_13TeV-madgraph/AODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/120000/1A83C5CE-92B2-E611-9BC8-00266CFCC490.root',
#                                                 'root://cmsxrootd.fnal.gov//store/mc/RunIISummer16DR80Premix/BulkGravTohhTohbbhbb_narrow_M-2500_13TeV-madgraph/AODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/120000/1C4B543D-49B2-E611-805D-008CFA1983E0.root',
#                                                 'root://cmsxrootd.fnal.gov//store/mc/RunIISummer16DR80Premix/BulkGravTohhTohbbhbb_narrow_M-2500_13TeV-madgraph/AODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/120000/24B383E8-3FB2-E611-B9F3-549F35AE4FFD.root',
#                                                 'root://cmsxrootd.fnal.gov//store/mc/RunIISummer16DR80Premix/BulkGravTohhTohbbhbb_narrow_M-2500_13TeV-madgraph/AODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/120000/402CD384-4BB2-E611-A05B-A0369F7FC608.root',
#                                                 'root://cmsxrootd.fnal.gov//store/mc/RunIISummer16DR80Premix/BulkGravTohhTohbbhbb_narrow_M-2500_13TeV-madgraph/AODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/120000/8A8BE646-4FB2-E611-BA79-008CFA05E874.root',
#                                                 'root://cmsxrootd.fnal.gov//store/mc/RunIISummer16DR80Premix/BulkGravTohhTohbbhbb_narrow_M-2500_13TeV-madgraph/AODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/120000/E2ACFBFE-44B2-E611-B744-00266CFCC860.root',
#                                                 'root://cmsxrootd.fnal.gov//store/mc/RunIISummer16DR80Premix/BulkGravTohhTohbbhbb_narrow_M-2500_13TeV-madgraph/AODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/120000/DE3318DE-43B2-E611-93E8-A0369F7FC210.root')

process.maxEvents.input = -1

from Configuration.EventContent.EventContent_cff import MINIAODSIMEventContent
#process.out.outputCommands = MINIAODSIMEventContent.outputCommands
process.out.outputCommands = patEventContentNoCleaning
process.out.outputCommands.append('keep *_ak8PFJetsCHS*_*_*')
process.out.outputCommands.append('keep *_offlinePrimaryVertices*_*_*')
process.out.outputCommands.append('keep *_inclusiveCandidateSecondaryVertices*_*_*')
process.out.outputCommands.append('keep *_selectedPatJets*_*_*')
process.out.outputCommands.append('keep *_selectedUpdatedPatJets*_*_*')
process.out.outputCommands.append('keep *_pfBoostedDoubleSVAK8TagInfos*_*_*')
process.out.outputCommands.append('keep *_pfDeepDoubleXTagInfos*_*_*')
process.out.outputCommands.append('keep *_updatedPatJets*_*_*')

print(process.out.outputCommands)
process.out.fileName = 'test_deep_doubleb_AODSIM.root'

#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)
# process.add_(cms.Service("InitRootHandlers", DebugLevel =cms.untracked.int32(3)))


#Trick to make it work in >=9_1_X
process.tsk = cms.Task()
for mod in six.itervalues(process.producers_()):
    process.tsk.add(mod)
for mod in six.itervalues(process.filters_()):
    process.tsk.add(mod)

process.p = cms.Path(
    process.jetSequence,
    process.tsk
    )
