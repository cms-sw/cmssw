import FWCore.ParameterSet.Config as cms
process = cms.Process("RERECO")

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
process.load('RecoLocalTracker.Configuration.RecoLocalTracker_cff')
process.load('RecoHI.HiMuonAlgos.HiReRecoMuon_cff')

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load("Configuration.EventContent.EventContentHeavyIons_cff")  
#global tags for conditions data: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START50_V2::All'#'START44_V7::All'

##################################################################################
# setup 'standard'  options
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

# Input source
# the fiel should have the STA in them; this is a RE-reco
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('/store/relval/CMSSW_5_0_0_pre3/HIAllPhysics/RECO/GR_R_50_V1_RelVal_hi2010-v2/0000/F8A98914-08FD-E011-A27B-003048F1BF66.root'),
#rfio:/castor/cern.ch/user/m/mironov/regittest/rawrecohltdebug_98_1_jYu.root'),
                            noEventSort = cms.untracked.bool(True),
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
                            skipEvents=cms.untracked.uint32(0)
                            )

process.output = cms.OutputModule("PoolOutputModule",
                                  splitLevel = cms.untracked.int32(0),
                                  outputCommands = cms.untracked.vstring('drop *',
                                                                         'keep *_*_*_RERECO',
                                                                         ),
                                  fileName = cms.untracked.string('RErecoFullTest.root')
                                  )

##################################################################################
'''
process.SimpleMemoryCheck = cms.Service('SimpleMemoryCheck',
                                        ignoreTotal=cms.untracked.int32(0),
                                        oncePerEventMode = cms.untracked.bool(False)
                                        )

process.Timing = cms.Service("Timing")

process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
'''
process.siliconRecHits= cms.Path(process.siPixelRecHits*process.siStripMatchedRecHits)
#process.raw2digi_step = cms.Path(process.RawToDigi)
#process.localtracker  = cms.Path(process.trackerlocalreco)
process.reMuons       = cms.Path(process.reMuonRecoPbPb)

process.endjob_step   = cms.Path(process.endOfProcess)
process.out_step      = cms.EndPath(process.output)

#process.raw2digi_step,process.localtracker,
process.schedule = cms.Schedule(process.siliconRecHits,process.reMuons)
process.schedule.extend([process.endjob_step,process.out_step])


from Configuration.PyReleaseValidation.ConfigBuilder import MassReplaceInputTag
MassReplaceInputTag(process)
