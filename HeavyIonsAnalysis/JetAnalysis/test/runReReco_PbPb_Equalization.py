# Auto generated configuration file
# using: 
# Revision: 1.381.2.28 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step4 --conditions INSERT_GT_HERE -s RAW2DIGI,L1Reco,RECO --scenario HeavyIons --datatier GEN-SIM-RECO --himix --eventcontent RECODEBUG -n 100 --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO3')

# Jet Reconstruction

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContentHeavyIons_cff')
process.load('SimGeneral.MixingModule.HiEventMixing_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
                            secondaryFileNames = cms.untracked.vstring(),
                            fileNames = cms.untracked.vstring(
                            # 'file:/afs/cern.ch/user/d/dgulhan/workDir/hiReco_RAW2DIGI_L1Reco_RECO_1000_1_S4K.root',
                            'file:/afs/cern.ch/user/d/dgulhan/workDir/public/0012F55C-97DD-E311-9781-00266CFAE144.root',
                            # 'file:/afs/cern.ch/user/d/dgulhan/workDir/big.root',
                     # "root://se1.accre.vanderbilt.edu:1095//store/hidata/HIRun2011/HIHighPt/RECO/14Mar2014-v2/00000/003B85C7-D8B9-E311-81DF-FA163EDC6F8F.root"
                                                              )
                            )


# Output definition
                                 
process.RECODEBUGEventContent.outputCommands.extend(cms.untracked.vstring('keep *'))

process.RECODEBUGoutput = cms.OutputModule("PoolOutputModule",
                                           splitLevel = cms.untracked.int32(0),
                                           eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
                                           outputCommands = process.RECODEBUGEventContent.outputCommands,
                                           fileName = cms.untracked.string('/afs/cern.ch/user/d/dgulhan/workDir/bigRECOWE.root'),
                                           dataset = cms.untracked.PSet(
    filterName = cms.untracked.string('MinBiasCollEvtSel'),
    dataTier = cms.untracked.string('GEN-SIM-RECO')
    ),
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'GR_R_53_LV6::All', '')
# Path and EndPath definitions

process.load('RecoHI.HiJetAlgos.HiRecoJets_cff')
process.load('RecoHI.HiJetAlgos.HiRecoPFJets_cff')

## background for HF/Voronoi-style subtraction
process.voronoiBackgroundCaloEqualizeR0p2= process.voronoiBackgroundCalo.clone(equalizeR=cms.double(0.2))
process.voronoiBackgroundCaloEqualizeR0p3= process.voronoiBackgroundCalo.clone(equalizeR=cms.double(0.3))
process.voronoiBackgroundCaloEqualizeR0p5= process.voronoiBackgroundCalo.clone(equalizeR=cms.double(0.5))
process.voronoiBackgroundCaloEqualizeR0p6= process.voronoiBackgroundCalo.clone(equalizeR=cms.double(0.6))
process.voronoiBackgroundCaloEqualizeR0p7= process.voronoiBackgroundCalo.clone(equalizeR=cms.double(0.7))
process.voronoiBackgroundCaloEqualizeR0p8= process.voronoiBackgroundCalo.clone(equalizeR=cms.double(0.8))

process.voronoiBackgroundCaloNE= process.voronoiBackgroundCalo.clone(doEqualize = cms.bool(False))
# process.voronoiBackgroundCaloGhost= process.voronoiBackgroundCalo.clone(doEqualize = cms.bool(False),                                                                        addNegativesFromCone = cms.bool(True))

process.voronoiBackgroundPFEqualizeR0p1= process.voronoiBackgroundPF.clone(equalizeR=cms.double(0.1))
process.voronoiBackgroundPFEqualizeR0p2= process.voronoiBackgroundPF.clone(equalizeR=cms.double(0.2))
process.voronoiBackgroundPFEqualizeR0p4= process.voronoiBackgroundPF.clone(equalizeR=cms.double(0.4))
process.voronoiBackgroundPFEqualizeR0p5= process.voronoiBackgroundPF.clone(equalizeR=cms.double(0.5))
process.voronoiBackgroundPFEqualizeR0p6= process.voronoiBackgroundPF.clone(equalizeR=cms.double(0.6))
process.voronoiBackgroundPFEqualizeR0p7= process.voronoiBackgroundPF.clone(equalizeR=cms.double(0.7))
                            
process.voronoiBackgroundPFNE = process.voronoiBackgroundPF.clone(doEqualize = cms.bool(False))
# process.voronoiBackgroundPFGhost = process.voronoiBackgroundPF.clone(doEqualize = cms.bool(False),                                                            addNegativesFromCone = cms.bool(True))


   # dropZeroTowers_(iConfig.getParameter<bool>("dropZeros")),
   # addNegativesFromCone_(iConfig.getParameter<bool>("addNegativesFromCone")),

process.akVs7CaloJetsEqualizeR0p8 = process.akVs7CaloJets.clone(bkg = cms.InputTag("voronoiBackgroundCaloEqualizeR0p8"))
process.akVs6CaloJetsEqualizeR0p7 = process.akVs6CaloJets.clone(bkg = cms.InputTag("voronoiBackgroundCaloEqualizeR0p7"))
process.akVs5CaloJetsEqualizeR0p6 = process.akVs5CaloJets.clone(bkg = cms.InputTag("voronoiBackgroundCaloEqualizeR0p6"))
process.akVs4CaloJetsEqualizeR0p5 = process.akVs4CaloJets.clone(bkg = cms.InputTag("voronoiBackgroundCaloEqualizeR0p5"))
process.akVs2CaloJetsEqualizeR0p3 = process.akVs2CaloJets.clone(bkg = cms.InputTag("voronoiBackgroundCaloEqualizeR0p3"))
process.akVs1CaloJetsEqualizeR0p2 = process.akVs1CaloJets.clone(bkg = cms.InputTag("voronoiBackgroundCaloEqualizeR0p2"))

process.akVs7PFJetsEqualizeR0p7 = process.akVs7PFJets.clone(bkg = cms.InputTag("voronoiBackgroundPFEqualizeR0p7"))
process.akVs6PFJetsEqualizeR0p6 = process.akVs6PFJets.clone(bkg = cms.InputTag("voronoiBackgroundPFEqualizeR0p6"))
process.akVs5PFJetsEqualizeR0p5 = process.akVs5PFJets.clone(bkg = cms.InputTag("voronoiBackgroundPFEqualizeR0p5"))
process.akVs4PFJetsEqualizeR0p4 = process.akVs4PFJets.clone(bkg = cms.InputTag("voronoiBackgroundPFEqualizeR0p4"))
process.akVs2PFJetsEqualizeR0p2 = process.akVs2PFJets.clone(bkg = cms.InputTag("voronoiBackgroundPFEqualizeR0p2"))
process.akVs1PFJetsEqualizeR0p1 = process.akVs1PFJets.clone(bkg = cms.InputTag("voronoiBackgroundPFEqualizeR0p1"))

process.akVs7CaloJetsNE = process.akVs7CaloJets.clone(bkg = cms.InputTag("voronoiBackgroundCaloNE"))
process.akVs6CaloJetsNE = process.akVs6CaloJets.clone(bkg = cms.InputTag("voronoiBackgroundCaloNE"))
process.akVs5CaloJetsNE = process.akVs5CaloJets.clone(bkg = cms.InputTag("voronoiBackgroundCaloNE"))
process.akVs3CaloJetsNE = process.akVs3CaloJets.clone(bkg = cms.InputTag("voronoiBackgroundCaloNE"))
process.akVs4CaloJetsNE = process.akVs4CaloJets.clone(bkg = cms.InputTag("voronoiBackgroundCaloNE"))
process.akVs2CaloJetsNE = process.akVs2CaloJets.clone(bkg = cms.InputTag("voronoiBackgroundCaloNE"))
process.akVs1CaloJetsNE = process.akVs1CaloJets.clone(bkg = cms.InputTag("voronoiBackgroundCaloNE"))

process.akVs7PFJetsNE = process.akVs7PFJets.clone(bkg = cms.InputTag("voronoiBackgroundPFNE"))
process.akVs6PFJetsNE = process.akVs6PFJets.clone(bkg = cms.InputTag("voronoiBackgroundPFNE"))
process.akVs5PFJetsNE = process.akVs5PFJets.clone(bkg = cms.InputTag("voronoiBackgroundPFNE"))
process.akVs3PFJetsNE = process.akVs3PFJets.clone(bkg = cms.InputTag("voronoiBackgroundPFNE"))
process.akVs4PFJetsNE = process.akVs4PFJets.clone(bkg = cms.InputTag("voronoiBackgroundPFNE"))
process.akVs2PFJetsNE = process.akVs2PFJets.clone(bkg = cms.InputTag("voronoiBackgroundPFNE"))
process.akVs1PFJetsNE = process.akVs1PFJets.clone(bkg = cms.InputTag("voronoiBackgroundPFNE"))

process.endjob_step = cms.EndPath(process.endOfProcess)
process.RECODEBUGoutput_step = cms.EndPath(process.RECODEBUGoutput)

process.jetsequence = cms.Sequence(
                                   process.voronoiBackgroundCaloEqualizeR0p2
                                   *process.voronoiBackgroundCaloEqualizeR0p3
                                   *process.voronoiBackgroundCaloEqualizeR0p5
                                   *process.voronoiBackgroundCaloEqualizeR0p6
                                   *process.voronoiBackgroundCaloEqualizeR0p7
                                   *process.voronoiBackgroundCaloEqualizeR0p8
                                   *process.voronoiBackgroundCaloNE
                                   *process.voronoiBackgroundPFEqualizeR0p1
                                   *process.voronoiBackgroundPFEqualizeR0p2
                                   *process.voronoiBackgroundPFEqualizeR0p4
                                   *process.voronoiBackgroundPFEqualizeR0p5
                                   # *process.voronoiBackgroundPFEqualizeR0p6
                                   # *process.voronoiBackgroundPFEqualizeR0p7
                                   *process.voronoiBackgroundPFNE
                                   *process.akVs1CaloJetsEqualizeR0p2
                                   *process.akVs2CaloJetsEqualizeR0p3
                                   *process.akVs4CaloJetsEqualizeR0p5
                                   *process.akVs5CaloJetsEqualizeR0p6
                                   *process.akVs6CaloJetsEqualizeR0p7
                                   *process.akVs7CaloJetsEqualizeR0p8
                                   *process.akVs1PFJetsEqualizeR0p1
                                   *process.akVs2PFJetsEqualizeR0p2
                                   *process.akVs4PFJetsEqualizeR0p4
                                   *process.akVs5PFJetsEqualizeR0p5
                                   *process.akVs1CaloJetsNE
                                   *process.akVs2CaloJetsNE
                                   *process.akVs3CaloJetsNE
                                   *process.akVs4CaloJetsNE
                                   *process.akVs5CaloJetsNE
                                   *process.akVs6CaloJetsNE
                                   *process.akVs7CaloJetsNE
                                   *process.akVs1PFJetsNE
                                   *process.akVs2PFJetsNE
                                   *process.akVs3PFJetsNE
                                   *process.akVs4PFJetsNE
                                   *process.akVs5PFJetsNE
                                   *process.akVs6PFJetsNE
                                   *process.akVs7PFJetsNE
                                   )

process.jetrecostep = cms.Path(process.jetsequence)

# Schedule definition
process.schedule = cms.Schedule(process.jetrecostep,process.endjob_step,process.RECODEBUGoutput_step)

process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                      oncePerEventMode=cms.untracked.bool(False))

process.Timing=cms.Service("Timing",
                           useJobReport = cms.untracked.bool(True)
                           )