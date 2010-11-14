#
#  SUSY-PAT configuration file
#
#  PAT configuration for the SUSY group - 38X series
#  More information here:
#  https://twiki.cern.ch/twiki/bin/view/CMS/SusyPatLayer1DefV9
#

# Starting with a skeleton process which gets imported with the following line
from PhysicsTools.PatAlgos.patTemplate_cfg import *

#-- Meta data to be logged in DBS ---------------------------------------------
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.33 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/PhysicsTools/Configuration/test/SUSY_pattuple_cfg.py,v $'),
    annotation = cms.untracked.string('SUSY pattuple definition')
)

#-- Message Logger ------------------------------------------------------------
process.MessageLogger.categories.append('PATSummaryTables')
process.MessageLogger.cerr.PATSummaryTables = cms.untracked.PSet(
    limit = cms.untracked.int32(-1),
    reportEvery = cms.untracked.int32(1)
    )
process.MessageLogger.cerr.FwkReport.reportEvery = 100

#-- VarParsing ----------------------------------------------------------------
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing ('standard')

options.output = "SUSYPAT.root"
options.maxEvents = 100
#  for SusyPAT configuration
options.register('GlobalTag', "", VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "GlobalTag to use (if empty default Pat GT is used)")
options.register('mcInfo', True, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "process MonteCarlo data")
options.register('jetCorrections', 'L2Relative', VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.string, "Level of jet corrections to use: Note the factors are read from DB via GlobalTag")
options.jetCorrections.append('L3Absolute')
options.register('hltName', 'HLT', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "HLT menu to use for trigger matching")
options.register('mcVersion', '', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "'35X' for samples from Spring10 production, not needed for new productions")
options.register('jetTypes', 'AK5JPT', VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.string, "Additional jet types that will be produced (AK5Calo and AK5PF, cross cleaned in PF2PAT, are included anyway)")
options.register('hltSelection', '', VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.string, "hlTriggers (OR) used to filter events")
options.register('doValidation', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Include the validation histograms from SusyDQM (needs extra tags)")
options.register('doExtensiveMatching', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Matching to simtracks (needs extra tags)")
options.register('doSusyTopProjection', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Apply Susy selection in PF2PAT to obtain lepton cleaned jets (needs validation)")
options.register('electronHLTMatches', 'HLT_Ele15_LW_L1R', VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.string, "HLT paths matched to electrons")
options.electronHLTMatches.append('HLT_Ele15_SW_L1R')
options.electronHLTMatches.append('HLT_Ele15_SW_CaloEleId_L1R')
options.electronHLTMatches.append('HLT_Ele15_SW_EleId_L1R')
options.electronHLTMatches.append('HLT_Ele17_SW_TightEleId_L1R')
options.electronHLTMatches.append('HLT_Ele17_SW_TighterEleId_L1R_v1')
options.electronHLTMatches.append('HLT_Ele22_SW_TighterCaloIdIsol_L1R_v1')
options.electronHLTMatches.append('HLT_Ele22_SW_TighterCaloIdIsol_L1R_v2')
options.electronHLTMatches.append('HLT_Ele22_SW_TighterEleId_L1R_v1')
options.electronHLTMatches.append('HLT_Ele22_SW_TighterEleId_L1R_v2')
options.electronHLTMatches.append('HLT_Ele22_SW_TighterEleId_L1R_v3')
options.register('muonHLTMatches', 'HLT_Mu15_v1', VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.string, "HLT paths matched to muons")
options.muonHLTMatches.append('HLT_Mu_9')
options.muonHLTMatches.append('HLT_Mu_11')
options.register('tauHLTMatches', 'HLT_Jet15U', VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.string, "HLT paths matched to taus")
options.tauHLTMatches.append('HLT_Jet30U')
options.tauHLTMatches.append('HLT_Jet50U')
options.register('photonHLTMatches', 'HLT_Photon30', VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.string, "HLT paths matched to photons")
options.register('jetHLTMatches', 'HLT_Jet15U', VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.string, "HLT paths matched to jets")
options.jetHLTMatches.append('HLT_Jet30U')
options.jetHLTMatches.append('HLT_Jet50U')
options.register('addKeep', '', VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.string, "Additional keep and drop statements to trim the event content")

#---parse user input
options.parseArguments()
options._tagOrder =[]

#-- Input Source --------------------------------------------------------------
if options.files:
    process.source.fileNames = cms.untracked.vstring (options.files)
#process.source.fileNames = [
#     '/store/relval/CMSSW_3_7_0_pre5/RelValProdTTbar/GEN-SIM-RECO/MC_37Y_V4-v1/0023/BA92C6D3-8863-DF11-B3AF-002618943939.root'
#    ]

process.maxEvents.input = options.maxEvents
# Due to problem in production of LM samples: same event number appears multiple times
process.source.duplicateCheckMode = cms.untracked.string('noDuplicateCheck')

#-- Calibration tag -----------------------------------------------------------
if options.GlobalTag:
    process.GlobalTag.globaltag = options.GlobalTag

############################# START SUSYPAT specifics ####################################
from PhysicsTools.Configuration.SUSY_pattuple_cff import addDefaultSUSYPAT, getSUSY_pattuple_outputCommands
#Apply SUSYPAT
addDefaultSUSYPAT(process,options.mcInfo,options.hltName,options.jetCorrections,options.mcVersion,options.jetTypes,options.doValidation,options.doExtensiveMatching,options.doSusyTopProjection,options.electronHLTMatches,options.muonHLTMatches,options.tauHLTMatches,options.jetHLTMatches,options.photonHLTMatches)
SUSY_pattuple_outputCommands = getSUSY_pattuple_outputCommands( process )
############################## END SUSYPAT specifics ####################################

#-- HLT selection ------------------------------------------------------------
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
if options.hltSelection:
    process.hltFilter = hlt.hltHighLevel.clone(
        HLTPaths = cms.vstring(options.hltSelection),
        TriggerResultsTag = cms.InputTag("TriggerResults","",options.hltName),
        throw = False
    )
    process.susyPatDefaultSequence.replace(process.eventCountProducer, process.eventCountProducer * process.hltFilter)

#-- Output module configuration -----------------------------------------------
process.out.fileName = options.output

# Custom settings
process.out.splitLevel = cms.untracked.int32(99)  # Turn on split level (smaller files???)
process.out.overrideInputFileSplitLevels = cms.untracked.bool(True)
process.out.dropMetaData = cms.untracked.string('DROPPED')   # Get rid of metadata related to dropped collections
process.out.outputCommands = cms.untracked.vstring('drop *', *SUSY_pattuple_outputCommands )
if options.hltSelection:
    process.out.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("p"))
if options.addKeep:
    process.out.outputCommands.extend(options.addKeep)

#-- Execution path ------------------------------------------------------------
# Full path
process.p = cms.Path( process.susyPatDefaultSequence )
#-- Dump config ------------------------------------------------------------
file = open('SusyPAT_cfg.py','w')
file.write(str(process.dumpPython()))
file.close()
