import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

#This config files produces plots to debug APV shots

process = cms.Process("APVShotAnalyzer")


from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
process = cms.Process('APVShotAnalyzer',Run2_2016)

#prepare options

process.load("DQM.SiStripCommon.TkHistoMap_cff")

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "DONOTEXIST",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")


options.parseArguments()

#

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False),
    fileMode = cms.untracked.string("FULLMERGE")
    )

process.load("FWCore.MessageService.MessageLogger_cfi")


process.MessageLogger.debugModules=cms.untracked.vstring("apvshotfilter")
#------------------------------------------------------------------

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(options.inputFiles),
#                    skipBadFiles = cms.untracked.bool(True),
                    inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                    )

#--------------------------------------
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.Reconstruction_Data_cff")
process.load('Configuration.StandardSequences.L1Reco_cff')

process.froml1abcHEs = cms.EDProducer("EventWithHistoryProducerFromL1ABC",
                                      l1ABCCollection=cms.InputTag("scalersRawToDigi")
                                      )
process.load("DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi")
process.load("DPGAnalysis.SiStripTools.eventtimedistribution_cfi")

process.seqEventHistoryReco = process.seqEventHistoryReco = cms.Sequence(process.froml1abcHEs + process.APVPhases)
process.seqEventHistory = cms.Sequence(process.eventtimedistribution)

process.eventtimedistribution.historyProduct = cms.InputTag("froml1abcHEs")


process.load("DPGAnalysis.SiStripTools.digibigeventsdebugger_cfi")

process.digibigeventsdebugger.collection=cms.InputTag("siStripDigis","ZeroSuppressed")

#with the following option set to True, the configuration produces a set of histograms for each even with APV shots, setting to false it produces only one summary histograms where all the APV shots in data are folded
process.digibigeventsdebugger.singleEvents=cms.bool(True)
process.digibigeventsdebugger.selections=cms.VPSet(

#an example to consider only the module with the detId indicated (in this case 369141946), the hexadecimal number in the selection are as follow: the first is a bit mask to select single modules, the second is the detId in hexadecimal format
cms.PSet(label=cms.string("369141946"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1600A8BA"))
#cms.PSet(label=cms.string("369141949"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1600A8BD")),

#Examples below can be used to filter a single partition
#cms.PSet(label=cms.string("TIB"),selection=cms.untracked.vstring("0x1e000000-0x16000000")),
#cms.PSet(label=cms.string("TEC"),selection=cms.untracked.vstring("0x1e000000-0x1c000000")),
#cms.PSet(label=cms.string("TOB"),selection=cms.untracked.vstring("0x1e000000-0x1a000000")),
#cms.PSet(label=cms.string("TID"),selection=cms.untracked.vstring("0x1e000000-0x18000000"))

)

process.load("DPGAnalysis.SiStripTools.apvshotsfilter_cfi")
process.apvshotsfilter.useCabling = cms.untracked.bool(True)

process.load("DPGAnalysis.SiStripTools.apvshotsanalyzer_cfi")
process.apvshotsanalyzer.historyProduct = cms.InputTag("froml1abcHEs")
process.apvshotsanalyzer.useCabling = cms.untracked.bool(True)

process.load("DPGAnalysis.SiStripTools.eventtimedistribution_cfi")


import DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi 
process.APVPhases = DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi.APVPhases 

process.load("DPGAnalysis.SiStripTools.eventwithhistoryproducerfroml1abc_cfi")

process.eventtimedistributionAfter= process.eventtimedistribution.clone()


#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,options.globalTag, '')

process.p0 = cms.Path(
   process.siStripDigis + process.siStripZeroSuppression +
   process.scalersRawToDigi +
   process.seqEventHistoryReco +
   process.seqEventHistory +
   process.eventtimedistribution +
   process.apvshotsfilter +
   process.apvshotsanalyzer +
   process.eventtimedistributionAfter +
   process.digibigeventsdebugger

   )

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('APVShotAnalyzer.root')
                                   )

