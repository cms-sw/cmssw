## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *

## to run in un-scheduled mode uncomment the following lines
process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")

process.options.allowUnscheduled = cms.untracked.bool(True)
process.Tracer = cms.Service("Tracer")

## ------------------------------------------------------
#  NOTE: you can use a bunch of core tools of PAT to
#  taylor your PAT configuration; for a few examples
#  uncomment the lines below
## ------------------------------------------------------
#from PhysicsTools.PatAlgos.tools.coreTools import *

## remove MC matching from the default sequence
# removeMCMatching(process, ['Muons'])
# runOnData(process)

## remove certain objects from the default sequence
# removeAllPATObjectsBut(process, ['Muons'])
# removeSpecificPATObjects(process, ['Electrons', 'Muons', 'Taus'])




#add pat conversions
process.patConversions = cms.EDProducer("PATConversionProducer",
    # input collection
    #electronSource = cms.InputTag("gsfElectrons"),
    electronSource = cms.InputTag("selectedPatElectrons")
    # this should be your last selected electron collection name since currently index is used to match with electron later. We can fix this using reference pointer. ,
)

process.mvaTrigNoIPPAT = cms.EDProducer("ElectronPATIdMVAProducer",
                                    verbose = cms.untracked.bool(True),
                                    electronTag = cms.InputTag('selectedPatElectrons'),
                                    method = cms.string("BDT"),
                                    Rho = cms.InputTag("kt6PFJets", "rho"),
                                    mvaWeightFile = cms.vstring(
    "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_TrigNoIPV0_2012_Cat1.weights.xml",
    "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_TrigNoIPV0_2012_Cat2.weights.xml",
    "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_TrigNoIPV0_2012_Cat3.weights.xml",
    "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_TrigNoIPV0_2012_Cat4.weights.xml",
    "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_TrigNoIPV0_2012_Cat5.weights.xml",
    "EgammaAnalysis/ElectronTools/data/Electrons_BDTG_TrigNoIPV0_2012_Cat6.weights.xml",
    ),
                                    Trig = cms.bool(True),
                                    NoIP = cms.bool(True),
                                    )


process.out.outputCommands.append( 'keep *_patConversions_*_*' )
process.out.outputCommands.append( 'keep *_mvaTrigNoIPPAT_*_*' )

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS1')
#                                         ##
#from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarPileUpGENSIMRECO
#process.source.fileNames = filesRelValTTbarPileUpGENSIMRECO # currently not available at CERN
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValProdTTbarAODSIM
process.source.fileNames = filesRelValProdTTbarAODSIM
#                                         ##
process.maxEvents.input = 100
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'patToPat_electronId.root'
#                                         ##
#   process.options.wantSummary = True    ##  (to suppress the long output at the end of the job)
