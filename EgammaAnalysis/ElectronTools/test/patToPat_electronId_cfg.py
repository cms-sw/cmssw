## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *


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
    electronSource = cms.InputTag("cleanPatElectrons")  
    # this should be your last selected electron collection name since currently index is used to match with electron later. We can fix this using reference pointer. ,
)

process.mvaTrigNoIPPAT = cms.EDProducer("ElectronPATIdMVAProducer",
                                    verbose = cms.untracked.bool(False),
                                    electronTag = cms.InputTag('cleanPatElectrons'),
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



## let it run
process.p = cms.Path(
    process.patDefaultSequence+
    process.patConversions+
    process.mvaTrigNoIPPAT
    )

## process.out.outputCommands +=[
##      'keep *_patConversions*_*_*'
## ]
## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
process.source.fileNames = [          ##
    '/store/mc/Summer12_DR53X/DYJetsToLL_M-50_TuneZ2Star_8TeV-madgraph-tarball/AODSIM/PU_S10_START53_V7A-v1/0001/FE4B9392-D8D3-E111-8789-0025B3E05D8C.root'
    ]                                     ##  (e.g. 'file:AOD.root')
#                                         ##
#   process.maxEvents.input = ...         ##  (e.g. -1 to run on all events)
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
#   process.out.fileName = ...            ##  (e.g. 'myTuple.root')
#                                         ##
#   process.options.wantSummary = True    ##  (to suppress the long output at the end of the job)    
