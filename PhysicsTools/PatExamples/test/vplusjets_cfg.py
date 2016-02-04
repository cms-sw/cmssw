# This is an example PAT configuration showing the usage of PAT on full sim samples

# Starting with a skeleton process which gets imported with the following line
from PhysicsTools.PatAlgos.patTemplate_cfg import *

# note that you can use a bunch of core tools of PAT 
# to taylor your PAT configuration; for a few examples
# uncomment the following lines

from PhysicsTools.PatAlgos.tools.coreTools import *
#removeMCMatching(process, 'Muons')
#removeAllPATObjectsBut(process, ['Muons'])
#removeSpecificPATObjects(process, ['Electrons', 'Muons', 'Taus'])

# add the trigger information to the configuration
from PhysicsTools.PatAlgos.tools.trigTools import *
switchOnTrigger( process )
from PhysicsTools.PatAlgos.patEventContent_cff import patTriggerEventContent

# add the flavor history
process.load("PhysicsTools.HepMCCandAlgos.flavorHistoryPaths_cfi")

# switch from "ak5" to "antikt5" if running on 31x samples. 
run33xOn31xMC = False
if run33xOn31xMC :
    switchJetCollection(process, 
                    cms.InputTag('antikt5CaloJets'),   
                    doJTA            = True,            
                    doBTagging       = True,            
                    jetCorrLabel     = ('AK5','Calo'),  
                    doType1MET       = True,            
                    genJetCollection = cms.InputTag("antikt5GenJets")
                    )
    process.cFlavorHistoryProducer.matchedSrc = cms.InputTag("antikt5GenJets")
    process.bFlavorHistoryProducer.matchedSrc = cms.InputTag("antikt5GenJets")


# shrink the event content
# jets
process.allLayer1Jets.tagInfoSources  = cms.VInputTag(
        cms.InputTag("secondaryVertexTagInfos")
    )
process.selectedLayer1Jets.cut = cms.string("pt > 20 & abs(eta) < 5")
process.allLayer1Jets.embedGenJetMatch = cms.bool(False)
# electrons
process.allLayer1Electrons.isoDeposits = cms.PSet()
process.allLayer1Electrons.embedGsfTrack = cms.bool(False)
process.allLayer1Electrons.embedSuperCluster = cms.bool(False)
# muons
process.allLayer1Muons.isoDeposits = cms.PSet()
# photons
process.allLayer1Photons.isoDeposits = cms.PSet()
process.allLayer1Photons.embedSuperCluster = cms.bool(False)
#taus
process.allLayer1Taus.isoDeposits = cms.PSet()

# let it run
process.p = cms.Path(
    process.flavorHistorySeq *
    process.patDefaultSequence
    )

# In addition you usually want to change the following parameters:
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
process.source.fileNames = [
    '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V7-v1/0002/BA958CA5-B19B-DE11-90C6-0018F3D0961A.root'
    ]
#   process.maxEvents.input = ...         ##  (e.g. -1 to run on all events)
process.out.outputCommands += ['keep *_flavorHistoryFilter_*_*']
process.out.outputCommands += patTriggerEventContent
process.out.dropMetaData = cms.untracked.string("DROPPED")
#process.out.outputCommands += patTriggerStandAloneEventContent
process.out.fileName = 'vplusjets.root'
process.options.wantSummary = True        ##  (to suppress the long output at the end of the job)    
