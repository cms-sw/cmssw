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


process.load("PhysicsTools.PatExamples.patJPsiProducer_cfi")

# let it run
process.p = cms.Path(
    process.patDefaultSequence*
    process.patJPsiCandidates
    )

# In addition you usually want to change the following parameters:
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
process.source.fileNames = [
    '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V7-v1/0002/BA958CA5-B19B-DE11-90C6-0018F3D0961A.root'
    ]
#   process.maxEvents.input = ...         ##  (e.g. -1 to run on all events)
process.out.dropMetaData = cms.untracked.string("DROPPED")
process.out.outputCommands += ['keep *_patJPsiCandidates_*_*']
process.out.fileName = 'jpsi.root'
process.options.wantSummary = True        ##  (to suppress the long output at the end of the job)    


