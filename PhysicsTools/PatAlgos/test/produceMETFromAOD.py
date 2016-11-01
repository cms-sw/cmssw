from PhysicsTools.PatAlgos.patTemplate_cfg import cms, process

# Set the process options -- Display summary at the end, enable unscheduled execution
process.options.allowUnscheduled = cms.untracked.bool(True)
process.options.wantSummary = cms.untracked.bool(False) 

# How many events to process
process.maxEvents = cms.untracked.PSet( 
   input = cms.untracked.int32(10)
)

#configurable options =======================================================================
usePrivateSQlite=False #use external JECs (sqlite file)
useHFCandidates=True #create an additionnal NoHF slimmed MET collection if the option is set to false
redoPuppi=False # rebuild puppiMET
#===================================================================


### External JECs =====================================================================================================
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
if runOnData:
  process.GlobalTag.globaltag = autoCond['run2_data']
else:
  process.GlobalTag.globaltag = autoCond['run2_mc']

if usePrivateSQlite:
    from CondCore.DBCommon.CondDBSetup_cfi import *
    import os
    if runOnData:
      era="Summer15_25nsV6_DATA"
    else:
      era="Summer15_25nsV6_MC"
      
    process.jec = cms.ESSource("PoolDBESSource",CondDBSetup,
                               connect = cms.string( "frontier://FrontierPrep/CMS_COND_PHYSICSTOOLS"),
                               toGet =  cms.VPSet(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                tag = cms.string("JetCorrectorParametersCollection_"+era+"_AK4PF"),
                label= cms.untracked.string("AK4PF")
                ),
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                tag = cms.string("JetCorrectorParametersCollection_"+era+"_AK4PFchs"),
                label= cms.untracked.string("AK4PFchs")
                ),
            )
                               )
    process.es_prefer_jec = cms.ESPrefer("PoolDBESSource",'jec')



### =====================================================================================================
# Define the input source
fname = 'root://eoscms.cern.ch//store/relval/CMSSW_8_0_20/RelValTTbar_13/GEN-SIM-RECO/80X_mcRun2_asymptotic_2016_TrancheIV_v4_Tr4GT_v4-v1/00000/1E399A96-C47A-E611-A718-0025905B8572.root'

# Define the input source
process.source = cms.Source("PoolSource", 
    fileNames = cms.untracked.vstring([ fname ])
)


### ---------------------------------------------------------------------------
### Removing the HF from the MET computation
### ---------------------------------------------------------------------------
if not useHFCandidates:
    process.noHFCands = cms.EDFilter("CandPtrSelector",
                                     src=cms.InputTag("packedPFCandidates"),
                                     cut=cms.string("abs(pdgId)!=1 && abs(pdgId)!=2 && abs(eta)<3.0")
                                     )

#jets are rebuilt from those candidates by the tools, no need to do anything else
### =================================================================================

from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMETCorrectionsAndUncertainties

process.load("PhysicsTools.PatAlgos.producersLayer1.jetProducer_cff")
process.load("PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi")
#default configuration for miniAOD reprocessing, change the isData flag to run on data
#for a full met computation, remove the pfCandColl input
runMETCorrectionsAndUncertainties(process,
                           #isData=runOnData,
                           )

if not useHFCandidates:
  runMETCorrectionsAndUncertainties(process,
                                    #isData=runOnData,
                                    pfCandColl=cms.InputTag("noHFCands"),
                                    reclusterJets=True, #needed for NoHF
                                    recoMetFromPFCs=True, #needed for NoHF
                                    postfix="NoHF"
                                    )

if redoPuppi:
  from PhysicsTools.PatAlgos.slimming.puppiForMET_cff import makePuppiesFromMiniAOD
  makePuppiesFromMiniAOD( process );

  runMETCorrectionsAndUncertainties(process,
                             #isData=runOnData,
                             pfCandColl=cms.InputTag("puppiForMET"),
                             recoMetFromPFCs=True, 
                             reclusterJets=True,
                             jetFlavor="AK4PFPuppi",
                             postfix="Puppi"
                             )

if runOnData:
  from PhysicsTools.PatAlgos.tools.coreTools import runOnData
  runOnData( process )


process.out.outputCommands = cms.untracked.vstring( "keep *_patPFMet*_*_*",
                                                    )
process.out.fileName = cms.untracked.string('corMETMiniAOD.root')
  


process.MINIAODSIMoutput_step = cms.EndPath(process.out)
