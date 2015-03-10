import FWCore.ParameterSet.Config as cms

process = cms.Process("TIMECALIBANALYSISELE")


# Output - dummy
process.out = cms.OutputModule(
    "PoolOutputModule",
    outputCommands = cms.untracked.vstring(),
    fileName = cms.untracked.string('file:pippo.root'),
    )


# gfworks: to get clustering 

# Geometry
process.load("Configuration.Geometry.GeometryIdeal_cff")

process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi") # gfwork: need this?
process.CaloTowerConstituentsMapBuilder = cms.ESProducer("CaloTowerConstituentsMapBuilder")


# pat needed to work out electron id/iso
from PhysicsTools.PatAlgos.tools.metTools import *
from PhysicsTools.PatAlgos.tools.tauTools import *
from PhysicsTools.PatAlgos.tools.jetTools import *
from PhysicsTools.PatAlgos.tools.coreTools import *
from PhysicsTools.PatAlgos.tools.pfTools import *

from PhysicsTools.PatAlgos.selectionLayer1.leptonCountFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.photonCountFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.electronCountFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.jetCountFilter_cfi import *


# Global Tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag( process.GlobalTag, 'GR_R_53_V18::All' )
# tag below tested in CMSSW_4_3_0_pre3
#process.GlobalTag.globaltag = 'GR_R_42_V14::All'

# this is for jan16 reprocessing - tested in CMSSW_4_3_0_pre3
#process.GlobalTag.globaltag = 'FT_R_42_V24::All'

process.load('Configuration.StandardSequences.MagneticField_38T_cff')


# Trigger
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtBoardMapsConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v2_Unprescaled_cff")
import FWCore.Modules.printContent_cfi
process.dumpEv = FWCore.Modules.printContent_cfi.printContent.clone()

import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
process.gtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()



#------------------
#Load PAT sequences
process.load("PhysicsTools.PatAlgos.patSequences_cff")
process.load("PhysicsTools.PatAlgos.tools.pfTools")
#
## THis is NOT MC => remove matching
removeMCMatching(process, ['All'])
#
#
## bugfix for DATA Run2011 (begin)
removeSpecificPATObjects( process, ['Taus'] )
process.patDefaultSequence.remove( process.patTaus )

#
###
process.patElectrons.isoDeposits = cms.PSet()
#
process.patElectrons.addElectronID = cms.bool(True)
process.patElectrons.electronIDSources = cms.PSet(
        simpleEleId95relIso= cms.InputTag("simpleEleId95relIso"),
            simpleEleId90relIso= cms.InputTag("simpleEleId90relIso"),
            simpleEleId85relIso= cms.InputTag("simpleEleId85relIso"),
            simpleEleId80relIso= cms.InputTag("simpleEleId80relIso"),
            simpleEleId70relIso= cms.InputTag("simpleEleId70relIso"),
            simpleEleId60relIso= cms.InputTag("simpleEleId60relIso"),
            simpleEleId95cIso= cms.InputTag("simpleEleId95cIso"),
            simpleEleId90cIso= cms.InputTag("simpleEleId90cIso"),
            simpleEleId85cIso= cms.InputTag("simpleEleId85cIso"),
            simpleEleId80cIso= cms.InputTag("simpleEleId80cIso"),
            simpleEleId70cIso= cms.InputTag("simpleEleId70cIso"),
            simpleEleId60cIso= cms.InputTag("simpleEleId60cIso"),
            )
###
process.load("ElectroWeakAnalysis.WENu.simpleEleIdSequence_cff")
process.patElectronIDs = cms.Sequence(process.simpleEleIdSequence)
process.makePatElectrons = cms.Sequence(process.patElectronIDs *
                                        process.patElectrons)
process.makePatCandidates = cms.Sequence( process.makePatElectrons   )
process.patMyDefaultSequence = cms.Sequence(process.makePatCandidates)



# this is the ntuple producer
process.load("CalibCalorimetry.EcalTiming.ecalTimeEleTree_cfi")
process.ecalTimeEleTree.OutfileName = 'EcalTimeTree'
process.ecalTimeEleTree.muonCollection = cms.InputTag("muons")
process.ecalTimeEleTree.runNum = 999999
#process.ecalTimeTree.endcapSuperClusterCollection = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower","")



process.dumpEvContent = cms.EDAnalyzer("EventContentAnalyzer")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.p = cms.Path(
    process.patMyDefaultSequence *
    # process.dumpEvContent  *
    process.ecalTimeEleTree
    )

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 250

# dbs search --query "find file where dataset=/ExpressPhysics/BeamCommissioning09-Express-v2/FEVT and run=124020" | grep store | awk '{printf "\"%s\",\n", $1}'
process.source = cms.Source(
    "PoolSource",
    skipEvents = cms.untracked.uint32(0),

    fileNames = (cms.untracked.vstring(
    #'/store/data/Commissioning10/MinimumBias/RAW-RECO/v9/000/135/494/A4C5C9FA-C462-DF11-BC35-003048D45F7A.root',
    #'/store/relval/CMSSW_4_2_0_pre8/EG/RECO/GR_R_42_V7_RelVal_wzEG2010A-v1/0043/069662C9-9A56-E011-9741-0018F3D096D2.root'
    #'/store/data/Run2010A/EG/RECO/v4/000/144/114/EEC21BFA-25B4-DF11-840A-001617DBD5AC.root'

   # 'file:/data/franzoni/data/Run2011A_DoubleElectron_AOD_PromptReco-v4_000_166_946_CE9FBCFF-4B98-E011-A6C3-003048F11C58.root'
   #     'file:/hdfs/cms/phedex/store/data/Run2012C/SinglePhoton/RECO/EXODisplacedPhoton-PromptSkim-v3/000/198/941/00000/0EA7C91A-B8CF-E111-9766-002481E150EA.root'  'dcache:/pnfs/cms/WAX/11/store/data/Run2012C/SinglePhoton/RECO/22Jan2013-v1/20000/0000FDB8-836E-E211-B773-002618943854.root',
'dcache:/pnfs/cms/WAX/11/store/data/Run2012C/SinglePhoton/RECO/22Jan2013-v1/20000/00114C6A-7C6E-E211-B59A-002618943946.root',
'dcache:/pnfs/cms/WAX/11/store/data/Run2012C/SinglePhoton/RECO/22Jan2013-v1/20000/001E9308-676E-E211-B7F0-00261894385D.root',
'dcache:/pnfs/cms/WAX/11/store/data/Run2012C/SinglePhoton/RECO/22Jan2013-v1/20000/00333CC0-7D6E-E211-9FAD-0026189438F4.root',
'dcache:/pnfs/cms/WAX/11/store/data/Run2012C/SinglePhoton/RECO/22Jan2013-v1/20000/00393040-776E-E211-904B-002618943809.root',
'dcache:/pnfs/cms/WAX/11/store/data/Run2012C/SinglePhoton/RECO/22Jan2013-v1/20000/0041AD3C-6E6E-E211-94FE-003048678FEA.root',
'dcache:/pnfs/cms/WAX/11/store/data/Run2012C/SinglePhoton/RECO/22Jan2013-v1/20000/0049FA8E-676E-E211-AA54-0026189438C4.root',
'dcache:/pnfs/cms/WAX/11/store/data/Run2012C/SinglePhoton/RECO/22Jan2013-v1/20000/006DFDA5-7E6E-E211-A274-00304867BFBC.root',
'dcache:/pnfs/cms/WAX/11/store/data/Run2012C/SinglePhoton/RECO/22Jan2013-v1/20000/00884BA4-496E-E211-875C-0026189437FE.root',
'dcache:/pnfs/cms/WAX/11/store/data/Run2012C/SinglePhoton/RECO/22Jan2013-v1/20000/00A473F6-676E-E211-95BE-002618FDA208.root',
'dcache:/pnfs/cms/WAX/11/store/data/Run2012C/SinglePhoton/RECO/22Jan2013-v1/20000/00C86187-656E-E211-A50D-002354EF3BDF.root',
'dcache:/pnfs/cms/WAX/11/store/data/Run2012C/SinglePhoton/RECO/22Jan2013-v1/20000/00EC4804-746E-E211-817E-003048FFD720.root',
'dcache:/pnfs/cms/WAX/11/store/data/Run2012C/SinglePhoton/RECO/22Jan2013-v1/20000/00FC51EF-836E-E211-ACF0-003048FFCB8C.root',
'dcache:/pnfs/cms/WAX/11/store/data/Run2012C/SinglePhoton/RECO/22Jan2013-v1/20000/021A22F0-7E6E-E211-90F8-003048FFD736.root',
'dcache:/pnfs/cms/WAX/11/store/data/Run2012B/SinglePhoton/RECO/22Jan2013-v1/20000/72F87251-F871-E211-83C3-002618943838.root'
    )
                 )
    )



