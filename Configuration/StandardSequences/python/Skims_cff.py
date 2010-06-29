import FWCore.ParameterSet.Config as cms


#from DPGAnalysis.Skims.MinBiasPDSkim_cfg import SkimCfg
from Configuration.EventContent.EventContent_cff import FEVTEventContent
skimContent = FEVTEventContent.clone()
skimContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimContent.outputCommands.append("drop *_*_*_SKIM")

#############
from FWCore.Modules.logErrorFilter_cfi import *
from Configuration.StandardSequences.RawToDigi_Data_cff import gtEvmDigis
stableBeam = cms.EDFilter("HLTBeamModeFilter",
                          L1GtEvmReadoutRecordTag = cms.InputTag("gtEvmDigis"),
                          AllowedBeamMode = cms.vuint32(11)
                          )

logerrorpath=cms.Path(gtEvmDigis+stableBeam+logErrorFilter)

SKIMStreamLogerror = cms.FilteredStream(
    responsible = 'reco convener',
    name = 'logerror',
    paths = (logerrorpath),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(SelectEvents=cms.vstring('logerrorpath')),
    dataTier = cms.untracked.string('RAW-RECO')
    )


##############
from HLTrigger.special.hltPhysicsDeclared_cfi import *
hltPhysicsDeclared.L1GtReadoutRecordTag = 'gtDigis'
hltbeamgas = cms.EDFilter("HLTHighLevel",
                          TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
                          HLTPaths = cms.vstring('HLT_L1_BptxXOR_BscMinBiasOR'), # provide list of HLT paths (or patterns) you want
                          eventSetupPathsKey = cms.string(''),
                          andOr              = cms.bool(True),
                          throw              = cms.bool(False)
                          )
pfgskim3noncross = cms.Path(hltPhysicsDeclared*hltbeamgas)

SKIMStreamBEAMBKGV3 = cms.FilteredStream(
    responsible = 'PFG',
    name = 'BEAMBKGV3',
    paths = (pfgskim3noncross),
    content = skimContent.outputCommands,
    selectEvents = cms.untracked.PSet(SelectEvents=cms.vstring('pfgskim3noncross')),
    dataTier = cms.untracked.string('RAW-RECO')
    )

###########
    


