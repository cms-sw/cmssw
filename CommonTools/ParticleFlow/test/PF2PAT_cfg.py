# official example for PF2PAT

import FWCore.ParameterSet.Config as cms

process = cms.Process("PF2PAT")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)


# the following line does not work anymore:
# process.load("CommonTools.ParticleFlow.Sources/Data/source_124120_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
      # you need to be a CAF user to be able to use these files. They're good to test pile-up
      '/store/data/Run2011A/HT/RAW-RECO/HighMET-PromptSkim-v4/0000/6C54BC2D-B194-E011-9D43-0026189438BD.root',
      '/store/data/Run2011A/HT/RAW-RECO/HighMET-PromptSkim-v4/0000/6A651E6D-898E-E011-A520-0018F3D096DC.root',
      '/store/data/Run2011A/HT/RAW-RECO/HighMET-PromptSkim-v4/0000/6215E4FB-5796-E011-B8EB-002618943831.root',
      '/store/data/Run2011A/HT/RAW-RECO/HighMET-PromptSkim-v4/0000/620A0295-BB95-E011-8351-002618943976.root',
      '/store/data/Run2011A/HT/RAW-RECO/HighMET-PromptSkim-v4/0000/6007B811-6C8F-E011-9E4C-00304867904E.root',
      '/store/data/Run2011A/HT/RAW-RECO/HighMET-PromptSkim-v4/0000/5E284375-D496-E011-908C-002618943919.root',
      '/store/data/Run2011A/HT/RAW-RECO/HighMET-PromptSkim-v4/0000/5E1E3DA5-0A96-E011-9150-00304867920C.root',
      '/store/data/Run2011A/HT/RAW-RECO/HighMET-PromptSkim-v4/0000/58932B3B-868B-E011-9F16-003048678AE4.root',
      '/store/data/Run2011A/HT/RAW-RECO/HighMET-PromptSkim-v4/0000/56B040A4-5C8F-E011-BCBC-0018F3D095FA.root',
      '/store/data/Run2011A/HT/RAW-RECO/HighMET-PromptSkim-v4/0000/5281ACC9-E386-E011-9789-002618943836.root',
      '/store/data/Run2011A/HT/RAW-RECO/HighMET-PromptSkim-v4/0000/5063E157-B58F-E011-B15A-0018F3D09620.root'
      # but you can always use a relval like this one:
      # '/store/relval/CMSSW_4_2_3/RelValZTT/GEN-SIM-RECO/START42_V12-v2/0062/4CEA9C47-287B-E011-BAB7-00261894396B.root')
      ))
    
     
                   
print process.source

# path ---------------------------------------------------------------


process.load("CommonTools.ParticleFlow.PF2PAT_cff")

from CommonTools.ParticleFlow.Tools.enablePileUpCorrection import enablePileUpCorrectionInPF2PAT

# the following is advocated by JetMET, but leads to include very far tracks in the no pile up collection
# enablePileUpCorrectionInPF2PAT( process, postfix='')

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.p = cms.Path(
    process.PF2PAT 
    )

# output ------------------------------------------------------------

#process.load("FastSimulation.Configuration.EventContent_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.pf2pat = cms.OutputModule("PoolOutputModule",
                                  #    process.AODSIMEventContent,
                                  outputCommands = cms.untracked.vstring('drop *'),
                                  fileName = cms.untracked.string('PF2PAT.root')
)
process.aod = cms.OutputModule("PoolOutputModule",
                               process.AODSIMEventContent,
                               #outputCommands = cms.untracked.vstring('drop *'),
                               fileName = cms.untracked.string('aod.root')
)
process.load("CommonTools.ParticleFlow.PF2PAT_EventContent_cff")
process.pf2pat.outputCommands.extend( process.PF2PATEventContent.outputCommands )
process.pf2pat.outputCommands.extend( process.PF2PATStudiesEventContent.outputCommands )


process.outpath = cms.EndPath(
    process.pf2pat 
#    process.aod
    )


# other stuff

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10

# the following are necessary for taus:

# process.load("Configuration.StandardSequences.GeometryPilot2_cff")
# process.load("Configuration.StandardSequences.MagneticField_cff")
# process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# do not forget to set the global tag according to the
# release you are using and the sample you are reading (data or MC)
# global tags can be found here:
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions#Valid_Global_Tags_by_Release
# process.GlobalTag.globaltag = cms.string('GR09_R_34X_V2::All')
