# official example for PF2PAT

import FWCore.ParameterSet.Config as cms

process = cms.Process("ELECISO")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
      '/store/relval/CMSSW_3_9_5/RelValTTbar/GEN-SIM-RECO/START39_V6-v1/0008/0AEEDFA4-88FA-DF11-B6FF-001A92811718.root'
    )
)


# path ---------------------------------------------------------------

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.load("RecoParticleFlow.PFProducer.pfBasedElectronIso_cff")


process.p = cms.Path(
    process.pfBasedElectronIsoSequence
    )

# output ------------------------------------------------------------

#process.load("FastSimulation.Configuration.EventContent_cff")
process.load("Configuration.EventContent.EventContent_cff")

process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      'keep *_*_*_ELECISO'),
                               fileName = cms.untracked.string('electronIsolation.root')
)



process.outpath = cms.EndPath(
    process.out 
#    process.aod
    )


# other stuff

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10

# the following are necessary for taus:

#process.load("Configuration.StandardSequences.GeometryPilot2_cff")
#process.load("Configuration.StandardSequences.MagneticField_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# do not forget to set the global tag according to the
# release you are using and the sample you are reading (data or MC)
# global tags can be found here:
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions#Valid_Global_Tags_by_Release
#process.GlobalTag.globaltag = cms.string('GR09_R_34X_V2::All')
