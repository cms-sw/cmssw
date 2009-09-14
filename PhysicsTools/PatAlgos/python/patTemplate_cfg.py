import FWCore.ParameterSet.Config as cms

process = cms.Process("PAT")

# initialize MessageLogger and output report
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_3_0_pre2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V7-v1/0002/BA958CA5-B19B-DE11-90C6-0018F3D0961A.root'
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('STARTUP31X_V1::All')
process.load("Configuration.StandardSequences.MagneticField_cff")


# load the standard PAT config 
process.load("PhysicsTools.PatAlgos.patSequences_cff")

from PhysicsTools.PatAlgos.tools.jetTools import *
print "*********************************************************************"
print "Switching all processes to use the anti-kT algorithm by default."
print "Switch the jet collection to your desired algorithm if this is not"
print "what you want to use."
print "*********************************************************************"
switchJetCollection(process, 
                    cms.InputTag('ak5CaloJets'),   
                    doJTA            = True,            
                    doBTagging       = True,            
                    jetCorrLabel     = ('AK5','Calo'),  
                    doType1MET       = True,            
                    genJetCollection = cms.InputTag("ak5GenJets")
                    ) 

# Output module configuration
from PhysicsTools.PatAlgos.patEventContent_cff import patEventContent
process.out = cms.OutputModule("PoolOutputModule",
                                   fileName = cms.untracked.string('PATLayer1_Output.fromAOD_full.root'),
                                   # save only events passing the full path
                                   SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
                                   # save PAT Layer 1 output
                                   outputCommands = cms.untracked.vstring('drop *', *patEventContent ) # you need a '*' to unpack the list of commands 'patEventContent'
                               )
process.outpath = cms.EndPath(process.out)
