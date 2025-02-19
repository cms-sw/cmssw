import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')
process.GlobalTag.globaltag = 'GR10_E_V6::All'

process.load("RecoMuon.MuonIdentification.links_cfi")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    # categories = cms.untracked.vstring('MuonIdentification','TrackAssociator'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        # threshold = cms.untracked.string('DEBUG'),
	# noTimeStamps = cms.untracked.bool(True),
	# noLineBreaks = cms.untracked.bool(True)
	DEBUG = cms.untracked.PSet(
           limit = cms.untracked.int32(0)
	   ),
	#MuonIdentification = cms.untracked.PSet(
	#   limit = cms.untracked.int32(-1)
	#),
	#TrackAssociator = cms.untracked.PSet(
	#   limit = cms.untracked.int32(-1)
	#),
    ),
    debugModules = cms.untracked.vstring("muons")
)

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
options = cms.untracked.PSet(
    Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Commissioning10/MinimumBias/RECO/v9/000/134/704/F6771A49-8455-DF11-8FA7-00304879EDEA.root'
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('out2.root'),
)

process.load("RecoMuon.MuonIdentification.cosmics_id")
process.p = cms.Path(process.globalMuonLinks*process.muons*process.cosmicsMuonIdSequence)

process.muons.inputCollectionLabels = cms.VInputTag(cms.InputTag("generalTracks"),
						    cms.InputTag("globalMuonLinks"), 
						    cms.InputTag("standAloneMuons","UpdatedAtVtx"))
process.muons.inputCollectionTypes = cms.vstring('inner tracks', 
						 'links', 
						 'outer tracks')

process.muons.fillIsolation = False
# process.muons.minPt = 0.
# process.muons.minP = 0.

process.e = cms.EndPath(process.out)
