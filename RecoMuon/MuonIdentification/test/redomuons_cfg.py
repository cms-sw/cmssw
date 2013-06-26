import FWCore.ParameterSet.Config as cms
process = cms.Process("MUO")
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
 

# process.load('Configuration.StandardSequences.Services_cff')
# process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
# process.load('FWCore.MessageService.MessageLogger_cfi')
# process.load('Configuration.EventContent.EventContentHeavyIons_cff')
# process.load('SimGeneral.MixingModule.mixNoPU_cfi')
# process.load('Configuration.StandardSequences.GeometryDB_cff')
# process.load('Configuration.StandardSequences.MagneticField_38T_cff')
# process.load('Configuration.StandardSequences.RawToDigi_cff')
# process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
# process.load('Configuration.StandardSequences.EndOfProcess_cff')
# process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')




process.GlobalTag.globaltag = 'START44_V7::All'


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    "/store/relval/CMSSW_4_4_0/RelValZMM/GEN-SIM-RECO/START44_V5-v2/0045/7C73A5A1-0AE6-E011-992F-00261894380A.root"
    )
)

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('/tmp/bellan/MUO.root'))


process.load("RecoMuon.MuonIdentification.links_cfi")
process.load("RecoMuon.MuonIdentification.muonIdProducerSequence_cff")


process.muons1stStep.inputCollectionLabels = cms.VInputTag(cms.InputTag("generalTracks"),
						    cms.InputTag("globalMuonLinks"),
                                                           
						    cms.InputTag("standAloneMuons","UpdatedAtVtx"))
process.muons1stStep.inputCollectionTypes = cms.vstring('inner tracks', 
                                                        'links', 
                                                        'outer tracks')

process.muons.PFCandidates = cms.InputTag("particleFlow")
process.muons.FillPFIsolation = False
process.muons.FillSelectorMaps = False 
process.muons.FillCosmicsIdMap =  False
process.muons.FillTimingInfo = False
process.muons.FillDetectorBasedIsolation = False 
process.muons.FillShoweringInfo = False
   
process.p = cms.Path(process.globalMuonLinks*
                     process.muons1stStep *
                     process.muons)

process.e = cms.EndPath(process.out)
