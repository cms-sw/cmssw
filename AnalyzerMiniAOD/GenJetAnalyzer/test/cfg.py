import FWCore.ParameterSet.Config as cms

processName = "JetStudy"
process = cms.Process(processName)


process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring("file:/afs/cern.ch/work/g/gimandor/private/Hmumu/DY_check_MiniAOD/DY_M-50_LO_MiniAOD.root"))
    #fileNames = cms.untracked.vstring("file:/afs/cern.ch/work/g/gimandor/private/Hmumu/DY_check_MiniAOD/DY_M-50_NLO_MiniAOD.root"))
    #fileNames = cms.untracked.vstring("file:/afs/cern.ch/work/g/gimandor/private/Hmumu/DY_check_MiniAOD/DY_M-50_NLO_MiniAOD_2.root"))  # NLO
    fileNames = cms.untracked.vstring("/store/mc/RunIISummer15GS/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/GEN-SIM/MCRUN2_71_V1_ext1-v1/20008/1E146496-C6C5-E511-A00F-02163E0167D0.root"))
    #fileNames = cms.untracked.vstring("file:/afs/cern.ch/work/g/gimandor/private/Hmumu/GenSimSample/1E146496-C6C5-E511-A00F-02163E0167D0.root"))
     #fileNames = cms.untracked.vstring("file:/afs/cern.ch/work/g/gimandor/private/Hmumu/DY_check_MiniAOD/1E146496-C6C5-E511-A00F-02163E0167D0.root"))
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(200))  



process.MyModule = cms.EDAnalyzer('GenJetAnalyzer',
)

#process.MyFilter = cms.EDFilter('VBFGenJetFilter',
#)

process.MyFilter = cms.EDFilter("VBFGenJetFilter",
  inputTag_GenJetCollection = cms.untracked.InputTag('ak4GenJets'),
  leadJetsNoLepMass  = cms.untracked.bool  ( True),  # Require j1_eta*j2_eta<0
  minLeadingJetsInvMass     = cms.untracked.double(   0.0), # Minimum dijet invariant mass
  maxLeadingJetsInvMass     = cms.untracked.double(99999.), # Maximum dijet invariant mass
)



#process.MyFilter = cms.EDFilter("VBFGenJetFilter",

  #inputTag_GenJetCollection = cms.untracked.InputTag('ak4GenJets'),

  #oppositeHemisphere = cms.untracked.bool  ( False), # Require j1_eta*j2_eta<0
  #minPt              = cms.untracked.double(    40), # Minimum dijet jet_pt
  #minEta             = cms.untracked.double(  -4.8), # Minimum dijet jet_eta
  #maxEta             = cms.untracked.double(   4.8), # Maximum dijet jet_eta
  #minInvMass         = cms.untracked.double( 1000.), # Minimum dijet invariant mass
  #maxInvMass         = cms.untracked.double(99999.), # Maximum dijet invariant mass
  #minDeltaPhi        = cms.untracked.double(  -1.0), # Minimum dijet delta phi
  #maxDeltaPhi        = cms.untracked.double(   3.2), # Maximum dijet delta phi
  #minDeltaEta        = cms.untracked.double(   3.0), # Minimum dijet delta eta
  #maxDeltaEta        = cms.untracked.double(99999.)  # Maximum dijet delta eta

#)



process.TFileService = cms.Service("TFileService",
	fileName = cms.string('test.root') )

#process.path = cms.Path(process.MyModule)
process.path = cms.Path(process.MyFilter)
