import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("USER")
process.maxEvents = cms.untracked.PSet(
      input = cms.untracked.int32(-1)
)

process.load("ElectroWeakAnalysis.WMuNu.wmunusProducer_cfi")

process.source = cms.Source("PoolSource",
      fileNames = cms.untracked.vstring(
#      'file:/data1/degrutto/CMSSW_3_5_6/src/ElectroWeakAnalysis/Skimming/test/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133483_3.root'
#   'file:/ciet3a/data4/EWK_SubSkim_Summer09_7TeV/WMuNu_7TeV_10invpb_1.root',
     ),
      inputCommands = cms.untracked.vstring(
      'keep *',
      'drop *_MEtoEDMConverter_*_*'
      )
)

import os,re
#file_directory = "/data4/Skimming/SkimResults/133/"
#file_directory="/data4/InclusiveMu15_Summer09-MC_31X_V3_AODSIM-v1/0024/"
#file_directory=/ciet3a/data4/EWK_SubSkim_Summer09_7TeV/"
file_directory="/ciet3b/data4/Spring10_10invpb_AODRED/" 
for file in os.listdir(file_directory):
     match = re.search(r'(InclusiveMu15)', file)
     if not match: continue
     process.source.fileNames.append("file:" + file_directory + "/" + file)




# Debug/info printouts
process.MessageLogger = cms.Service("MessageLogger",
      debugModules = cms.untracked.vstring('correlcorMet'),
      cout = cms.untracked.PSet(
             default = cms.untracked.PSet( limit = cms.untracked.int32(20) ),
             threshold = cms.untracked.string('ERROR')
       #      threshold = cms.untracked.string('DEBUG')
      ),
      destinations = cms.untracked.vstring('cout')
)





process.PRESEL_corMet = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      WMuNuCollectionTag = cms.untracked.InputTag("corMetWMuNus"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),

      PtCut = cms.untracked.double(0),
      EtaCut = cms.untracked.double(10),
      MtMin = cms.untracked.double(0.0),
      MtMax = cms.untracked.double(1000.0),
      AcopCut = cms.untracked.double(2.0),
)

process.SEL_corMet = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      WMuNuCollectionTag = cms.untracked.InputTag("corMetWMuNus"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),
      AcopCut = cms.untracked.double(2.0),
#      PtCut = cms.untracked.double(20),


)

process.corMet = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      WMuNuCollectionTag = cms.untracked.InputTag("corMetWMuNus"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),
      AcopCut = cms.untracked.double(2.0),
#      PtCut = cms.untracked.double(20),
)



process.PRESEL_pfMet = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      WMuNuCollectionTag = cms.untracked.InputTag("pfMetWMuNus"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),
      PtCut = cms.untracked.double(0),
      EtaCut = cms.untracked.double(10),
      MtMin = cms.untracked.double(0.0),
      MtMax = cms.untracked.double(1000.0),
      AcopCut = cms.untracked.double(2.0),

)

process.SEL_pfMet = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      WMuNuCollectionTag = cms.untracked.InputTag("pfMetWMuNus"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),
      AcopCut = cms.untracked.double(2.0),
#      PtCut = cms.untracked.double(20),

)

process.pfMet = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      WMuNuCollectionTag = cms.untracked.InputTag("pfMetWMuNus"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),
      AcopCut = cms.untracked.double(2.0),
#      PtCut = cms.untracked.double(20),

)


process.PRESEL_tcMet = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      WMuNuCollectionTag = cms.untracked.InputTag("tcMetWMuNus"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),

      PtCut = cms.untracked.double(0),
      EtaCut = cms.untracked.double(10),
      MtMin = cms.untracked.double(0.0),
      MtMax = cms.untracked.double(1000.0),
      AcopCut = cms.untracked.double(2.0),

)

process.SEL_tcMet = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      WMuNuCollectionTag = cms.untracked.InputTag("tcMetWMuNus"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),
      AcopCut = cms.untracked.double(2.0),
#      PtCut = cms.untracked.double(20),

)

process.tcMet = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      WMuNuCollectionTag = cms.untracked.InputTag("tcMetWMuNus"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),
      AcopCut = cms.untracked.double(2.0),
#      PtCut = cms.untracked.double(20),

)

# Create a new reco::Muon collection with PFLow Iso information
process.muonsWithPFIso = cms.EDFilter("MuonWithPFIsoProducer",
        MuonTag = cms.untracked.InputTag("muons")
      , PfTag = cms.untracked.InputTag("particleFlow")
      , UsePfMuonsOnly = cms.untracked.bool(False)
      , TrackIsoVeto = cms.untracked.double(0.01)
      , GammaIsoVeto = cms.untracked.double(0.07)
      , NeutralHadronIsoVeto = cms.untracked.double(0.1)
)

process.pfWMuNus = cms.EDProducer("WMuNuProducer",
      # Input collections ->
      MuonTag = cms.untracked.InputTag("muonsWithPFIso"),
      METTag = cms.untracked.InputTag("pfMet")
)


process.pfMetAndMuons = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      MuonTag = cms.untracked.InputTag("muonsWithPFIso"),
      WMuNuCollectionTag = cms.untracked.InputTag("pfWMuNus"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),
      AcopCut = cms.untracked.double(2.0),
)

process.load("ElectroWeakAnalysis.WMuNu.wmunusValidation_cfi") #load validation sequence (for WMunu & ZMuMu)
process.wmnVal_pfMet.JetTag = cms.untracked.InputTag("ak5CaloJets")
process.wmnVal_corMet.JetTag = cms.untracked.InputTag("ak5CaloJets") 
process.wmnVal_tcMet.JetTag = cms.untracked.InputTag("ak5CaloJets") 
process.wmnVal_pfMetAndMuons=process.wmnVal_pfMet.clone()
process.wmnVal_pfMetAndMuons.JetTag = cms.untracked.InputTag("ak5CaloJets")
process.wmnVal_pfMetAndMuons.MuonTag=cms.untracked.InputTag("muonsWithPFIso")


process.TFileService = cms.Service("TFileService", fileName = cms.string('QCD_10pb.root') )


# Steering the process
process.path=cms.Path(process.muonsWithPFIso)
process.path1 = cms.Path(process.wmnVal_corMet)
process.path2 = cms.Path(process.wmnVal_pfMet)
process.path3 = cms.Path(process.wmnVal_tcMet)
process.path4 = cms.Path(process.wmnVal_pfMetAndMuons)

process.path5 = cms.Path(process.corMetWMuNus+process.corMet)
process.path6 = cms.Path(process.pfMetWMuNus+process.pfMet)
process.path7 = cms.Path(process.tcMetWMuNus+process.tcMet)
process.path8 = cms.Path(process.pfWMuNus*process.pfMetAndMuons)





