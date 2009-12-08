import FWCore.ParameterSet.Config as cms

process = cms.Process("HITdqm")
 
process.load("FWCore.MessageService.MessageLogger_cfi")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#RelVal RAW-HLTDEBUG file, or:
'/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/3C9BA2AC-12DC-DE11-B849-001731AF6789.root',

'/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/F6CE4237-D2DB-DE11-BA20-001A92810ADE.root',

'/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/EE41D155-D3DB-DE11-9356-00304867BEE4.root',

'/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/E64B45AC-D9DB-DE11-9CB9-0026189437F9.root',

'/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/BCE1E137-D2DB-DE11-87F8-0018F3D09702.root',

'/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/96BFAC37-D2DB-DE11-9707-0018F3D09670.root',

'/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/9216F9B6-D1DB-DE11-89BD-003048678FB2.root',

'/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/82ED12BD-D1DB-DE11-ABD5-003048678CA2.root',

'/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/7C4EF951-D3DB-DE11-ADCC-0026189438A2.root'

))

process.load("DQM.HLTEvF.HLTMonHcalIsoTrack_cfi")

process.hltMonHcalIsoTrack.SaveToRootFile=cms.bool(True)
process.hltMonHcalIsoTrack.triggers=cms.VPSet(
        cms.PSet(
        triggerName=cms.string('HLT_IsoTrackHE_1E31'),
        l2collectionLabel=cms.string("hltIsolPixelTrackProdHE1E31"),
        l3collectionLabel=cms.string("hltHITIPTCorrectorHE1E31"),

        hltL3filterLabel=cms.string("hltIsolPixelTrackL3FilterHE1E31"),
        hltL2filterLabel=cms.string("hltIsolPixelTrackL2FilterHE1E31"),
        hltL1filterLabel=cms.string("hltL1sIsoTrack1E31")
        ),
        cms.PSet(
        triggerName=cms.string('HLT_IsoTrackHB_1E31'),
        l2collectionLabel=cms.string("hltIsolPixelTrackProdHB1E31"),
        l3collectionLabel=cms.string("hltHITIPTCorrectorHB1E31"),

        hltL3filterLabel=cms.string("hltIsolPixelTrackL3FilterHB1E31"),
        hltL2filterLabel=cms.string("hltIsolPixelTrackL2FilterHB1E31"),
        hltL1filterLabel=cms.string("hltL1sIsoTrack1E31")
        )
)

process.hltMonHcalIsoTrack.hltProcessName=cms.string("HLT")

process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.MEtoEDMConverter.verbose = cms.untracked.int32(1)
 
process.dqmOut = cms.OutputModule("PoolOutputModule",
      fileName = cms.untracked.string('dqmHltHIT.root'),
      outputCommands = cms.untracked.vstring("drop *", "keep *_MEtoEDMConverter_*_*")
  )
 
process.p = cms.Path(process.hltMonHcalIsoTrack+process.MEtoEDMConverter)
 
process.ep=cms.EndPath(process.dqmOut) 
