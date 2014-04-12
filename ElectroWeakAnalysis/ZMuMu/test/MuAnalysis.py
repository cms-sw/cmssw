import FWCore.ParameterSet.Config as cms

process = cms.Process("MuonAnalysis")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

#"file:~/www/2010/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133483_331.root" 
# "file:MuTriggerReco_1_1.root"


    )
)

#import os
#dirname = "/data4/Skimming/SkimResults/135"
#dirlist = os.listdir(dirname)
#basenamelist = os.listdir(dirname + "/")
#for basename in basenamelist:
#                    process.source.fileNames.append("file:" + dirname + "/" + basename)
#                    print "Number of files to process is %s" % (len(process.source.fileNames))


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 10000


process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('START3X_V21::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("ElectroWeakAnalysis.Skimming.patCandidatesForZMuMuSubskim_cff")

process.selectedPatMuons.cut = 'pt > 5. & abs(eta) < 100.0'

process.load("ElectroWeakAnalysis.Skimming.zMuMuMuonUserData")










### CandViewNtpProducer Configuration - common to all categories.

process.goodMuonsEdmNtuple = cms.EDProducer(
    "CandViewNtpProducer", 
    src=cms.InputTag("userDataMuons"),
    lazyParser=cms.untracked.bool(True),
    prefix=cms.untracked.string("Mu"),
    eventInfo=cms.untracked.bool(True),
    variables = cms.VPSet(
    cms.PSet(
    tag = cms.untracked.string("Pt"),
    quantity = cms.untracked.string("pt")
    ),
    cms.PSet(
    tag = cms.untracked.string("Eta"),
    quantity = cms.untracked.string("eta")
    ),
    cms.PSet(
    tag = cms.untracked.string("Phi"),
    quantity = cms.untracked.string("phi")
    ),
    cms.PSet(
    tag = cms.untracked.string("Q"),
    quantity = cms.untracked.string("charge")
    ),
    cms.PSet(
    tag = cms.untracked.string("Iso"),
    quantity = cms.untracked.string("userIso(3)")
    ),
    cms.PSet(
    tag = cms.untracked.string("RelIso"),
    quantity = cms.untracked.string("userIso(4)")
    ),
    cms.PSet(
    tag = cms.untracked.string("TrkIso"),
    quantity = cms.untracked.string("userIso(0)")
    ),
    cms.PSet(
    tag = cms.untracked.string("EcalIso"),
    quantity = cms.untracked.string("userIso(1)")
    ),
    cms.PSet(
    tag = cms.untracked.string("HcalIso"),
    quantity = cms.untracked.string("userIso(2)")
    ),
    cms.PSet(
    tag = cms.untracked.string("DxyFromBS"),
    quantity = cms.untracked.string("userFloat('zDau_dxyFromBS')")
    ),
    cms.PSet(
    tag = cms.untracked.string("DzFromBS"),
    quantity = cms.untracked.string("userFloat('zDau_dzFromBS')")
    ),
    cms.PSet(
    tag = cms.untracked.string("DxyFromPV"),
    quantity = cms.untracked.string("userFloat('zDau_dxyFromPV')")
    ),
    cms.PSet(
    tag = cms.untracked.string("DzFromPV"),
    quantity = cms.untracked.string("userFloat('zDau_dzFromPV')")
    ),
    cms.PSet(
    tag = cms.untracked.string("HLTBit"),
    quantity = cms.untracked.string("userFloat('zDau_HLTBit')")
    ),
    cms.PSet(
    tag = cms.untracked.string("Chi2"),
    quantity = cms.untracked.string("userFloat('zDau_Chi2')")
    ),
    cms.PSet(
    tag = cms.untracked.string("TrkChi2"),
    quantity = cms.untracked.string("userFloat('zDau_TrkChi2')")
    ),
    cms.PSet(
    tag = cms.untracked.string("SaChi2"),
    quantity = cms.untracked.string("userFloat('zDau_SaChi2')")
    ),
    cms.PSet(
    tag = cms.untracked.string("NofMuonHits"),
    quantity = cms.untracked.string("userFloat('zDau_NofMuonHits')")
    ),
    cms.PSet(
    tag = cms.untracked.string("SaNofMuonHits"),
    quantity = cms.untracked.string("userFloat('zDau_SaNofMuonHits')")
    ),  
    cms.PSet(
    tag = cms.untracked.string("NofStripHits"),
    quantity = cms.untracked.string("userFloat('zDau_NofStripHits')")
    ),
    cms.PSet(
    tag = cms.untracked.string("NofPixelHits"),
    quantity = cms.untracked.string("userFloat('zDau_NofPixelHits')")
    ),
    cms.PSet(
    tag = cms.untracked.string("TrkNofStripHits"),
    quantity = cms.untracked.string("userFloat('zDau_TrkNofStripHits')")
    ),
    cms.PSet(
    tag = cms.untracked.string("NofMuChambers"),
    quantity = cms.untracked.string("userFloat('zDau_NofMuChambers')")
    ),
    cms.PSet(
    tag = cms.untracked.string("NofMuMatches"),
    quantity = cms.untracked.string("userFloat('zDau_NofMuMatches')")
    ),
    cms.PSet(
    tag = cms.untracked.string("EnergyEm"),
    quantity = cms.untracked.string("userFloat('zDau_MuEnergyEm')")
    ),
    cms.PSet(
    tag = cms.untracked.string("GlobalMuonBit"),
    quantity = cms.untracked.string("isGlobalMuon")
    ),
    cms.PSet(
    tag = cms.untracked.string("StandAloneBit"),
    quantity = cms.untracked.string("isStandAloneMuon")
    ),
    cms.PSet(
    tag = cms.untracked.string("TrackerMuonBit"),
    quantity = cms.untracked.string("isTrackerMuon")
    ),

    
  )
 )
    





# Output module configuration
from Configuration.EventContent.EventContent_cff import *

EventContent = cms.PSet(
        outputCommands = cms.untracked.vstring(
                    )
        )

## ntpEventContent = cms.PSet(
##         outputCommands = cms.untracked.vstring(
##         "keep *_goodMuonsNtuples_*_*"
##              )
##         )

EventContent.outputCommands.extend(RECOEventContent.outputCommands)
## EventContent.outputCommands.extend(ntpEventContent.outputCommands)


EventSelection = cms.PSet(
        SelectEvents = cms.untracked.PSet(
            SelectEvents = cms.vstring(
               'muonsNtpPath')
                )
        )


process.OutputModule = cms.OutputModule("PoolOutputModule",
                                EventContent,
                                EventSelection,
                                dataset = cms.untracked.PSet(
                                  filterName = cms.untracked.string('muonsNtpPath'),
                                  dataTier = cms.untracked.string('USER')
               ),
                                                     fileName = cms.untracked.string('MuReco.root')

                                                  )


process.ntuplesOut = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('MuonsNtuple.root'),
    outputCommands = cms.untracked.vstring(
    "drop *",
    "keep *_goodMuonsEdmNtuple_*_*"
      
    )
    )



process.muonsNtpPath = cms.Path(
    process.goodMuonRecoForDimuon *
    process.userDataMuons  * 
    process.goodMuonsEdmNtuple
    )




process.outpath = cms.EndPath(process.OutputModule)

process.ntpoutpath = cms.EndPath(process.ntuplesOut)



