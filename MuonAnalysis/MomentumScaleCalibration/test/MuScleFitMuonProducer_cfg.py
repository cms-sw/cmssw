import FWCore.ParameterSet.Config as cms

process = cms.Process("MUSCLEFITMUONPRODUCER")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_1.root",
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_10.root",
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_11.root",
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_12.root",
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_13.root",
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_14.root",
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_15.root",
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_16.root",
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_17.root",
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_18.root",
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_19.root",
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_2.root",
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_20.root",
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_21.root",
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_3.root",
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_4.root",
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_5.root",
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_6.root",
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_7.root",
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_8.root",
    "rfio:/castor/cern.ch/user/d/demattia/MuScleFit/Summer09/Zmumu/Filter_Zmumu_9.root"
    )
)

process.MuScleFitMuonProducer = cms.EDProducer(
    'MuScleFitMuonProducer',
    MuonLabel = cms.InputTag("muons")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('myOutputFile.root')
)
  
process.p = cms.Path(process.MuScleFitMuonProducer)

process.e = cms.EndPath(process.out)
