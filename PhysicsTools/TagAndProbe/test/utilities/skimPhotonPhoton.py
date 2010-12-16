import FWCore.ParameterSet.Config as cms
process = cms.Process("SKIM")

#------------------------------------------
# Load standard sequences.
#------------------------------------------
process.load('FWCore.MessageService.MessageLogger_cfi')


#  Photons!!! ################ 
process.goodPhotons = cms.EDFilter("PhotonSelector",
                                   src = cms.InputTag("photons"),
                                   cut = cms.string("hadronicOverEm<0.15"
                                                    " && (superCluster.rawEnergy*sin(superCluster.position.theta)>15.0)")
                                   )


process.ZToEE = cms.EDProducer("CandViewShallowCloneCombiner",
    cut = cms.string('40 < mass < 7000'),
    decay = cms.string('goodPhotons goodPhotons'),
    checkCharge = cms.bool(False),
    
)

process.ZToEEFilter = cms.EDFilter("CandViewCountFilter",
                                    src = cms.InputTag("ZToEE"),
                                    minNumber = cms.uint32(1)
                                    )

process.ZeeSequence = cms.Sequence(process.goodPhotons *
                                    process.ZToEE)




process.mypath = cms.Path(
    process.ZeeSequence*
    process.ZToEEFilter)




#############  output module if just want to skim by HLT path ##############
process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring('mypath')
       ), 
    fileName = cms.untracked.string('mySkim.root')
)
process.p = cms.EndPath(process.out)


#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 100



#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
#############   Define the source file ###############
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Run2010A/EG/RECO/v4/000/139/195/FA470877-BE85-DF11-840C-003048F117B4.root',
        '/store/data/Run2010A/EG/RECO/v4/000/139/195/ACECA275-D885-DF11-931C-003048F11DE2.root',
        '/store/data/Run2010A/EG/RECO/v4/000/139/195/A6DB3770-D885-DF11-9A0B-003048F024DE.root',

     )
)
process.source.inputCommands = cms.untracked.vstring("keep *","drop *_MEtoEDMConverter_*_*")



