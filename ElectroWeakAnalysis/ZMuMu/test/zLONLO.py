import FWCore.ParameterSet.Config as cms

process = cms.Process("zlonlo")
process.source = cms.Source("PoolSource",
 fileNames = cms.untracked.vstring(
##    'rfio:/castor/cern.ch/user/d/degrutto/MCatNLOzToMuMum20/dimuons_1.root',
##    'rfio:/castor/cern.ch/user/d/degrutto/MCatNLOzToMuMum20/dimuons_2.root',
##    'rfio:/castor/cern.ch/user/d/degrutto/MCatNLOzToMuMum20/dimuons_3.root'
                                   
 'rfio:/castor/cern.ch/user/d/degrutto/zToMuMum20v2/dimuons_1.root',
   'rfio:/castor/cern.ch/user/d/degrutto/zToMuMum20v2/dimuons_2.root',
   'rfio:/castor/cern.ch/user/d/degrutto/zToMuMum20v2/dimuons_3.root'
  
                                   )
 )


process.evtInfo = cms.OutputModule("AsciiOutputModule")


process.TFileService = cms.Service("TFileService",
    fileName = cms.string('ZLO_10pb.root')
)



process.zHistos = cms.EDAnalyzer("ZLONLOHistogrammer",
    RecZ = cms.InputTag("dimuons"),
    genParticles = cms.InputTag("genParticles"),
    weights = cms.InputTag("genEventWeight"),                             
    nbinsMass=cms.untracked.uint32(200),
    nbinsPt=cms.untracked.uint32(200),
    nbinsAng=cms.untracked.uint32(200),
    massMax =  cms.untracked.double(200.),
    ptMax=  cms.untracked.double(200.),
    angMax = cms.untracked.double(6.),
    #parameter for the geometric acceptance
    accPtMin = cms.untracked.double(0.0),
    accMassMin = cms.untracked.double(40.0),
    accMassMax = cms.untracked.double(12000.0),                             
    accEtaMin = cms.untracked.double(0.0),
    accEtaMax = cms.untracked.double(2.5),
    isMCatNLO= cms.untracked.bool(False) 
    
)






process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True)
)
                                                  

                                                  
process.maxEvents = cms.untracked.PSet(
  input =cms.untracked.int32(19440)
)

process.path=cms.Path(process.zHistos)

process.end = cms.EndPath(process.evtInfo )



